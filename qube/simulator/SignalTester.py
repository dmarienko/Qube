import logging
import uuid
from copy import deepcopy
from datetime import timedelta
from logging import Logger
from typing import Union, Dict, List

import numpy as np
import pandas as pd

from qube.datasource.DataSource import DataSource
from qube.portfolio.PortfolioLogger import PortfolioLogger
from qube.portfolio.Position import Position
from qube.simulator.Brokerage import BrokerInfo
from qube.simulator.core import Tracker, ExecutionLogger, Terminator, SimulationResult, TradingService
from qube.simulator.utils import (dstype,
                                  recognize_datasource_structure,
                                  load_tick_price_block,
                                  dsinfo)
from qube.utils import QubeLogger
from qube.utils.DateUtils import DateUtils

VERBOSE_MODE = 'verbose'
JUPYTER_PROGRESS_LISTENER = 'jupyter_progress_listener'

_pd_ver = int(pd.__version__.split('.')[0])


class _InstrumentTrack(TradingService):
    """
    Market data holder with attached instrument position. It contains last known
    top of the book quotes update
    """

    def __init__(self, position: Position, aux=None,
                 verbose_logger: Logger = None,
                 execution_logger: ExecutionLogger = None,
                 tracker: Tracker = None
                 ):
        self.position = position
        self.verbose_logger = verbose_logger
        self.execution_logger = execution_logger
        self.time = None
        self.bid = None
        self.ask = None
        self.bid_vol = 10 ** 6
        self.ask_vol = 10 ** 6
        self.tracker: Tracker = tracker
        self.instrument = position.symbol
        self.last_trade_time = None

        # auxiliary instrument for crosses and if instrument's quoted in non-USD
        self.aux = aux
        self.is_straight = position.is_straight

        # quote data class (to avoid dict creation on every call of get_aux_quote() )
        self.quote_info = {'a_is_straight': self.aux.is_straight if self.aux is not None else False,
                           'a_bid': None,
                           'a_ask': None}

        # initialize tracker
        if tracker is not None:
            tracker.setup(self)

        # on new execution we may want to post log messages
        if self.execution_logger or self.verbose_logger:
            self.position.attach_execution_callback(self.__log_execution)

    def get_position(self) -> Position:
        return self.position

    def __log_execution(self, pos: Position, exec_time, pos_change, exec_price, comms, comment):
        if self.execution_logger:
            new_exec_record = pd.Series(name=exec_time,
                                        data=[pos.symbol, pos_change, exec_price, comms, comment],
                                        index=['instrument', 'quantity', 'exec_price', 'commissions', 'comment'])
            self.execution_logger._exec_log.append(new_exec_record)

        if self.verbose_logger:
            action = 'SLD' if pos_change < 0 else 'BOT'
            self.verbose_logger.info('%s [%s] %+d %s @ %.5f' % (action, exec_time.strftime('%d-%b-%Y %H:%M:%S.%f'),
                                                                pos_change, pos.symbol, exec_price))

    def process_trading_signal(self, signal_time, signal_value) -> float:
        """
        Process new trading signal
        """
        # if tracker is passed first we pre-process signals there
        processed_signal = self.tracker.on_signal(
            signal_time, signal_value, self.time, self.bid, self.ask, self.bid_vol, self.ask_vol
        ) if self.tracker else signal_value

        # tracker may return None so we skip this signal
        if processed_signal is not None and np.isfinite(processed_signal):
            # store last trading time in tracker
            self.last_trade_time = signal_time

            # keep last pricessed signal (not processed value)
            if self.tracker:
                self.tracker.last_signal = signal_value
                self.tracker.last_signal_time = signal_time

            # execute trade (by default we are using market orders)
            return self.position.update_position_bid_ask(
                signal_time, processed_signal, self.bid, self.ask, **self.get_aux_quote(),
                comment='', crossed_market=True
            )
        return 0

    def update_pm_pnl(self, pm_log_time, trace_log=False) -> float:
        pnl = self.position.update_pnl_bid_ask(pm_log_time, self.bid, self.ask, **self.get_aux_quote())
        # if trace_log:
        #     print('%s(%+d | %.2f) ' % (self.position.instrument, self.position.quantity, pnl), end='')
        return pnl

    def get_aux_quote(self):
        """
        Get aux instrument's quotes if presented
        """
        if self.aux is not None:
            self.quote_info['a_bid'] = self.aux.bid
            self.quote_info['a_ask'] = self.aux.ask
        return self.quote_info


class SignalTester:
    """
    Backtester for fast strategy backtest on generated signals.
    """

    def __init__(self,
                 broker_info: BrokerInfo,
                 datasource: Union[str, DataSource],
                 ds_info: dsinfo = None,
                 logger=None):
        if isinstance(datasource, str):
            self.__data_src = DataSource(datasource)
        elif isinstance(datasource, DataSource):
            self.__data_src = datasource
        else:
            raise ValueError("Datasource must be either string or instance of qube.datasource.DataSource class!")

        self.__ds_info: dsinfo = ds_info
        self.__broker_info: BrokerInfo = broker_info
        if logger is None:
            self.__logger = QubeLogger.getLogger(self.__module__)
        else:
            self.__logger = logger

    def run_signals(self, 
                    signals: pd.Series | pd.DataFrame,
                    portfolio_logger=None,
                    run_id=None,
                    progress_listener=None,
                    terminator: Terminator = None,
                    execution_logger: ExecutionLogger = None,
                    tracker: Union[Tracker, Dict[str, Tracker]] = None,
                    single_tracker_for_all=False,
                    trace_log=False,
                    exec_by_new_update=False,
                    name: str = None,
                    **kwargs) -> SimulationResult:
        """
        Execute prepared signals of specified datasource data

        TODO: write detailed doc + examples
        TODO:  progress_listener NEED to rework from func with predefined args to qube.utils.utils.IProgressListener

        :param signals: pd.DataFrame or Series within trading signals to test
        :param portfolio_logger: portfolio logging object
        :param run_id: simulation ID
        :param terminator: termination provider
        :param execution_logger: executions logger
        :param progress_listener:
        :param name: simulation custom name
        :param trackers: trackers for signals (may be dict of trackers for each symbol)
        :param single_tracker_for_all: if true and trackers is just object simulator will use it for tracking whole portfolio
        :kwargs param exec_by_new_update: execution by new next price
        :kwargs param warnings: if False all warnings will be disabled
        :return: struct with backtest data
        """

        self.__logger.setLevel(logging.INFO if kwargs.get(VERBOSE_MODE, True) else logging.WARN)

        # we will use default portfolio logger if not specified
        if portfolio_logger is None:
            portfolio_logger = PortfolioLogger()

        # let's have unique identifier for the run_signals if it's not specified
        sim_start_time = pd.Timestamp.now()
        run_id = 'run_id_%s' % str(uuid.uuid4()) if not run_id else run_id

        if signals.empty:
            self.warn(f'Simulation {run_id} gets empty signals data frame !')
            return SimulationResult(
                name, broker=self.__broker_info.__class__.__name__,
                portfolio_logger=portfolio_logger,
                execution_logger=execution_logger,
                sim_start_time=sim_start_time, sim_start=None, sim_end=None,
                instruments=[], tracks=[], number_processed_signals=0
            )

        signals = SignalTester.validate_and_format_signals(signals)
        sim_start = signals.index[0].to_pydatetime()
        sim_end = signals.index[-1].to_pydatetime()
        self.info('Running simulation on interval [%s : %s]' % (DateUtils.get_as_string(sim_start),
                                                                DateUtils.get_as_string(sim_end)))

        # get instruments list
        instruments = signals.columns.tolist()
        if not instruments:
            raise ValueError("Couldn't find any named columns in signals data frame !")

        if terminator and terminator.is_terminated():
            return portfolio_logger

        # first time we try to get info about specified datasource
        if self.__ds_info is None:

            # getting datasource info
            self.__ds_info = recognize_datasource_structure(self.__data_src, instruments, sim_start, sim_end, self.__logger)

            if self.__ds_info.type == dstype.UNKNOWN:
                raise ValueError("Can't recognize datasource structure: it doesn't provide OHLC, bid/ask or price data")

        # create Positions array, so positions[0] is first instrument from positions frame etc
        positions = [self.__broker_info.create_position(instr) for instr in instruments]

        # attach positions to portfolio logger
        portfolio_logger.add_positions_for_watching(positions)

        # if self.__ds_info.type == dstype.OHLC:
        #     exec_by_new_update = True

        # process signals on tick data
        tracks = self.__run_simulation_on_ticks(
            positions, instruments, signals, portfolio_logger, run_id,
            progress_listener, terminator, execution_logger,
            tracker=tracker, single_tracker_for_all=single_tracker_for_all,
            exec_by_new_update=exec_by_new_update,
            trace_log=trace_log, **kwargs
        )

        return SimulationResult(
            name,
            broker=self.__broker_info.__class__.__name__,
            portfolio_logger=portfolio_logger,
            execution_logger=execution_logger,
            sim_start_time=sim_start_time,
            sim_start=sim_start, sim_end=sim_end,
            instruments=instruments,
            tracks=tracks,
            number_processed_signals=len(signals)
        )

    @staticmethod
    def validate_and_format_signals(signals):
        # if passed dictionary of signals
        if isinstance(signals, dict):
            if not all([isinstance(s, pd.Series) for s in signals.values()]):
                raise ValueError("All signals containers from dict must be instances of Series")
            signals = pd.concat([pd.Series(data=s.data, index=s.index, name=n) for (n, s) in signals.items()], axis=1)

        # if passed collection of signals
        if isinstance(signals, (list, tuple)):
            if not all([isinstance(s, (pd.DataFrame, pd.Series)) for s in signals]):
                raise ValueError("All signals containers must be instances of DataFrame or Series")

            if any([s.name is None for s in signals if isinstance(s, pd.Series)]):
                raise ValueError("All signals passed as Series object must have name")

            signals = pd.concat(signals, axis=1)

        # all signals must be series or dataframe
        if not isinstance(signals, (pd.DataFrame, pd.Series)):
            raise ValueError("Signals argument must be instance of either DataFrame or Series classes")

        # make dataframe from series for unify access to signals
        if isinstance(signals, pd.Series):
            if not signals.name:
                raise ValueError("Name must be specified for signal series !")
            signals = pd.DataFrame(signals, columns=[signals.name])

        # check how signals is indexed
        if signals.empty:
            raise ValueError("Signals can't be empty !")

        # check how signals is indexed
        if not isinstance(signals.index, pd.DatetimeIndex):
            raise ValueError("Signal dataframe must be indexed by pandas.DatetimeIndex object !")

        return signals

    def __run_simulation_on_ticks(self, pos_trackings: List[Position], instruments, signals: pd.DataFrame,
                                  portfolio_logger,
                                  run_id, progress_listener, terminator=None,
                                  execution_logger=None,
                                  tracker=None,
                                  single_tracker_for_all=False,
                                  exec_by_new_update=False,
                                  fill_latency_msec=timedelta(milliseconds=0),
                                  trace_log=False, warnings=True,
                                  **kwargs) -> List[_InstrumentTrack]:
        """
        Test trading signals on tick data. For OHLC tester see __OHLC_signals_runner() method.

        :param pos_trackings:
        :param instruments:
        :param signals:
        :param portfolio_logger:
        :param run_id:
        :param progress_listener:
        :param terminator:
        :param execution_logger:
        :param kwargs:
        :return: list of instruments trackings
        """
        aux_instrs_names = set([t.aux_instrument for t in pos_trackings if t.aux_instrument is not None])

        if trace_log and len(aux_instrs_names) > 0:
            print(' -> Auxiliary instruments: %s' % ','.join(aux_instrs_names))

        verbose_logger = self.__logger if kwargs.get(VERBOSE_MODE) else None
        jupyter_progress_listener = kwargs.get(JUPYTER_PROGRESS_LISTENER, None)

        def _copy_or_dispatch_tracker(tracker: Tracker, instrument: str, is_aux: bool) -> Union[Tracker, None]:
            new_tracker = tracker
            if isinstance(tracker, Tracker):
                new_tracker = tracker.__on_tracker_cloning__(instrument, is_aux)
                new_tracker = new_tracker if new_tracker is not None else deepcopy(tracker)
            return new_tracker

        # create tracks for every symbol
        if isinstance(tracker, dict):
            aux_instr = {
                ai: _InstrumentTrack(
                    self.__broker_info.create_position(ai),
                    None, verbose_logger, execution_logger, tracker.get(ai, None)
                ) for ai in aux_instrs_names}
            tob_trackings = [
                _InstrumentTrack(
                    t, aux_instr.get(t.aux_instrument), verbose_logger, execution_logger,
                    tracker.get(t.symbol, None)
                ) for t in pos_trackings]
        else:
            aux_instr = {
                ai: _InstrumentTrack(
                    self.__broker_info.create_position(ai), None,
                    verbose_logger, execution_logger, 
                    tracker if single_tracker_for_all else _copy_or_dispatch_tracker(tracker, ai, True)
                ) for ai in aux_instrs_names}
            tob_trackings = [
                _InstrumentTrack(
                    t, aux_instr.get(t.aux_instrument), verbose_logger, execution_logger,
                    tracker if single_tracker_for_all else _copy_or_dispatch_tracker(tracker, t.symbol, True)
                ) for t in pos_trackings]

        instruments += aux_instrs_names
        tob_md_update = tob_trackings + [aux_instr[i] for i in aux_instrs_names]

        # collect spreads info
        bid_ask_spreads = {t.instrument: t.position.half_spread for t in tob_md_update}

        log_freq_msec = portfolio_logger._log_freq_msec
        last_pm_log_time = None

        # iterating indexes: i_signals - indexing signals, i_prices - indexing prices
        i_signals, i_prices = 0, 0

        # buffer for prices
        prices_buffer = pd.DataFrame()
        prices_buffer_matrix = prices_buffer.values
        is_prices_buffer_empty = True
        prices_buffer_len = 0
        prices_buffer_index = []

        # iterating through list is much faster. conv. to list!
        signals_timeline = pd.to_datetime(signals.index, infer_datetime_format=True) if _pd_ver <= 1 else pd.to_datetime(signals.index)

        signals_matrix = signals.values
        signals_length = len(signals)

        # number of days being loaded per time (not limited for daily/weekly timeframes)
        ld_blk = self.__ds_info.load_block_amnt

        # price's start loading time - first signal's time and rewind it to start of nearest minute
        loading_price_data_start = DateUtils.round_time(
            signals_timeline[0] - self.__ds_info.freq - timedelta(seconds=60), log_freq_msec)

        # number of columns per single instrument
        # may be: ('ask', 'bid'), ('ask', 'bid', 'askvol', 'bidvol'),
        #         ('ask', 'bid', 'is_real'), ('ask', 'bid', 'askvol', 'bidvol', 'is_real')
        n_cols = None
        dirty_flag_idx = -1

        # start iterating: we have to process all signals
        #  for that we are running 2 iterators - one through signals another through prices
        while i_signals < signals_length:
            if terminator and terminator.is_terminated():
                self.info("SignalTest %s was terminated by user's command" % run_id)
                break

            self.__notify_progress_listener(progress_listener, portfolio_logger, run_id,
                                            i_signals, signals_length, jupyter_progress_listener)
            signal_time = signals_timeline[i_signals]

            if is_prices_buffer_empty:
                # load prices block from datasource
                prices_buffer = load_tick_price_block(self.__data_src, self.__ds_info, instruments,
                                                      loading_price_data_start,
                                                      bid_ask_spreads,
                                                      exec_by_new_update,
                                                      logger=self.__logger, broker_info=self.__broker_info)
                prices_buffer_matrix = prices_buffer.values
                is_prices_buffer_empty = prices_buffer.empty
                prices_buffer_len = len(prices_buffer)

                # will start iterating prices from begin of new block
                i_prices = 0

                if is_prices_buffer_empty:

                    while i_signals < signals_length and signals_timeline[
                        i_signals] < loading_price_data_start + ld_blk:
                        # if no prices for current signal time then skip this signal
                        if warnings:
                            self.warn("Can't find prices for signals at time [%s]" % str(loading_price_data_start))
                        i_signals += 1
                    if i_signals < signals_length:
                        loading_price_data_start = loading_price_data_start + ld_blk
                    continue

                prices_buffer_index = prices_buffer.index.to_pydatetime()

                # how many columns per instrument
                if n_cols is None:
                    n_cols = prices_buffer.shape[1] // len(instruments)
                    if n_cols == 3 or n_cols == 5:
                        dirty_flag_idx = n_cols - 1
                    if n_cols < 2 or n_cols > 5:
                        raise ValueError("Wrong number of columns in loaded price data: %s" %
                                         ','.join(set(prices_buffer.columns.get_level_values(1))))

            # time of price data
            prices_time = prices_buffer_index[i_prices]

            # is dirty update skip not real updates (on Hi/Low data from ohlc generated ticks)
            if dirty_flag_idx > 0 and not prices_buffer_matrix[i_prices, dirty_flag_idx]:
                # todo: here we need to update trackers if presented
                self._update_tick_tracked_positions(tob_md_update, i_prices, n_cols,
                                                    prices_buffer_matrix, prices_time, True)
                i_prices += 1
                continue

            # check if we can 'execute' signals
            # we change position only after fill_latency_msec is passed after signal's time so introducing
            # latency emulation
            if prices_time >= signal_time + fill_latency_msec:

                if exec_by_new_update:
                    self._update_tick_tracked_positions(tob_md_update, i_prices, n_cols, prices_buffer_matrix,
                                                        prices_time, False)

                # execute signals
                for (instr_idx, track) in enumerate(tob_trackings):
                    new_signal = signals_matrix[i_signals, instr_idx]

                    # here we want to handle str signals as some additional information updates
                    # we do all the work here to speed up by avoid calling method on _InstrumentTrack object
                    if isinstance(new_signal, str) and track.tracker:
                        track.tracker.on_info(signal_time, new_signal)

                    # otherwise we just process it as usual postion
                    elif not np.isnan(new_signal) and (new_signal != track.position.quantity):
                        track.process_trading_signal(signal_time, new_signal)

                # take next signal
                i_signals += 1
            else:
                self._update_tick_tracked_positions(tob_md_update, i_prices, n_cols, prices_buffer_matrix, prices_time,
                                                    False)

                i_prices += 1

                # if we processed all available price data try to load next price block
                if i_prices >= prices_buffer_len:
                    i_prices = 0
                    # take next time as loading start (+1 msec to avoid loading last quote)
                    loading_price_data_start = prices_time + timedelta(milliseconds=1)
                    prices_buffer = pd.DataFrame()
                    prices_buffer_matrix = prices_buffer.values
                    is_prices_buffer_empty = True
                    prices_buffer_len = 0
                    prices_buffer_index = []

            if last_pm_log_time is None:
                last_pm_log_time = DateUtils.round_time(prices_time, log_freq_msec)

            # check if it's time for logging according logging frequency
            # we update all positions by last known market data by this bar
            # so if we get quoted at 13:59:57, then at 14:00:10, we consider 14:00:00 as end of the logging period
            # and update positions PnL by quotes from 13:59:57 (last quote before period's close)
            if prices_time >= last_pm_log_time + timedelta(milliseconds=log_freq_msec):
                pm_log_time = DateUtils.round_time(prices_time, log_freq_msec)
                self.__notify_pm_logger_ticks(portfolio_logger, pm_log_time, tob_trackings, trace_log)
                last_pm_log_time = pm_log_time

        # at the end we seek for last logging period and update PM logger
        for i in range(i_prices, prices_buffer_len):
            prices_time = prices_buffer_index[i]
            if prices_time >= last_pm_log_time + timedelta(milliseconds=log_freq_msec):
                pm_time = DateUtils.round_time(prices_time, log_freq_msec)
                self.__notify_pm_logger_ticks(portfolio_logger, pm_time, tob_trackings, trace_log)
                break

            is_service_update = dirty_flag_idx > 0 and not prices_buffer_matrix[i_prices, dirty_flag_idx]
            self._update_tick_tracked_positions(tob_md_update, i, n_cols, prices_buffer_matrix, prices_time,
                                                is_service_update)

        self.__notify_progress_listener(progress_listener, portfolio_logger, run_id, i_signals, signals_length,
                                        jupyter_progress_listener)

        return tob_trackings

    def _update_tick_tracked_positions(self, tob_md_update, i_prices, n_cols, prices_buffer_matrix, prices_time,
                                       is_service_data):
        # update all tracked positions by current market's TOB data
        for (instr_idx, track) in enumerate(tob_md_update):

            # skip data getting/setting when we don't need that
            if is_service_data and not track.tracker:
                continue

            # get new TOB data
            i_idx = instr_idx * n_cols

            bid = prices_buffer_matrix[i_prices, i_idx]
            if np.isnan(bid):
                # 2022-12-28: we shouldn't skip other instruments - so let's check them too
                continue

            ask = bid
            if n_cols > 1:
                ask = prices_buffer_matrix[i_prices, i_idx + 1]

            bid_vol = 10 ** 6  # use large bid|ask volumes if they are not presented in data
            ask_vol = 10 ** 6
            if n_cols >= 4:  # also volumes are presented
                bid_vol = prices_buffer_matrix[i_prices, i_idx + 2]
                ask_vol = prices_buffer_matrix[i_prices, i_idx + 3]

            # update market data

            # huge optimization by eliminating method invocation
            if not is_service_data:
                track.time = prices_time
                track.bid = bid
                track.ask = ask
                track.bid_vol = bid_vol
                track.ask_vol = ask_vol

            # send update
            if track.tracker:
                track.tracker.update_market_data(track.position.symbol,
                                                 prices_time, bid, ask, bid_vol, ask_vol, is_service_data)

    def __notify_progress_listener(self, progress_listener, current_portfolio_logger, run_identifier,
                                   current, total, jupyter_progress_listener=None):
        total_num = total if isinstance(total, int) else len(total)
        if progress_listener:
            progress_listener(run_identifier, current_portfolio_logger, current, total_num)
        if jupyter_progress_listener:
            progress = 100 * current / total_num
            jupyter_progress_listener(progress)

    def __notify_pm_logger_ticks(self, portfolio_logger, pm_log_time, tob_trackings, trace_log):
        # if trace_log: print('LOG [%s] ' % self.__tm_log_fmt(pm_log_time), end='')

        # do update PnL data for PM logger
        [track.update_pm_pnl(pm_log_time, trace_log) for track in tob_trackings if track.bid is not None]

        # notify logger
        portfolio_logger.notify_update()

        # if trace_log: print()

    def warn(self, msg):
        self.__logger.warning(msg)

    def info(self, msg):
        self.__logger.info(msg)

    def err(self, msg):
        self.__logger.error(msg)
