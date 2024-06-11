from typing import List, Union, Dict

import numpy as np
import pandas as pd

from qube.portfolio.Position import Position
from qube.portfolio.performance import split_cumulative_pnl, portfolio_stats
from qube.series.BarSeries import BarSeries
from qube.simulator.utils import generate_simulation_identificator
from qube.utils.utils import dict2struct

DB_SIMULATION_RESULTS = "simdb"


class TradingService:
    def __init__(self):
        self.bid = None
        self.ask = None
        self.last_trade_time = None

    def get_position(self) -> Position:
        pass

    def get_aux_quote(self) -> Dict:
        pass


class Tracker:
    """
    Signals tracker used for preprocessing signals before pushing them into backtesting engine
    """

    def setup(self, service: TradingService):
        """
        Method must be called before this class can be used
        """
        # some service information
        self._position: Position = service.get_position()
        self._instrument: str = self._position.symbol
        self._service: TradingService = service

        # last processed signal
        self.last_signal = None
        self.last_signal_time = None

        # holder for custom OHLC series
        self._ohlc_series = dict()

        # cutom initialization
        self.initialize()

    def initialize(self):
        pass

    def get_tracker(self, instrument: str):
        # todo: when it needs to get access to different tracker
        # something like that: self._service.tracker[instrument]
        return

    def trade(self, trade_time, quantity, comment="", exact_price=None, market_order=True):
        """
        Process trade from on_quote
        :param trade_time: time when trade occured
        :param quantity: signed postion size (negative for sell, zero - flat position)
        :param comment: user custom comment (for execution log)
        :param exact_price: pass exact price if we need to register this trade at accurate price.
                            If passed None (default) last quote will be used.
        :param market_order: if true it used market order (crossed the spread)
        """
        pnl = 0
        if np.isfinite(quantity):
            pnl = self._position.update_position_bid_ask(
                trade_time,
                quantity,
                self._service.bid,
                self._service.ask,
                exec_price=exact_price,
                **self._service.get_aux_quote(),
                comment=comment,
                crossed_market=market_order,
            )

            # set last trade time
            self._service.last_trade_time = trade_time
        return pnl

    def get_ohlc_series(self, timeframe: Union[str, pd.Timedelta], series_length=np.inf):
        if timeframe not in self._ohlc_series:
            series = BarSeries(timeframe, max_series_length=series_length)
            self._ohlc_series = {timeframe: series}
        else:
            series = self._ohlc_series[timeframe]
        return series

    def _update_series(self, quote_time, bid, ask, bid_size, ask_size):
        for s in self._ohlc_series.values():
            s.update_by_data(quote_time, bid, ask, bid_size, ask_size)

    def update_market_data(self, instrument: str, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs):
        self._update_series(quote_time, bid, ask, bid_size, ask_size)

        # call handler if it's not service quote
        if not is_service_quote:
            self.on_quote(quote_time, bid, ask, bid_size, ask_size, **kwargs)

    def on_info(self, info_time, info_data, **kwargs):
        """
        Callback on new information update (market regime/econ news etc)
        """
        pass

    def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
        """
        Callback on new quote (service quotes will be skipped)
        """
        pass

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        """
        Callback on new signal
        """
        return signal_qty

    def statistics(self) -> Dict:
        """
        Custom statistic generator
        """
        return None

    def __on_tracker_cloning__(self, instrument: str, is_aux=False) -> Union["Tracker", None]:
        """
        It's possible to override this method to control what's returned on cloning this tracker.
        For example in case when we want to control all cloned trackers in parent one for multi assets portfolio
        """
        return None

    def __repr__(self):
        """
        Tracker representation
        """
        import inspect

        r = []
        for a, p in inspect.signature(self.__init__).parameters.items():
            if not a.startswith("debug") and hasattr(self, a):
                v = getattr(self, a)
                if p.default != v:
                    r.append(f"{a}={repr(v)}")
        return f'{str(self.__class__.__name__)}({",".join(r)})'


class ExecutionLogger:
    def __init__(self):
        # we use raw list for appending log records instead pandas DataFrame
        self._exec_log = list()

    def get_execution_log(self):
        return pd.DataFrame(self._exec_log).sort_index()


class Terminator:
    def __init__(self):
        self.__terminated = False

    def is_terminated(self):
        return self.__terminated

    def terminate(self):
        self.__terminated = True
        self.on_terminate()

    def on_terminate(self):
        pass


class SimulationResult:
    """
    Simulation results storage
    """

    def __init__(
        self,
        name,
        broker,
        portfolio_logger,
        execution_logger,
        sim_start_time,
        sim_start,
        sim_end,
        instruments,
        tracks,
        number_processed_signals,
    ):
        self.name = name
        self.id = generate_simulation_identificator("", broker, sim_start_time)
        # we keep non-cumulative (PnL/Commissions) portfolio here
        portf_log = portfolio_logger.get_portfolio_log()
        self.portfolio = split_cumulative_pnl(portf_log) if not portf_log.empty else pd.DataFrame()
        self.executions = execution_logger.get_execution_log() if execution_logger is not None else None
        self.simulation_start_time = sim_start_time
        self.simulation_time = pd.Timestamp.now() - sim_start_time
        self.start = sim_start
        self.end = sim_end
        self.instruments = instruments
        self.trackers: Dict[str, Tracker] = {
            t.instrument: repr(t.tracker) if t.tracker is not None else None for t in tracks
        }
        self.trackers_stat = {t.instrument: t.tracker.statistics() if t.tracker is not None else None for t in tracks}
        self.number_processed_signals = number_processed_signals

        if isinstance(instruments, (list, tuple, set)):
            _symbols_list = ",".join(instruments) if len(instruments) < 5 else f"{len(instruments)}"
        else:
            _symbols_list = instruments

        self.description = (
            f"Interval: {sim_start} ~ {sim_end} | "
            f"Time: {str(self.simulation_time)} | "
            f"Signals: {number_processed_signals} | "
            f"Execs: {len(self.executions) if self.executions is not None else '???'} | "
            f"Symbols: {_symbols_list}"
        )

    def performance(
        self,
        init_cash,
        risk_free=0,
        rolling_sharpe_window=252,
        account_transactions=True,
        margin_call_level=0.33,
        drop_margin_call=True,
        performance_statistics_period=365,
    ):
        """
        Calculate simulation potfolio performance
        If drop_margin_call is set (default value) then it returns Sharpe=-np.inf if equity has margin call event.
        :param init_cash: invested deposit
        :param risk_free: risk-free (0 by default)
        :param rolling_sharpe_window: rolling Sharpe window in days (252)
        :param account_transactions: take in account transaction costs (False)
        :param margin_call_level: level of margin call (33%)
        :param drop_margin_call: if true it set -inf to Sharpe ratio
        :param performance_statistics_period: annualization period for performance statistics (default 252)
        :return: portfolio performance statistics (as mstruct)
        """
        portfolio = self.portfolio

        # quickly calculate equity first to find out if margin call was issued
        rets = self.returns(account_transactions=account_transactions)
        equity = rets.cumsum() + init_cash

        # now we must take in account only data where equity is > minimal required margin
        below_mc = equity[equity <= init_cash * margin_call_level]
        if not below_mc.empty:
            portfolio = portfolio[: below_mc.index[0]]

        stats = dict2struct(
            portfolio_stats(
                portfolio,
                init_cash,
                risk_free=risk_free,
                rolling_sharpe_window=rolling_sharpe_window,
                account_transactions=account_transactions,
                performance_statistics_period=performance_statistics_period,
                benchmark=None,
            )
        )

        # reset Sharpe to -inf if MC observed
        if not below_mc.empty and drop_margin_call:
            stats.sharpe = -np.inf

        return stats

    def equity(self, account_transactions=True):
        """
        Simulation equity
        :param account_transactions: True if we want to take in account trasaction costs (default False)
        """
        return self.returns(account_transactions=account_transactions).cumsum()

    def returns(self, resample=None, account_transactions=True):
        """
        Simulation returns series

        :param resample: resample timeframe (for example 1D) defualt (None)
        :param account_transactions: True if we want to take in account trasaction costs (default False)
        """
        rets = self.portfolio.filter(regex=".*_PnL").sum(axis=1)

        # if it's asked to calculate commissions
        if account_transactions:
            rets -= self.portfolio.filter(regex=".*_Commissions").sum(axis=1)

        return rets.resample(resample).sum() if resample is not None else rets

    def trackers_stats(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.trackers_stat, orient="index")

    def __repr__(self):
        desc = self.description.replace(" | ", "\n\t")
        return f" Simulation {self.name}.{self.id} \n\t{desc}"
