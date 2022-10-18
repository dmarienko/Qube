from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Union, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from qube.learn.core.base import MarketDataComposer, SingleInstrumentComposer, PortfolioComposer
from qube.portfolio.commissions import TransactionCostsCalculator, ZeroTCC
from qube.simulator.backtester import backtest
from qube.simulator.core import Tracker, SimulationResult, DB_SIMULATION_RESULTS
from qube.simulator.multiproc import Task, RunningInfoManager
from qube.utils.ui_utils import red, green, yellow, blue
from qube.utils.utils import mstruct, runtime_env
from qube.utils.QubeLogger import getLogger

_LOGGER = getLogger('Simulator')

_has_method = lambda obj, op: callable(getattr(obj, op, None))


class _Types(Enum):
    UKNOWN = 'unknown'
    LIST = 'list'
    TRACKER = 'tracker'
    SIGNAL = 'signal'
    ESTIMATOR = 'estimator'
    TRACKED_ESTIMATOR = 'tracked_estimator'


def _type(obj) -> _Types:
    if obj is None:
        t = _Types.UKNOWN
    elif isinstance(obj, (list, tuple)):
        t = _Types.LIST
    elif isinstance(obj, Tracker):
        t = _Types.TRACKER
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        t = _Types.SIGNAL
    elif isinstance(obj, (Pipeline, BaseEstimator)):
        # generator now can have a tracker
        if _has_method(obj, 'tracker'):
            t = _Types.TRACKED_ESTIMATOR
        else:
            t = _Types.ESTIMATOR
    elif isinstance(obj, dict):
        # when tracker has a setup for each intrument {str -> Tracker}
        if all([isinstance(k, str) & isinstance(v, Tracker) for k, v in obj.items()]):
            t = _Types.TRACKER
        else:
            t = _Types.UKNOWN
    else:
        t = _Types.UKNOWN
    return t


def start_stop_sigs(data: Dict[pd.DataFrame, Dict], start=None, stop=None):
    """
    Generate stub signals (NaNs mainly for backtester progress)
    """
    r = None

    if stop is not None:
        try:
            stop = str(pd.Timestamp(start) + pd.Timedelta(stop))
        except:
            pass

    for i, d in data.items():
        start = d.index[0] if start is None else start
        stop = d.index[-1] if stop is None else stop

        d_sel = d[start:stop]
        if d_sel.empty:
            raise ValueError(f">>> There is no '{i}' historical data for period {start} : {stop} !")

        dx = max(len(d_sel) // 99, 1)
        ix = d_sel.index[::dx]
        last_idx = d_sel.index[-1]
        if last_idx not in ix:
            ix = ix.append(pd.DatetimeIndex([last_idx]))
        r = pd.concat((r, pd.Series(np.nan, ix, name=i)), axis=1)
    return r


class SimSetup:
    def __init__(self, signals, trackers, experiment_name, estimator_portfolio_composer: str, used_data: str):
        self.signals = signals
        self.signal_type: _Types = _type(signals)
        self.trackers = trackers
        self.name = experiment_name
        self.estimator_portfolio_composer = estimator_portfolio_composer
        self.used_data = used_data

    def _check_is_fitted(self, estimator):
        if isinstance(estimator, MarketDataComposer):
            if not estimator.fitted_predictors_:
                raise NotFittedError(f"MarketDataComposer '{type(estimator).__name__}' is not fitted ")

        check_is_fitted(estimator)

    def get_signals(self, data: Union[pd.DataFrame, pd.Series, Dict],
                    start: Union[str, pd.Timestamp, None],
                    stop: Union[str, pd.Timestamp, None],
                    fit_stop: Union[str, pd.Timestamp, None]):
        sx = self.signals

        # - unknown type of signals
        if sx is None or self.signal_type == _Types.UKNOWN:
            return start_stop_sigs(data, start, stop)

        # - if we got estimator / tracked estimator here
        if self.signal_type == _Types.ESTIMATOR or self.signal_type == _Types.TRACKED_ESTIMATOR:

            # - check if we need to wrap estimator into MarketDataComposer
            if not isinstance(sx, MarketDataComposer) and self.estimator_portfolio_composer is not None:

                # - when we want to process one by one
                if self.estimator_portfolio_composer == 'single':
                    sx = SingleInstrumentComposer(sx, column=self.used_data)

                # - when we want to process as portfolio
                elif self.estimator_portfolio_composer == 'portfolio':
                    sx = PortfolioComposer(sx, column=self.used_data)

                else:
                    raise ValueError(f"Unsupported estimator composer: {self.estimator_portfolio_composer}")

            # - let's check if this estimator is fitted
            try:
                self._check_is_fitted(sx)
            except NotFittedError as err:
                # - try to fit predictor
                _LOGGER.info(f'Fitting estimator {type(sx).__name__} for {start} : {fit_stop} ... ')
                sx = sx.for_interval(start, fit_stop).fit(data, None)

            # - if we need to select some date range
            if isinstance(sx, MarketDataComposer):
                sx = sx.for_interval(start, stop)

            # get signals
            sx = sx.predict(data)

        _z = slice(start, stop) if start is not None and stop is not None else None
        return sx[_z] if _z is not None else sx

    def __repr__(self):
        return f'{self.name}.{self.signal_type}.[{repr(self.trackers) if self.trackers is not None else "-"}]'


def _is_signal_or_generator(obj):
    return _type(obj) in [_Types.SIGNAL, _Types.ESTIMATOR]


def _is_generator(obj):
    return _type(obj) == _Types.ESTIMATOR


def _is_generator_with_tracker(obj):
    return _type(obj) == _Types.TRACKED_ESTIMATOR


def _is_tracker(obj):
    return _type(obj) == _Types.TRACKER


def _recognize(setup: Union[Dict, List, Tuple, pd.DataFrame, pd.Series, BaseEstimator, Pipeline],
               name: str,
               portfolio_composer: str,
               used_data: str,
               **kwargs) -> List[SimSetup]:
    r = list()

    if isinstance(setup, dict):
        for n, v in setup.items():
            r.extend(_recognize(
                v, name + '/' + n, portfolio_composer=portfolio_composer, used_data=used_data, **kwargs
            ))

    elif isinstance(setup, (list, tuple)):
        if len(setup) == 2 and _is_signal_or_generator(setup[0]) and _is_tracker(setup[1]):
            r.append(SimSetup(setup[0], setup[1], name, portfolio_composer, used_data))
        else:
            for j, s in enumerate(setup):
                r.extend(_recognize(
                    s, name + '/' + str(j), portfolio_composer=portfolio_composer, used_data=used_data, **kwargs)
                )

    elif _is_tracker(setup):
        r.append(SimSetup(None, setup, name, portfolio_composer, used_data))

    elif isinstance(setup, (pd.DataFrame, pd.Series)):
        r.append(SimSetup(setup, None, name, portfolio_composer, used_data))

    elif _is_generator(setup):
        r.append(SimSetup(setup, None, name, portfolio_composer, used_data))

    elif _is_generator_with_tracker(setup):
        r.append(SimSetup(setup, setup.tracker(**kwargs), name, portfolio_composer, used_data))

    return r


def _proc_run(s: SimSetup, data, start, stop, fit_stop, broker, spreads, progress, tcc: TransactionCostsCalculator):
    """
    TODO: need to be running in separate process
    """
    b = backtest(
        s.get_signals(data, start, stop, fit_stop),
        data, broker, spread=spreads, name=s.name, execution_logger=True,
        trackers=s.trackers, progress=progress,
        tcc=tcc
    )
    return b


@dataclass
class MultiResults:
    """
    Store multiple simulations results
    """
    results: List[SimulationResult]
    project: str
    broker: str
    start: Union[str, pd.Timestamp]  # simulation / fit start
    stop: Union[str, pd.Timestamp]  # simulation end
    fit_stop: Union[str, pd.Timestamp]  # fit end (for estimators)

    def __add__(self, other):
        if not isinstance(other, MultiResults):
            raise ValueError(f"Don't know how to add {type(other)} data to results !")

        if self.project != other.project:
            raise ValueError(f"You can't add results from another project {other.project}")

        brok = self.broker if self.broker == other.broker else f'{self.broker},{other.broker}'
        return MultiResults(self.results + other.results, self.project, brok, self.start, self.stop, self.fit_stop)

    def __getitem__(self, key):
        s = self.results[key]
        return MultiResults(s if isinstance(s, list) else [s], self.project, self.broker, self.start, self.stop,
                            self.fit_stop)

    def report(self, init_cash=0, risk_free=0.0, margin_call_pct=0.33, only_report=False,
               only_positive=False, account_transactions=True) -> pd.DataFrame:
        import matplotlib.pyplot as plt
        rs = self.results

        def _fmt(x, f='.2f'):
            xs = f'%{f}' % x
            return green(xs) if x >= 0 else red(xs)

        # max simulation name length for making more readable report
        max_len = max([len(x.name) for x in rs if x.name is not None]) + 1

        # here we will collect report's data
        rpt = {}

        for k, _r in enumerate(rs):
            eqty = init_cash + _r.equity(account_transactions=account_transactions)
            o_num = f'{k:2d}'
            print(f'{yellow(o_num)}: {blue(_r.name.ljust(max_len))} : ', end='')

            # skip negative results
            if only_positive and eqty[-1] < 0:
                print(red('[SKIPPED]'))
                continue

            if init_cash > 0:
                prf = _r.performance(init_cash, risk_free, margin_call_level=margin_call_pct,
                                     account_transactions=account_transactions)
                n_execs = len(_r.executions) if _r.executions is not None else 0

                rpt[_r.name] = {
                    'sharpe': prf.sharpe, 'sortino': prf.sortino, 'cagr': 100 * prf.cagr,
                    'dd': prf.mdd_usd, 'dd_pct': prf.drawdown_pct,
                    'gain': eqty[-1] - eqty[0], 'number_executions': n_execs,
                    'comm': prf.broker_commissions,
                }

                print(
                    f'Sharpe: {_fmt(prf.sharpe)} | Sortino: {_fmt(prf.sortino)} | CAGR: {_fmt(100 * prf.cagr)} | '
                    f'DD: ${_fmt(prf.mdd_usd)} ({_fmt(prf.drawdown_pct)}%) | '
                    f'Gain: ${_fmt(eqty[-1] - eqty[0])} | Execs: {n_execs} | Comm: {_fmt(prf.broker_commissions)}[{"inc" if account_transactions else "noincl"}]',
                    end=''
                )

            if not only_report:
                plt.plot(eqty, label=_r.name)

            print(yellow('[OK]'))

        if not only_report:
            plt.title(f'Comparison simualtions for {self.project} @ {self.broker}')
            plt.legend()

        # return report as dataframe
        df = pd.DataFrame.from_dict(rpt, orient='index')
        df.index.name = 'Name'
        return df.reset_index()


class __ForeallProgress:
    def __init__(self, n_sims, descr='backtest'):
        if runtime_env() == 'notebook':
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        self.p = tqdm(total=100 * n_sims, unit_divisor=1, unit_scale=1, unit=' signals', desc=descr)
        self.sim = 0
        self.i_in_sim = 0

    def set_descr(self, descr):
        self.p.desc = descr

    def close(self):
        self.p.update(self.p.total - self.p.n)
        self.p.close()

    def __call__(self, i, label=None):
        if i < self.i_in_sim and i < 10:
            self.sim += 1

        d = self.sim * 100 + i - self.p.n
        if d > 0:
            self.p.update(d)

        self.i_in_sim = i


def simulation(setup, data, broker, project='', start=None, stop=None, fit_stop=None,
               spreads: Union[Dict, float] = 0,
               portfolio_composer: str = 'single',
               used_data: str = 'close',
               tcc: TransactionCostsCalculator = None) -> MultiResults:
    """
    Simulate different setups
    """
    sims = _recognize(setup, project, portfolio_composer, used_data)
    results = []
    progress = __ForeallProgress(len(sims))

    for i, s in enumerate(sims):
        # print(s)
        if True:
            progress.set_descr(s.name)
            b = _proc_run(s, data, start, stop, fit_stop, broker, spreads, progress, tcc)
            results.append(b)

    progress.set_descr(f'Backtest {project}')
    progress.close()
    return MultiResults(results=results, project=project, broker=broker, start=start, stop=stop, fit_stop=fit_stop)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# New experimental multisim implementation (WIP)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class _InfoProgress:
    """
    Progress bar implementation for simulations running in multiproc
    It just updates progress of simulation in memcached
    """

    def __init__(self, run_name, run_id, t_id, task_name, ri: RunningInfoManager):
        self.run_name = run_name
        self.run_id = run_id
        self.t_id = t_id
        self.task_name = task_name
        self.ri = ri
        self._prev_i = 0

    def __call__(self, i, label=None):
        if i > self._prev_i:
            self.ri.update_task_info(self.run_id, self.t_id, {
                'task': self.task_name, 'id': self.t_id, 'update_time': str(datetime.now()), 'progress': i
            })
            self._prev_i = i


class IMarketDataProvider:
    """
    Some generic interface for providing access to historical data

    # TODO: Temp loader hack !!
    """

    def ticks(self):
        raise ValueError("Method 'ticks()' must be implemented !!!")

    def ohlc(self, timeframe, **kwargs):
        raise ValueError("Method 'ohlc(timeframe)' must be implemented !!!")

    def __getitem__(self, idx):
        raise ValueError("Method '__getitem__(idx)' must be implemented !!!")


class _LoaderCallable:
    """
    # TODO: Temp loader hack !!
    """

    def __call__(self, a: List[str], start: str, end: str, **kwargs) -> IMarketDataProvider:
        pass


@dataclass
class _SimulationConfigDescriptor:
    broker: str
    tcc: TransactionCostsCalculator
    loader: _LoaderCallable
    start: str
    stop: str
    fit_stop: str
    spreads: Union[float, Dict[str, float]]
    test_timeframe: str
    estimator_portfolio_composer: str
    estimator_used_data: str


class _SimulationTrackerTask(Task):
    """
    Task for simulations run (tracker only)
    """

    def __init__(
            self, instrument, simualtion_cfg: _SimulationConfigDescriptor, simulations_storage_db,
            tracker_class, *tracker_args, **tracker_kwargs
    ):
        super().__init__(tracker_class, *tracker_args, **tracker_kwargs)
        self.instrument = instrument
        self.broker = simualtion_cfg.broker
        self.start = simualtion_cfg.start
        self.stop = simualtion_cfg.stop
        self.fit_stop = simualtion_cfg.fit_stop
        self.spreads = simualtion_cfg.spreads
        self.tcc = simualtion_cfg.tcc
        self.timeframe = simualtion_cfg.test_timeframe
        self.estimator_portfolio_composer = simualtion_cfg.estimator_portfolio_composer
        self.estimator_used_data = simualtion_cfg.estimator_used_data
        self.save(True, simulations_storage_db)

        # TODO: [1] Temp loader hack !!
        self.loader: _LoaderCallable = simualtion_cfg.loader

    def run(self, tracker_instance, run_name, run_id, t_id, task_name, ri: RunningInfoManager):
        # TODO: [1] Temp loader hack: very stupid raw implementation here - need to re-do it better !!!!

        # - loading data
        s_data = self.loader(self.instrument, start=self.start, end=self.stop)
        if self.timeframe is not None:
            data = s_data.ohlc(self.timeframe)
        else:
            data = s_data.ticks()

        s = _recognize({
            f"{task_name}.{t_id}": tracker_instance
        }, run_name, self.estimator_portfolio_composer, self.estimator_used_data)[0]

        sim_result = backtest(
            s.get_signals(data, self.start, self.stop, self.fit_stop), data, self.broker,
            spread=self.spreads, name=s.name, execution_logger=True, trackers=s.trackers,
            progress=_InfoProgress(run_name, run_id, t_id, task_name, ri),
            tcc=self.tcc
        )
        return sim_result


class Market:
    """
    Generic market descriptor
    """

    def __init__(self, broker: str, start: str, stop: str, fit_stop: str,
                 spreads: Union[float, Dict[str, float]], data_loader: _LoaderCallable,
                 tcc: TransactionCostsCalculator = None,
                 test_timeframe: Union[str, None] = None,
                 estimator_portfolio_composer: str = 'single',
                 estimator_used_data: str = 'close'):
        self.market_description: _SimulationConfigDescriptor = _SimulationConfigDescriptor(
            broker=broker, tcc=tcc, loader=data_loader,
            start=start, stop=stop, fit_stop=fit_stop,
            spreads=spreads, test_timeframe=test_timeframe,
            estimator_portfolio_composer=estimator_portfolio_composer, estimator_used_data=estimator_used_data
        )

    def new_simulation(self, instrument, tracker, *tracker_args, storage_db=DB_SIMULATION_RESULTS, **tracker_kwargs):
        return _SimulationTrackerTask(
            instrument, self.market_description, storage_db, tracker, *tracker_args, **tracker_kwargs
        )

    def new_simulations_set(self, instrument, tracker, tracker_args_permutations,
                            simulation_id_start=0, storage_db=DB_SIMULATION_RESULTS):
        return {
            f'sim.{k}.{instrument}': self.new_simulation(
                instrument, tracker, *[], **p, storage_db=storage_db
            ) for k, p in enumerate(tracker_args_permutations, simulation_id_start)
        }
