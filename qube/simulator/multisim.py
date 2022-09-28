from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Union, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from qube.learn.core.base import MarketDataComposer
from qube.portfolio.commissions import TransactionCostsCalculator, ZeroTCC
from qube.simulator.SignalTester import Tracker, SimulationResult
from qube.simulator.multiproc import Task, RunningInfoManager
from qube.utils.nb_functions import z_backtest
from qube.utils.ui_utils import red, green, yellow, blue
from qube.utils.utils import mstruct, runtime_env


class _Types(Enum):
    UKNOWN = 'unknown'
    LIST = 'list'
    TRACKER = 'tracker'
    SIGNAL = 'signal'
    ESTIMATOR = 'estimator'


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
        t = _Types.ESTIMATOR
    elif isinstance(obj, dict):
        # when tracker has setup for each intrument {str -> Tracker}
        if all([isinstance(k, str) & isinstance(v, Tracker) for k, v in obj.items()]):
            t = _Types.TRACKER
        else:
            t = _Types.UKNOWN
    else:
        t = _Types.UKNOWN
    return t


def start_stop_sigs(data: Dict[str, pd.DataFrame], start=None, stop=None):
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
    def __init__(self, signals, trackers, experiment_name=None):
        self.signals = signals
        self.signal_type: _Types = _type(signals)
        self.trackers = trackers
        self.name = experiment_name

    def get_signals(self, data, start, stop):
        sx = self.signals

        if sx is None or self.signal_type == _Types.UKNOWN:
            return start_stop_sigs(data, start, stop)

        if self.signal_type == _Types.ESTIMATOR:
            if isinstance(sx, MarketDataComposer):
                sx = sx.for_interval(start, stop)
            sx = sx.predict(data)

        _z = slice(start, stop) if start is not None and stop is not None else None
        return sx[_z] if _z is not None else sx

    def __repr__(self):
        return f'{self.name}.{self.signal_type}.[{repr(self.trackers) if self.trackers is not None else "-"}]'


def _is_signal_or_generator(obj):
    return _type(obj) in [_Types.SIGNAL, _Types.ESTIMATOR]


def _is_generator(obj):
    return _type(obj) == _Types.ESTIMATOR


def _is_tracker(obj):
    return _type(obj) == _Types.TRACKER


def _recognize(setup, data, name) -> List[SimSetup]:
    r = list()

    if isinstance(setup, dict):
        for n, v in setup.items():
            r.extend(_recognize(v, data, name + '/' + n))

    elif isinstance(setup, (list, tuple)):
        if len(setup) == 2 and _is_signal_or_generator(setup[0]) and _is_tracker(setup[1]):
            r.append(SimSetup(setup[0], setup[1], name))
        else:
            for j, s in enumerate(setup):
                r.extend(_recognize(s, data, name + '/' + str(j)))

    elif _is_tracker(setup):
        r.append(SimSetup(None, setup, name))

    elif isinstance(setup, (pd.DataFrame, pd.Series)):
        r.append(SimSetup(setup, None, name))

    elif _is_generator(setup):
        r.append(SimSetup(setup, None, name))

    return r


def _proc_run(s: SimSetup, data, start, stop, broker, spreads, progress, tcc: TransactionCostsCalculator):
    """
    TODO: need to be running in separate process
    """
    b = z_backtest(
        s.get_signals(data, start, stop),
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
    start: Union[str, pd.Timestamp]
    stop: Union[str, pd.Timestamp]

    def __add__(self, other):
        if not isinstance(other, MultiResults):
            raise ValueError(f"Don't know how to add {type(other)} data to results !")

        if self.project != other.project:
            raise ValueError(f"You can't add results from another project {other.project}")

        brok = self.broker if self.broker == other.broker else f'{self.broker},{other.broker}'
        return MultiResults(self.results + other.results, self.project, brok, self.start, self.stop)

    def __getitem__(self, key):
        s = self.results[key]
        return MultiResults(s if isinstance(s, list) else [s], self.project, self.broker, self.start, self.stop)

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


def simulation(setup, data, broker, project='', start=None, stop=None, spreads=0,
               tcc: TransactionCostsCalculator = None) -> MultiResults:
    """
    Simulate different setups
    """
    sims = _recognize(setup, data, project)
    results = []
    progress = __ForeallProgress(len(sims))

    for i, s in enumerate(sims):
        # print(s)
        if True:
            progress.set_descr(s.name)
            b = _proc_run(s, data, start, stop, broker, spreads, progress, tcc)
            results.append(b)

    progress.set_descr(f'Backtest {project}')
    progress.close()
    return MultiResults(results=results, project=project, broker=broker, start=start, stop=stop)


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


class _SimulationTrackerTask(Task):
    """
    Task for simulations run (tracker only)
    TODO: signal generator
    """

    def __init__(self, instrument, market_description, tracker_class, *tracker_args, **tracker_kwargs):
        super().__init__(tracker_class, *tracker_args, **tracker_kwargs)
        self.instrument = instrument
        self.broker = market_description.broker
        self.start = market_description.start
        self.stop = market_description.stop
        self.spreads = market_description.spreads
        self.tcc = market_description.tcc

        # TODO: Temp loader hack !!
        self.loader: _LoaderCallable = market_description.loader

    def run(self, tracker_instance, run_name, run_id, t_id, task_name, ri: RunningInfoManager):
        # TODO: Temp loader hack: very stupid raw implementation here - need to re-do it better !!!!
        data = self.loader(self.instrument, start=self.start, end=self.stop).ticks()

        s = _recognize({f"{task_name}.{t_id}": tracker_instance}, data, run_name)[0]
        sim_result = z_backtest(
            s.get_signals(data, self.start, self.stop), data, self.broker,
            spread=self.spreads, name=s.name, execution_logger=True, trackers=s.trackers,
            progress=_InfoProgress(run_name, run_id, t_id, task_name, ri),
            tcc=self.tcc
        )
        return sim_result


class Market:
    """
    Generic market descriptor
    """

    def __init__(self, broker, start, stop, spreads,
                 data_loader: _LoaderCallable,
                 tcc: TransactionCostsCalculator = ZeroTCC()):
        self.market_description = mstruct(
            broker=broker, tcc=tcc, loader=data_loader, start=start, stop=stop, spreads=spreads
        )

    def new_simulation(self, instrument, tracker, *tracker_args, **tracker_kwargs):
        return _SimulationTrackerTask(instrument, self.market_description, tracker, *tracker_args, **tracker_kwargs)

    def new_simulations_set(self, instrument, tracker, tracker_args_permutations, simulation_id_start=0):
        return {
            f'sim.{k}.{instrument}': self.new_simulation(instrument, tracker, *[], **p) for k, p in
            enumerate(tracker_args_permutations, simulation_id_start)
        }
