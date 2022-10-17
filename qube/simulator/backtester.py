from typing import Union, Dict

import pandas as pd

from qube.portfolio.commissions import TransactionCostsCalculator, ZeroTCC
from qube.simulator import SignalTester

from qube.simulator.core import ExecutionLogger, Tracker, SimulationResult
from qube.simulator.utils import __instantiate_simulated_broker, _progress_bar


def backtest_signals_inplace(signals, datasource, broker, spread: Union[float, Dict[str, float]] = None,
                             execution_logger: ExecutionLogger = None,
                             trackers: Union[Tracker, Dict[str, Tracker]] = None,
                             verbose=False,
                             **kwargs) -> SimulationResult:
    """
    Shortcut for test trading signals in current notebook process.
    It's good for fast testing on small set of instruments and history.
    Just for compatibility

    :return: portfolio log dataframe(PnL is split)
    """
    r = backtest(signals, datasource, broker, spread, execution_logger is not None, trackers, verbose, **kwargs)
    return r.portfolio


def backtest(signals, datasource, broker,
             spread: Union[float, Dict[str, float]] = None,
             execution_logger: bool = False,
             trackers: Union[Tracker, Dict[str, Tracker]] = None,
             single_tracker_for_all=False,
             progress=None,
             name: str = None,
             tcc: TransactionCostsCalculator = ZeroTCC(),
             verbose=False,
             **kwargs) -> SimulationResult:
    """
    Shortcut for test trading signals in current notebook process.
    It's good for fast testing on small set of instruments and history.

    :param signals: set of signals
    :param datasource: datasource to be used for testing
    :param broker: brokerage details ('forex' or 'stock' are supported now)
    :param spread: default spread size (only for OHLC cases)
    :param execution_logger: execution logger
    :param trackers: custom positions trackers
    :param single_tracker_for_all: if true and tracker is object do not make copy for each symbol
    :param progress: custom progress indicator
    :param name: name of simualtion
    :param tcc: transaction costs calculator (default is None i.e. zero commissions)
    :param verbose: true if needed more information
    :return: simulation results structure
    """
    if isinstance(datasource, (pd.DataFrame, dict)):
        # not needed to import this class to notebook
        from qube.datasource.InMemoryDataSource import InMemoryDataSource
        datasource = InMemoryDataSource(datasource)

    tester = SignalTester(__instantiate_simulated_broker(broker, spread, tcc), datasource)
    r = tester.run_signals(signals,
                           jupyter_progress_listener=_progress_bar() if progress is None else progress,
                           tracker=trackers,
                           single_tracker_for_all=single_tracker_for_all,
                           execution_logger=ExecutionLogger() if execution_logger else None,
                           verbose=verbose, name=name, **kwargs)
    return r
