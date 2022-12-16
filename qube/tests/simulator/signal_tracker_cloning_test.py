import unittest

import numpy as np
import pandas as pd
from typing import Any, Union
from collections import defaultdict
import mongomock
from mongomock.gridfs import enable_gridfs_integration
from qube.datasource.controllers.MongoController import MongoController
from qube.learn.core.utils import debug_output

from qube.tests.utils_for_tests import _read_timeseries_data

enable_gridfs_integration()

from qube.portfolio.performance import portfolio_stats
from qube.quantitative.tools import scols
from qube.datasource import DataSource
from qube.portfolio.PortfolioLogger import PortfolioLogger
from qube.portfolio.commissions import TransactionCostsCalculator
from qube.series.Indicators import ATR
from qube.simulator import SignalTester
from qube.simulator.Brokerage import GenericStockBrokerInfo
from qube.simulator.core import Tracker, ExecutionLogger
from qube.simulator.multisim import simulation
from qube.tests.simulator.utilities import cross_ma_signals_generator, portfolio_from_executions


class IMonitor:
    def on_tracker_event(self, tracker: Tracker, data: Any):
        pass

    def on_tracker_position_opening(self, tracker: Tracker, data: Any):
        pass

class TestingTracker(Tracker):
    def __init__(self, parameter: Any, monitor: IMonitor, debug=False):
        self.monitor = monitor 
        self.parameter = parameter
        self.debug = debug

    def initialize(self):
        self.h = self.get_ohlc_series('1d')
        self.vol = ATR(12)
        self.h.attach(self.vol)
        self.stop = None
        self.take = None
        self.monitor.on_tracker_event(self, 'Initialized !')
        self.n_openings = 0

    def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
        if self._position.quantity > 0:
            if bid >= self.take:
                if self.debug:
                    print(f' + take long [{self._instrument}] at {bid}')
                self.trade(quote_time, 0, comment="Take Long", market_order=False)
                return

            if ask <= self.stop:
                if self.debug:
                    print(f' - stop long [{self._instrument}] at {ask}')
                self.trade(quote_time, 0, comment="Stop Long")
                return

        if self._position.quantity < 0:
            if ask <= self.take:
                if self.debug:
                    print(f' + take short [{self._instrument}] at {ask}')
                self.trade(quote_time, 0, comment="Take Short", market_order=False)
                return

            if bid >= self.stop:
                if self.debug:
                    print(f' - stop short [{self._instrument}] at {bid}')
                self.trade(quote_time, 0, comment="Stop Short")
                return

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        v = self.vol[0]
        v = 0 if np.isnan(v) else v
        signal_qty = int(signal_qty * 1000 * v)
        
        if self.debug:
            print(f'\t[{quote_time} | {signal_time}] -> {self._instrument} {signal_qty}', end='')

        if signal_qty != 0:
            self.stop = self.h[0].low - v if signal_qty > 0 else self.h[0].high + v
            self.take = self.h[0].high + 2 * v if signal_qty > 0 else self.h[0].low - 2 * v
            entry = ask if signal_qty > 0 else bid

            if self.debug:
                print(f' @ {entry} x {self.stop:.2f} + {self.take:.2f}', end='')

            # - notify monitor -
            self.monitor.on_tracker_event(self, ' <<<< SIGNAL ' + str(signal_qty) + " >>>>")
            self.monitor.on_tracker_position_opening(self, signal_qty)
            self.n_openings += 1

        if self.debug:
            print()

        return signal_qty
    
    def statistics(self) -> dict:
        return {'openings': self.n_openings}


class MultiTrackersDispatcher(Tracker, IMonitor):
    def __init__(self, param) -> None:
        super().__init__()
        self.param = param
        self._trackers = {}
        self._n_positions = defaultdict(lambda: 0)

    def on_tracker_event(self, tracker: Tracker, data: Any):
        print(f" >> {str(tracker._position.symbol)} -> event {data} ")

    def on_tracker_position_opening(self, tracker: Tracker, data: Any):
        symbol = tracker._position.symbol
        print(f" >> {symbol} -> {data} ")
        self._n_positions[symbol] += 1

    def __on_tracker_cloning__(self, instrument: str, is_aux=False) -> Tracker:
        tr = TestingTracker(self.param, self)
        print(f'{instrument} --> {str(tr)}')
        self._trackers[instrument] = tr
        return tr 


class TestSimulatorTracker(unittest.TestCase):
    DS_CFG_FILE = 'qube/tests/data/simulator/sim_test_datasource_cfg.json'

    def setUp(self):
        # set wide print options
        pd.set_option('display.max_columns', 240)
        pd.set_option('display.width', 520)

    def test_cloned_tracker(self):
        """
        New functionality test
        """
        with DataSource('simulator::ohlc_data', self.DS_CFG_FILE) as ds_daily:
            prices = ds_daily.load_data(['XOM', 'AAPL', 'MSFT'], start='2016-01-01', end='2017-01-01')

            # entry signals on moving crossing
            positions1 = cross_ma_signals_generator(prices, 'XOM', p_fast=7, p_slow=21)
            positions2 = cross_ma_signals_generator(prices, 'AAPL', p_fast=7, p_slow=14)
            positions3 = cross_ma_signals_generator(prices, 'MSFT', p_fast=7, p_slow=14)
            positions = scols(positions1, positions2, positions3)

            # some custom TCC example with rebates
            class TCC_test(TransactionCostsCalculator):
                def __init__(self, fees):
                    super().__init__(fees / 100, -fees / 100)

            sim = SignalTester(GenericStockBrokerInfo(spread=0, tcc=TCC_test(0.01)), ds_daily)
            exec_log = ExecutionLogger()

            multitrack = MultiTrackersDispatcher(1234)

            W = sim.run_signals(positions, portfolio_logger=PortfolioLogger(), execution_logger=exec_log, tracker=multitrack)

            r = portfolio_from_executions(exec_log, prices)
            equity = W.equity(account_transactions=False)

            pst0 = portfolio_stats(W.portfolio, 50000, account_transactions=False)
            pst1 = portfolio_stats(W.portfolio, 50000, account_transactions=True)
            # print(W.executions)
            # print(" -> Broker Commissions: ", pst1['broker_commissions'])

            # check portfolio stats + commissions
            self.assertAlmostEqual(pst1['broker_commissions'], 528.092445)
            self.assertAlmostEqual(pst0['equity'][-1] - pst1['equity'][-1], pst1['broker_commissions'])
            self.assertAlmostEqual(equity[-1], r[:equity.index[-1]].sum(), delta=0.01)

            print(dict(multitrack._n_positions))
            print(W.trackers_stat)

    def __initialize_mongo_db(self):
        print('Initializing database ...')
        d1 = _read_timeseries_data('solusdt_15min', compressed=True)
        d2 = _read_timeseries_data('ethusdt_15min', compressed=True)

        self.mongo = MongoController('md_test')
        self.mongo.save_data('m1/BINANCEF:SOLUSDT', d1, is_serialize=True)
        self.mongo.save_data('m1/BINANCEF:ETHUSDT', d2, is_serialize=True)
        self.mongo.close()

    @mongomock.patch()
    def test_tracker_on_mongo(self):

        self.__initialize_mongo_db()

        with DataSource('simulator::mongo-market-data-1min', self.DS_CFG_FILE) as ds:
            prices = ds.load_data(['ETHUSDT', 'SOLUSDT'], start='2021-01-01', end='2021-02-01')

            # entry signals on moving crossing
            positions1 = cross_ma_signals_generator(prices, 'ETHUSDT', p_fast=7, p_slow=30)
            positions2 = cross_ma_signals_generator(prices, 'SOLUSDT', p_fast=7, p_slow=30)
            positions = scols(positions1, positions2)

            multitrack = MultiTrackersDispatcher(1234)
            r = simulation(
                { 'Test0': [positions, multitrack] }, 
                ds, 
                'binance_um_vip0_usdt', 
                start='2021-01-01', stop='2021-02-01',
                instruments=['ETHUSDT', 'SOLUSDT']
            )
            # print(r.results[0].trackers_stat)
            # debug_output(r.results[0].executions, 'Executions')
            pst1 = portfolio_stats(r.results[0].portfolio, 50000, account_transactions=True)
            self.assertAlmostEquals(pst1['sharpe'], -0.05029371157299659, places=4)


if __name__ == '__main__':
    unittest.main()
        