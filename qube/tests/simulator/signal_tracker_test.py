import unittest

import numpy as np
import pandas as pd

from qube.portfolio.performance import portfolio_stats
from qube.quantitative.tools import scols
from qube.datasource import DataSource
from qube.portfolio.PortfolioLogger import PortfolioLogger
from qube.portfolio.commissions import TransactionCostsCalculator
from qube.series.Indicators import ATR
from qube.simulator import SignalTester
from qube.simulator.Brokerage import GenericStockBrokerInfo
from qube.simulator.core import Tracker, ExecutionLogger
from qube.tests.simulator.utilities import cross_ma_signals_generator, portfolio_from_executions


class TestingTracker(Tracker):
    def initialize(self):
        self.h = self.get_ohlc_series('1d')
        self.vol = ATR(12)
        self.h.attach(self.vol)
        self.stop = None
        self.take = None

    def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
        if self._position.quantity > 0:
            if bid >= self.take:
                print(f' + take long [{self._instrument}] at {bid}')
                self.trade(quote_time, 0, comment="Take Long", market_order=False)
                return

            if ask <= self.stop:
                print(f' - stop long [{self._instrument}] at {ask}')
                self.trade(quote_time, 0, comment="Stop Long")
                return

        if self._position.quantity < 0:
            if ask <= self.take:
                print(f' + take short [{self._instrument}] at {ask}')
                self.trade(quote_time, 0, comment="Take Short", market_order=False)
                return

            if bid >= self.stop:
                print(f' - stop short [{self._instrument}] at {bid}')
                self.trade(quote_time, 0, comment="Stop Short")
                return

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        v = self.vol[0]
        v = 0 if np.isnan(v) else v
        signal_qty = int(signal_qty * 1000 * v)
        print(f'\t[{quote_time} | {signal_time}] -> {self._instrument} {signal_qty}', end='')

        if signal_qty != 0:
            self.stop = self.h[0].low - v if signal_qty > 0 else self.h[0].high + v
            self.take = self.h[0].high + 2 * v if signal_qty > 0 else self.h[0].low - 2 * v
            entry = ask if signal_qty > 0 else bid
            print(f' @ {entry} x {self.stop:.2f} + {self.take:.2f}', end='')
        print()

        return signal_qty


class TestSimulatorTracker(unittest.TestCase):
    DS_CFG_FILE = 'qube/tests/data/simulator/sim_test_datasource_cfg.json'

    def setUp(self):
        # set wide print options
        pd.set_option('display.max_columns', 240)
        pd.set_option('display.width', 520)

    def test_quoted_tracker(self):
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
            W = sim.run_signals(
                positions, portfolio_logger=PortfolioLogger(),
                execution_logger=exec_log,
                tracker={s: TestingTracker() for s in positions.columns}
            )

            # pfl = calculate_total_pnl(W.portfolio)
            r = portfolio_from_executions(exec_log, prices)

            # el = exec_log.get_execution_log()
            # s2 = {s: el[el.instrument == s].quantity.cumsum() for s in set(el.instrument)}
            #
            # for k, v in s2.items():
            #     v.index = v.index.round('2s')
            #     s2[k] = v
            #
            # prep_prices = lambda s: srows(
            #     shift_signals(prices[s]['open'], '9:30:00'),
            #     shift_signals(prices[s]['close'], '16:00:00'))
            #
            # d1 = scols(prep_prices('AAPL'), srows(s2['AAPL'], keep='last'), names=['s', 'q']).ffill().fillna(0)
            # d2 = scols(prep_prices('XOM'), srows(s2['XOM'], keep='last'), names=['s', 'q']).ffill().fillna(0)
            # d3 = scols(prep_prices('MSFT'), srows(s2['MSFT'], keep='last'), names=['s', 'q']).ffill().fillna(0)
            # r = (d1.s.diff() * d1.q.shift()) + (d2.s.diff() * d2.q.shift()) + (d3.s.diff() * d3.q.shift())

            equity = W.equity(account_transactions=False)

            pst0 = portfolio_stats(W.portfolio, 50000, account_transactions=False)
            pst1 = portfolio_stats(W.portfolio, 50000, account_transactions=True)
            print(W.executions)
            print(" -> Broker Commissions: ", pst1['broker_commissions'])

            # check portfolio stats + commissions
            self.assertAlmostEqual(pst1['broker_commissions'], 528.092445)
            self.assertAlmostEqual(pst0['equity'][-1] - pst1['equity'][-1], pst1['broker_commissions'])
            self.assertAlmostEqual(equity[-1], r[:equity.index[-1]].sum(), delta=0.01)


from pytest import main
if __name__ == '__main__':
    main()