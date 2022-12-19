import unittest
from os.path import join

import pandas as pd

from qube.configs.Properties import get_root_dir
from qube.portfolio import commissions
from qube.portfolio.Instrument import Instrument
from qube.portfolio.Position import CryptoFuturesPosition
from qube.portfolio.commissions import BinanceRatesCommon


class CommissionsTest(unittest.TestCase):

    def test_crypto_commissions(self):
        portfolio = pd.read_csv(join(get_root_dir(), r'tests/data/portfolios/crypto_portfolio.csv'),
                                index_col='Date', parse_dates=True)
        calculator = commissions.get_calculator('hitbtc')
        self.assertAlmostEqual(0.0112331778, commissions.get_total_commissions(calculator.calculate(portfolio)))

        calculator = commissions.get_calculator('bitfinex')
        self.assertAlmostEqual(0.0224663556, commissions.get_total_commissions(calculator.calculate(portfolio)))

    def test_dukas_commissions(self):
        portfolio = pd.read_csv(join(get_root_dir(), r'tests/data/portfolios/dukas_portfolio.csv'),
                                index_col='Date', parse_dates=True)
        calculator = commissions.get_calculator('dukas')
        self.assertAlmostEqual(2.1681414617, commissions.get_total_commissions(calculator.calculate(portfolio)))

    def test_tcc(self):
        def to_uXBT(tc, instr, exec_price, amount, crossed_market):
            return 1e6 * tc.get_execution_cost(instr, exec_price, amount, crossed_market=crossed_market) / exec_price

        tc = commissions.BitmexTCC()
        exec_price = 58644
        instr1 = Instrument('XBTUSDT', True, 0.5, 0.5)
        cms1 = to_uXBT(tc, instr1, exec_price, -1400, crossed_market=True)
        cms2 = to_uXBT(tc, instr1, exec_price, +1400, crossed_market=True)
        cms3 = to_uXBT(tc, instr1, exec_price, -1400, crossed_market=False)
        cms4 = to_uXBT(tc, instr1, exec_price, +1400, crossed_market=False)
        print("\n --- \n")

        print(f"c1: {cms1} ")
        print(f"c2: {cms2} ")
        print(f"c3: {cms3} ")
        print(f"c4: {cms4} ")
        self.assertAlmostEqual(11.9364, cms1, places=3)
        self.assertAlmostEqual(11.9364, cms2, places=3)
        self.assertAlmostEqual(-2.3873, cms3, places=3)
        self.assertAlmostEqual(-2.3873, cms4, places=3)

    def test_binance_tcc(self):
        # - Crypto futures test
        p = CryptoFuturesPosition(Instrument('BTCUSDT', True, 0, 0), BinanceRatesCommon('um', 'vip0', 'USDT'))

        p.update_position_bid_ask(0, 2500, 20000, 20000)
        pnl = p.update_position_bid_ask(1, 0, 19800, 19800)

        self.assertAlmostEqual(-24.9999, pnl, places=3)
        self.assertAlmostEqual(2.0, p.commissions, places=3)


from pytest import main
if __name__ == '__main__':
    main()