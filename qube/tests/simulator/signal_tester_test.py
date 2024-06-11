import unittest
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from qube.learn.core.base import signal_generator
from qube.portfolio.performance import calculate_total_pnl
from qube.datasource import DataSource
from qube.portfolio.PortfolioLogger import PortfolioLogger
from qube.simulator import SignalTester
from qube.simulator.Brokerage import GenericStockBrokerInfo, GenericForexBrokerInfo
from qube.simulator.core import Tracker, ExecutionLogger
from qube.simulator.multisim import simulation
from qube.simulator.utils import convert_ohlc_to_ticks, load_tick_price_block
from qube.tests.simulator.utilities import MockDataSource, gen_pos, cumulative_pnl_calcs_eod, TickMockDataSource, cross_ma
from qube.utils import QubeLogger


class ITracker(Tracker):
    def __init__(self, silent=False):
        self.silent = silent

    def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
        if not self.silent:
            print(f'\t>-Q-> [{self._instrument}] {quote_time} : {bid},{ask}')

    def on_signal(self, signal_time, signal_qty,
                  quote_time, bid, ask, bid_size, ask_size):
        if not self.silent:
            print(f'\t>>> [{self._instrument}] {signal_time} : {signal_qty} | {quote_time}: {bid},{ask}')

        return signal_qty


@signal_generator
class TestNone(BaseEstimator):

    def fit(self, x, y): return self

    def predict(self, xs: pd.DataFrame):
        return None #pd.Series()

@signal_generator
class TestEmpty(BaseEstimator):

    def fit(self, x, y): return self

    def predict(self, xs: pd.DataFrame):
        return pd.Series()


class TestSimulator(unittest.TestCase):
    DS_CFG_FILE = 'qube/tests/data/simulator/sim_test_datasource_cfg.json'

    def setUp(self):
        # set wide print options
        pd.set_option('display.max_columns', 240)
        pd.set_option('display.width', 520)

    def test_sim_pnl_daily(self):
        ds = MockDataSource('2000-01-01', 100, amplitudes=(10, 30, 5), freq='1d')

        positions = gen_pos({
            'XXX': [
                '2000-01-01 9:30', 100,
                '2000-01-01 16:00', 0,
                '2000-01-02 9:30', -100,
                '2000-01-03 16:00', 0,
            ],
            'YYY': [
                '2000-01-01 9:30', 100,
                '2000-01-01 16:00', 0,
                '2000-01-02 9:30', 200,
                '2000-01-05 16:00', 0,
            ],
        })
        print(positions)

        plogger = PortfolioLogger(log_frequency_sec=60)
        sim = SignalTester(GenericStockBrokerInfo(spread=0), ds)
        elogger = ExecutionLogger()
        w = sim.run_signals(positions, portfolio_logger=plogger, execution_logger=elogger,
                            tracker=ITracker(),
                            trace_log=True)
        r = w.portfolio

        print('----------------------------')
        print(elogger.get_execution_log())
        print('----------------------------')
        print(plogger.get_portfolio_log())
        print('----------------------------')
        self.assertEqual(r.filter(regex=r'.*_PnL').sum(axis=1).sum(axis=0), 2000,
                         'Error calculating Total PnL on test signals')

        # check execution logger work
        exec_log = elogger.get_execution_log()
        # print(exec_log)
        self.assertEqual(len(exec_log), 8)
        self.assertEqual(exec_log.index.dtype.type, np.datetime64)
        xxx_execs = exec_log[exec_log.instrument == 'XXX']
        yyy_execs = exec_log[exec_log.instrument == 'YYY']
        self.assertEqual(list(yyy_execs.iloc[0])[0], 'YYY')
        self.assertEqual(list(yyy_execs.iloc[-1])[0], 'YYY')
        # added commissions
        np.testing.assert_array_almost_equal(yyy_execs.iloc[0][1:-1], [100.0, 10.0, 0.5])
        np.testing.assert_array_almost_equal(yyy_execs.iloc[-1][1:-1], [-200.0, 19.0, 19.0 * 0.05 / 100 * 200])

    def test_ohlc_to_ticks(self):
        ds_daily = DataSource('simulator::ohlc_data', self.DS_CFG_FILE)
        ohlc = ds_daily.load_data(['AAPL', 'MSFT'], start='2015-01-01', end='2017-01-01')

        tcks = convert_ohlc_to_ticks(ohlc, default_size=1e-6, reverse_order_for_bullish_bars=False)
        print(tcks)

        # - test ordinary tick generating logic (before 2022-Sep-05)
        np.testing.assert_array_almost_equal(
            tcks['AAPL'].ask[:20],
            # here we expect series of O,L,H,C ... for default behaviour
            [
                #    O       L       H       C
                118.32, 117.81, 119.86, 118.28,
                117.52, 116.86, 118.60, 118.23,
                115.30, 115.08, 117.69, 115.62,
                116.04, 115.51, 116.94, 116.17,
                115.19, 112.85, 115.39, 113.18,
            ]
        )

        # - test advanced tick generating logic: O,H,L,C for bearish and O,L,H,C for bullish bars
        tcks2 = convert_ohlc_to_ticks(ohlc, default_size=1e-6, reverse_order_for_bullish_bars=True)
        np.testing.assert_array_almost_equal(
            tcks2['AAPL'].ask[:20],
            # here we expect series of O,L,H,C for bullish & O
            [
                #  O     L|H      H|L     C
                118.32, 119.86, 117.81, 118.28,  # Bear O,H,L,C
                117.52, 116.86, 118.60, 118.23,  # Bull O,L,H,C
                115.30, 115.08, 117.69, 115.62,  # Bull O,L,H,C
                116.04, 115.51, 116.94, 116.17,  # Bull O,L,H,C
                115.19, 115.39, 112.85, 113.18,  # Bear O,H,L,C
            ]
        )

        ds = MockDataSource('2000-01-01', 100, amplitudes=(10, 30, 5), freq='5Min')
        ohlc = ds.load_data(['XXX'], start='2000-01-01', end='2000-01-10')
        tcks = convert_ohlc_to_ticks(ohlc, 1e-6)
        print(tcks)

    def test_ticks_loader(self):
        logger = QubeLogger.getLogger(self.__module__)

        ds_daily = DataSource('simulator::ohlc_data', self.DS_CFG_FILE)
        ticks = load_tick_price_block(ds_daily, None, ['AAPL', 'MSFT'], '2016-01-01',
                                      spread_info={
                                          'AAPL': 0.25, 'MSFT': 0.25
                                      }, exec_by_new_update=True, logger=logger)
        self.assertTrue(len(ticks['AAPL']) > 20)
        self.assertTrue(len(ticks['MSFT']) > 20)
        print(ticks['AAPL'].head())

        ds_qts = DataSource('simulator::quotes_data', self.DS_CFG_FILE)
        ticks = load_tick_price_block(ds_qts, None, ['GBPUSD', 'USDCHF'], '2017-03-21',
                                      exec_by_new_update=True, logger=logger)
        self.assertTrue(len(ticks['GBPUSD']) > 20)
        self.assertTrue(len(ticks['USDCHF']) > 20)
        print(ticks['GBPUSD'].head())
        print(ticks['USDCHF'].head())

    def test_ticks_loader_arr_concat(self):
        ds_qts = DataSource('simulator::ohlc_data', self.DS_CFG_FILE)
        ticks = load_tick_price_block(
            ds_qts, None, ['XOM', 'MSFT', 'EURGBP'], '2015-01-01', exec_by_new_update=True, logger=QubeLogger.getLogger(self.__module__)
        )
        print(ticks['XOM'].head())
        print(ticks['MSFT'].head())
        print(ticks['EURGBP'].head())

    def test_simulator_ohlc(self):
        ds_daily = DataSource('simulator::ohlc_data', self.DS_CFG_FILE)

        prices = ds_daily.load_data(['XOM', 'AAPL', 'MSFT'], start='2016-01-01', end='2017-01-01')
        positions1 = cross_ma(prices, 'XOM', p_fast=5, p_slow=50)
        positions2 = cross_ma(prices, 'AAPL', p_fast=5, p_slow=50)
        positions3 = cross_ma(prices, 'MSFT', p_fast=5, p_slow=50)
        positions = pd.concat([positions1, positions2, positions3], axis=1)

        sim = SignalTester(GenericStockBrokerInfo(spread=0), ds_daily)
        W = sim.run_signals(positions, portfolio_logger=PortfolioLogger(1), tracker=ITracker(True))
        equity = W.equity(account_transactions=False)

        # print(pfl.tail(20))

        # test signals
        right_pnl_1 = cumulative_pnl_calcs_eod(positions1, prices['XOM'])
        right_pnl_2 = cumulative_pnl_calcs_eod(positions2, prices['AAPL'])
        right_pnl_3 = cumulative_pnl_calcs_eod(positions3, prices['MSFT'])
        right_pnl = right_pnl_1 + right_pnl_2 + right_pnl_3

        self.assertAlmostEqual(equity[-1], right_pnl[-1], delta=0.01)

        # c, o = prices['XOM'].close, prices['XOM'].open
        # p = srows(shift_signals(c, '16H'), shift_signals(o, '9H30Min')).rename('price')
        # m = scols(p, positions1.rename('position')).ffill().dropna()
        # eqO = (m.price.diff() * m.position.shift(1)).cumsum()
        # eqS = pfl.Total_PnL.cumsum()

        ds_daily.close()

    def test_simulator_intraday(self):
        ds5m = MockDataSource('2000-01-01', 10000, amplitudes=(10, 30, 5), freq='5 min')

        pos5m = gen_pos({
            'A': [
                '2000-01-01 9:30', 100,
                '2000-01-01 16:00', 0,
                '2000-01-09 10:00', 0,
                '2000-02-10 16:00', 0,
            ],
            'B': [
                '2000-01-01 9:30', 100,
                '2000-01-10 16:00', 0,
            ],
        })

        plogger = PortfolioLogger()
        sim = SignalTester(GenericStockBrokerInfo(spread=0), ds5m)
        sim.run_signals(pos5m, portfolio_logger=plogger)
        r = calculate_total_pnl(plogger.get_portfolio_log(), split_cumulative=False)

        self.assertAlmostEqual(r['Total_PnL'][-1], 1200.0, delta=0.0001)

    def test_simulator_ticks(self):
        rd_prices = lambda *r: pd.DataFrame(
            data=np.array([r[1:len(r):5], r[2:len(r):5], r[3:len(r):5], r[4:len(r):5]]).T,
            index=pd.to_datetime(r[0:-1:5]), columns=['bid', 'ask', 'bidvol', 'askvol'])

        ds_tick = TickMockDataSource({
            'A': rd_prices(
                '2000-01-01 09:30:00.000100', 50, 51.1, 1000, 1000,
                '2000-01-01 09:30:01.100000', 51, 52.1, 1000, 1100,
                '2000-01-01 09:30:01.200000', 52, 53.1, 1000, 1200,
                '2000-01-01 09:30:05.100000', 53, 54.1, 1000, 1300,
                '2000-01-01 09:30:06.000000', 54, 55.1, 1000, 1400,
                '2000-01-01 09:30:06.500000', 55, 55.1, 1000, 1500,
                '2000-01-01 09:30:07.100000', 56, 56.1, 1000, 1600,
                '2000-01-01 15:59:59.000000', 56, 56.1, 1000, 1600,
                '2000-01-02 10:30:00.100000', 60, 60.1, 2000, 2700,
                '2000-01-02 10:30:01.100000', 61, 61.1, 2000, 2800,
                '2000-01-02 10:30:02.100000', 62, 62.1, 2000, 2900,
                '2000-01-02 10:30:03.100000', 63, 63.1, 2000, 2900,
                '2000-01-02 10:30:05.100000', 64, 64.1, 2000, 2900,
                '2000-01-02 10:30:06.100000', 65, 65.1, 2000, 2900,
                '2000-01-02 10:30:07.100000', 66, 66.1, 2000, 2900,
                '2000-01-02 10:30:10.100000', 67, 67.1, 2000, 3000,
            ),

            'B': rd_prices(
                '2000-01-01 09:30:01.000100', 30, 31.1, 1000, 2000,
                '2000-01-01 09:30:01.100000', 31, 32.1, 1000, 3000,
                '2000-01-01 09:30:01.200000', 32, 33.1, 1000, 4000,
                '2000-01-01 09:30:10.120000', 33, 34.1, 1000, 5000,
                '2000-01-01 09:30:11.120000', 33, 34.1, 1000, 6000,
            ),
        })

        pos_tick = gen_pos({
            'A': [
                '2000-01-01 9:30:00.5', 0,
                '2000-01-01 9:30:01', 0,
                '2000-01-01 9:30:05', 50,
                '2000-01-01 9:30:07', 0,
                '2000-01-02 10:30:01', 100,
                '2000-01-02 10:30:05', 0,
            ],

            'B': [
                '2000-01-01 9:30:01.5', 100,
                '2000-01-01 9:30:10.150', 0,
            ]

        })

        plogger_tick = PortfolioLogger(0)
        sim = SignalTester(GenericStockBrokerInfo(spread=0), ds_tick)
        sim.run_signals(pos_tick, portfolio_logger=plogger_tick,
                        fill_latency_msec=timedelta(milliseconds=100),
                        trace_log=True)
        r_tick = calculate_total_pnl(plogger_tick.get_portfolio_log(), split_cumulative=False)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(r_tick)

        self.assertAlmostEqual(385.0, r_tick['A_PnL'][-1], delta=0.001)
        self.assertAlmostEqual(-10.0, r_tick['B_PnL'][-1], delta=0.001)

    def test_simulator_ticks_with_exec_by_new_update(self):
        rd_prices = lambda *r: pd.DataFrame(
            data=np.array([r[1:len(r):5], r[2:len(r):5], r[3:len(r):5], r[4:len(r):5]]).T,
            index=pd.to_datetime(r[0:-1:5], format='mixed'), columns=['bid', 'ask', 'bidvol', 'askvol'])

        ds_tick = TickMockDataSource({
            'A': rd_prices(
                '2000-01-01 09:30:00.000100', 50, 51.1, 1000, 1000,
                '2000-01-01 09:30:01.100000', 51, 52.1, 1000, 1100,
                '2000-01-03 09:30:01', 150, 155, 1000, 1200,
            ),
        })

        pos_tick = gen_pos({
            'A': [
                '2000-01-01 9:30:00.5', 1000,
                '2000-01-01 9:30:01', 2000,
                '2000-01-03 09:29:01', 3000,

            ]

        })
        elogger = ExecutionLogger()
        sim = SignalTester(GenericStockBrokerInfo(spread=0), ds_tick)
        sim.run_signals(pos_tick, execution_logger=elogger, exec_by_new_update=True)
        elog = elogger.get_execution_log()
        # it is check that executions by next price
        self.assertEqual(elog.iloc[0]['exec_price'], 52.1)
        self.assertEqual(elog.iloc[1]['exec_price'], 52.1)
        self.assertEqual(elog.iloc[2]['exec_price'], 155)

    def test_simulator_aux_instr_ticks(self):
        ds_quotes = DataSource('simulator::quotes_data', self.DS_CFG_FILE)

        pos_tick = gen_pos({
            'EURCHF': [
                '2000-01-01 9:30:00.5', 1000,
                '2000-01-01 9:30:01', 0,
                '2000-01-01 9:30:05', 0,
            ]
        })

        plogger_tick = PortfolioLogger()
        sim = SignalTester(GenericForexBrokerInfo(spread=0.0001), ds_quotes)
        W = sim.run_signals(pos_tick, portfolio_logger=plogger_tick, fill_latency_msec=timedelta(milliseconds=0), )
        r_tick = calculate_total_pnl(W.portfolio, split_cumulative=False)
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        print(r_tick)
        self.assertAlmostEqual(1.6456, r_tick['Total_PnL'][-1], delta=0.0001)

    def test_simulator_aux_instr_ticks_straight(self):
        ds_quotes = DataSource('simulator::quotes_data', self.DS_CFG_FILE)

        pos_tick = gen_pos({
            'EURGBP': [
                '2017-03-21 20:37:49.500', 1000000,
                '2017-03-21 20:37:58.392', 0,
                '2017-03-21 20:39:45.400', -1000000,
                '2017-03-21 20:39:51.533', 0
            ]
        })

        plogger_tick = PortfolioLogger(0)
        sim = SignalTester(GenericForexBrokerInfo(spread=0.0001), ds_quotes)
        W = sim.run_signals(pos_tick, portfolio_logger=plogger_tick, fill_latency_msec=timedelta(milliseconds=0),
                            trace_log=True)
        r_tick = calculate_total_pnl(W.portfolio, split_cumulative=False)
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        print(r_tick)
        self.assertAlmostEqual(r_tick['Total_PnL'].sum(), -62.4495 - 162.3518, delta=0.0001)

    def test_simulator_aux_instr_OHLC(self):
        ds_ohlc = DataSource('simulator::ohlc_data', self.DS_CFG_FILE)
        spread = 0.00006
        sim = SignalTester(GenericForexBrokerInfo(spread=spread), ds_ohlc)
        pos_ohlc = gen_pos({
            'EURGBP': [
                '2017-03-01 00:00:01', 10 ** 6,
                '2017-03-19 23:59:59', 0,
                '2017-03-23 00:00:01', -10 ** 6,
                '2017-04-07 23:59:59', 0
            ]
        })

        def _Q(symb, t, spread=spread):
            class Struct(object): pass

            x = Struct()
            p = ds_ohlc.load_data(symb, t, t)[symb]['open'].values[0]
            x.b = p - spread / 2
            x.a = p + spread / 2
            return x

        W = sim.run_signals(pos_ohlc, trace_log=True)
        r_tick = calculate_total_pnl(W.portfolio, split_cumulative=False)
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        print(r_tick)
        pl_0 = +10 ** 6 * (_Q('EURGBP', '2017-03-19').b - _Q('EURGBP', '2017-03-01').a) * _Q('GBPUSD', '2017-03-19').a
        pl_1 = -10 ** 6 * (_Q('EURGBP', '2017-04-07').a - _Q('EURGBP', '2017-03-23').b) * _Q('GBPUSD', '2017-04-07').b
        self.assertAlmostEqual(r_tick['Total_PnL'].sum(), pl_0 + pl_1, delta=0.0001)

    def test_simulataor_with_aux_in_instrs(self):
        quotes_pos = gen_pos({
            'GBPUSD': [
                '2017-03-21 20:37:49.500', -1000,
                '2017-03-21 20:38:02.500', 1000,
            ],
            'EURGBP': [
                '2017-03-21 20:37:49.500', 1000,
                '2017-03-21 20:38:02.500', -1000,

            ]
        })
        ds_quotes = DataSource('simulator::quotes_data', self.DS_CFG_FILE)
        sim_quotes = SignalTester(GenericForexBrokerInfo(spread=0), ds_quotes)
        w = sim_quotes.run_signals(quotes_pos, trace_log=True)
        print(w.portfolio)
        total_pnl_quotes = calculate_total_pnl(w.portfolio, split_cumulative=False)
        self.assertAlmostEqual(total_pnl_quotes['Total_PnL'][-1], -0.214899, delta=0.0001)

        ohlcv_pos = gen_pos({
            'GBPUSD': [
                '2017-03-01 00:00:01', -1000,
                '2017-03-21 00:00:01', 1000,
            ],
            'EURGBP': [
                '2017-03-01 00:00:01', 1000,
                '2017-03-15 00:00:01', -1000,

            ],
        })
        ds_ohlcv = DataSource('simulator::ohlc_data', self.DS_CFG_FILE)
        sim_ohlcv = SignalTester(GenericForexBrokerInfo(spread=0), ds_ohlcv)
        w = sim_ohlcv.run_signals(ohlcv_pos, trace_log=True)
        total_pnl_ohlcv = calculate_total_pnl(w.portfolio, split_cumulative=False)
        self.assertAlmostEqual(total_pnl_ohlcv['Total_PnL'].sum(), -0.053692, delta=0.0001)

    def test_simulator_empty_price_block(self):
        pos = gen_pos({
            'GBPUSD': [
                '2017-03-20 09:37:47.645', 1000,
                '2017-03-20 09:37:48.645', 0,
                '2017-03-21 20:37:47.925', 1000,

            ]
        })
        ds = DataSource('simulator::quotes_data', self.DS_CFG_FILE)
        sim = SignalTester(GenericForexBrokerInfo(spread=0), ds)
        W = sim.run_signals(pos)
        total_pnl = calculate_total_pnl(W.portfolio, split_cumulative=False)
        self.assertAlmostEqual(total_pnl['Total_PnL'].sum(), -0.03999, delta=0.0001)

    def test_simulator_empty_signals(self):        
        ds = MockDataSource('2016-01-01', 10000, amplitudes=(10, 30, 5), freq='5 min')
        prices = ds.load_data(['Test1', 'Test2'], start='2016-01-01', end='2016-02-01')
        r2 = simulation( {
            'test-none': TestNone(), 
            'test-empty': TestEmpty(), 
            }, prices, 'binance_um_vip0_usdt')


from pytest import main
if __name__ == '__main__':
    main()