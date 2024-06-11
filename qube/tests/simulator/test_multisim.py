from unittest import TestCase

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline

from qube.examples.learn.generators import RangeBreakoutDetector
from qube.examples.learn.transformers import RollingRange
from qube.learn.core.base import MarketDataComposer, signal_generator
from qube.learn.core.pickers import SingleInstrumentPicker
from qube.learn.core.utils import debug_output
from qube.simulator.multisim import simulation, simulation_mt
from qube.simulator.tracking.sizers import FixedRiskSizer
from qube.simulator.tracking.trackers import TimeExpirationTracker, FixedRiskTrader, ATRTracker
from qube.tests.utils_for_tests import _read_timeseries_data
from pytest import fixture


@fixture(autouse=True)
def mock_pool_imap_unordered(monkeypatch):
    """
    Making single process from multiproc for being able to use mocked mongo
    """
    def _mock_start(obj): obj._target(*obj._args, **obj._kwargs)
    def _mock_join(obj): pass
    def _mock_imap_unordered(obj, func, args): return [func(a) for a in args]
    monkeypatch.setattr(
        "multiprocess.pool.Pool.imap_unordered", 
        lambda self, func, args=(): _mock_imap_unordered(self, func, args))
    monkeypatch.setattr(
        "multiprocessing.pool.Pool.imap_unordered", 
        lambda self, func, args=(): _mock_imap_unordered(self, func, args))
    monkeypatch.setattr("multiprocess.context.Process.start", lambda self: _mock_start(self))
    monkeypatch.setattr("multiprocess.context.Process.join", lambda self: _mock_join(self))

# def _read_csv_ohlc(symbol):
    # return {symbol: pd.read_csv(f'../data/{symbol}.csv', parse_dates=True, header=0, index_col='time')}


def _signals(sdata, as_series=False):
    if as_series:
        s = pd.Series(sdata)
    else:
        s = pd.DataFrame.from_dict(sdata, orient='index')
    s.index = pd.DatetimeIndex(s.index)
    return s


class Test(TestCase):
    def setUp(self):
        self.ds = _read_timeseries_data('ES', compressed=True, as_dict=True)
        self.data_bnc = _read_timeseries_data(
            'binance_perpetual_futures__BTCUSDT_ohlcv_M1', compressed=True, as_dict=False)
        self.ds_bnc = {'BTCUSDT': self.data_bnc}

    def test_simulation(self):
        m1 = MarketDataComposer(make_pipeline(RollingRange('1h', 12), RangeBreakoutDetector()),
                                SingleInstrumentPicker(), debug=True).fit(self.ds)

        m2 = MarketDataComposer(make_pipeline(RollingRange('3h', 4), RangeBreakoutDetector()), SingleInstrumentPicker(),
                                debug=True).fit(self.ds)

        r = simulation({
            'exp1 [simple break]': m1,
            'exp2 [time tracker]': [m2, TimeExpirationTracker('5h')]
        }, self.ds, 'forex', 'Test1')
        # debug_output(r.results[0].portfolio, 'Portfolio')

        self.assertAlmostEqual(24.50, r.results[0].portfolio['ES_PnL'].sum())
        self.assertAlmostEqual(46.75, r.results[1].portfolio['ES_PnL'].sum())

        r.report(1000, only_report=True)

    def test_simulation_threads(self):
        m1 = MarketDataComposer(make_pipeline(RollingRange('1h', 12), RangeBreakoutDetector()),
                                SingleInstrumentPicker(), debug=True).fit(self.ds)

        m2 = MarketDataComposer(make_pipeline(RollingRange('3h', 4), RangeBreakoutDetector()), SingleInstrumentPicker(),
                                debug=True).fit(self.ds)

        r = simulation({
            'exp1 [simple break]': m1,
            'exp2 [time tracker]': [m2, TimeExpirationTracker('5h')]
        }, self.ds, 'forex', 'Test1', ncpus=3)
        # debug_output(r.results[0].portfolio, 'Portfolio')

        self.assertAlmostEqual(24.50, r.results[0].portfolio['ES_PnL'].sum())
        self.assertAlmostEqual(46.75, r.results[1].portfolio['ES_PnL'].sum())

        r.report(1000, only_report=True)

    def test_simulation_fixed_risk_trader(self):
        m1 = MarketDataComposer(make_pipeline(RollingRange('1h', 12), RangeBreakoutDetector()),
                                SingleInstrumentPicker(), debug=True).fit(self.ds)

        r = simulation({
            'exp1 [FIXED TRADER]': [m1, FixedRiskTrader(10, 30, 10)],
        }, self.ds, 'forex', 'Test1')

        r.report(1000, only_report=True)

        print(" - - - - - - - - - - - - - - - - - - - ")
        print(r.results[0].trackers_stat)
        print(" - - - - - - - - - - - - - - - - - - - ")
        debug_output(r.results[0].executions, 'Execs', 5)
        print(" - - - - - - - - - - - - - - - - - - - ")
        self.assertEqual(9, r.results[0].trackers_stat['ES']['takes'])
        self.assertEqual(29, r.results[0].trackers_stat['ES']['stops'])
        self.assertAlmostEqual(-455.0, r.results[0].portfolio['ES_PnL'].sum())

    def test_simulation_fixed_risk_trader_pct(self):
        m1 = MarketDataComposer(make_pipeline(RollingRange('1h', 12), RangeBreakoutDetector()),
                                SingleInstrumentPicker(), debug=True).fit(self.ds)

        r = simulation({
            'FIXED TRADER PCT': [m1, FixedRiskTrader(10, 0.75, 0.5, in_percentage=True, accurate_stops=True)]
        }, self.ds, 'forex', 'Test1')

        r.report(1000, only_report=True)

        print(" - - - - - - - - - - - - - - - - - - - ")
        print(r.results[0].trackers_stat)
        print(" - - - - - - - - - - - - - - - - - - - ")
        debug_output(r.results[0].executions, 'Execs', 5)
        print(" - - - - - - - - - - - - - - - - - - - ")
        self.assertEqual(12, r.results[0].trackers_stat['ES']['takes'])
        self.assertEqual(18, r.results[0].trackers_stat['ES']['stops'])

        self.assertAlmostEqual(-929.29999, r.results[0].portfolio['ES_PnL'].sum(), 4)
        op = r.results[0].executions.iloc[0].exec_price
        stp = r.results[0].executions.iloc[1].exec_price
        actual_loss_pct = 100 * (op / stp - 1)
        self.assertAlmostEqual(0.5, actual_loss_pct, 2)

    def test_start_stop(self):
        r = simulation({'simple tracker': FixedRiskTrader(10, 30, 10, tick_size=1)},
                       self.ds, 'forex', 'Test1',
                       start='2021-01-03', stop='2021-01-04')
        self.assertEqual(r.results[0].portfolio.index[-1], pd.Timestamp('2021-01-05 00:00:00'))

    def test_new_commissions_impl_binance(self):
        sigs = _signals({
            '2019-09-08 19:15': {'BTCUSDT': +1.0},
            '2019-09-10 22:00': {'BTCUSDT': 0.0}
        })

        r_spot = simulation({
            'simple tracker spot': [sigs, FixedRiskTrader(
                100 / 10345.41,  # here $100 converted to amount of BTC
                30, 10,
                in_percentage=False,
                tick_size=1,
                take_by_limit_orders=False,
                reset_risks_on_repeated_signals=True,
                debug=True)]
        }, self.ds_bnc,
            'binance_spot_vip0_usdt',
            'Test1', start='2019-09-08', stop='2019-09-11'
        )

        r_fut = simulation({
            'simple tracker spot': [sigs, FixedRiskTrader(
                100,  # here $100 traded on future contract
                30, 10,
                in_percentage=False,
                tick_size=1,
                take_by_limit_orders=False,
                reset_risks_on_repeated_signals=True,
                debug=True)]
        }, self.ds_bnc,
            'binance_um_vip0_usdt',
            'Test1', start='2019-09-08', stop='2019-09-11')

        debug_output(r_spot.results[0].executions, 'Executions SPOT')
        print(' - - - - - - - - - - - - - - - ')
        debug_output(r_fut.results[0].executions, 'Executions FUTURES')
        print(' - - - - - - - - - - - - - - - ')

        c_spt = r_spot.results[0].portfolio['BTCUSDT_Commissions'].sum()
        p_spt = r_spot.results[0].portfolio['BTCUSDT_PnL'].sum()

        c_fut = r_fut.results[0].portfolio['BTCUSDT_Commissions'].sum()
        p_fut = r_fut.results[0].portfolio['BTCUSDT_PnL'].sum()

        print(f"Commissions SPOT: {c_spt} FUT: {c_fut}")
        print(f"PnL SPOT: {p_spt} FUT: {p_fut}")
        self.assertAlmostEqual(p_spt, p_fut)

        # Spot is 0.1% futures is 0.04% so commissions must be equal by this ratio
        self.assertAlmostEqual(c_spt * 0.04 / 0.1, c_fut, places=3)

    def test_ATR_tracker(self):
        data = _read_timeseries_data('EURUSD', as_dict=True)

        s = _signals({
            '2020-08-17 08:25:01': {'EURUSD': +1},
            '2020-08-17 10:25:01': {'EURUSD': +1},
            '2020-08-17 11:50:59': {'EURUSD': -1},
            '2020-08-17 23:19:59': {'EURUSD': 0},
        })

        r = simulation({
            'ATR TEST': [s, ATRTracker(1000, '5Min', 15, 1, 3, take_by_limit_orders=False, accurate_stops=True,
                                       debug=True)]
        }, data,
            'forex',
            'Test1', start='2020-08-17 00:00:00', stop='2020-08-18 00:00:00'
        )
        print(" - - - - - - - - - - - - - - - - - - - ")
        print(r.results[0].trackers_stat)
        print(r.results[0].portfolio['EURUSD_PnL'].sum())
        print(" - - - - - - - - - - - - - - - - - - - ")
        debug_output(r.results[0].executions, 'Execs', 25)
        print(" - - - - - - - - - - - - - - - - - - - ")
        self.assertAlmostEqual(-2.981, r.results[0].portfolio['EURUSD_PnL'].sum())
        self.assertEqual(0, r.results[0].trackers_stat['EURUSD']['takes'])
        self.assertEqual(2, r.results[0].trackers_stat['EURUSD']['stops'])

    def test_simulation_fixed_risk_trader_pct_position_sizing(self):
        m1 = MarketDataComposer(
            make_pipeline(RollingRange('1H', 12), RangeBreakoutDetector()),
            SingleInstrumentPicker(), debug=True
        ).fit(self.ds)

        r = simulation({
            'POSITION CALCULATOR': [m1, FixedRiskTrader(
                FixedRiskSizer(1000, 0.5), 0.25, 5,
                in_percentage=True,
                accurate_stops=True)]
        }, self.ds, 'stock', 'Test1')

        r.report(1000, only_report=True)

        print(" - - - - - - - - - - - - - - - - - - - ")
        print(f" | {r.results[0].portfolio['ES_PnL'].sum()}")
        print(" - - - - - - - - - - - - - - - - - - - ")
        print("\t" + str(r.results[0].trackers_stat))
        print(" - - - - - - - - - - - - - - - - - - - ")
        debug_output(r.results[0].executions, 'Execs', 5)
        print(" - - - - - - - - - - - - - - - - - - - ")

    def test_signal_generator_with_tracker(self):
        SIM_START = '2020-08-17 00:00:00'
        FIT_END = '2020-08-17 15:00:00'
        SIM_END = '2020-08-18 00:00:00'

        @signal_generator
        class TestGenerator(BaseEstimator):
            def __init__(self, capital, signals: dict):
                self.signals = _signals(signals)
                self.capital = capital
                self._test_case = TestCase()

            def fit(self, x, y, **kwargs):
                # well let's test here ranges of fitting
                self._test_case.assertTrue(x.index[0] >= pd.to_datetime(SIM_START))
                self._test_case.assertTrue(x.index[-1] <= pd.to_datetime(FIT_END))

                # nothing to do
                return self

            def predict(self, x):
                # well let's test here ranges of prediction
                self._test_case.assertTrue(x.index[0] >= pd.to_datetime(SIM_START))
                self._test_case.assertTrue(x.index[-1] > pd.to_datetime(FIT_END))
                self._test_case.assertTrue(x.index[-1] <= pd.to_datetime(SIM_END))

                self.exact_time = True
                return self.signals[self.market_info_.symbols]

            def tracker(self, capital=None):
                return ATRTracker(self.capital, '5Min', 15, 1, 3,
                                  take_by_limit_orders=False, accurate_stops=True, debug=True)

        r = simulation({
            'POSITION CALCULATOR': TestGenerator(
                1000,
                {
                    '2020-08-17 08:25:01': {'EURUSD': +1},
                    '2020-08-17 10:25:01': {'EURUSD': +1},
                    '2020-08-17 11:50:59': {'EURUSD': -1},
                    '2020-08-17 23:19:59': {'EURUSD': 0}
                })
        }, _read_timeseries_data('EURUSD', as_dict=True),
            'forex', 'Test1', start=SIM_START, stop=SIM_END, fit_stop=FIT_END
        )

        # - same results as in usual way of using generator + tracker
        print(" - - - - - - - - - - - - - - - - - - - ")
        print(r.results[0].trackers_stat)
        print(" - - - - - - - - - - - - - - - - - - - ")
        debug_output(r.results[0].executions, 'Execs', 25)
        print(" - - - - - - - - - - - - - - - - - - - ")

        self.assertAlmostEqual(-2.981, r.results[0].portfolio['EURUSD_PnL'].sum())
        self.assertEqual(0, r.results[0].trackers_stat['EURUSD']['takes'])
        self.assertEqual(2, r.results[0].trackers_stat['EURUSD']['stops'])


from pytest import main
if __name__ == '__main__':
    main()