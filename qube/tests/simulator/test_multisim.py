from unittest import TestCase

import pandas as pd
from sklearn.pipeline import make_pipeline

from qube.examples.learn.generators import RangeBreakoutDetector
from qube.examples.learn.transformers import RollingRange
from qube.learn.core.base import MarketDataComposer
from qube.learn.core.pickers import SingleInstrumentPicker
from qube.learn.core.utils import debug_output
from qube.simulator.multisim import simulation
from qube.simulator.tracking.trackers import TimeExpirationTracker, FixedTrader


class Test(TestCase):
    def setUp(self):
        self.data = pd.read_csv('../data/ES.csv.gz', parse_dates=True, index_col=['time'])
        self.ds = {'ES': self.data}

        self.data_bnc = pd.read_csv(
            '../data/binance_perpetual_futures__BTCUSDT_ohlcv_M1.csv.gz',
            parse_dates=True, index_col=['time']
        )
        self.ds_bnc = {'BTCUSDT': self.data_bnc}

    def test_simulation(self):
        m1 = MarketDataComposer(make_pipeline(RollingRange('1H', 12), RangeBreakoutDetector()),
                                SingleInstrumentPicker(), debug=True).fit(self.ds)

        m2 = MarketDataComposer(make_pipeline(RollingRange('3H', 4), RangeBreakoutDetector()), SingleInstrumentPicker(),
                                debug=True).fit(self.ds)

        r = simulation({
            'exp1 [simple break]': m1,
            'exp2 [time tracker]': [m2, TimeExpirationTracker('5H')]
        }, self.ds, 'forex', 'Test1')
        # debug_output(r.results[0].portfolio, 'Portfolio')

        self.assertAlmostEqual(24.50, r.results[0].portfolio['ES_PnL'].sum())
        self.assertAlmostEqual(46.75, r.results[1].portfolio['ES_PnL'].sum())

        r.report(1000, only_report=True)

    def test_simulation_fixed(self):
        m1 = MarketDataComposer(make_pipeline(RollingRange('1H', 12), RangeBreakoutDetector()),
                                SingleInstrumentPicker(), debug=True).fit(self.ds)

        r = simulation({
            'exp1 [FIXED TRADER]': [m1, FixedTrader(10, 30, 10, 1)]
        }, self.ds, 'forex', 'Test1')

        r.report(1000, only_report=True)

        print(r.results[0].trackers_stat)
        self.assertEqual(3, r.results[0].trackers_stat['ES']['takes'])
        self.assertEqual(42, r.results[0].trackers_stat['ES']['stops'])
        self.assertAlmostEqual(1160.0, r.results[0].portfolio['ES_PnL'].sum())

    def test_start_stop(self):
        r = simulation({'simple tracker': FixedTrader(10, 30, 10, 1)},
                       self.ds, 'forex', 'Test1',
                       start='2021-01-03', stop='2021-01-04')

    def test_new_commissions_impl_binance(self):
        def _signals(sdata):
            s = pd.DataFrame.from_dict(sdata, orient='index')
            s.index = pd.DatetimeIndex(s.index)
            return s

        sigs = _signals({
            '2019-09-08 19:15': {'BTCUSDT': +1.0},
            '2019-09-10 22:00': {'BTCUSDT': 0.0}
        })

        r_spot = simulation({
            'simple tracker spot': [sigs, FixedTrader(
                100 / 10345.41,  # here $100 converted to amount of BTC
                30, 10, 1, take_by_limit_orders=False, debug=True)]
        }, self.ds_bnc,
            'binance_spot_vip0_usdt',
            'Test1', start='2019-09-08', stop='2019-09-11'
        )

        r_fut = simulation({
            'simple tracker spot': [sigs, FixedTrader(
                100,  # here $100 traded on future contract
                30, 10, 1, take_by_limit_orders=False, debug=True)]
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
