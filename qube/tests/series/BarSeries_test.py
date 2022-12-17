from unittest import TestCase
from pytest import main

import numpy as np
from pandas.testing import assert_frame_equal

from qube.series.BarSeries import BarSeries, PriceType
from qube.series.Quote import Quote
from qube.tests.data.test_mdf import generate_feed
from qube.utils.DateUtils import DateUtils


class TestBarsSeries(TestCase):

    def test_basic(self):
        start_time = DateUtils.get_datetime('2017-08-24 12:58:22')
        df = generate_feed(start_time, 10.0, 10000)

        s1 = BarSeries(300, df)
        s2 = BarSeries(300)

        for time, row in df.iterrows():
            quote = Quote(time, row['bid'], row['ask'], row['bidvol'], row['askvol'])
            s2.update_by_quote(quote)

        resampled = df[['ask', 'bid']].mean(axis=1)
        resampled = resampled.resample('5Min').agg('ohlc')
        ohlcv_frame = s1.to_frame()
        ohlcv_frame.drop('volume', axis=1, inplace=True)

        assert_frame_equal(resampled, ohlcv_frame, check_names=False)
        self.assertAlmostEqual(s1.opens(), s2.opens(), delta=1e-5)
        self.assertAlmostEqual(s1.highs(), s2.highs(), delta=1e-5)
        self.assertAlmostEqual(s1.lows(), s2.lows(), delta=1e-5)
        self.assertAlmostEqual(s1.closes(), s2.closes(), delta=1e-5)
        self.assertAlmostEqual(s1.volumes(), s2.volumes(), delta=1e-5)
        self.assertEqual(s1.last_bar_volume, s2.last_bar_volume)
        self.assertEqual(s1.last_price, s2.last_price)
        self.assertEqual(s1.last_updated, s2.last_updated)
        self.assertEqual(s1.is_new_bar, s2.is_new_bar)

    def test_bars(self):
        start_time = DateUtils.get_datetime('2017-08-24 00:00:00')
        df = generate_feed(start_time, 10.0, 60 * 5 * 10)
        s1 = BarSeries(300, df)
        self.assertEqual(len(s1), 10)

        # first bar
        self.assertEqual(s1[-1].time, start_time)

        # last recent bar and previous one
        self.assertEqual(s1[0].time, DateUtils.get_datetime('2017-08-24 00:45:00'))
        self.assertEqual(s1[1].time, DateUtils.get_datetime('2017-08-24 00:40:00'))

        # test slices
        l_bars = s1[0:11]
        self.assertEqual(len(l_bars), 10)
        self.assertEqual(l_bars[-1], s1[9])

    def test_from_ohlcv_df(self):
        start_time = DateUtils.get_datetime('2017-08-24 13:01:12')
        df = generate_feed(start_time, 10.0, 10000)
        s1 = BarSeries('5Min', df, price_type=PriceType.VWMPT)
        ohlcv_df = s1.to_frame()
        s2 = BarSeries('5Min', ohlcv_df, price_type=PriceType.VWMPT)
        assert_frame_equal(ohlcv_df, s2.to_frame())
        self.assertEqual(s1.last_bar_volume, s2.last_bar_volume)
        self.assertEqual(s1.last_price, s2.last_price)
        self.assertEqual(s1.last_updated, s2.last_updated)
        self.assertEqual(s1.is_new_bar, s2.is_new_bar)
        self.assertEqual(s2.to_frame().index.dtype.type, np.datetime64)

    def test_freq(self):
        self.assertEqual(BarSeries.get_pd_freq_by_seconds_freq_val(300), '5Min')
        self.assertEqual(BarSeries.get_pd_freq_by_seconds_freq_val(60), '1Min')
        self.assertEqual(BarSeries.get_pd_freq_by_seconds_freq_val(3600), '1H')
        self.assertEqual(BarSeries.get_pd_freq_by_seconds_freq_val(3600 * 24 * 300), '300D')
        self.assertEqual(BarSeries.get_pd_freq_by_seconds_freq_val(3600 * 24 * 365), '1A')
        self.assertEqual(BarSeries.get_pd_freq_by_seconds_freq_val(10), '10s')

    def test_bars_2(self):
        start_time = DateUtils.get_datetime('2017-08-01 00:00:00')
        df_1 = generate_feed(start_time, 10.0, 10000)
        s1: BarSeries = BarSeries(60, df_1)

        df_2 = generate_feed(df_1.index[-1], 10.0, 10000)
        for r in df_2.iterrows():
            rdata = r[1]
            q = Quote(r[0], rdata['bid'], rdata['ask'], rdata['bidvol'], rdata['askvol'])
            s1.update_by_quote(q)

        s1f = s1.to_frame()

        # it will show that all data aggregated from df_2 has same values for open and closes !!!!
        self.assertTrue(s1f.close[-2] - s1f.open[-2] != 0.0)

    def test_max_series_len(self):
        start_time = DateUtils.get_datetime('2017-08-24 12:58:22')
        df = generate_feed(start_time, 10.0, 10000)

        s1 = BarSeries(300, df)
        s2 = BarSeries(300, max_series_length=5)

        for time, row in df.iterrows():
            quote = Quote(time, row['bid'], row['ask'], row['bidvol'], row['askvol'])
            s2.update_by_quote(quote)
        assert_frame_equal(s1.to_frame()[-5:], s2.to_frame(), check_dtype=False)


if __name__ == '__main__':
    main()