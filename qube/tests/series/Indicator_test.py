import unittest

import numpy as np
import pandas as pd

from qube.quantitative.stats.stats import percentile_rank
from qube.quantitative.ta.indicators import (ema, tema, dema, atr, kama, bollinger, macd,
                                             trend_detector, bollinger_atr, sma, denoised_trend, rolling_std_with_mean)
from qube.series.BarSeries import BarSeries
from qube.series.DoubleSeries import DoubleSeries
from qube.series.Indicators import (Indicator, Sma, Ema, Tema, Dema, ATR, MovingMinMax, KAMA, Bollinger, Returns,
                                    DailyHighLow, MACD, TrendDetector, BollingerATR, DenoisedTrend, DenoisedDeltaRank,
                                    RollingStd, WilliamsR)
from qube.series.Quote import Quote
from qube.tests.data.test_mdf import generate_feed
from qube.utils.DateUtils import DateUtils


class Adder(Indicator):
    def __init__(self, add_v):
        super().__init__(2)
        self.add = add_v

    def calculate(self, v): return v + self.add


class IndicatorTest(unittest.TestCase):
    @staticmethod
    def __update_by_quotes(series: BarSeries, quotes):
        for r in quotes.iterrows():
            rdata = r[1]
            q = Quote(r[0], rdata['bid'], rdata['ask'], rdata['bidvol'], rdata['askvol'])
            series.update_by_quote(q)
        return series

    def test_indicator(self):
        s = Ema(2)
        a = Adder(100).attach(s)

        a.update(1)
        print(a[0], s[0])

        a.update(2)
        print(a[0], s[0])

        a.update(3)
        print(a[0], s[0])

        a.update(4)
        print(a[0], s[0])

        print(len(a))

    def test_ind_2(self):
        P = 4
        ema0 = Ema(P, False)
        ema1 = Ema(P, False)

        ema0.update(1);
        print(ema0[:])
        ema0.update(2);
        print(ema0[:])
        ema0.update(3, False);
        print(ema0[:])
        ema0.update(100, False);
        print(ema0[:])
        ema0.update(-100, False);
        print(ema0[:])
        ema0.update(4);
        print(ema0[:])
        ema0.update(5);
        print(ema0[:])

        ema1.update(1)
        ema1.update(2)
        ema1.update(4)
        ema1.update(5)
        print(ema1[:])
        self.assertAlmostEqual(ema0[:], ema1[:])

        # compare to standard ema
        x_ema = ema(np.array([1, 2, 4, 5]), P, init_mean=False).flatten().tolist()[::-1]
        self.assertAlmostEqual(x_ema, ema0[:], delta=1e-6)

        sma0 = Sma(4)
        sma0.update(1)
        sma0.update(2)
        sma0.update(3)
        sma0.update(4)
        sma0.update(5);
        print(sma0[0])
        sma0.update(611, False);
        print(sma0[0])
        sma0.update(711, False);
        print(sma0[0])
        sma0.update(6);
        print(sma0[0])
        sma0.update(7)
        print(sma0[:])

        tema0 = Tema(4, False)
        xx = np.array([1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3])
        [tema0.update(x) for x in xx]
        self.assertAlmostEqual(tema(xx, 4, init_mean=False).flatten().tolist()[::-1], tema0[:], delta=1e-6)

    def test_ind_init_values(self):
        P = 3
        e0 = Ema(P, True)

        vv = np.array([1, 2, 4, 5, 6, 5, 4, 3, 2, 1])

        [e0.update(v) for v in vv]
        print('\n\n Ema0:', e0[:])

        x_ema = ema(vv, P, init_mean=True).flatten().tolist()[::-1]
        print(' XEma:', x_ema)

        np.testing.assert_almost_equal(x_ema, e0[:], 5, err_msg='Ema values are not equal')

        # test when we have some non-appendable data on initial period
        e1 = Ema(P, True)
        e1.update(100, False)
        e1.update(200, False)
        e1.update(vv[0])

        e1.update(200, False)
        e1.update(-300, False)
        e1.update(vv[1])
        e1.update(vv[2])
        [e1.update(v) for v in vv[3:]]
        print(' Ema1:', e1[:])
        np.testing.assert_almost_equal(x_ema, e1[:], 5, err_msg='Ema1 values are not equal')

        e2 = Ema(P, True)
        xx = np.array([np.nan, np.nan, np.nan, 1, 2, 3, 4, 5, 6])
        xe2 = ema(xx, P, init_mean=True).flatten().tolist()[::-1]
        [e2.update(x) for x in xx]
        print(e2[:])
        print(xe2[:])
        np.testing.assert_almost_equal(e2[:], xe2[:], 5, err_msg='Ema (nan inits) values are not equal')

        tema0 = Tema(P, True)
        xx = np.array([1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1])
        [tema0.update(x) for x in xx]
        print(tema0[:])
        np.testing.assert_almost_equal(tema(xx, P, init_mean=True).flatten().tolist()[::-1],
                                       tema0[:], 5, err_msg='Tema values are not equal')

    def test_ind_dema(self):
        P = 3
        e0 = Dema(P, True)
        vv = np.array([1, 2, 4, 5, 6, 5, 4, 3, 2, 1])

        e0.update(vv)
        print('\n\n Dema0:', e0[:])

        x_dema = dema(vv, P, init_mean=True).flatten().tolist()[::-1]
        print(' XDEma:', x_dema)

        np.testing.assert_almost_equal(x_dema, e0[:], 5, err_msg='Dema values are not equal')

    def test_atr(self):
        start_time = DateUtils.get_datetime('2017-08-01 00:00:00')
        df = generate_feed(start_time, 10.0, 10000)
        s1 = BarSeries('1Min', df)

        a0 = ATR(14, 'ema')
        a0.update(s1[::-1])
        print('\nATR:', a0[::-1])

        x_atr = atr(s1.to_frame(), 14, 'ema').values
        np.testing.assert_almost_equal(x_atr, a0[::-1], 5, err_msg='ATR values are not equal')

    def test_streaming_barseries_updates(self):
        start_time = DateUtils.get_datetime('2017-08-01 00:00:00')
        df_1 = generate_feed(start_time, 10.0, 10000)

        # here we create bar series push some data and attach ATR indicator
        s1 = BarSeries(60, df_1)

        a0 = ATR(14, 'ema')
        s1.attach(a0)

        e0 = Ema(5)
        s1.attach(e0)

        # later we emulate updating series by streaming quotes
        df_2 = generate_feed(df_1.index[-1], 10.0, 10000)
        s1 = IndicatorTest.__update_by_quotes(s1, df_2)

        print(a0, e0)

        # finally we compare atr calculated from bar series and atr calculated by closing prices
        x_atr = atr(s1.to_frame(), 14, 'ema').values
        np.testing.assert_almost_equal(x_atr, a0[::-1], 5, err_msg='ATR values are not equal')

        x_ema = ema(s1.to_frame().close, 5).flatten()
        np.testing.assert_almost_equal(x_ema, e0[::-1], 5, err_msg='EMA values are not equal')

    def test_min_max(self):
        start_time = DateUtils.get_datetime('2017-08-01 00:00:00')
        df_1 = generate_feed(start_time, 10.0, 10000)

        # here we create bar series push some data and attach ATR indicator
        s1 = BarSeries(60, df_1)

        mm = MovingMinMax(7)
        s1.attach(mm)

        # TODO: here we need test
        print(mm[:])

    def test_daily_hl(self):
        start_time = DateUtils.get_datetime('2017-08-01 23:55:00')
        df_1 = generate_feed(start_time, 10.0, 10000)

        s1 = BarSeries(300, df_1)

        hl = DailyHighLow()
        s1.attach(hl)

        self.assertAlmostEqual(hl[0][0], 5.910, msg="Low in DailyHighLow are not equal")
        self.assertAlmostEqual(hl[0][1], 9.925, msg="High in DailyHighLow are not equal")

        self.assertAlmostEqual(hl[-1][0], 9.565, msg="Low in DailyHighLow are not equal")
        self.assertAlmostEqual(hl[-1][1], 10.100, msg="High in DailyHighLow are not equal")

    def test_sma(self):
        xin = np.random.randn(100, 15)
        xin[0:10, 0] = np.nan
        np.testing.assert_almost_equal(np.nansum(sma(xin, 5) - (pd.DataFrame(xin).rolling(window=5).sum() / 5).values),
                                       0, 10, err_msg='sma values are not equal')

    def test_kama(self):
        k_period = 7
        start_time = DateUtils.get_datetime('2017-08-01 00:00:00')
        df_1 = generate_feed(start_time, 10.0, 10000)

        # here we create bar series push some data and attach KAMA indicator
        s1: BarSeries = BarSeries(60, df_1)

        k1 = KAMA(k_period)
        s1.attach(k1)

        x_kama = kama(s1.to_frame().close, k_period).flatten()
        print(np.array([k1[::-1], x_kama.T]).T)

        # here we drop first k_period bars because standard kama doesn't include first value
        np.testing.assert_almost_equal(x_kama[k_period:], k1[::-1][k_period:], 5, err_msg='KAMA values are not equal')

        # stage 2: pushing streaming updates
        df_2 = generate_feed(df_1.index[-1], 10.0, 10000)
        s1 = IndicatorTest.__update_by_quotes(s1, df_2)

        # compare final results
        x_kama = kama(s1.to_frame().close, k_period).flatten()
        np.testing.assert_almost_equal(x_kama[k_period:], k1[::-1][k_period:], 5, err_msg='KAMA values are not equal')

    def test_bollinger(self):
        b_period = 7
        b_std = 2
        b_mm = 'ema'

        start_time = DateUtils.get_datetime('2017-08-01 00:00:00')
        df_1 = generate_feed(start_time, 10.0, 10000)

        # here we create bar series push some data and attach KAMA indicator
        s1: BarSeries = BarSeries(300, df_1)

        b1 = Bollinger(b_period, b_std, b_mm)
        s1.attach(b1)

        x_m, x_u, x_l = bollinger(s1.to_frame().close, b_period, b_std, b_mm)
        print(np.concatenate((np.asarray(b1[::-1]), np.array([x_l, x_u]).T), axis=1))

        np.testing.assert_almost_equal(np.asarray(b1[::-1]),
                                       np.array([x_l, x_u]).T, 5, err_msg='Bollinger values are not equal')

        # stage 2: pushing streaming updates
        df_2 = generate_feed(df_1.index[-1], 10.0, 10000)
        s1 = IndicatorTest.__update_by_quotes(s1, df_2)

        x_m, x_u, x_l = bollinger(s1.to_frame().close, b_period, b_std, b_mm)
        np.testing.assert_almost_equal(np.asarray(b1[::-1]),
                                       np.array([x_l, x_u]).T, 5, err_msg='Bollinger values are not equal')

    def test_returns(self):
        start_time = DateUtils.get_datetime('2017-08-01 00:00:00')
        df_1 = generate_feed(start_time, 10.0, 10000)

        # here we create bar series push some data and attach KAMA indicator
        s1: BarSeries = BarSeries(300, df_1)

        r1 = Returns(1, True)
        s1.attach(r1)

        r2 = Returns(1, False)
        s1.attach(r2)

        # stage 2: pushing streaming updates
        df_2 = generate_feed(df_1.index[-1], 10.0, 10000)
        s1 = IndicatorTest.__update_by_quotes(s1, df_2)

        f1 = s1.to_frame()
        x_r_1 = f1.close / f1.open - 1.0

        np.testing.assert_almost_equal(np.asarray(r1[::-1]), x_r_1.values, 5, err_msg='Returns 1 values are not equal')

        x_r_2 = f1.close / f1.shift(1).close - 1.0

        np.testing.assert_almost_equal(np.asarray(r2[::-1]), x_r_2.values, 5, err_msg='Returns 2 values are not equal')

    def test_chained_indicators(self):
        b_period = 7
        b_std = 2.5
        b_mm = 'ema'
        start_time = DateUtils.get_datetime('2017-08-01 00:00:00')
        df_1 = generate_feed(start_time, 10.0, 10000)

        # here we create bar series push some data
        s1: BarSeries = BarSeries(60, df_1)

        r1 = Returns(100.0, True)
        s1.attach(r1)

        b1 = Bollinger(period=b_period, nstd=b_std, mean_model=b_mm)
        r1.attach(b1)

        x_m, x_u, x_l = bollinger(pd.Series(r1[::-1]), b_period, b_std, b_mm)
        # print(np.concatenate((np.asarray(b1[::-1]), np.array([x_l, x_u]).T), axis=1))

        np.testing.assert_almost_equal(np.asarray(b1[::-1]),
                                       np.array([x_l, x_u]).T, 5, err_msg='Bollinger values are not equal')

        # stage 2: pushing streaming updates
        df_2 = generate_feed(df_1.index[-1], 10.0, 10000)
        IndicatorTest.__update_by_quotes(s1, df_2)

        x_m, x_u, x_l = bollinger(pd.Series(r1[::-1]), b_period, b_std, b_mm)
        print(np.concatenate((np.asarray(b1[::-1]), np.array([x_l, x_u]).T), axis=1))

        np.testing.assert_almost_equal(np.asarray(b1[::-1]),
                                       np.array([x_l, x_u]).T, 5, err_msg='Bollinger values are not equal')

    def test_macd_indicator(self):
        start_time = DateUtils.get_datetime('2017-08-01 00:00:00')
        df = generate_feed(start_time, 10.0, 10000)
        s1 = BarSeries(60, df)
        s2 = BarSeries(60)
        mcd_1 = MACD(12, 26, 9, 'ema', 'sma')
        mcd_2 = MACD(12, 26, 9, 'ema', 'sma')
        s1.attach(mcd_1)
        s2.attach(mcd_2)

        s2 = IndicatorTest.__update_by_quotes(s2, df)
        rm = macd(s2.to_frame()['close'], 12, 26, 9, 'ema', 'sma')

        np.testing.assert_almost_equal(np.asarray(mcd_1[::-1]), rm.values, 5, err_msg='MACD values are not equal')
        np.testing.assert_almost_equal(np.asarray(mcd_2[::-1]), rm.values, 5, err_msg='MACD values are not equal')

    def test_bollinger_atr(self):
        start_time = DateUtils.get_datetime('2017-08-01 00:00:00')
        df_1 = generate_feed(start_time, 10.0, 10000)

        # here we create bar series push some data and attach KAMA indicator
        s1: BarSeries = BarSeries(300, df_1)

        b1 = BollingerATR(12, 5, 2, 'ema', 'kama')
        s1.attach(b1)

        x_m, x_u, x_l = bollinger_atr(s1.to_frame(), 12, 5, 2, 'ema', 'kama')
        print(np.concatenate((np.asarray(b1[::-1]), np.array([x_l, x_u]).T), axis=1))

        np.testing.assert_almost_equal(np.asarray(b1[::-1]),
                                       np.array([x_l, x_u]).T, 5, err_msg='BollingerATR values are not equal')

        # stage 2: pushing streaming updates
        df_2 = generate_feed(df_1.index[-1], 10.0, 10000)
        s1 = IndicatorTest.__update_by_quotes(s1, df_2)

        x_m, x_u, x_l = bollinger_atr(s1.to_frame(), 12, 5, 2, 'ema', 'kama')
        np.testing.assert_almost_equal(np.asarray(b1[::-1]),
                                       np.array([x_l, x_u]).T, 5, err_msg='BollingerATR values are not equal')

    def test_trend_detector(self):
        df = generate_feed(DateUtils.get_datetime('2017-08-01 00:00:00'), 10.0, 10000)
        s1 = BarSeries('5Min', df)
        s2 = BarSeries('5Min')

        td1 = TrendDetector(20, 1, 'ema', True)
        td2 = TrendDetector(20, 1, 'ema', True)
        s1.attach(td1)
        s2.attach(td2)

        s2 = IndicatorTest.__update_by_quotes(s2, df)
        test = trend_detector(s2.to_frame()['close'], 20, 1, 'bb', True, 'ema')

        test = test.fillna(0.0)
        td1_f = pd.DataFrame(td1[::-1], index=s1.times(), columns=['uts', 'dts', 'trend', 'utl', 'dtl']).fillna(0.0)
        td2_f = pd.DataFrame(td2[::-1], index=s2.times(), columns=['uts', 'dts', 'trend', 'utl', 'dtl']).fillna(0.0)

        np.testing.assert_almost_equal(td1_f.values, test.values, 5, err_msg='TrendDetector values 1 are not equal')
        np.testing.assert_almost_equal(td2_f.values, test.values, 5, err_msg='TrendDetector values 2 are not equal')

    def test_trend_detector_atr(self):
        df = generate_feed(DateUtils.get_datetime('2017-08-01 00:00:00'), 10.0, 10000)
        s1 = BarSeries('5Min', df)
        s2 = BarSeries('5Min')

        td1 = TrendDetector(20, 1, 'ema', True, 1, True, 5, 'ema')
        td2 = TrendDetector(20, 1, 'ema', True, 1, True, 5, 'ema')
        s1.attach(td1)
        s2.attach(td2)

        s2 = IndicatorTest.__update_by_quotes(s2, df)
        test = trend_detector(s2.to_frame(), 20, 1, 'bbatr', True, 'ema', 5, 'ema')

        test = test.fillna(0.0)
        td1_f = pd.DataFrame(td1[::-1], index=s1.times(), columns=['uts', 'dts', 'trend', 'utl', 'dtl']).fillna(0.0)
        td2_f = pd.DataFrame(td2[::-1], index=s2.times(), columns=['uts', 'dts', 'trend', 'utl', 'dtl']).fillna(0.0)

        np.testing.assert_almost_equal(td1_f.values, test.values, 5, err_msg='TrendDetector values 1 are not equal')
        np.testing.assert_almost_equal(td2_f.values, test.values, 5, err_msg='TrendDetector values 2 are not equal')

    def test_denoised_trend_detector(self):
        df = generate_feed(DateUtils.get_datetime('2017-08-01 00:00:00'), 10.0, 10000)
        s1 = BarSeries('5Min', df)
        s2 = BarSeries('5Min')

        dt1 = DenoisedTrend(10, 0, None, False)
        s1.attach(dt1)
        dt1_test = denoised_trend(s1.to_frame(), 10, 0, None, False)
        df1 = pd.DataFrame(dt1[::-1], index=s1.times())

        np.testing.assert_almost_equal(df1.values.flatten(), dt1_test.values.flatten(), 5,
                                       err_msg="DenoisedTrend indicator 1")

        dt2 = DenoisedTrend(10, 0, None, True)
        s1.attach(dt2)
        dt2_test = denoised_trend(s1.to_frame(), 10, 0, None, True)
        df2 = pd.DataFrame(dt2[::-1], index=s1.times())

        np.testing.assert_almost_equal(df2.values.flatten(), dt2_test.values.flatten(), 5,
                                       err_msg="DenoisedTrend indicator 2")

        dt3 = DenoisedTrend(8, 0, None, True)
        s2.attach(dt3)
        s2 = IndicatorTest.__update_by_quotes(s2, df)
        dt3_test = denoised_trend(s2.to_frame(), 8, 0, None, True)
        df3 = pd.DataFrame(dt3[::-1], index=s2.times())

        np.testing.assert_almost_equal(df3.values.flatten(), dt3_test.values.flatten(), 5,
                                       err_msg="DenoisedTrend indicator 3")

    def test_denoised_delta_rank(self):
        def delta_rank(x, f_p, s_p, n_back=30, qts=np.arange(10, 100, 10)):
            if qts[-1] < 100:
                qts = np.append(qts, 100)
            fast_trend, slow_trend = (denoised_trend(x, f_p, 0, None), denoised_trend(x, s_p, 0, None))
            delta = fast_trend - slow_trend
            slot = delta.rolling(window=n_back).apply(lambda x: percentile_rank(x[:-1], x[-1], qts))
            slot = slot * np.sign(slow_trend)
            return slot

        df = generate_feed(DateUtils.get_datetime('2017-08-01 00:00:00'), 10.0, 10000)
        s1 = BarSeries('5Min', df)
        dd1 = DenoisedDeltaRank(10, 5, 10, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        s1.attach(dd1)
        dd1_test = delta_rank(s1.to_frame(), 10, 5, 10, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        df1 = pd.DataFrame(dd1[::-1], index=s1.times())
        np.testing.assert_almost_equal(df1.values.flatten(), dd1_test.values.flatten(), 5,
                                       err_msg="DenoisedDeltaRank indicator 1")

        dd2 = DenoisedDeltaRank(8, 12, 11, [10, 20, 30, 100])
        s1.attach(dd2)
        df2 = pd.DataFrame(dd2[::-1], index=s1.times())
        dd2_test = delta_rank(s1.to_frame(), 8, 12, 11, [10, 20, 30, 100])
        np.testing.assert_almost_equal(df2.values.flatten(), dd2_test.values.flatten(), 5,
                                       err_msg="DenoisedDeltaRank indicator 2")

        dd3 = DenoisedDeltaRank(5, 10, 5, [10, 20, 30, 100])
        s2 = BarSeries('5Min')
        s2.attach(dd3)
        s2 = IndicatorTest.__update_by_quotes(s2, df)
        df3 = pd.DataFrame(dd3[::-1], index=s2.times())
        dd3_test = delta_rank(s2.to_frame(), 5, 10, 5, [10, 20, 30, 100])

        np.testing.assert_almost_equal(df3.values.flatten(), dd3_test.values.flatten(), 5,
                                       err_msg="DenoisedDeltaRank indicator 3")

    def test_rolling_std(self):
        _x = pd.Series(np.random.rand(2000), index=pd.date_range('2000-01-01', periods=2000, freq='1d'))
        isma = Sma(200)
        rtsd = RollingStd(200, isma)
        xs = DoubleSeries('1D')
        xs.attach(rtsd)
        xs.attach(isma)

        for xi, ti in zip(_x, _x.index):
            xs.update_by_value(ti, xi)

        np.testing.assert_almost_equal(
            pd.Series(rtsd[::-1], _x.index).values[-1600:],
            rolling_std_with_mean(_x, sma(_x, 200).flatten(), 200).values[-1600:],
            4
        )

    def test_william_r(self):
        df = generate_feed(DateUtils.get_datetime('2017-08-01 00:00:00'), 10.0, 10000)
        s1 = BarSeries('5Min', df)
        s2 = BarSeries('5Min')

        wr1 = WilliamsR(14)
        wr2 = WilliamsR(14)
        s1.attach(wr1)
        s2.attach(wr2)
        IndicatorTest.__update_by_quotes(s2, df)
        np.testing.assert_almost_equal(wr1[::], wr2[::])
        self.assertAlmostEqual(wr1[2], -80.093, delta=0.01)
        self.assertAlmostEqual(wr1[6], -83.7209, delta=0.01)


from pytest import main
if __name__ == '__main__':
    main()