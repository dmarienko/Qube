import unittest

import numpy as np
import pandas as pd
from numpy import nan

from qube.quantitative.ta.indicators import ema, shift, moving_ols, series_halflife, kama, dema, tema, denoised_trend, \
    pivot_point
from qube.quantitative.tools import add_constant, ohlc_resample


class TestTimeSeriesUtils(unittest.TestCase):

    def test_ema(self):
        r1 = ema(np.array([1, 5, 10, 4, 3]), 3, init_mean=True)
        r2 = ema(np.array([1, 5, 10, 4, 3]), 3, init_mean=False, min_periods=3)

        np.testing.assert_almost_equal(r1, np.array([[nan], [nan], [5.33333333], [4.66666667], [3.83333333]]),
                                       5, err_msg='EMA init_mean are not equal')

        np.testing.assert_almost_equal(r2, np.array([[nan], [nan], [6.5], [5.25], [4.125]]),
                                       5, err_msg='EMA init_mean False are not equal')

        r3 = ema(np.array([[1, 5, 10, 4, 3, 4, 5],
                           [nan, nan, 1, 5, 10, 4, 3]]).T, 3, init_mean=True)
        np.testing.assert_almost_equal(r3, np.array([[nan, nan],
                                                     [nan, nan],
                                                     [5.33333333, nan],
                                                     [4.66666667, nan],
                                                     [3.83333333, 5.33333333],
                                                     [3.91666667, 4.66666667],
                                                     [4.45833333, 3.83333333]]), 5,
                                       err_msg='EMA multiple columns are not equal')

    def test_shift(self):
        np.testing.assert_almost_equal(
            shift(np.array([[1., 2.],
                            [11., 22.],
                            [33., 44.]]), 1),
            np.array([[nan, nan],
                      [1., 2.],
                      [11., 22.]]),
            1, err_msg='Shifted values are not equal')

    def test_moving_ols(self):
        np.random.seed(123)
        N = 1000
        WINSIZE = 100
        # some linear function with minimal noise
        x = np.linspace(1, 2, N)
        y = 2 * x + 5 + 0.01 * np.random.randn(N)
        b, e, s = moving_ols(y, add_constant(x), WINSIZE)
        np.testing.assert_almost_equal(np.array([5., 2.]), b[~np.isnan(b)[:, 0]].mean(axis=0), decimal=2)

    def test_series_halflife(self):
        np.testing.assert_almost_equal(series_halflife(np.array([1, 0, 2, 3, 2, 1, -1, -2, 0, 1])), 2.0)

    def test_kama(self):
        x = np.array([
            [1., 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        ]).T
        x2 = np.array([
            [11., 22, 33, 44, 55, 66, 77, 88, 99, 88, 77, 66, 55, 44, 33, 22, 11]
        ]).T

        our_data_1 = kama(x, 5, 2, 30)
        our_data_2 = kama(x2, 5, 2, 30)

        talib_data_1 = np.array([nan, nan, nan, nan, nan, 5.44444444, 6.13580247, 6.96433471, 7.86907484, 7.89281303,
                                 7.86227426, 7.79857496, 7.29116216, 5.82842342, 4.57134635, 3.42852575, 2.34918097])
        talib_data_2 = np.array(
            [nan, nan, nan, nan, nan, 59.88888889, 67.49382716, 76.60768176, 86.5598232, 86.82094329,
             86.48501684, 85.78432457, 80.20278377, 64.11265765, 50.28480981, 37.71378323, 25.84099068])
        np.testing.assert_almost_equal(our_data_1.flatten(), talib_data_1, decimal=3)
        np.testing.assert_almost_equal(our_data_2.flatten(), talib_data_2, decimal=3)

    def test_dema(self):
        x = np.array([1., 5, 10, 4, 3, 4, 5, 11, 10, 8, 8, 18, 21, 1, 5])
        our_data = dema(x, 3)
        talib_data = np.array([nan, nan, nan, nan, 3.05555556, 3.56944444, 4.55555556, 9.41319444, 10.27430556,
                               8.92100694, 8.35243056, 15.62217882, 20.28407118, 6.62852648, 4.80750868])

        np.testing.assert_almost_equal(our_data.flatten(), talib_data, decimal=3)

    def test_tema(self):
        x = np.array([1., 5, 10, 4, 3, 4, 5, 11, 10, 8, 8, 18, 21, 1, 5])
        our_data = tema(x, 3)
        talib_data = np.array([nan, nan, nan, nan, nan, nan, 4.7037037, 10.2806713,
                               10.5708912, 8.6087963, 8.02010995, 16.64492911, 21.15341073,
                               4.24893302, 3.71395761])

        np.testing.assert_almost_equal(our_data.flatten(), talib_data, decimal=3)

    def test_macd(self):
        # TODO: need to compare it against TA-LIB output like we did it with kama
        pass

    def test_denoised_trend(self):
        o1 = np.array([10., 11, 12, 13, 14, 15, 16])

        c1 = o1 + 0.5
        data = pd.DataFrame({
            'open': o1,
            'close': c1
        }, index=pd.date_range('2000-01-01 00:00', periods=len(o1), freq='1H'))

        np.testing.assert_almost_equal(denoised_trend(data, 6)[-1], c1[6] - o1[0])
        np.testing.assert_almost_equal(denoised_trend(data, 6, bar_returns=False)[-1], c1[6] - c1[0])

        c1 = o1 + 1.5
        data = pd.DataFrame({
            'open': o1,
            'close': c1
        }, index=pd.date_range('2000-01-01 00:00', periods=len(o1), freq='1H'))

        np.testing.assert_equal(denoised_trend(data, 6)[-1] < c1[6] - o1[0], True)
        np.testing.assert_almost_equal(denoised_trend(data, 6, bar_returns=False)[-1], c1[6] - c1[0])

    def test_pivot_point(self):
        prices = np.arange(50)

        data = pd.DataFrame({
            'ask': prices,
            'bid': prices
        }, index=pd.date_range('2019-01-01 00:00', periods=len(prices), freq='1H'))

        r = pivot_point(data)

        first_day = ohlc_resample(data, '1D', resample_tz='EET').loc['2018-12-31 22:00:00']
        self.assertEqual(r.loc['2019-01-01 22:00']['P'], (first_day.high + first_day.low + first_day.close) / 3)
