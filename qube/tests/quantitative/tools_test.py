import unittest

from qube.quantitative.ta.indicators import ema
from qube.quantitative.tools import *


class ToolsTests(unittest.TestCase):

    def test_add_constant(self):
        x = np.array([1, 2, 3, 4])
        np.testing.assert_almost_equal(
            add_constant(add_constant(x, 1, prepend=False), 5),
            np.array([
                [5., 1., 1.],
                [5., 2., 1.],
                [5., 3., 1.],
                [5., 4., 1.],
            ]), decimal=1
        )

    def test_apply_to_frame(self):
        x = np.array([1, 2, 3, 4])
        y1 = apply_to_frame(ema, x, 2)
        self.assertTrue(isinstance(y1, np.ndarray))

        y2 = apply_to_frame(ema, pd.Series(x), 2)
        self.assertTrue(isinstance(y2, pd.Series))

        y3 = apply_to_frame(ema, pd.DataFrame([x, x]), 2)
        self.assertTrue(isinstance(y3, pd.DataFrame))

    def test_ohlc_resample(self):
        prices = np.arange(50)

        data = pd.DataFrame({
            'ask': prices,
            'bid': prices
        }, index=pd.date_range('2019-01-01 00:00', periods=len(prices), freq='1H'))

        # Resample quotes without tz. Daily bars start at 00
        resampled = ohlc_resample(data, "1D")
        self.assertEqual(resampled.iloc[1].open, 24)
        self.assertEqual(resampled.index[1], pd.Timestamp('2019-01-02 00:00:00'))

        resampled_eet = ohlc_resample(data, "1D", resample_tz="EET")

        # Resample quotes in EET. Bars start at 22 hours instead of 00
        self.assertEqual(resampled_eet.iloc[1].open, 22)
        self.assertEqual(resampled_eet.index[1], pd.Timestamp('2019-01-01 22:00:00'))
        self.assertEqual(resampled_eet.index.tz, data.index.tz)

        # Resample ohlc in EET. Same result as quotes
        resampled2 = ohlc_resample(data, "2H")
        resampled_eet2 = ohlc_resample(resampled2, "1D", resample_tz="EET")
        self.assertEqual(resampled_eet2.iloc[1].open, 22)
        self.assertEqual(resampled_eet2.index[1], pd.Timestamp('2019-01-01 22:00:00'))
        self.assertEqual(resampled_eet2.index.tz, resampled2.index.tz)


from pytest import main
if __name__ == '__main__':
    main()