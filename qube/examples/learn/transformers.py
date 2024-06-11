import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from qube.quantitative.ta.indicators import pivot_point
from qube.quantitative.tools import ohlc_resample, scols
from qube.learn.core.base import signal_generator
from qube.learn.core.utils import _check_frame_columns


@signal_generator
class RollingRange(TransformerMixin, BaseEstimator):
    """
    Produces rolling high/low range (top/bottom) indicator
    """

    def __init__(self, timeframe, period, forward_shift_periods=1, tz='UTC'):
        self.period = period
        self.timeframe = timeframe
        self.forward_shift_periods = forward_shift_periods
        self.tz = tz

    def fit(self, x, y, **kwargs):
        return self

    def transform(self, x):
        """
        Attaches RangeTop, RangeBot columns to source dataframe
        """
        try:
            # we can use either OHLC
            _check_frame_columns(x, 'open', 'high', 'low', 'close')
        except ValueError:
            # or quotes data to be resampled
            _check_frame_columns(x, 'bid', 'ask')

        ohlc = ohlc_resample(x, self.timeframe, resample_tz=self.tz)
        hilo = scols(
            ohlc.rolling(self.period, min_periods=self.period).high.max(),
            ohlc.rolling(self.period, min_periods=self.period).low.min(),
            names=['RangeTop', 'RangeBot'])

        # move to 'future'
        hilo.index = hilo.index + self.forward_shift_periods * pd.Timedelta(self.timeframe)

        # and combine with existing data
        return x.combine_first(hilo).ffill()


@signal_generator
class FractalsRange(TransformerMixin, BaseEstimator):
    """
    Produces rolling range (top/bottom) indicator based on fractal indicator
    """

    def __init__(self, timeframe, nf=2, tz='UTC'):
        self.timeframe = timeframe
        self.nf = nf
        self.tz = tz

    def fit(self, x, y, **kwargs):
        return self

    def transform(self, x):
        """
        Attaches RangeTop, RangeBot columns to source dataframe
        """
        try:
            # we can use either OHLC
            _check_frame_columns(x, 'open', 'high', 'low', 'close')
            shift_forward = self.nf
        except ValueError:
            # or quotes data to be resampled
            _check_frame_columns(x, 'bid', 'ask')
            # on tick series we need to shift on one more bar ahead
            # becuse we use time of bar open
            shift_forward = self.nf + 1

        ohlc = ohlc_resample(x, self.timeframe, resample_tz=self.tz)
        ohlc = scols(ohlc.close.ffill(), ohlc, keys=['A', 'ohlc']).ffill(axis=1)['ohlc']

        ru, rd = None, None
        for i in range(1, self.nf + 1):
            ru = scols(
                ru,
                (ohlc.high - ohlc.high.shift(i)).rename(f'p{i}'),
                (ohlc.high - ohlc.high.shift(-i)).rename(f'p_{i}')
            )
            rd = scols(
                rd,
                (ohlc.low.shift(i) - ohlc.low).rename(f'p{i}'),
                (ohlc.low.shift(-i) - ohlc.low).rename(f'p_{i}')
            )

        ru = ru.dropna()
        rd = rd.dropna()

        upF = pd.Series(+1, ru[((ru > 0).all(axis=1))].index)
        dwF = pd.Series(-1, rd[((rd > 0).all(axis=1))].index)

        ht = ohlc.loc[upF.index].reindex(ohlc.index).shift(shift_forward).ffill().high
        lt = ohlc.loc[dwF.index].reindex(ohlc.index).shift(shift_forward).ffill().low

        hilo = scols(ht.rename('RangeTop'), lt.rename('RangeBot'))

        # and combine with existing data
        return x.combine_first(hilo).fillna(method='ffill')


class Pivots(TransformerMixin):
    """
    Produces pivots levels
    """

    def __init__(self, timeframe, method='classic', tz='UTC'):
        self.timeframe = timeframe
        self.method = method
        self.tz = tz

    def fit(self, x, y, **kwargs):
        return self

    def transform(self, x):
        pp = pivot_point(x, method=self.method, timeframe=self.timeframe, timezone=self.tz)
        return pd.concat((x, pp), axis=1)
