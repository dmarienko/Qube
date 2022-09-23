import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from qube.quantitative import column_vector
from qube.quantitative.ta.indicators import adx, atr
from qube.quantitative.tools import ohlc_resample, rolling_sum
from qube.learn.core.base import signal_generator


@signal_generator
class AdxFilter(BaseEstimator):
    """
    ADX based trend filter. When adx > threshold
    """

    def __init__(self, timeframe, period, threshold, smoother='ema'):
        self.timeframe = timeframe
        self.period = period
        self.threshold = threshold
        self.smoother = smoother

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x):
        a = adx(ohlc_resample(x, self.timeframe), self.period, smoother=self.smoother, as_frame=True).shift(1)
        return a.ADX > self.threshold


@signal_generator
class AcorrFilter(BaseEstimator):
    """
    Autocorrelation filter on returns series
     If above is True (default) returns True for acorr > threshold
     If above is False returns True for acorr < threshold
    """

    def __init__(self, timeframe, lag, period, threshold, above=True):
        self.lag = lag
        self.period = period
        self.threshold = threshold
        self.timeframe = timeframe
        self.above = above

    def fit(self, x, y, **kwargs):
        return self

    def rolling_autocorrelation(self, x, lag, period):
        """
        Timeseries rolling autocorrelation indicator
        :param period: rolling window
        :param lag: lagged shift used for finding correlation coefficient
        """
        return x.rolling(period).corr(x.shift(lag))

    def predict(self, x):
        xr = ohlc_resample(x[['open', 'high', 'low', 'close']], self.timeframe)
        returns = xr.close.pct_change()
        ind = self.rolling_autocorrelation(returns, self.lag, self.period).shift(1)
        return (ind > self.threshold) if self.above else (ind < self.threshold)


@signal_generator
class VolatilityFilter(BaseEstimator):
    """
    Regime based on volatility
       False: flat
       True:  volatile market
    """

    def __init__(self, timeframe, instant_period, typical_period, factor=1):
        self.instant_period = instant_period
        self.typical_period = typical_period
        self.factor = factor
        self.timeframe = timeframe

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x):
        xr = ohlc_resample(x, self.timeframe)
        inst_vol = atr(xr, self.instant_period).shift(1)
        typical_vol = atr(xr, self.typical_period).shift(1)
        return inst_vol > typical_vol * self.factor


@signal_generator
class AtrFilter(BaseEstimator):
    """
    Raw ATR filter
    """

    def __init__(self, timeframe, period, threshold, tz='UTC'):
        self.timeframe = timeframe
        self.period = period
        self.threshold = threshold
        self.tz = tz

    def fit(self, x, y, **kwargs):
        return self

    def get_filter(self, x):
        a = atr(ohlc_resample(x, self.timeframe, resample_tz=self.tz), self.period).shift(1)
        return a > self.threshold


@signal_generator
class ChoppinessFilter(BaseEstimator):
    """
    Volatile market leads to false breakouts, and not respecting support/resistance levels (being choppy),
    We cannot know whether we are in a trend or in a range.

    Values above 61.8% indicate a choppy market that is bound to breakout. We should be ready for some directional.
    Values below 38.2% indicate a strong trending market that is bound to stabilize.
    """

    def __init__(self, timeframe, period, upper=61.8, lower=38.2, tz='UTC', atr_smoother='sma'):
        self.period = period
        self.upper = upper
        self.lower = lower
        self.timeframe = timeframe
        self.tz = tz
        self.atr_smoother = atr_smoother

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x):
        xr = ohlc_resample(x[['open', 'high', 'low', 'close']], self.timeframe, resample_tz=self.tz)
        a = atr(xr, self.period, self.atr_smoother)

        rng = xr.high.rolling(window=self.period, min_periods=self.period).max() \
              - xr.low.rolling(window=self.period, min_periods=self.period).min()

        rs = pd.Series(rolling_sum(column_vector(a.copy()), self.period).flatten(), a.index)
        ci = 100 * np.log(rs / rng) * (1 / np.log(self.period))

        f0 = pd.Series(np.nan, ci.index)
        f0[ci >= self.upper] = True
        f0[ci <= self.lower] = False
        return f0.ffill().fillna(False)
