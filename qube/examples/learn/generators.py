from typing import Union

import pandas as pd
from sklearn.base import BaseEstimator

from qube.quantitative.ta.indicators import smooth, rsi, ema
from qube.quantitative.tools import srows, scols, apply_to_frame, ohlc_resample
from qube.learn.core.base import signal_generator
from qube.learn.core.data_utils import pre_close_time_shift
from qube.learn.core.utils import _check_frame_columns
from qube.simulator.utils import rolling_forward_test_split


def crossup(x, t: Union[pd.Series, float]):
    t1 = t.shift(1) if isinstance(t, pd.Series) else t
    return x[(x > t) & (x.shift(1) <= t1)].index


def crossdown(x, t: Union[pd.Series, float]):
    t1 = t.shift(1) if isinstance(t, pd.Series) else t
    return x[(x < t) & (x.shift(1) >= t1)].index


@signal_generator
class RangeBreakoutDetector(BaseEstimator):
    """
    Detects breaks of rolling range. +1 for breaking upper range and -1 for bottom one.
    """

    def __init__(self, threshold=0):
        self.threshold = threshold

    def fit(self, X, y, **fit_params):
        return self

    def _ohlc_breaks(self, X):
        U, B = X.RangeTop + self.threshold, X.RangeBot - self.threshold
        open, close, high, low = X.open, X.close, X.high, X.low

        b1_bU = high.shift(1) <= U.shift(1)
        b1_aL = low.shift(1) >= B.shift(1)
        l_c = (b1_bU | (open <= U)) & (close > U)
        s_c = (b1_aL | (open >= B)) & (close < B)
        l_o = (b1_bU & (open > U))
        s_o = (b1_aL & (open < B))

        pre_close = pre_close_time_shift(X)

        return srows(
            pd.Series(+1, X[l_o].index), pd.Series(+1, X[(l_c & ~l_o)].index + pre_close),
            pd.Series(-1, X[s_o].index), pd.Series(-1, X[(s_c & ~s_o)].index + pre_close),
        )

    def _ticks_breaks(self, X):
        U, B = X.RangeTop + self.threshold, X.RangeBot - self.threshold
        a, b = X.ask, X.bid

        break_up = (a.shift(1) <= U.shift(1)) & (a > U)
        break_dw = (b.shift(1) >= B.shift(1)) & (b < B)

        return srows(pd.Series(+1, X[break_up].index), pd.Series(-1, X[break_dw].index))

    def predict(self, X):
        # take control on how we produce timestamps for signals
        self.exact_time = True

        try:
            _check_frame_columns(X, 'RangeTop', 'RangeBot', 'open', 'high', 'low', 'close')
            y0 = self._ohlc_breaks(X)

        except ValueError:
            _check_frame_columns(X, 'RangeTop', 'RangeBot', 'bid', 'ask')
            y0 = self._ticks_breaks(X)

        return y0


@signal_generator
class PivotsBreakoutDetector(BaseEstimator):
    @staticmethod
    def _tolist(x):
        return [x] if not isinstance(x, (list, tuple)) else x

    def __init__(self, resistances, supports):
        self.resistances = self._tolist(resistances)
        self.supports = self._tolist(supports)

    def fit(self, X, y, **fit_params):
        return self

    def predict(self, x):
        _check_frame_columns(x, 'open', 'close')

        t = scols(x, x.shift(1)[['open', 'close']].rename(columns={'open': 'open_1', 'close': 'close_1'}))
        cols = x.columns
        breaks = srows(
            # breaks up levels specified as resistance
            *[pd.Series(+1, t[(t.open_1 < t[ul]) & (t.close_1 < t[ul]) & (t.close > t[ul])].index) for ul in
              self.resistances if ul in cols],

            # breaks down levels specified as supports
            *[pd.Series(-1, t[(t.open_1 > t[bl]) & (t.close_1 > t[bl]) & (t.close < t[bl])].index) for bl in
              self.supports if bl in cols],
            keep='last')
        return breaks


@signal_generator
class CrossingMovings(BaseEstimator):
    def __init__(self, fast, slow, fast_type='sma', slow_type='sma'):
        self.fast = fast
        self.slow = slow
        self.fast_type = fast_type
        self.slow_type = slow_type

    def fit(self, x, y, **fit_args):
        return self

    def predict(self, x):
        price_col = self.market_info_.column
        fast_ma = smooth(x[price_col], self.fast_type, self.fast)
        slow_ma = smooth(x[price_col], self.slow_type, self.slow)

        return srows(
            pd.Series(+1, crossup(fast_ma, slow_ma)),
            pd.Series(-1, crossdown(fast_ma, slow_ma))
        )


@signal_generator
class Rsi(BaseEstimator):
    """
    Classical RSI entries generator
    """

    def __init__(self, period, lower=25, upper=75, smoother='sma'):
        self.period = period
        self.upper = upper
        self.lower = lower
        self.smoother = smoother

    def fit(self, x, y, **fit_args):
        return self

    def predict(self, x):
        price_col = self.market_info_.column
        r = rsi(x[price_col], self.period, smoother=self.smoother)
        return srows(pd.Series(+1, crossup(r, self.lower)), pd.Series(-1, crossdown(r, self.upper)))


@signal_generator
class OsiMomentum(BaseEstimator):
    """
    Outstretched momentum contrarian generator

    The idea is to mark rising and falling momentum and then calculate an exponential moving average based on the sum
    of the different momentum moves.

    The steps can be summed up as follows:

    - Select a momentum lookback and a moving average lookback. By default we can use 3 and 5.

    - Create two columns called the Positive Stretch and the Negative Stretch,
      where the first one has 1’s if the current closing price is greater than the closing price 3 periods ago
      and the other one has 1’s if the current closing price is lower than the closing price 3 periods ago.

    - Sum the latest three positive and negative stretches and subtract the results from each other.
      This is called the Raw Outstretch.

    - Finally, to get the Outstretched Indicator, we take the 5-period exponential moving average of the Raw Outstretch.

    """

    def __init__(self, period, smoothing, threshold=0.05):
        """
        :param period: period of momentum
        :param smoothing: period of ema smoothing
        :param threshold: threshold for entries. Max abs indicator value is period we generate long entries
                          when indicator crosses down lower threshold (period-T) and
                          short whem it crosses up upper threshold (-(period-T))
        """
        self.period = period
        self.smoothing = smoothing
        self.threshold = threshold
        if threshold > 1:
            raise ValueError(f'Threshold parameter {threshold} exceedes 1 !')

    def fit(self, x, y, **fit_args):
        return self

    def predict(self, x):
        price_col = self.market_info_.column
        c = x[price_col]

        pos = (c > c.shift(self.period)) + 0
        neg = (c < c.shift(self.period)) + 0
        osi = apply_to_frame(ema, pos.rolling(self.period).sum() - neg.rolling(self.period).sum(), self.smoothing)

        kt = self.period * (1 - self.threshold)
        return srows(
            pd.Series(+1, osi[(osi.shift(2) > -kt) & (osi.shift(1) > -kt) & (osi <= -kt)].index),
            pd.Series(-1, osi[(osi.shift(2) < +kt) & (osi.shift(1) < +kt) & (osi >= +kt)].index)
        )


@signal_generator
class InternalBarStrength(BaseEstimator):
    """
    Internal bar strength mean reverting generator.
    when:
        | close is > (high - T) -> -1
        | close is < (low + T)  -> +1

    T in (0 ... 1/2)
    """

    def __init__(self, timeframe, threshold, tz='UTC'):
        self.timeframe = timeframe
        self.threshold = threshold
        self.tz = tz
        self.exact_time = True
        if threshold >= 0.5 or threshold <= 0:
            raise ValueError(f'Threshold parameter {threshold} must be in (0 ... 0.5) range !')

    def fit(self, x, y, **fit_args):
        return self

    def predict(self, x):
        _check_frame_columns(x, 'open', 'close', 'high', 'low')

        xf = ohlc_resample(x, self.timeframe, resample_tz=self.tz)

        # on next bar openinig
        self.exact_time = True
        ibs = ((xf.close - xf.low) / (xf.high - xf.low)).shift(1)

        return srows(
            pd.Series(+1, ibs[ibs < self.threshold].index),
            pd.Series(-1, ibs[ibs > 1 - self.threshold].index)
        )


@signal_generator
class Equilibrium(BaseEstimator):
    """
        - Calculate a simple N-period moving average of the market price.
        - Subtract the current market price from its moving average.
        - Calculate a N-period exponential moving average on the subtracted values.

        The result is the N-period Equilibrium Indicator that we will use to generate mean-reverting signals.
    """

    def __init__(self, period, threshold, smoother='sma'):
        self.period = period
        self.smoother = smoother
        self.threshold = threshold

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x):
        c = x[self.market_info_.column]
        k1 = smooth(c, self.smoother, self.period)
        dK = smooth(k1 - c, 'ema', self.period)

        return srows(
            pd.Series(-1, dK[
                ((dK.shift(2) < +self.threshold) & ((dK.shift(1) < +self.threshold) & (dK > +self.threshold)))
            ].index),

            pd.Series(+1, dK[
                ((dK.shift(2) > -self.threshold) & ((dK.shift(1) > -self.threshold) & (dK < -self.threshold)))
            ].index)
        )


@signal_generator
class WalkForwardTest(BaseEstimator):
    """
    Walk Forward Test
    """

    def __init__(self, estimator: BaseEstimator, train_period=4, test_period=1, 
                 units='W', next_period_close_posititons=True):
        """
        Create new WFT using provided estimator and train/test windows
        By default it uses 4 weeks for training and 1 week for prediction

        :param training_period: number observations for training period 
        :param test_period: number observations for testing (aka trading) period  
        :param units: period units if training_period and test_period is the period date: {'H', 'D', 'W', 'M', 'Q', 'Y'}
        :param next_period_close_posititons: close all positions at the start of new test period (true)
        """
        if estimator is None or not isinstance(estimator, BaseEstimator):
            raise ValueError(f"Estimator must be non empty and be derived from BaseEstimator")

        if train_period <= 0 or test_period <= 0:
            raise ValueError(f"Train and Test periods must be postive numbers: {train_period} / {test_period}")

        self.estimator = estimator
        self.train_period = train_period
        self.test_period = test_period
        self.units = units
        self.close_positions_next_trade_period = next_period_close_posititons
        self.sigs = None

    def fit(self, x: pd.DataFrame, y, **kwargs):
        self.sigs = pd.Series(dtype='float64')
        signals = []
        
        # - we may want to attach indicator as additional coliumn to avoid recalcuting it on every chank
        #   because some indicators (SMA, RSI, ...) may have empty data in the begging 
        #   nans for first 'period' bars for SMA for example
        if hasattr(self.estimator, 'aux_data'):
            aux_d = self.estimator.aux_data(x, **kwargs)
            if aux_d is not None:
                if isinstance(aux_d, pd.DataFrame):
                    if [c for c in x.columns if c in aux_d.columns]:
                        raise ValueError(f"Some aux dataframe column names '{aux_d.columns}' intersect with original dataframe columns !")
                elif isinstance(aux_d, pd.Series):
                    if aux_d.name in x.columns:
                        raise ValueError(f"Aux data series name '{aux_d.name}' intersects with original dataframes columns !")
                else:
                    raise ValueError(f"Aux data has unrecognized type '{type(aux_d)}' !")

                # attach aux data 
                x = scols(x.copy(), aux_d)
        
        for trn, tst in rolling_forward_test_split(x, self.train_period, self.test_period, units=self.units):
            self.estimator.fit(x.loc[trn], y, **kwargs)
            signals.append(self.estimator.predict(x.loc[tst]))
        self.sigs = srows(*signals)

        # - check if need to close positions were open previously 
        #   when new test (trade) period is started
        if self.close_positions_next_trade_period:
            # on new period positions from prev period becomes NaN after concatenating 
            # so we could just replace them to 0 it's equal to flatting
            self.sigs = self.sigs.fillna(0.0)

        return self

    def predict(self, x):
        if hasattr(self.estimator, 'exact_time'):
            self.exact_time = self.estimator.exact_time
        return self.sigs

    def tracker(self, **kwargs):
        if hasattr(self.estimator, 'tracker'):
            return self.estimator.tracker(**kwargs)
        return None
