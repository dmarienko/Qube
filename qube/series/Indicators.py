from collections import deque
from typing import Union, List

import numpy as np

from qube.quantitative.stats.stats import percentile_rank
from qube.quantitative.ta.indicators import nans
from qube.series.Bar import Bar

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


class Indicator:
    """
    Abstract indicator as basic class for any streaming data indicator
    """

    def __init__(self, max_values_length=np.inf):
        self.indicators = []
        self.values = []
        self.last_value = None
        self._is_being_appended = False
        self._name = '<<Undefined>>'
        self._max_values_length = max_values_length

    def need_bar_input(self):
        return False

    def last(self):
        return self.__value_at(0)

    def __value_at(self, idx: int):
        is_last = self.last_value is not None

        if idx == 0 and is_last:
            return self.last_value

        if idx > 0 and is_last: idx = idx - 1

        try:
            idx = (len(self.values) - idx - 1) if idx >= 0 else abs(1 + idx)
            return self.values[idx]
        except IndexError:
            return None

    def __len__(self):
        return len(self.values) + (1 if self.last_value is not None else 0)

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, slice):
            return [self.__value_at(i) for i in range(*idx.indices(len(self)))]
        else:
            return self.__value_at(idx)

    def update(self, xs, append=True):
        """
        Update indicator by new value. If append flag is set we appending new calculated value to underlying series
        otherwise it just updates last indicator's value. This needs for processing data arriving `inside` bar.

        :param xs: new input value (or collection of values)
        :param append: if true we append indicator's value to indicator series
        :return: self
        """
        if isinstance(xs, Iterable) and not isinstance(xs, Bar):
            [self.__update(x, append) for x in xs]
            return self

        return self.__update(xs, append)

    def __update(self, new_input, append=True):
        """
        Update indicator by new value. If append flag is set we appending new calculated value to underlying series
        otherwise it just updates last indicator's value. This needs for processing data arriving `inside` bar.

        :param new_input: new value
        :param append: if true we append indicator's value to indicator series
        :return: self
        """
        self._is_being_appended = append

        # calculate indicator's value
        self.last_value = self.calculate(new_input)

        if append:
            if len(self.values) >= self._max_values_length:
                self.values.pop(0)
            self.values.append(self.last_value)

        # call all dependent indicators
        for i in self.indicators:
            i.update(self.last_value, append)

        if append:
            self.last_value = None

        return self

    def attach(self, indicator):
        """
        Attaches new indicator to this one. It will be updated by values of 'host' indicator (for ema(ema(...)))

        :param indicator:
        :return:
        """
        if indicator is not None and isinstance(indicator, Indicator):
            # if we already have some data in this series
            if len(self) > 0:
                bars = self[::-1]

                # push all formed bars
                [indicator.update(z, True) for z in bars[:-1]]

                # finally push last bar
                indicator.update(bars[-1], False)

            # finally attach it to list
            self.indicators.append(indicator)
        else:
            raise ValueError("Can't attach empty indicator or non-Indicator object")

        return self

    def calculate(self, v):
        raise NotImplementedError("Method 'calculate(v)' must be implemented in this class")

    def name(self):
        return self._name

    def __repr__(self):
        return self.name() + ' : %d values' % len(self)


class Sma(Indicator):
    """
    Simple moving average
    """

    def __init__(self, period):
        super().__init__()
        self.period = period
        self.__s = nans(period)
        self.__i = 0
        self._name = 'sma(%d)' % period

    def calculate(self, x):
        _x = x / self.period
        self.__s[self.__i] = _x

        if self._is_being_appended:
            self.__i += 1
            if self.__i >= self.period:
                self.__i = 0

        return np.sum(self.__s)


class Ema(Indicator):
    """
    Exponential moving average
    """

    def __init__(self, period, init_mean=True):
        super().__init__()
        self.period = period

        # when it's required to initialize this ema by mean on first period
        self.init_mean = init_mean
        if init_mean:
            self.__s = nans(period)
            self.__i = 0

        self.alpha = 2.0 / (1.0 + period)
        self.alpha_1 = (1 - self.alpha)
        self._name = 'ema(%d,%s)' % (period, init_mean)

    def calculate(self, x):
        # when we need to initialize ema by average from initial period
        if self.init_mean and self.__i < self.period:
            # we skip any nans on initial period (tema, dema, ...)
            if np.isnan(x):
                return np.nan

            self.__s[self.__i] = x / self.period
            if self._is_being_appended:
                self.__i += 1

                if self.__i >= self.period:
                    self.init_mean = False
                    return np.sum(self.__s)

            return np.nan

        if len(self) == 0:
            return x

        return self.alpha * x + self.alpha_1 * self.values[-1]


class Tema(Indicator):
    """
    Triple exponential moving average
    """

    def __init__(self, period, init_mean=True):
        super().__init__()
        self.period = period
        self._name = 'tema(%d,%s)' % (period, init_mean)

        self.__e1 = Ema(period, init_mean)
        self.__e2 = Ema(period, init_mean)
        self.__e3 = Ema(period, init_mean)

        self.__e1.attach(self.__e2)
        self.__e2.attach(self.__e3)

    def calculate(self, x):
        self.__e1.update(x, self._is_being_appended)
        return 3 * self.__e1.last() - 3 * self.__e2.last() + self.__e3.last()


class Dema(Indicator):
    """
    Double exponential moving average
    """

    def __init__(self, period, init_mean=True):
        super().__init__()
        self._name = 'dema(%d,%s)' % (period, init_mean)
        self.period = period

        self.__e1 = Ema(period, init_mean)
        self.__e2 = Ema(period, init_mean)
        self.__e1.attach(self.__e2)

    def calculate(self, x):
        self.__e1.update(x, self._is_being_appended)
        return 2 * self.__e1.last() - self.__e2.last()


class KAMA(Indicator):
    """
    Kaufman Adaptive Moving Average
    """

    def __init__(self, period, fast_span=2, slow_span=30):
        super().__init__()
        self.period = period

        # here we will collect past input data
        self.__x_past = deque(nans(period + 1), period + 1)
        self.__collecting_init_data = True

        self._S1 = 2.0 / (slow_span + 1)
        self._K1 = 2.0 / (fast_span + 1) - self._S1
        self._name = 'kama(%d,%d,%d)' % (period, fast_span, slow_span)

    def calculate(self, x):
        self.__x_past[-1] = x

        if self._is_being_appended:
            if self.__collecting_init_data and not np.isnan(self.__x_past[1]):
                # seems we finished collecting `period` bars of data - ready to go
                self.__collecting_init_data = False

                # attaching new slot for future data
                self.__x_past.append(np.nan)

                # first kama value is first input value
                return x

        # do kama factors calculations (before attaching new empty slot !)
        rs = np.sum(np.abs(np.diff(self.__x_past)))
        er = (np.abs(x - self.__x_past[0]) / rs) if not np.isnan(rs) else np.nan
        # er = (np.abs(x - self.__x_past[0]) / rs)
        sc = np.square(er * self._K1 + self._S1)

        # attaching new slot for future data
        if self._is_being_appended:
            self.__x_past.append(np.nan)

        # if there is not enough data
        if self.__collecting_init_data:
            return np.nan

        return sc * x + (1 - sc) * self.values[-1]


class ATR(Indicator):
    """
    Average True Range indicator. It requires Bar object as inputs.
    """
    __smoothers = {'sma': Sma, 'ema': Ema, 'tema': Tema, 'dema': Dema, 'kama': KAMA}

    def __init__(self, period, smoother='sma'):
        super().__init__()
        self.period = period
        self._name = 'atr(%d,%s)' % (period, smoother)

        if smoother not in ATR.__smoothers:
            raise ValueError("ATR: Unknown smoother '%s'" % smoother)

        self.__smoother = ATR.__smoothers[smoother](period)
        self.__prev_close = None

    def need_bar_input(self):
        return True

    def calculate(self, x: Bar):
        if not isinstance(x, Bar):
            raise ValueError("ATR indicator requires Bar object as input value ! [%s]" % x)

        d = abs(x.high - x.low)
        pc = self.__prev_close
        if pc is not None:
            d = max(d, abs(x.high - pc), abs(x.low - pc))

        if self._is_being_appended:
            self.__prev_close = x.close

        self.__smoother.update(d, self._is_being_appended)
        return self.__smoother.last()


class MovingMinMax(Indicator):
    """
    Tracking minimal low and maximal high during last `period` bars
    """

    def __init__(self, period):
        super().__init__()
        self.period = period
        self._name = 'MinMax(%d)' % period
        self.__highs = nans(period)
        self.__lows = nans(period)
        self.__i = 0

    def need_bar_input(self):
        return True

    def calculate(self, x):
        if not isinstance(x, Bar):
            raise ValueError("MovingMinMax indicator requires Bar object as input value ! [%s]" % x)
        self.__highs[self.__i] = x.high
        self.__lows[self.__i] = x.low

        if self._is_being_appended:
            self.__i += 1
            if self.__i >= self.period:
                self.__i = 0

        return np.min(self.__lows), np.max(self.__highs)


class DailyHighLow(Indicator):
    """
    Tracking daily minimal low and maximal high
    """

    def __init__(self):
        super().__init__()
        self._name = 'DailyHighLow'
        self.__today = False
        self.__high = 0
        self.__low = 0

    def need_bar_input(self):
        return True

    def calculate(self, x):
        if not isinstance(x, Bar):
            raise ValueError("DailyHighLow indicator requires Bar object as input value ! [%s]" % x)
        if self.__today != x.time.dayofyear:
            self.__today = x.time.dayofyear
            self.__high = x.high
            self.__low = x.low
        else:
            self.__high = max(self.__high, x.high)
            self.__low = min(self.__low, x.low)

        return self.__low, self.__high


class Bollinger(Indicator):
    """
    Bollinger bands
    """
    __mean_methods = {'sma': Sma, 'ema': Ema, 'tema': Tema, 'dema': Dema, 'kama': KAMA}

    def __init__(self, period, nstd: float = 2, mean_model='sma'):
        super().__init__()
        self.period = period
        self.nstd = nstd

        if mean_model not in Bollinger.__mean_methods:
            raise ValueError("Bollinger: Unknown mean model '%s'" % mean_model)

        self.__s = nans(period)
        self.__i = 0
        self.__mean = Bollinger.__mean_methods[mean_model](period)
        self._name = 'bollinger(%d,%.1f,%s)' % (period, nstd, mean_model)

    def calculate(self, x):
        self.__mean.update(x, self._is_being_appended)
        _mean_val = self.__mean.last()
        self.__s[self.__i] = (x - _mean_val) ** 2

        if self._is_being_appended:
            self.__i += 1
            if self.__i >= self.period:
                self.__i = 0

        _std_val = self.nstd * np.sqrt(np.sum(self.__s) / (self.period - 1))

        return _mean_val - _std_val, _mean_val + _std_val


class BollingerATR(Indicator):
    """
    Bollinger bands based on Average True Range value
    """
    __mean_methods = {'sma': Sma, 'ema': Ema, 'tema': Tema, 'dema': Dema, 'kama': KAMA}

    def __init__(self, period, atr_period, nstd: float = 2, mean_model='sma', atr_model='ema'):
        super().__init__()
        self.period = period
        self.nstd = nstd

        if mean_model not in BollingerATR.__mean_methods:
            raise ValueError("BollingerATR: Unknown mean model '%s'" % mean_model)

        self.__mean = BollingerATR.__mean_methods[mean_model](period)
        self.__atr = ATR(atr_period, atr_model)
        self._name = 'BollingerATR(%d,%d,%.1f,%s,%s)' % (period, atr_period, nstd, mean_model, atr_model)

    def need_bar_input(self):
        return True

    def calculate(self, x: Bar):
        self.__mean.update(x.close, self._is_being_appended)
        self.__atr.update(x, self._is_being_appended)

        _mean_val = self.__mean.last()
        _std_val = self.nstd * self.__atr.last()

        return _mean_val - _std_val, _mean_val + _std_val


class Returns(Indicator):
    """
    Returns on bar series. It requires Bar object as inputs. It calculates open ~ close returns by default
    """

    def __init__(self, multiplyer=1.0, open_close=False):
        """
        Create new Returns indicator

        return = multiplyer * (b.close/b.open - 1) if open_close = True
        return = multiplyer * (b.close/b.close[-1] - 1) if open_close = False

        :param multiplyer: return nultiplicator
        :param open_close: if true it calculates open/close return for every bar
        """
        super().__init__()
        self.__mx = multiplyer
        self.__open_close = open_close
        self.__prev_close = np.nan
        self._name = 'returns(mx=%f,oc=%s)' % (multiplyer, open_close)

    def need_bar_input(self):
        return True

    def calculate(self, x: Bar):
        if not isinstance(x, Bar):
            raise ValueError("Returns indicator requires Bar object as input value ! [%s]" % x)

        if self.__open_close:
            v = x.open
        else:
            v = self.__prev_close

        if self._is_being_appended:
            self.__prev_close = x.close

        return self.__mx * (x.close / v - 1.0)


class MACD(Indicator):
    """
    MACD Indicator
    """
    __methods = {'ema': Ema, 'tema': Tema, 'dema': Dema, 'sma': Sma, 'kama': KAMA}

    def __init__(self, fast=12, slow=26, signal=9, method='ema', signal_method='ema'):
        super().__init__()
        if method not in MACD.__methods:
            raise ValueError("MACD: Unknown method '%s'" % method)

        if signal_method not in MACD.__methods:
            raise ValueError("MACD: Unknown signal method '%s'" % signal_method)

        self.__fast = self.__methods[method](fast)
        self.__slow = self.__methods[method](slow)
        self.__signal = self.__methods[signal_method](signal)
        self._name = 'MACD(%d,%d,%d,%s,%s)' % (fast, slow, signal, method, signal_method)

    def calculate(self, x):
        self.__fast.update(x, self._is_being_appended)
        self.__slow.update(x, self._is_being_appended)
        self.__signal.update(self.__fast.last() - self.__slow.last(), self._is_being_appended)
        return self.__signal.last()


class TrendDetector(Indicator):
    """
    Simple trend detecting method based on BB approach
    """

    def __init__(self, period: int, nstd: float, smoother: str, trend_ends_at_mid=False, k_ext=1,
                 use_atr=False, atr_period: int = 14, atr_smoother: str = 'ema'):
        super().__init__()
        self.__exit_on_mid = trend_ends_at_mid
        self.k0 = 0.5 * (k_ext - 1)

        if use_atr:
            self._name = 'TrendDetector(atr, %d,%d,%1.f,%s,%s)' % (period, atr_period, nstd, smoother, atr_smoother)
            self.__bb = BollingerATR(period, atr_period, nstd, mean_model=smoother, atr_model=atr_smoother)
            self.__use_atr = True
        else:
            self._name = 'TrendDetector(%d,%1.f,%s)' % (period, nstd, smoother)
            self.__bb = Bollinger(period, nstd, mean_model=smoother)
            self.__use_atr = False

        self.__t = 0
        self.__smin1, self.__smax1, self.__bsmin1, self.__bsmax1 = nans(4)
        self.__utb1, self.__dtb1 = nans(2)

    def need_bar_input(self):
        # we need bars as input when using BollingerATR
        return self.__use_atr

    def calculate(self, x):
        self.__bb.update(x, self._is_being_appended)
        bsmin, bsmax, up_trend_line, up_trend_sig, down_trend_line, down_trend_sig = nans(6)
        smin, smax = self.__bb[0]
        smin1, smax1 = (self.__smin1, self.__smax1)

        # take close price if input is Bar object
        x = x.close if self.__use_atr else x

        # detecting trend
        trend = 1 if x > smax1 else -1 if x < smin1 else self.__t
        if trend > 0 and smin < smin1: smin = smin1
        if trend < 0 and smax > smax1: smax = smax1

        if self.__exit_on_mid:
            midle = (smin + smax) / 2.0
            if trend > 0 and x < midle: trend = 0
            if trend < 0 and x > midle: trend = 0

        dlta = self.k0 * (smax - smin)
        bsmax = smax + dlta
        bsmin = smin - dlta

        if trend > 0:
            if bsmin < self.__bsmin1: bsmin = self.__bsmin1
            if np.isnan(self.__utb1): up_trend_sig = bsmin
            up_trend_line = bsmin

        if trend < 0:
            if bsmax > self.__bsmax1: bsmax = self.__bsmax1
            if np.isnan(self.__dtb1): down_trend_sig = bsmax
            down_trend_line = bsmax

        if self._is_being_appended:
            self.__t, self.__smin1, self.__smax1 = (trend, smin, smax)
            self.__utb1, self.__dtb1 = (up_trend_line, down_trend_line)
            self.__bsmin1, self.__bsmax1 = (bsmin, bsmax)

        return up_trend_sig, down_trend_sig, trend, up_trend_line, down_trend_line


class DenoisedTrend(Indicator):

    def __init__(self, period: int, window=0, mean: str = 'kama', bar_returns: bool = True):
        super().__init__()
        self._name = 'DenoisedTrend(%d,%d,%s,%s)' % (period, window, mean, bar_returns)
        self.period = period

        self.__use_bar_returns = bar_returns
        self.__prev_close = np.nan
        self.__pr_buff = nans(period + 1)
        if bar_returns:
            self.__ri_buff = nans(period + 1)
        else:
            self.__ri_buff = nans(period)
        self.__pi = 0
        self.__ri = 0

    def need_bar_input(self):
        # we need bars as input when using Bar returns
        return self.__use_bar_returns

    def calculate(self, x):
        c_price = x.close if self.__use_bar_returns else x
        ri = c_price - (x.open if self.__use_bar_returns else self.__prev_close)

        # if not np.isnan(ri):
        self.__ri_buff[self.__ri] = np.abs(ri)
        self.__pr_buff[self.__pi] = c_price

        di_lag_idx = (self.__pi + 1) if self.__pi < self.period else 0
        di = c_price - self.__pr_buff[di_lag_idx]
        si = np.sum(self.__ri_buff)

        # restrict si for bar returns
        if self.__use_bar_returns and not (np.isnan(si) or np.isnan(di)):
            si = max(si, np.abs(di))

        # we got new bar so shift pointer to next slot
        if self._is_being_appended:
            # store open if we need bar's return
            self.__pr_buff[self.__pi] = x.open if self.__use_bar_returns else c_price

            # keep prev close price
            self.__prev_close = c_price

            # jump to next slot
            self.__pi += 1
            self.__ri += 1
            if self.__pi >= len(self.__pr_buff):
                self.__pi = 0
            if self.__ri >= len(self.__ri_buff):
                self.__ri = 0

        filtered_trend = abs(di) * (di / si)
        return filtered_trend if np.isfinite(filtered_trend) else np.nan


class DenoisedDeltaRank(Indicator):

    def __init__(self, fast_period: int, slow_period: int, lookback: int,
                 qts: List = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
        super().__init__()
        self._name = 'DenoisedDeltaRank(%d,%d,%s,%s)' % (fast_period, slow_period, lookback, qts)
        self.__fdt = DenoisedTrend(fast_period, 0, None)
        self.__sdt = DenoisedTrend(slow_period, 0, None)
        self.__lookback = lookback
        self.__qts = qts
        self.__delta_buff = nans(lookback - 1)
        self.__i = 0
        self.__delta_i = max(fast_period, slow_period) + lookback - 1

    def need_bar_input(self):
        return True

    def calculate(self, x):
        self.__fdt.update(x, self._is_being_appended)
        self.__sdt.update(x, self._is_being_appended)
        slow_trend = self.__sdt.last()
        fast_trend = self.__fdt.last()
        delta = fast_trend - slow_trend
        if self.__delta_i <= 0:
            rank = percentile_rank(self.__delta_buff, delta, self.__qts)
        else:
            rank = np.nan
            if self._is_being_appended:
                self.__delta_i -= 1

        if self._is_being_appended:
            self.__delta_buff[self.__i] = delta
            self.__i += 1
            if self.__i >= self.__lookback - 1:
                self.__i = 0

        return rank * np.sign(slow_trend)


class RollingPercentile(Indicator):

    def __init__(self, period, pctls=(0, 1, 2, 3, 5, 10, 15, 25, 45, 50, 55, 75, 85, 90, 95, 97, 98, 99, 100)):
        super().__init__()
        self.period = period
        self.pctls = pctls
        self._name = 'RollingPercentile(%d)' % period
        self.__cbuff = nans(period)
        self.__i = 0

    def calculate(self, x):
        self.__cbuff[self.__i] = x

        if self._is_being_appended:
            self.__i += 1
            if self.__i >= self.period:
                self.__i = 0

        return np.percentile(self.__cbuff, self.pctls)


class RollingStd(Indicator):
    """
    Simple moving std
    """

    def __init__(self, period, mean: Union[float, Indicator] = 0):
        super().__init__()
        self.period = period
        self.__init_not_passed = True
        self.__s = nans(period)
        self.__i = 0
        self.__mean = mean
        self._name = 'rolling_std(%d)' % period
        self._m_is_indicator = isinstance(self.__mean, Indicator)

    def __mean_value(self):
        _m = self.__mean[0] if self._m_is_indicator else self.__mean
        return np.nan if _m is None else _m

    def calculate(self, x):
        self.__s[self.__i] = (x - self.__mean_value()) ** 2

        if self._is_being_appended:
            self.__i += 1
            if self.__i >= self.period:
                self.__init_not_passed = False
                self.__i = 0

        # we skip values untill there is not all data in frame
        return np.nan if self.__init_not_passed \
            else np.sqrt(np.nansum(self.__s) / (self.period - 1))


class WilliamsR(Indicator):
    """
    WilliamsR Indicator
    """

    def __init__(self, period):
        super().__init__()
        self.__i = 0
        self._name = 'WilliamR(%d)' % period
        self.period = period
        self.highs = nans(period)
        self.lows = nans(period)

    def need_bar_input(self):
        return True

    def calculate(self, x: Bar):
        if self._is_being_appended:
            self.__i += 1

            if self.__i >= self.period:
                self.__i = 0

            self.highs[self.__i] = x.high
            self.lows[self.__i] = x.low

            hh = max(self.highs)
            ll = min(self.lows)

            return (hh - x.close) / (hh - ll) * -100
        else:
            return self.last_value
