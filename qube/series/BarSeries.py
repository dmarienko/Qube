from enum import Enum
from typing import Union

import numpy as np
import pandas as pd

from qube.series.Quote import Quote
from qube.series.Bar import Bar
from qube.series.Indicators import Indicator
from qube.series.Series import Series
from qube.utils.DateUtils import DateUtils


class PriceType(Enum):
    """
    Enumeration for price type
    """
    MIDPRICE = 0  # (ask + bid) / 2
    BID = 1
    ASK = 2
    VWMPT = 3  # volume weighted prices


class BarSeries(Series):
    """
    OHLCV (bar) series builder from streaming quotes
    """
    _OP_ID = 'open'
    _HI_ID = 'high'
    _LO_ID = 'low'
    _CL_ID = 'close'
    _VO_ID = 'volume'
    _TM_ID = 'time'

    def __init__(self, timeframe: Union[int, str],
                 series: pd.DataFrame = None,
                 max_series_length=np.inf,
                 price_type=PriceType.MIDPRICE):

        """
        Constructs new OHLCV series object.

        :param timeframe: series timeframe from 1 second to 24 hours. In seconds format (60) or pandas ofsset alias format (1Min)
        :param series: incoming series
        :param estimate_volumes: true if we need to estimate volumes from quotes stream
        :param time_threshold: threshold for trades estimating algo
        :param price_type: price type for building bars
        """

        super().__init__(timeframe)

        if price_type == PriceType.MIDPRICE:
            self.mp_func = BarSeries.__mid_price

        elif price_type == PriceType.BID:
            self.mp_func = BarSeries.__bid_price

        elif price_type == PriceType.ASK:
            self.mp_func = BarSeries.__ask_price

        elif price_type == PriceType.VWMPT:
            self.mp_func = BarSeries.__vw_mid__price

        self.price_type = price_type
        self.max_series_length = max_series_length

        # just for fast access
        self.last_bar_volume = 0
        self.last_updated = None
        self.last_price = None

        if isinstance(series, pd.DataFrame):
            ohlcv_keys = ['open', 'high', 'low', 'close', 'volume']
            quote_keys = ['ask', 'bid', 'askvol', 'bidvol']
            if all(key in series for key in quote_keys):
                self.series = self._from_quoted_df(series)
            elif all(key in series for key in ohlcv_keys):
                self.series = self._from_ohlcv_df(series)
            else:
                raise TypeError('Type DataFrame series is not recognized')
        else:
            self.series = {self._TM_ID: [],
                           self._OP_ID: [], self._HI_ID: [], self._LO_ID: [], self._CL_ID: [], self._VO_ID: []}

    @staticmethod
    def __bid_price(bid, ask, bid_size, ask_size) -> float:
        return bid

    @staticmethod
    def __ask_price(bid, ask, bid_size, ask_size) -> float:
        return ask

    @staticmethod
    def __mid_price(bid, ask, bid_size, ask_size) -> float:
        """
        Calculate raw midprice
        """
        return 0.5 * (ask + bid)

    @staticmethod
    def __vw_mid__price(bid, ask, bid_size, ask_size) -> float:
        """
        Calculate volume weighted mid price
        """
        return (ask * ask_size + bid * bid_size) / (ask_size + bid_size)

    def update_by_quote(self, quote: Quote) -> bool:
        """
        Update series by new arrived quote. Returns True if new bar is added.

        :param quote: quote from datafeed
        :return: True if new bar is added
        """
        return self.update_by_data(quote.time, quote.bid, quote.ask, quote.bid_size, quote.ask_size)

    def update_by_data(self, timestamp, bid, ask, bid_size, ask_size) -> bool:
        """
        Update series by new quote's data. Returns True if new bar is added. It might be called if we don't need to
        wrap data into Quote object (for speed up in simulator)

        :param timestamp: data timestamp
        :param bid: bid price
        :param ask: ask price
        :param bid_size: bid size
        :param ask_size: ask size
        :return: True if new bar is added
        """

        # Volume weighted mid-price
        price = self.mp_func(bid, ask, bid_size, ask_size)

        # if we need estimated trade sizes
        volume = 0

        if not self.series[self._TM_ID]:
            self.__add_new_bar(timestamp, price, volume)

            # Here we disable first notification because first bar may be incomplete
            self.is_new_bar = False
        elif timestamp - self.series[self._TM_ID][-1] >= self._timeframe:

            # first we update indicators by currect last bar
            self.__update_all_indicators(self.__bar_at(0), True)

            # then add new bar
            self.__add_new_bar(timestamp, price, volume)
        else:
            self.__update_last_bar(timestamp, price, volume)

        # update indicators by new data
        self.__update_all_indicators(self.__bar_at(0), False)

        return self.is_new_bar

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, slice):
            return [self.__bar_at(i) for i in range(*idx.indices(len(self.series[self._TM_ID])))]
        return self.__bar_at(idx)

    def __len__(self):
        return len(self.series[self._TM_ID])

    def __bar_at(self, idx: int) -> Union[Bar, None]:
        """
        Get bar at specified index. Indexing order is reversed for convenient accessing to previous bars :
         - bar_at(0) returns most recent (current) bar
         - bar_at(1) returns previous bar
         - bar_at(-1) returns first bar in the series

        :param idx: bar index
        :return: bar at index specified or none if bar not exists
        """
        n_len = len(self)
        if n_len == 0 or (idx > 0 and idx > (n_len - 1)) or (idx < 0 and abs(idx) > n_len):
            return None
        idx = (n_len - idx - 1) if idx >= 0 else abs(1 + idx)
        return Bar(time=self.series[self._TM_ID][idx],
                   open=self.series[self._OP_ID][idx],
                   high=self.series[self._HI_ID][idx],
                   low=self.series[self._LO_ID][idx],
                   close=self.series[self._CL_ID][idx],
                   volume=self.series[self._VO_ID][idx])

    def times(self) -> list:
        return self.series[self._TM_ID]

    def opens(self) -> list:
        return self.series[self._OP_ID]

    def highs(self) -> list:
        return self.series[self._HI_ID]

    def lows(self) -> list:
        return self.series[self._LO_ID]

    def closes(self) -> list:
        return self.series[self._CL_ID]

    def volumes(self) -> list:
        return self.series[self._VO_ID]

    def __add_new_bar(self, timestamp, price, volume):
        bar_start_time = DateUtils.round_time_by(timestamp, -self._timeframe_value, self._timeframe_units)
        if len(self.series[self._TM_ID]) >= self.max_series_length:
            self.series[self._TM_ID].pop(0)
            self.series[self._OP_ID].pop(0)
            self.series[self._HI_ID].pop(0)
            self.series[self._LO_ID].pop(0)
            self.series[self._CL_ID].pop(0)
            self.series[self._VO_ID].pop(0)

        self.series[self._TM_ID].append(bar_start_time)
        self.series[self._OP_ID].append(price)
        self.series[self._HI_ID].append(price)
        self.series[self._LO_ID].append(price)
        self.series[self._CL_ID].append(price)
        self.series[self._VO_ID].append(volume)
        self.last_bar_volume = volume
        self.last_updated = timestamp
        self.last_price = price
        self.is_new_bar = True

    def __update_last_bar(self, timestamp, price, volume):
        self.series[self._CL_ID][-1] = price
        self.series[self._HI_ID][-1] = max(self.series[self._HI_ID][-1], price)
        self.series[self._LO_ID][-1] = min(self.series[self._LO_ID][-1], price)
        self.last_bar_volume += volume
        self.series[self._VO_ID][-1] = self.last_bar_volume
        self.last_updated = timestamp
        self.last_price = price
        self.is_new_bar = False

    def _from_quoted_df(self, series: pd.DataFrame):
        self.last_updated = series.index[-1]
        last_time = DateUtils.round_time_by(series.index[-1], -self._timeframe_value, self._timeframe_units)
        penult_time = DateUtils.round_time_by(series.index[-2], -self._timeframe_value, self._timeframe_units)
        self.is_new_bar = last_time - penult_time >= self._timeframe

        if self.price_type == PriceType.MIDPRICE:
            series['price'] = series[['ask', 'bid']].mean(axis=1)

        elif self.price_type == PriceType.BID:
            series['price'] = series['bid']

        elif self.price_type == PriceType.ASK:
            series['price'] = series['ask']

        elif self.price_type == PriceType.VWMPT:
            series['price'] = (series['ask'] * series['askvol'] + series['bid'] * series['bidvol']) / (
                    series['askvol'] + series['bidvol'])

        resampled_series = series['price'].resample(self._resample_rule).agg('ohlc')
        resampled_series['time'] = resampled_series.index
        resampled_series.dropna(inplace=True)
        volumes = np.zeros(len(resampled_series.index))
        resampled_series['volume'] = volumes
        series_dict = self._from_ohlcv_df(resampled_series)
        return series_dict

    def _from_ohlcv_df(self, ohlcv_series: pd.DataFrame) -> dict:
        time = ohlcv_series['time'] if 'time' in ohlcv_series else ohlcv_series.index
        time = pd.to_datetime(time)
        series_dict = {
            self._TM_ID: time.tolist(),
            self._OP_ID: ohlcv_series['open'].tolist(),
            self._HI_ID: ohlcv_series['high'].tolist(),
            self._LO_ID: ohlcv_series['low'].tolist(),
            self._CL_ID: ohlcv_series['close'].tolist(),
            self._VO_ID: ohlcv_series['volume'].tolist()
        }
        self.last_bar_volume = ohlcv_series['volume'].iloc[-1]
        self.last_price = ohlcv_series['close'].iloc[-1]
        if hasattr(ohlcv_series, 'last_updated'):
            self.last_updated = ohlcv_series.last_updated
        if hasattr(ohlcv_series, 'is_new_bar'):
            self.is_new_bar = ohlcv_series.is_new_bar
        return series_dict

    def to_frame(self) -> pd.DataFrame:
        """
        Convert OHLCV series object to pandas DataFrame with following columns: time,open,high,low,close,volume
        :return: dataframe
        """
        frame_dict = {'time': self.series[self._TM_ID],
                      'open': self.series[self._OP_ID],
                      'high': self.series[self._HI_ID],
                      'low': self.series[self._LO_ID],
                      'close': self.series[self._CL_ID],
                      'volume': self.series[self._VO_ID]}
        # TODO: here we need to add all available indicators to this dataframe too as separate column
        frame = pd.DataFrame(frame_dict, columns=frame_dict.keys()).set_index('time').asfreq(self._timeframe)
        frame.index = pd.to_datetime(frame.index)
        frame.is_new_bar = self.is_new_bar
        frame.last_updated = self.last_updated
        return frame

    def attach(self, indicator: Indicator):
        if indicator is not None and isinstance(indicator, Indicator):
            self.indicators.append(indicator)

            # and we already have some data in this series
            if len(self) > 0:
                bars = self[::-1]

                # push all formed bars
                [self.__update_indicator(indicator, z, True) for z in bars[:-1]]

                # finally push last bar
                self.__update_indicator(indicator, bars[-1], False)
        else:
            raise ValueError("Can't attach empty indicator or non-Indicator object")

        return self

    def __update_all_indicators(self, bar: Bar, is_new_bar: bool):
        _last_close = bar.close
        for i in self.indicators:
            if i.need_bar_input():
                i.update(bar, is_new_bar)
            else:
                i.update(_last_close, is_new_bar)

    @staticmethod
    def __update_indicator(indicator: Indicator, bar: Bar, is_new_bar: bool):
        indicator.update(bar if indicator.need_bar_input() else bar.close, is_new_bar)
