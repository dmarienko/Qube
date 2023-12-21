import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from qube.quantitative.tools import srows, scols, ohlc_resample
from qube.quantitative.ta.indicators import smooth, ema, sma, rsi, super_trend
from qube.simulator.utils import shift_signals
from qube.learn.core.base import signal_generator
from qube.simulator.tracking.trackers import FixedRiskTrader
from qube.examples.learn.generators import crossup, crossdown 


@signal_generator
class SuperRsi(BaseEstimator):
    """
    Trade in general trend direction on RSI confirmations 
    """
    VERSION = '0.1'

    def __init__(self, trend_period=25, trend_nstd=2, trend_smoother='kama',
                 rsi_period=14, rsi_entry=10, 
                 trade_size=1000, take_pct=5, stop_pct=3):
        self.trade_size = trade_size
        self.take_pct = take_pct
        self.stop_pct = stop_pct
        
        self.trend_period = trend_period
        self.trend_nstd = trend_nstd 
        self.trend_smoother = trend_smoother
        self.rsi_period = rsi_period
        self._upper = 100 - rsi_entry
        self._lower = rsi_entry

    def fit(self, x, y, **fit_args):
        # we don't need to fit it
        return self

    def predict(self, ohlc):
        price_col = self.market_info_.column
        
        # 1. entries by rsi 
        r = rsi(ohlc[price_col], self.rsi_period)
        entries = srows(pd.Series(+1, crossup(r,  self._lower)), pd.Series(-1, crossdown(r, self._upper)))
        
        # 2. filter only ones that follows trend from supertrend indicator
        trend = super_trend(ohlc, self.trend_period, self.trend_nstd, atr_smoother=self.trend_smoother)
        entries_by_trend = entries[((entries > 0) & (trend.trend > 0)) | ((entries < 0) & (trend.trend < 0))]
        
        return entries_by_trend
    
    def tracker(self):
        return FixedRiskTrader(self.trade_size, take=self.take_pct, stop=self.stop_pct, 
                               in_percentage=True, accurate_stops=True, reset_risks_on_repeated_signals=True)

    