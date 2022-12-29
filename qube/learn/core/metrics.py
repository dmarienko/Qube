from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from qube.quantitative.tools import scols
from qube.learn.core.data_utils import detect_data_type, ohlc_to_flat_price_series, forward_timeseries
from qube.learn.core.structs import MarketInfo, _FIELD_MARKET_INFO
from qube.learn.core.utils import debug_output


class ForwardReturnsCalculator:

    def extract_market_info(self, estimator) -> MarketInfo:
        def _find_estimator(x):
            """
            Tries to find estimator in nested pipelines
            """
            if isinstance(x, Pipeline):
                return _find_estimator(x.steps[-1][1])
            return x

        mi: MarketInfo = getattr(_find_estimator(estimator), _FIELD_MARKET_INFO, None)
        if mi is None:
            raise Exception(f"Can't exctract market info data from {estimator}")
        return mi

    def _cleanup_data(self, series, keep='last'):
        """
        Remove duplicated indexes
        """
        return series[~series.index.duplicated(keep=keep)]

    def get_prices(self, x, market_info: MarketInfo):
        dt = detect_data_type(x)
        s0, s1 = (market_info.session_start, market_info.session_end) if market_info is not None else (None, None)
        if dt.type == 'ohlc':
            prices = ohlc_to_flat_price_series(x, dt.frequency(), s0, s1)
        elif dt.type == 'ticks':
            # here we will use midprices as some first approximation
            prices = 0.5 * (x.bid + x.ask)
        else:
            raise ValueError(f"Don't know how to derive forward returns from '{dt.type}' data")
        return prices

    def get_forward_returns(self, data, signals, market_info: MarketInfo):
        raise ValueError('>>> get_forward_returns method should be implemented')


class ForwardDirectionScoring(ForwardReturnsCalculator):

    def __init__(self, period: Union[str, pd.Timedelta], min_threshold=0):
        self.period = pd.Timedelta(period) if isinstance(period, str) else period
        self.min_threshold = min_threshold

    def get_forward_returns(self, data, signals, market_info: MarketInfo):
        prices = self.get_prices(data, market_info)
        forward_prices = forward_timeseries(prices, self.period)
        dp = forward_prices - prices
        if self.min_threshold > 0:
            dp = dp.where(abs(dp) >= self.min_threshold, 0)

        # drop duplicated signals to avoid unnecessary collisions
        signals = self._cleanup_data(signals, 'last')

        # drop nan's
        dp[np.isnan(dp)] = 0
        returns = np.sign(dp)
        _returns = returns[~returns.index.duplicated(keep='first')]
        return _returns.reindex(signals.index).dropna()

    def __call__(self, estimator, data, _):
        pred = estimator.predict(data)

        if isinstance(pred, pd.DataFrame):
            pred = pred[pred.columns[0]]

        # we skip empty signals set
        if len(pred) == 0 or all(np.isnan(pred)):
            return 0

        market_info = self.extract_market_info(estimator)
        rets = self.get_forward_returns(data, pred, market_info)

        yc = scols(rets, pred, keys=['rp', 'pred']).dropna()
        return accuracy_score(yc.rp, yc.pred)


class ForwardReturnsSharpeScoring(ForwardReturnsCalculator):
    COMMS = {'bitmex': (0.075, True), 'okex': (0.05, True),
             'binance': (0.04, True), 'dukas': (35 * 100 / 1e6, False)}

    def __init__(self, period: Union[str, pd.Timedelta], commissions=0, crypto_futures=False, debug=False):
        self.period = pd.Timedelta(period) if isinstance(period, str) else period

        # possible to pass name of exchange
        comm = commissions
        if isinstance(commissions, str):
            comm_info = ForwardReturnsSharpeScoring.COMMS.get(commissions, (0, False))
            comm = comm_info[0]
            crypto_futures = comm_info[1]

            # commissions are required in percentages
        self.commissions = comm / 100
        self.crypto_futures = crypto_futures
        self.debug = debug

    def _returns_with_commissions_crypto_aware(self, prices, f_prices, signals):
        # drop duplicated signals to avoid unnecessary collisions
        signals = self._cleanup_data(signals, 'last')

        if self.crypto_futures:
            # pnl on crypto is calculated as following

            # in BTC
            # dp = 1 / prices - 1 / f_prices

            # in USD
            dp = f_prices / prices - 1

            # commissions are dependent on prices
            # dpc = scols(dp, self.commissions * 1 / f_prices, names=['D', 'C'])
            dpc = scols(dp, self.commissions * abs(signals), names=['D', 'C'])
        else:
            dp = f_prices - prices

            # commissions are fixed
            dpc = scols(dp, pd.Series(self.commissions, dp.index), names=['D', 'C'])

        # drop duplicated indexes if exist (may happened on tick data)
        dpc = self._cleanup_data(dpc, 'first')
        rpc = dpc.reindex(signals.index).dropna()

        yc = scols(rpc, signals.rename('pred')).dropna()
        return yc.D * yc.pred - yc.C

    def get_forward_returns(self, data, signals, market_info: MarketInfo):
        prices = self.get_prices(data, market_info)
        f_prices = forward_timeseries(prices, self.period)
        return self._returns_with_commissions_crypto_aware(prices, f_prices, signals)

    def __call__(self, estimator, data, _):
        sharpe_metric = -1e6
        pred = estimator.predict(data)

        if isinstance(pred, pd.DataFrame):
            pred = pred[pred.columns[0]]

        # we skip empty signals set
        if len(pred) == 0 or all(np.isnan(pred)):
            return sharpe_metric

        market_info = self.extract_market_info(estimator)
        rets = self.get_forward_returns(data, pred, market_info)

        if rets is not None:
            # measure is proratio to Sharpe
            if not all(np.isnan(rets)):
                std = np.nanstd(rets)
                sharpe_metric = (np.nanmean(rets) / std) if std != 0 else -1e6

            if self.debug:
                debug_output(data, 'Metric data', time_info=True)
                print(f'\t->> Estimator: {estimator}')
                print(f'\t->> Metric: {sharpe_metric:.4f}')

        return sharpe_metric


class ReverseSignalsSharpeScoring(ForwardReturnsSharpeScoring):
    def __init__(self, commissions=0, crypto_futures=False, debug=False):
        super().__init__(None, commissions, crypto_futures, debug)

    def get_forward_returns(self, data, signals, market_info: MarketInfo):
        prices = self.get_prices(data, market_info)

        # we need only points where position is reversed
        revere_pts = signals[signals.diff() != 0].dropna()
        # prices = prices.loc[revere_pts.index]
        prices = prices.reindex(index=revere_pts.index, method='ffill')
        f_prices = prices.shift(-1)
        return self._returns_with_commissions_crypto_aware(prices, f_prices, signals)
    

class TripleBarrierScoring:
    """
    TODO: implement tripple barrier scoring
    """
    pass


class SwingReversalsScoring:
    """
    TODO: implement swings reversal scoring
           +.5 +1 +.5
          \    /\
           \  /  \
            \/
        +.5 +1 +.5 

        We assign +1 if signal hits exact point of reversal (bar where reversal happened)
        and +.5 if signal in the neighborhood of reversal point (N bars before / after)  
        otherwise we use -1
        Probably we need to use adjustable weights ?
    """
    pass
