import unittest

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline

from qube.quantitative.tools import srows, ohlc_resample
from qube.examples.learn.generators import RangeBreakoutDetector
from qube.examples.learn.transformers import RollingRange
from qube.learn.core.base import MarketDataComposer, signal_generator
from qube.learn.core.metrics import ForwardDirectionScoring, ForwardReturnsSharpeScoring
from qube.learn.core.pickers import SingleInstrumentPicker
from qube.learn.core.utils import debug_output
from qube.tests.utils_for_tests import _read_timeseries_data


@signal_generator
class MidDayPrice(TransformerMixin, BaseEstimator):
    def fit(self, x, y, **kwargs):
        return self

    def transform(self, x):
        return x.assign(mid=0.5 * (x.high + x.low))


@signal_generator
class TesterSingle(BaseEstimator):
    def __init__(self, period):
        self.period = period

    def fit(self, X, y, **fit_params):
        return self

    def predict(self, X):
        price = X[self.market_info_.column]
        return srows(
            pd.Series(+1, price[(price > price.shift(self.period))].index),
            pd.Series(-1, price[(price < price.shift(self.period))].index),
        )


class ScoringTests(unittest.TestCase):

    def test_scorer(self):
        # data = pd.read_csv('../data/ES.csv.gz', parse_dates=True, index_col=['time'])
        data = _read_timeseries_data("ES", compressed=True)

        debug_output(data, "Test OHLC")

        # wor = make_pipeline(_WeekOpenRange('4Min', 0.25), _RangeBreakoutDetector().fillna(0).as_classifier())
        # m1 = MarketDataComposer(TesterSingle(15), SingleInstrumentPicker(), None, debug=True)
        # debug_output(m1.fit(data, None).predict(data), 'Close signals')

        g1 = GridSearchCV(
            cv=TimeSeriesSplit(5),
            estimator=make_pipeline(MidDayPrice(), TesterSingle(5)),
            scoring=ForwardDirectionScoring("1H"),
            param_grid={
                "testersingle__period": np.arange(2, 60),
            },
            verbose=True,
        )

        mds = MarketDataComposer(
            g1, SingleInstrumentPicker(), column="close", debug=True
        )
        mds.fit(data, None)
        print(g1.best_params_)
        print(g1.best_score_)
        self.assertAlmostEqual(0.48656, g1.best_score_, delta=1e-5)

    def test_scorer_open_close(self):
        data = ohlc_resample(_read_timeseries_data("ES", compressed=True), "5Min")
        # data = ohlc_resample(pd.read_csv('c:/data/ohlc/ES.csv.gz', parse_dates=True, index_col=['time']), '5Min')

        bs = make_pipeline(RollingRange("1h", 12), RangeBreakoutDetector())
        m2 = MarketDataComposer(bs, SingleInstrumentPicker(), None, debug=True)
        y0 = m2.fit(data, None).predict(data)
        print(sum(y0.index.second == 0))

        g1 = GridSearchCV(
            cv=TimeSeriesSplit(5),
            estimator=bs,
            scoring=ForwardDirectionScoring("3h"),
            param_grid={
                "rollingrange__period": np.arange(12, 15),
                "rollingrange__timeframe": [pd.Timedelta("1h")],
            },
            verbose=True,
        )

        mds = MarketDataComposer(
            g1, SingleInstrumentPicker(), column="close", debug=True
        )
        mds.fit(data, None)
        print(g1.best_params_)
        print(g1.best_score_)
        self.assertAlmostEqual(0.45124, g1.best_score_, delta=1e-5)

    def test_scorer_ticks(self):
        data = _read_timeseries_data("XBTUSD", compressed=True)

        bs = make_pipeline(RollingRange("10S", 30, 6), RangeBreakoutDetector(0.5))

        m2 = MarketDataComposer(bs, SingleInstrumentPicker(), None, debug=True)
        y0 = m2.fit(data, None).predict(data)
        debug_output(y0, "TestPrediction")

        g1 = GridSearchCV(
            n_jobs=10,
            cv=TimeSeriesSplit(3),
            estimator=bs,
            scoring=ForwardDirectionScoring("1Min"),
            param_grid={
                "rollingrange__period": np.arange(10, 30),
                "rollingrange__forward_shift_periods": np.arange(5, 10),
                "rollingrange__timeframe": ["10S", "15S"],
            },
            verbose=True,
        )

        mds = MarketDataComposer(
            g1, SingleInstrumentPicker(), column="close", debug=True
        )
        mds.fit(data, None)
        print(g1.best_params_)
        print(g1.best_score_)
        self.assertAlmostEqual(0.68888, g1.best_score_, delta=1e-5)

    def test_sharpe_scorer_ticks(self):
        data = _read_timeseries_data("XBTUSD", compressed=True)

        bs = make_pipeline(RollingRange("10S", 30, 6), RangeBreakoutDetector(0.5))

        m2 = MarketDataComposer(bs, SingleInstrumentPicker(), None, debug=True)
        y0 = m2.fit(data, None).predict(data)
        debug_output(y0, "TestPrediction")

        g1 = GridSearchCV(
            n_jobs=10,
            cv=TimeSeriesSplit(3),
            estimator=bs,
            scoring=ForwardReturnsSharpeScoring("1Min", commissions=0.17),
            param_grid={
                "rollingrange__period": np.arange(10, 30),
                "rollingrange__forward_shift_periods": np.arange(5, 10),
                "rollingrange__timeframe": ["10S", "15S"],
            },
            verbose=True,
        )

        mds = MarketDataComposer(
            g1, SingleInstrumentPicker(), column="close", debug=True
        )
        mds.fit(data, None)
        print(g1.best_params_)
        print(g1.best_score_)
        self.assertAlmostEqual(1.07625, g1.best_score_, delta=1e-5)


from pytest import main

if __name__ == "__main__":
    main()
