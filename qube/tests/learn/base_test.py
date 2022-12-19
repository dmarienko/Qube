import unittest

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline

from qube.quantitative.tools import srows, drop_duplicated_indexes
from qube.examples.learn.filters import AdxFilter
from qube.examples.learn.generators import CrossingMovings, Rsi
from qube.learn.core.base import MarketDataComposer, signal_generator, SingleInstrumentComposer
from qube.learn.core.metrics import ForwardDirectionScoring
from qube.learn.core.operations import Imply, Neg, Or, And
from qube.learn.core.pickers import SingleInstrumentPicker
from qube.learn.core.utils import debug_output, ls_params
from qube.tests.utils_for_tests import _read_timeseries_data


@signal_generator
class WeekOpenRangeTest(TransformerMixin, BaseEstimator):

    @staticmethod
    def find_week_start_time(data, week_start_day=6):
        d1 = data.assign(time=data.index)
        return d1[d1.index.weekday == week_start_day].groupby(pd.Grouper(freq='1d')).first().dropna().time.values

    def __init__(self, open_interval, tick_size=0.25):
        self.open_interval = open_interval
        self.tick_size = tick_size

    def fit(self, x, y, **kwargs):
        return self

    def transform(self, data):
        ul = {}
        op_int = pd.Timedelta(self.open_interval) if isinstance(self.open_interval, str) else self.open_interval
        for wt0 in self.find_week_start_time(data):
            wt1 = data[wt0 + op_int:].index[0]
            ws = data[wt0:wt1]
            ul[wt1] = {'RangeTop': ws.high.max() + self.tick_size,
                       'RangeBot': ws.low.min() - self.tick_size}
        ulf = pd.DataFrame.from_dict(ul, orient='index')
        return data.combine_first(ulf).fillna(method='ffill')


@signal_generator
class RangeBreakoutDetectorTest(BaseEstimator):
    def fit(self, X, y, **fit_params):
        return self

    def predict(self, X):
        if not all([c in X.columns for c in ['RangeTop', 'RangeBot']]):
            raise ValueError("Can't find 'RangeTop', 'RangeBot' in input data !")
        U, B = X.RangeTop, X.RangeBot
        l_i = ((X.close.shift(1) <= U.shift(1)) & (X.close > U)) | ((X.open <= U) & (X.close > U))
        s_i = ((X.close.shift(1) >= B.shift(1)) & (X.close < B)) | ((X.open >= B) & (X.close < B))
        return srows(pd.Series(+1, X[l_i].index), pd.Series(-1, X[s_i].index))


@signal_generator
class Fp(BaseEstimator):
    """
    Some testing filter
    """

    def __init__(self, s, n):
        self.s = s
        self.n = n

    def fit(self, x, y, **fit_params):
        return self

    def predict(self, x):
        f = pd.Series(False, x.index)
        b = len(x) // self.n
        f[self.s * b: self.s * b + b] = True
        return f


@signal_generator
class Mp(TransformerMixin):
    """
    Produces test transformations
    """

    def __init__(self, timeframe):
        self.timeframe = timeframe

    def fit(self, x, y, **kwargs):
        return self

    def transform(self, x):
        return x.assign(midprice=(x.close + x.open) / 2)


@signal_generator
class Gp(BaseEstimator):
    def __init__(self, idxs):
        self.idxs = idxs

    def fit(self, x, y, **fit_params):
        return self

    def predict(self, x):
        f = pd.Series(np.nan, x.index)
        for i in self.idxs:
            f.iloc[np.abs(i)] = np.sign(i)
        return f.dropna()


class BaseFunctionalityTests(unittest.TestCase):
    def setUp(self):
        self.data = _read_timeseries_data('ES', compressed=True, as_dict=False)
        debug_output(self.data, 'ES')

    def test_prediction_alignment(self):
        wor = make_pipeline(WeekOpenRangeTest('4Min', 0.25), RangeBreakoutDetectorTest())
        m1 = MarketDataComposer(wor, SingleInstrumentPicker(), debug=True)
        debug_output(m1.fit(self.data, None).predict(self.data), 'Predicted')

        g1 = GridSearchCV(
            cv=TimeSeriesSplit(2),
            estimator=wor,
            scoring=ForwardDirectionScoring('30Min'),
            param_grid={
                'weekopenrangetest__open_interval': [pd.Timedelta(x) - pd.Timedelta('1Min') for x in [
                    '5Min', '10Min', '15Min', '20Min', '25Min', '30Min', '35Min', '40Min', '45Min'
                ]],
            }, verbose=True
        )

        mds = MarketDataComposer(g1, SingleInstrumentPicker(), 'close', debug=True)
        mds.fit(self.data, None)
        print(g1.best_score_)
        print(g1.best_params_)
        self.assertAlmostEqual(0.4893939, g1.best_score_, delta=1e-5)

    def test_filters(self):
        f_wor = make_pipeline(
            WeekOpenRangeTest('4Min', 0.25),
            RangeBreakoutDetectorTest() & AdxFilter('1Min', 20, 10, 'ema')
        )

        m1 = MarketDataComposer(f_wor, SingleInstrumentPicker(), debug=True)
        debug_output(m1.fit(self.data, None).predict(self.data).dropna(), 'Predicted')

        g1 = GridSearchCV(
            cv=TimeSeriesSplit(2),
            estimator=f_wor,
            scoring=ForwardDirectionScoring('30Min'),
            param_grid={
                'weekopenrangetest__open_interval': [pd.Timedelta(x) - pd.Timedelta('1Min') for x in [
                    '5Min', '10Min', '15Min',
                ]],
                'and__right__period': [20, 30],
                'and__right__threshold': [15, 25],
                'and__right__timeframe': ['1Min', '5Min', '15Min'],
            }, verbose=True
        )

        mds = MarketDataComposer(g1, SingleInstrumentPicker(), 'close')
        mds.fit(self.data, None)
        print(g1.best_score_)
        print(g1.best_params_)
        self.assertAlmostEqual(0.62637, g1.best_score_, delta=1e-4)

    def test_ops(self):
        cross = CrossingMovings(5, 15, 'ema', 'ema')
        sup = Rsi(15, smoother='ema')
        trend = AdxFilter('5Min', 20, 1, 'ema')

        brk = make_pipeline(
            WeekOpenRangeTest('4Min', 0.25),
            (RangeBreakoutDetectorTest() >> cross >> sup) & trend
        )

        mds = SingleInstrumentComposer(brk, 'close')
        mds.fit(self.data, None)

        print(ls_params(brk))
        print(mds.predict(self.data).dropna())

    def test_ops2(self):
        def S(est):
            return SingleInstrumentComposer(est).fit(self.data[:7000], None).predict(self.data[:7000])

        f0 = Fp(1, 5)
        f1 = Fp(3, 5)
        g0 = Gp([100, 1500, -1900, 2900, 5000, 6000])
        g1 = Gp([300, -2500, 3000, 5100, 6100])

        g21 = And(f0, Imply(g0, g1))
        g22 = And(f1, Imply(g0, g1))
        g23 = And((Or(Neg(f0), Neg(f1))), Imply(g0, g1))

        r = drop_duplicated_indexes(
            srows(S(g21).dropna(), S(g22).dropna(), S(g23).dropna())
        )
        self.assertTrue(
            all(np.array([1, -1, 1, 1, 1]) == r.values.flatten())
        )

        # sugar test
        sg21 = (g0 >> g1) & f0
        sg22 = (g0 >> g1) & f1
        sg23 = (g0 >> g1) & ~(f0 | f1)

        sr = drop_duplicated_indexes(
            srows(S(sg21).dropna(), S(sg22).dropna(), S(sg23).dropna())
        )
        print(sr)

        self.assertTrue(
            all(np.array([1, -1, 1, 1, 1]) == sr.values.flatten())
        )

    def test_ops_imply_memory(self):
        def S(est):
            return SingleInstrumentComposer(est).fit(self.data[:1000], None).predict(self.data[:1000])

        # memory parameter passing test
        g01 = Gp([100])
        g11 = Gp([110])

        self.assertTrue(S(
            (g01 >> g11)(memory=3)
        ).dropna().empty)

        self.assertFalse(S(
            (g01 >> g11)(memory=11)
        ).dropna().empty)

    def test_ops_add_mul(self):
        def S(est):
            return SingleInstrumentComposer(est).fit(self.data[:1000], None).predict(self.data[:1000])

        # memory parameter passing test
        g01 = Gp([100])
        g11 = Gp([110])
        g12 = Gp([130])

        sr = S(
            g01 + g11 * 2 + g12 * 100
        )

        print(sr)

        self.assertTrue(
            all(np.array([1, 2, 100]) == sr.values.flatten())
        )

    def test_operations_on_mdc(self):
        def F(est):
            return SingleInstrumentComposer(est)

        data = {'ES': self.data}

        g01 = F(Gp([100]))
        g11 = F(Gp([110]))

        p0 = F(make_pipeline(Mp('1Min'), Gp([100])))
        p1 = F(make_pipeline(Mp('1Min'), Gp([110])))
        p2 = F(make_pipeline(Mp('1Min'), Gp([-115])))

        print(
            F(((p0 >> g11) + p2) * 10).fit(data, None).predict(data),
            '\n--------------'
        )

        print(
            # p0 -> g11 -> opposite p2
            F(-(p0 >> g11 >> -p2) * 10).fit(data, None).predict(data)
        )

        print(
            (p0 + p1 * 100).fit(data, None).predict(data)
        )

        print(
            (g01 + g11 * 100).fit(data, None).predict(data)
        )

        print(
            (g01 + g11 * 100).fit(self.data, None).predict(self.data)
        )

        print(
            ((g01 >> g11) * 2).fit(self.data, None).predict(self.data)
        )


from pytest import main
if __name__ == '__main__':
    main()