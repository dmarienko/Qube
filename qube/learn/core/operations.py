from sklearn.base import BaseEstimator

from qube.quantitative.tools import scols, srows
from qube.learn.core.base import _FIELD_MARKET_INFO
from qube.learn.core.base import signal_generator


class __Operations:
    def fit(self, x, y, **fit_args):
        mkt_info = getattr(self, _FIELD_MARKET_INFO)
        for k, v in self.__dict__.items():
            if isinstance(v, BaseEstimator) or hasattr(v, 'fit'):
                v.market_info_ = mkt_info
                v.fit(x, y, **fit_args)

        # prevent time manipulations
        self.exact_time = True
        return self


@signal_generator
class Imply(BaseEstimator, __Operations):
    """
    Implication operator
    """

    def __init__(self, first, second, memory=0):
        self.first = first
        self.second = second
        self.memory = memory

    def predict(self, x):
        f, s = self.first.predict(x), self.second.predict(x)

        ws = scols(x, f, s, keys=['____X_', 's_first', 's_second'])
        f2, s2 = ws['s_first'].iloc[:, 0], ws['s_second'].iloc[:, 0]

        track_memory = self.memory if self.memory > 0 else len(ws)
        ws2 = scols(f2.ffill(limit=track_memory), s2, names=['s_first', 's_second']).dropna()

        impl = ws2[(ws2['s_first'] != 0) & (ws2['s_first'] == ws2['s_second'])]
        return impl['s_second']

    def __call__(self, **kwargs):
        """
        It's posisble to pass additional memory parameter like that:
        (gen1 >> gen2)(memory=10)
        """
        if 'memory' in kwargs:
            self.memory = kwargs.get('memory')
        return self


@signal_generator
class And(BaseEstimator, __Operations):
    """
    AND operator for filters or for signal and filter
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def predict(self, x):
        lft = self.left.predict(x)
        rgh = self.right.predict(x)

        # if there is no filtering series raise exception
        if lft.dtype != bool and rgh.dtype != bool:
            raise Exception(
                f"At least one of arguments of And must be series of booleans for using as filter\n"
                f"Received {lft.dtype} and {rgh.dtype}"
            )

        flt_on, subj = (lft, rgh) if lft.dtype == bool else (rgh, lft)

        # mx = scols(flt_on, subj, names=['F', 'S'])
        # return mx[mx.F == True].S

        sf = scols(flt_on, subj, names=['F', 'S'])
        return sf['S'].reindex(sf['F'].ffill().replace({False: None}).dropna().index).dropna()


@signal_generator
class Mul(BaseEstimator, __Operations):
    """
    Mult operator for multiplication of signal
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def predict(self, x):
        if isinstance(self.left, (int, float, bool)):
            lft = self.left
        else:
            lft = self.left.predict(x)

        if isinstance(self.right, (int, float, bool)):
            rgh = self.right
        else:
            rgh = self.right.predict(x)

        return lft * rgh


@signal_generator
class Join(BaseEstimator, __Operations):
    """
    Operator for joining signals
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def predict(self, x):
        lft = self.left.predict(x)
        rgh = self.right.predict(x)
        # joins two series for duplicated keep last one
        return srows(lft, rgh, keep='last')


@signal_generator
class Or(BaseEstimator, __Operations):
    """
    OR operator on filters
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def predict(self, x):
        lft = self.left.predict(x)
        rgh = self.right.predict(x)

        # if there is no filtering series raise exception
        if lft.dtype != bool and rgh.dtype != bool:
            raise Exception(
                f"Both arguments of Or must be series of booleans\n"
                f"Received {lft.dtype} and {rgh.dtype}"
            )
        mx = scols(lft, rgh, names=['F1', 'F2'])
        return (mx.F1 == True) | (mx.F2 == True)


@signal_generator
class Neg(BaseEstimator, __Operations):
    """
    Just reverses signals/filters
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, x):
        return -self.predictor.predict(x)
