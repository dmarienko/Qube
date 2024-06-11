import copy
import inspect
from typing import Union, Dict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline

from qube.learn.core.data_utils import make_dataframe_from_dict, pre_close_time_shift
from qube.learn.core.metrics import ForwardReturnsCalculator
from qube.learn.core.pickers import AbstractDataPicker, SingleInstrumentPicker, PortfolioPicker
from qube.learn.core.structs import MarketInfo, _FIELD_MARKET_INFO, _FIELD_EXACT_TIME, QLEARN_VERSION
from qube.utils.ui_utils import green


def predict_and_postprocess(class_predict_function):
    def wrapped_predict(obj, xp, *args, **kwargs):
        # run original predict method
        yh = class_predict_function(obj, xp, *args, **kwargs)

        # if this predictor doesn't provide tag we operate with closes
        if yh is not None and not yh.empty and not getattr(obj, _FIELD_EXACT_TIME, False) and obj.market_info_.column == "close":
            yh = yh.shift(1, freq=pre_close_time_shift(xp))

        return yh

    return wrapped_predict


def preprocess_fitargs_and_fit(class_fit_function):
    def wrapped_fit(obj, x, y, **fit_params):
        # intercept market_info_
        if _FIELD_MARKET_INFO in fit_params:
            obj.market_info_ = fit_params.pop(_FIELD_MARKET_INFO)

        return class_fit_function(obj, x, y, **fit_params)

    return wrapped_fit


def _decorate_class_method_if_exist(cls, method_name, decorator):
    m = inspect.getmembers(cls, lambda x: (inspect.isfunction(x) or inspect.ismethod(x)) and x.__name__ == method_name)
    if m:
        setattr(cls, m[0][0], decorator(m[0][1]))


def operation(op, *args):
    """
    Loads operation for predictor
    """
    from qube.learn.core.operations import Imply, And, Or, Neg, Mul, Join

    ops = {
        "imply": Imply,
        "and": And,
        "or": Or,
        "neg": Neg,
        "mul": Mul,
        "join": Join,
    }

    if op.lower() in ops:
        return ops.get(op.lower())

    raise Exception("Unknown operation {op} !")


def __operator_impl_class__(op_class_name):
    def operator(obj, *args):
        op_class = operation(op_class_name)
        return op_class(obj, *args)

    return operator


def signal_generator(cls):
    cls.__qlearn__ = QLEARN_VERSION
    setattr(cls, _FIELD_MARKET_INFO, None)
    setattr(cls, _FIELD_EXACT_TIME, False)
    _decorate_class_method_if_exist(cls, "predict", predict_and_postprocess)
    _decorate_class_method_if_exist(cls, "predict_proba", predict_and_postprocess)
    _decorate_class_method_if_exist(cls, "fit", preprocess_fitargs_and_fit)

    # syntax sugar
    setattr(cls, "Imply", __operator_impl_class__("imply"))
    setattr(cls, "__rshift__", __operator_impl_class__("imply"))

    setattr(cls, "And", __operator_impl_class__("and"))
    setattr(cls, "__and__", __operator_impl_class__("and"))

    setattr(cls, "Or", __operator_impl_class__("or"))
    setattr(cls, "__or__", __operator_impl_class__("or"))

    setattr(cls, "Neg", __operator_impl_class__("neg"))
    setattr(cls, "__neg__", __operator_impl_class__("neg"))
    setattr(cls, "__invert__", __operator_impl_class__("neg"))

    setattr(cls, "Mul", __operator_impl_class__("mul"))
    setattr(cls, "__mul__", __operator_impl_class__("mul"))

    setattr(cls, "Add", __operator_impl_class__("join"))
    setattr(cls, "__add__", __operator_impl_class__("join"))

    return cls


def collect_qlearn_estimators(p, estimators_list, step=""):
    if isinstance(p, BaseEstimator) and hasattr(p, "__qlearn__"):
        estimators_list.append((step, p))
        return estimators_list

    if isinstance(p, Pipeline):
        for sn, se in p.steps:
            collect_qlearn_estimators(se, estimators_list, (step + "__" + sn) if step else sn)
        return estimators_list

    if isinstance(p, BaseSearchCV):
        return collect_qlearn_estimators(p.estimator, estimators_list, step)

    if isinstance(p, MarketDataComposer):
        return collect_qlearn_estimators(p.predictor, estimators_list, step)

    return estimators_list


class MarketDataComposer(BaseEstimator):
    """
    Market data composer for any predictors related to trading signals generation
    """

    def __init__(self, predictor, selector: AbstractDataPicker, column="close", debug=False):
        self.column = column
        self.predictor = predictor
        self.selector = selector
        self.fitted_predictors_ = {}
        self.best_params_ = None
        self.best_score_ = None
        self.estimators_ = collect_qlearn_estimators(predictor, list())
        self.debug = debug

    def __prepare_market_info_data(self, symbol, kwargs) -> dict:
        self.market_info_ = MarketInfo(symbol, self.column, debug=self.debug)
        new_kwargs = dict(**kwargs)
        for name, _ in self.estimators_:
            mi_name = f"{name}__{_FIELD_MARKET_INFO}" if name else _FIELD_MARKET_INFO
            new_kwargs[mi_name] = MarketInfo(symbol, self.column)
        return new_kwargs

    def for_interval(self, start, stop):
        """
        Setup dates interval for fitting/prediction
        """
        if self.debug:
            print(" > Selected [" + green(f"{start}:{stop}") + "]")
        self.selector.for_range(start, stop)
        return self

    def take(self, data, nth: Union[str, int] = 0):
        """
        Helper method to take n-th iteration from data picker

        :param data: data to be iterated
        :param nth: if int it returns n-th iteration of data
        :return: data or none if not matched
        """
        return self.selector.take(data, nth)

    def as_datasource(self, data) -> Dict:
        """
        Return prepared data ready to be used in simulator

        :param data: input data
        :return: {symbol : preprocessed_data}
        """
        return self.selector.as_datasource(data)

    def fit(self, X, y=None, **fit_params):
        # reset fitted predictors
        self.fitted_predictors_ = {}
        self.best_params_ = {}
        self.best_score_ = {}

        for symbol, xp in self.selector.iterate(X):
            # propagate market info meta-data to be passed to fit method of all qlearn estimators
            n_fit_params = self.__prepare_market_info_data(symbol, fit_params)

            # in case we still have nothing we need to feed it by some values
            # to avoid fit validation failure
            if y is None:
                y = np.zeros_like(xp)

            # process fitting on prepared data
            _f_p = self.predictor.fit(xp, y, **n_fit_params)

            # store best parameters for each symbol
            if hasattr(_f_p, "best_params_") and hasattr(_f_p, "best_score_"):
                self.best_params_[str(symbol)] = _f_p.best_params_
                self.best_score_[str(symbol)] = _f_p.best_score_

                # just some output on unclear situations
                if self.debug:
                    print(symbol, _f_p.best_params_)

            # here we need to keep a copy of fitted object
            self.fitted_predictors_[str(symbol)] = copy.deepcopy(_f_p)

        return self

    def __get_prediction(self, symbol, x):
        p_key = str(symbol)

        if p_key not in self.fitted_predictors_:
            raise ValueError(f"Can't find fitted predictor for '{p_key}' !")

        # run predictor
        predictor = self.fitted_predictors_[p_key]
        yh = predictor.predict(x)
        return yh

    def predict(self, x):
        """
        Get prediction on all market data from x
        """
        r = dict()

        for symbol, xp in self.selector.iterate(x):
            yh = self.__get_prediction(symbol, xp)
            if isinstance(symbol, str):
                r[symbol] = yh
            else:
                r = yh

        return make_dataframe_from_dict(r, "frame")

    def __rshift__(self, other):
        # TODO: Test -> it doesn't work !!!
        return operation("imply")(self, other)

    def __and__(self, other):
        return operation("and")(self, other)

    def __or__(self, other):
        return operation("or")(self, other)

    def __add__(self, other):
        """
        Support for joining of different predictions
        """
        return operation("join")(self, other)

    def __mul__(self, other):
        """
        Support for multiplication on constant
        """
        return operation("mul")(self, other)

    def __invert__(self):
        return operation("neg")(self)

    def __neg__(self):
        return operation("neg")(self)

    def estimated_portfolio(self, x, forwards_calculator: ForwardReturnsCalculator):
        """
        Get estimated portfolio based on forwards calculator
        """
        rets = {}
        if forwards_calculator is None or not hasattr(forwards_calculator, "get_forward_returns"):
            raise ValueError(
                "forwards_calculator parameter doesn't have get_forward_returns(price, signals, market_info) method"
            )
        for symbol, xp in self.selector.iterate(x):
            yh = self.__get_prediction(symbol, xp)
            rets[symbol] = forwards_calculator.get_forward_returns(xp, yh, MarketInfo(symbol, self.column))

        return make_dataframe_from_dict(rets, "frame")


class SingleInstrumentComposer(MarketDataComposer):
    """
    Shortcut for MarketDataComposer(x, SingleInstrumentPicker(), ...)
    """

    def __init__(self, predictor, column="close", debug=False):
        super().__init__(predictor, SingleInstrumentPicker(), column, debug)


class PortfolioComposer(MarketDataComposer):
    """
    Shortcut for MarketDataComposer(x, PortfolioPicker(), ...)
    """

    def __init__(self, predictor, column="close", debug=False):
        super().__init__(predictor, PortfolioPicker(), column, debug)

    def select(self, rules):
        self.selector.rules = rules
        return self
