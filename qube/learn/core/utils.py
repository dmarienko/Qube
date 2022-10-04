import inspect
import warnings
from typing import Dict, Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from qube.utils.ui_utils import red, blue


def _check_frame_columns(x, *args):
    if not (isinstance(x, pd.DataFrame) and sum(x.columns.isin(args)) == len(args)):
        raise ValueError(f"Input series must be DataFrame with {args} columns !")


def debug_output(data, name, start=3, end=3, time_info=True):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        t_info = f'{len(data)} records'
        if time_info:
            t_info += f' | {data.index[0]}:{data.index[-1]}'
        hdr = f'.-<{name} {t_info} records>-' + ' -' * 50
        sep = ' -' * 50
        print(blue(hdr[:len(sep)]))
        print(data.head(start).to_string(header=True))
        if start < len(data):
            print(' \t . . . . . . ')
            print(data.tail(end).to_string(header=True))
        print(blue(sep))
    else:
        print(data)


def get_object_params(obj, deep=True) -> dict:
    """
    Get parameter names for the object
    """
    cls = obj.__class__
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    if init is object.__init__:
        # No explicit constructor to introspect
        return {}

    init_signature = inspect.signature(init)
    # Consider the constructor parameters excluding 'self'
    parameters = [p for p in init_signature.parameters.values()
                  if p.name != 'self' and p.kind != p.VAR_KEYWORD]
    for p in parameters:
        if p.kind == p.VAR_POSITIONAL:
            raise RuntimeError("class should always "
                               "specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s with constructor %s doesn't "
                               " follow this convention."
                               % (cls, init_signature))

    names = sorted([p.name for p in parameters])

    # Extract and sort argument names excluding 'self'
    out = dict()
    for key in names:
        try:
            value = getattr(obj, key)
        except AttributeError:
            warnings.warn('get_class_params() will raise an '
                          'AttributeError if a parameter cannot be '
                          'retrieved as an instance attribute. Previously '
                          'it would return None.',
                          FutureWarning)
            value = None
        if deep and hasattr(value, 'get_params'):
            deep_items = value.get_params().items()
            out.update((key + '__' + k, val) for k, val in deep_items)
        out[key] = value

    return out


def ls_params(e, to_skip=list(['verbose', 'memory'])):
    """
    Show parameters of predictor as dictionary ready for GridSearchCV
    """
    ps0 = e.get_params()
    res = []

    for k, v in ps0.items():
        is_obj = isinstance(v, (BaseEstimator, Pipeline, TransformerMixin))
        is_iter = isinstance(v, (list, tuple))
        is_x = any([(k.endswith('__' + x) or k == x) for x in to_skip])

        if v is None or is_obj or is_iter or is_x:
            continue
        else:
            res.append(f"\t'{k}': [{repr(v)}]")

    res_s = ',\n'.join(res)
    print('{\n' + res_s + '\n}')


def replicate_object_with_mixings(obj, fields_to_mix: Dict[str, Any]):
    """
    Make object instance replica with mixed fields
    """
    p_cls = obj.__class__
    p_meths = {**p_cls.__dict__, **fields_to_mix}
    new_p_cls = type(p_cls.__name__, tuple(p_cls.mro()[1:]), p_meths)
    p_params = get_object_params(obj)

    return new_p_cls(**p_params)
