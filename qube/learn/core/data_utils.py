import re
from dataclasses import dataclass
from datetime import timedelta as timedelta_t
from typing import Union, List, Set

import numpy as np
import pandas as pd

from qube.learn.core.utils import _check_frame_columns
from qube.quantitative.tools import scols, infer_series_frequency
from qube.utils.utils import mstruct


@dataclass
class DataType:
    # data type: multi, ticks, ohlc
    type: str
    symbols: List[str]
    freq: str
    subtypes: Set[str]

    def frequency(self):
        return pd.Timedelta(self.freq)


_S1 = pd.Timedelta("1s")
_D1 = pd.Timedelta("1D")


def pre_close_time_delta(freq):
    """
    What is minimal time delta time bar's close
    It returns 1S for all timeframes > 1Min and F/10 otherwise

    TODO: take in account session start/stop times for daily freq
    """
    if freq >= _D1:
        raise ValueError("Data with daily frequency is not supported properly yet !")

    return _S1 if freq > _S1 else freq / 10


def pre_close_time_shift(bars):
    """
    What is interval to 'last' time before bar's close

    TODO: take in account session start/stop times for daily freq
    """
    _tshift = pd.Timedelta(infer_series_frequency(bars[:100]))
    return _tshift - pre_close_time_delta(_tshift)


def time_delta_to_str(d: Union[int, timedelta_t, pd.Timedelta]):
    """
    Convert timedelta object to pretty print format

    :param d:
    :return:
    """
    seconds = d.total_seconds() if isinstance(d, (pd.Timedelta, timedelta_t)) else int(d)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    r = ""
    if days > 0:
        r += "%dD" % days
    if hours > 0:
        r += "%dH" % hours
    if minutes > 0:
        r += "%dMin" % minutes
    if seconds > 0:
        r += "%dS" % seconds
    return r


def series_period_as_str(data: Union[pd.Series, pd.DataFrame], min_size=100) -> str:
    """
    Returns series/dataframe period as string
    """
    return time_delta_to_str(pd.Timedelta(infer_series_frequency(data[:min_size])))


def shift_for_timeframe(signals: pd.Series, data: pd.DataFrame, tf: Union[str, pd.Timedelta]) -> pd.Series:
    """
    Shift signals to future by timeframe - timeframe(data)
    """
    t = pd.Timedelta(infer_series_frequency(data[:100]))
    tf = pd.Timedelta(tf)
    return signals.shift(1, freq=tf - t) if tf > t else signals


def timeseries_density(dx, period="1Min"):
    """
    Detect average records density per period
    :param dx:
    :param period:
    :return:
    """
    return dx.groupby(pd.Grouper(freq=period)).count().mean().mean()


def inner_join_and_split(d1, d2, dropna=True, keep="last"):
    """
    Joins two series (frames) and reindex on most dense time index

    :param d1: first frame
    :param d2: second frame
    :param dropna: if need frop nans
    :param keep: what to keep on same timestamps
    :return: tuple of reindexed frames (d1, d2)
    """
    extract_by_key = lambda x, sfx: x.filter(regex=".*_%s" % sfx).rename(columns=lambda y: y.split("_")[0])

    # period for density of records: we take more sparsed one
    dens_period = max((d1.index[-1] - d1.index[0]) / len(d1), (d2.index[-1] - d2.index[0]) / len(d2))
    if timeseries_density(d1, dens_period) > timeseries_density(d2, dens_period):
        m1 = pd.merge_asof(d1, d2, left_index=True, right_index=True, suffixes=["_X1", "_X2"])
    else:
        m1 = pd.merge_asof(d2, d1, left_index=True, right_index=True, suffixes=["_X2", "_X1"])

    if dropna:
        m1.dropna(inplace=True)

    if m1.index.has_duplicates:
        m1 = m1[~m1.index.duplicated(keep=keep)]

    return extract_by_key(m1, "X1"), extract_by_key(m1, "X2")


def merge_ticks_from_dict(data, instruments, dropna=True, keep="last"):
    """
    :param data:
    :param instruments:
    :param dropna:
    :param keep:
    :return:
    """
    if len(instruments) == 1:
        return pd.concat(
            [
                data[instruments[0]],
            ],
            keys=instruments,
            axis=1,
        )

    max_dens_period = max([(d.index[-1] - d.index[0]) / len(d) for s, d in data.items() if s in instruments])
    densitites = {s: timeseries_density(data[s], max_dens_period) for s in instruments}
    pass_dens = dict(sorted(densitites.items(), key=lambda x: x[1], reverse=True))
    ins = list(pass_dens.keys())
    mss = list(inner_join_and_split(data[ins[0]], data[ins[1]], dropna=False, keep=keep))

    for s in ins[2:]:
        mss.append(inner_join_and_split(mss[0], data[s], dropna=False, keep=keep)[1])

    r = pd.concat(mss, axis=1, keys=ins)
    if dropna:
        r.dropna(inplace=True)

    return r


def _get_top_names(cols):
    return list(set(cols.get_level_values(0).values))


def make_dataframe_from_dict(data: dict, data_type: str):
    """
    Produses dataframe from dictionary
    :param data:
    :param data_type:
    :return:
    """
    if isinstance(data, dict):
        if data_type in ["ohlc", "frame"]:
            # return pd.concat(data.values(), keys=data.keys(), axis=1)
            xd = dict((k, v) for (k, v) in data.items() if v is not None)
            return pd.concat(xd.values(), keys=xd.keys(), axis=1) if xd else pd.DataFrame()
        elif data_type == "ticks":
            return merge_ticks_from_dict(data, list(data.keys()))
        else:
            raise ValueError(f"Don't know how to merge '{data_type}'")
    return data


def do_columns_contain(cols, keys):
    return all([c in cols for c in keys])


def detect_data_type(data) -> DataType:
    """
    Finds info about data structure

    :param data:
    :return:
    """
    dtype = re.findall(".*'(.*)'.*", f"{type(data)}")[0]
    freq = None
    symbols = []
    subtypes = None

    if isinstance(data, pd.DataFrame):
        cols = data.columns

        if isinstance(cols, pd.MultiIndex):
            # multi index dataframe
            dtype = "multi"
            symbols = _get_top_names(cols)
        else:
            # just dataframe
            dtype = "frame"
            if do_columns_contain(cols, ["open", "high", "low", "close"]):
                symbols = ["OHLC1"]
                dtype = "ohlc"
            elif do_columns_contain(cols, ["bid", "ask"]):
                symbols = ["TICKS1"]
                dtype = "ticks"

    elif isinstance(data, pd.Series):
        dtype = "series"
        symbols = [data.name if data.name is not None else "SERIES1"]

    elif isinstance(data, dict):
        dtype = "dict"
        symbols = list(data.keys())
        subtypes = {detect_data_type(v).type for v in data.values()}

    if isinstance(data, (pd.DataFrame, pd.Series)):
        freq = time_delta_to_str(infer_series_frequency(data[:100]))

    return DataType(type=dtype, symbols=symbols, freq=freq, subtypes=subtypes)


def ohlc_to_flat_price_series(ohlc: pd.DataFrame, freq: pd.Timedelta, sess_start, sess_end):
    """
    Make flat series from OHLC data.

    time     open close
    15:30:00 100  105
    15:35:00 106  107

    Flat series:
    15:30:00 100
    15:34:59 105
    15:35:00 106
    15:49:59 107
    """
    _check_frame_columns(ohlc, "open", "close")
    return pd.concat((ohlc.open, ohlc.close.shift(1, freq=freq - pre_close_time_delta(freq)))).sort_index()


def forward_timeseries(x: pd.Series, period):
    """
    Forward shifted timeseries for specified time period
    """
    if not isinstance(x, pd.Series):
        raise ValueError("forward_timeseries> Argument must be pd.Series !")
    f_x = x.asof(x.index + period).reset_index(drop=True)
    f_x.index = x.index
    # drop last points
    f_x[f_x.index[-1] - period :] = np.nan
    return f_x


def backward_timeseries(x: pd.Series, period):
    """
    Backward shifted timeseries for specified time period
    """
    if not isinstance(x, pd.Series):
        raise ValueError("backward_timeseries> Argument must be pd.Series !")
    f_x = x.asof(x.index - period).reset_index(drop=True)
    f_x.index = x.index
    # drop first points
    f_x[: f_x.index[0] + period] = np.nan
    return f_x


def put_under(section, x):
    """
    Put series under 'section' (higher level of multi index)

    put_under('MySect', pd.Series([1,2,3,4], name='xx'))

       MySect
           xx
    0       1
    1       2
    2       3
    3       4
    """
    s_name = x.name
    x = x.rename(section).to_frame() if isinstance(x, pd.Series) else x
    return pd.concat([x], axis=1, keys=[s_name]).swaplevel(0, 1, 1)
