from typing import Union

import numpy as np
import pandas as pd

from qube.quantitative.ta.swings.swings_splitter import find_movements
from qube.quantitative.ta.indicators import ema, sma
from qube.quantitative.tools import ohlc_resample, apply_to_frame
from qube.utils.utils import mstruct


def pullbacks_estimate(hmin, n_ema_days=14, f=0.1, smoother=ema):
    """
    Estimation of pullbacks on rolling basis as percentage of daily price movement
    """
    try:
        h1d = ohlc_resample(hmin, '1D')
    except:
        h1d = hmin.resample('1D').agg('ohlc').fillna(method='ffill')

    # averaged daily spread (on rolling window)
    return f * apply_to_frame(smoother, (h1d.high - h1d.low) / h1d.low, n_ema_days).shift(1)


def pullbacks_estimate_abs(ohlc, n_ema_days=7, smoother=sma, field='close'):
    """
    Estimate pullbacks as smoothed std deviation of price bouncing

    :param ohlc:
    :param n_ema_days:
    :param field:
    :param smoother:
    :return:
    """
    if field == 'typical':
        px = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
    else:
        px = ohlc[field]
    px = abs(px.diff())
    pb_ticks = px.groupby(px.index.date).std().shift(1)
    pb_ticks.index = pb_ticks.index.map(pd.Timestamp)
    return apply_to_frame(smoother, pb_ticks, n_ema_days) if smoother is not None else pb_ticks


def get_reversal_points(data: pd.DataFrame,
                        pb_roll_window=14, pb_factor=0.1, pb_smoother=ema,
                        pb_field: str = 'close', pb_method: str = 'default', pb_fixed=None,
                        field: str = 'close', split_interval='W'):
    """
    Find reversal points for series of data

    :param data: OHLC dataframe
    :param pb_roll_window:
    :param pb_factor:
    :param pb_field:
    :param pb_smoother:
    :param pb_fixed: fixed pullback value (used when pb_method is 'fixed abs' or 'fixed rel')
    :param pb_method: 'default' or 'abs' or 'fixed abs' or 'fixed rel'
    :param field: used field for OHLC dataframe
    :param split_interval: default W   
    :return: mstruct(points, tracking, thresholds, trends)
    """
    use_abs_pullback = False
    if pb_method and pb_method.startswith('abs'):
        pbacks = pb_factor * pullbacks_estimate_abs(data, pb_roll_window, smoother=pb_smoother, field=pb_field)
        use_abs_pullback = True
    elif pb_method.startswith('fixed'):
        if pb_fixed is not None:
            pbacks = pd.Series(pb_fixed, index=sorted(set(data.index.floor('D'))))
        else:
            raise ValueError("For Fixed pullback method pb_fixed argument must be specified !")
        use_abs_pullback == pb_method.endswith('abs')
    else:
        pbacks = pullbacks_estimate(data, pb_roll_window, pb_factor, smoother=pb_smoother)

    r_points, track, all_trends = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print('%s : ' % ('A' if use_abs_pullback else 'R'), end='')

    for _, i_data in data.sort_index().groupby(pd.Grouper(freq=split_interval)):
        if len(i_data) > 0:

            if isinstance(data, pd.DataFrame) and field in data.columns:
                i_data = i_data.ffill()[field]

            #             pb = pbacks.loc[pd.to_datetime(i_data.index[0]).date()]
            pb = pbacks.loc[pd.Timestamp(i_data.index[0].date())]

            if np.isfinite(pb):
                # run it on weekly set
                if use_abs_pullback:
                    trends, trk = find_movements(i_data, pb, use_prev_movement_size_for_percentage=False,
                                                 pcntg=np.inf, t_window=np.inf, drop_weekends_crossings=True,
                                                 drop_out_of_market=False, result_as_frame=True, silent=True,
                                                 collect_log=True)
                else:
                    trends, trk = find_movements(i_data, np.inf, use_prev_movement_size_for_percentage=False,
                                                 pcntg=pb, t_window=np.inf, drop_weekends_crossings=True,
                                                 drop_out_of_market=False, result_as_frame=True, silent=True,
                                                 collect_log=True)
                # get reversal points
                if len(trends) > 0 and isinstance(trends, pd.DataFrame):
                    u, d = trends.UpTrends.dropna(), trends.DownTrends.dropna()
                    d.delta = -d.delta
                    _w_points = pd.concat((u, d), axis=0).sort_index()
                    _w_points = _w_points[_w_points.delta != 0]
                    r_points = pd.concat((r_points, _w_points), axis=0).sort_index()
                    track = pd.concat((track, trk), axis=0).sort_index()
                    all_trends = pd.concat((all_trends, trends), axis=0).sort_index()
                    print('+', end='')
                else:
                    print('.', end='')

                # add duration
                if len(r_points) > 0:
                    r_points['duration'] = r_points.end - pd.Series(r_points.index, index=r_points.index)

    # add price at RP was occured
    track['PriceOccured'] = data.loc[track.index][field]

    return mstruct(points=r_points, tracking=track, thresholds=pbacks, trends=all_trends)


def split_by_reversal_points(x: Union[pd.Series, pd.DataFrame], points: pd.DataFrame, n_shift=1):
    """
    Split data from x (Series or DataFrame) by top and bottom reversal points

    :param x:
    :param points:
    :param n_shift:
    :return:
    """
    _x_len = len(x)
    _lo_pts = points[points.delta > 0].index
    _hi_pts = points[points.delta < 0].index

    _lo_idx = np.array([x.index.get_loc(i) for i in _lo_pts if i in x.index])
    _hi_idx = np.array([x.index.get_loc(i) for i in _hi_pts if i in x.index])

    _lo_idx_n = np.array(list(filter(lambda x: x >= 0 and x < _x_len, _lo_idx - n_shift)))
    _hi_idx_n = np.array(list(filter(lambda x: x >= 0 and x < _x_len, _hi_idx - n_shift)))

    # drop too old points
    _lo_idx_n = _lo_idx_n[abs((x.index[_lo_idx] - x.index[_lo_idx_n])) < pd.Timedelta('1D')]
    _hi_idx_n = _hi_idx_n[abs((x.index[_hi_idx] - x.index[_hi_idx_n])) < pd.Timedelta('1D')]

    # find shifted points
    _lo_n = x.iloc[_lo_idx_n]
    _hi_n = x.iloc[_hi_idx_n]

    return _lo_n, _hi_n


def select_points_without_extremums(x: Union[pd.Series, pd.DataFrame], points: pd.DataFrame, n_p: int, n_a: int):
    lo, hi = pd.DataFrame(), pd.DataFrame()
    for i in range(abs(n_p), -abs(n_a) - 1, -1):
        _l, _h = split_by_reversal_points(x, points, i)
        lo = pd.concat((lo, _l), axis=0)
        hi = pd.concat((hi, _h), axis=0)
    return x.loc[x.index.difference(lo.index).difference(hi.index)]


def select_neighborhood_points(x: Union[pd.Series, pd.DataFrame], points: pd.DataFrame, n_p: int, n_a: int):
    lo, hi = pd.DataFrame(), pd.DataFrame()
    for i in range(abs(n_p), -abs(n_a) - 1, -1):
        _l, _h = split_by_reversal_points(x, points, i)
        lo = pd.concat((lo, _l), axis=0)
        hi = pd.concat((hi, _h), axis=0)
    return lo, hi
