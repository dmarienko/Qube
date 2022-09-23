from collections import OrderedDict
from typing import Union, Tuple, List

import numpy as np
import pandas as pd

from qube.quantitative.tools import isscalar

try:
    from numba import njit
except:
    print('numba package is not found !')


    def njit(f):
        return f


def __has_columns(x, *args):
    return isinstance(x, pd.DataFrame) and sum(x.columns.isin(args)) == len(args)


def __check_frame_columns(x, *args):
    if not isinstance(x, pd.DataFrame):
        raise ValueError(f"Input data must be DataFrame but {type(x)} received !")

    if sum(x.columns.isin(args)) != len(args):
        required = [y for y in args if y not in x.columns]
        raise ValueError(f"> Required {required} columns not found in dataframe !")


def find_movements_hilo(x, threshold, pcntg=0.75,
                        t_window: Union[List, Tuple, int] = 10,
                        drop_out_of_market=False,  # not used
                        drop_weekends_crossings=False,  # not used
                        silent=False,
                        use_prev_movement_size_for_percentage=True,
                        result_as_frame=False, collect_log=False,
                        init_direction=0):
    """
    Finds all movements in DataFrame x (should be pandas Dataframe object with low & high columns) which have absolute magnitude >= threshold
    and lasts not more than t_window bars.
    
    # Example:
    # -----------------

    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from pylab import *

    z = 50 + np.random.normal(0, 0.2, 1000).cumsum()
    x = pd.Series(z, index=pd.date_range('1/1/2000 16:00:00', periods=len(z), freq='30s'))

    i_drops, i_grows, _, _ = find_movements(x, threshold=1, t_window=120, pcntg=.75)

    plt.figure(figsize=(15,10))

    # plot series
    plt.plot(x)

    # plot movements
    plt.plot(x.index[i_drops].T, x[i_drops].T, 'r--', lw=1.2);
    plt.plot(x.index[i_grows].T, x[i_grows].T, 'w--', lw=1.2);

    # or new version (after 2018-08-31)
    trends = find_movements(x, threshold=1, t_window=120, pcntg=.75, result_as_indexes=False)
    u, d = trends.UpTrends.dropna(), trends.DownTrends.dropna()
    plt.plot([u.index, u.end], [u.start_price, u.end_price], 'w--', lw=0.7, marker='.', markersize=5);
    plt.plot([d.index, d.end], [d.start_price, d.end_price], 'r--', lw=0.7);

    plt.draw()
    plt.show()

    # -----------------

    :param x: pandas DataFrame object
    :param threshold: movement minimal magnitude threshold
    :param pcntg: percentage of previous movement (if use_prev_movement_size_for_percentage is True) that considered as start of new movement (1 == 100%)
    :param use_prev_movement_size_for_percentage: False if use percentage from previous price extremum (otherwise it uses prev. movement) [True]
    :param t_window: movement's length filter in bars or range: 120 or (0, 100) or (100, np.inf) etc
    :param drop_out_of_market: True if need to drop movements between sessions
    :param drop_weekends_crossings: True if need to drop movemets crossing weekends (for intraday data)
    :param silent: if True it doesn't show progress bar [False by default]
    :param result_as_frame: if False (default) result returned as tuple of indexes otherwise as DataFrame
    :param collect_log: True if need to collect track of tops/bottoms at times when they appeared
    :param init_direction: initial direction, can be 0, 1, -1
    :return: tuple with indexes of (droping movements, growing movements, droping magnitudes, growing magnitudes)
    """

    # check input arguments
    __check_frame_columns(x, 'high', 'low')

    # drop nans (not sure about 0 as replacement)
    # if x.hasnans:
    #     x = x.fillna(0)

    direction = init_direction
    mi, mx = 0, 0
    i_drops, i_grows = [], []
    log_rec = OrderedDict()
    timeline = x.index

    # check filter values
    if isscalar(t_window):
        t_window = [0, t_window]
    elif len(t_window) != 2 or t_window[0] >= t_window[1]:
        raise ValueError("t_window must have 2 ascending elements")

    if not silent: print(' -[', end='')
    n_p_len = max(int(len(x) / 100), 1)

    prev_vL = 0
    prev_vH = 0
    prev_mx = 0
    prev_mi = 0
    last_drop = None
    last_grow = None

    xL_mi = x.low.values[mi]
    xH_mx = x.high.values[mx]

    # for i in range(1, len(x)):
    i = 1
    x_len = len(x)
    while i < x_len:
        vL = x.low.values[i]
        vH = x.high.values[i]

        if direction <= 0:
            if direction < 0 and vH > prev_vH and last_grow is not None:
                # extend to previous grow start
                last_grow[1] = i

                # extend to current point
                mx = i
                xH_mx = x.high.values[mx]
                prev_mx = mx
                prev_vH = xH_mx
                last_drop = None  # already added, reset to avoid duplicates

                mi = i
                xL_mi = x.low.values[mi]

            elif vL < xL_mi:
                mi = i
                xL_mi = x.low.values[mi]
                direction = -1

            else:
                # floating up
                if use_prev_movement_size_for_percentage:
                    l_mv = pcntg * (xH_mx - xL_mi)
                else:
                    l_mv = pcntg * xL_mi

                # check condition    
                if (vL - xL_mi >= threshold) or (l_mv < vL - xL_mi):

                    # case when HighLow of a one bar are extreme points, to avoid infinite loop
                    if mx == mi:
                        mi += 1
                        # xL_mi = x.low.values[mi]

                    last_drop = [mx, mi]
                    # i_drops.append([mx, mi])
                    if last_grow:
                        # check if not violate the previous drop
                        min_idx = np.argmin(x.low.values[last_grow[0]: last_grow[1] + 1])
                        if last_grow[1] > (last_grow[0] + 1) and min_idx > 0 and len(i_drops) > 0:
                            # we have low, which is lower than start of uptrend,
                            # remove the previous drop and replace it with the new one
                            new_drop = [i_drops[-1][0], last_grow[0] + min_idx]
                            i_drops[-1] = new_drop
                            last_grow[0] = last_grow[0] + min_idx

                        i_grows.append(last_grow)
                        last_grow = None

                    prev_vL = xL_mi
                    prev_mi = mi

                    if collect_log:
                        log_rec[timeline[i]] = {'Type': '-', 'Time': timeline[mi], 'Price': xL_mi}

                    # need to move back to the end of last drop
                    i = mi
                    mx = i
                    direction = 1
                    xH_mx = x.high.values[mx]
                    xL_mi = x.low.values[mi]

        if direction >= 0:
            if direction > 0 and vL < prev_vL and last_drop is not None:
                # extend to previous drop start
                last_drop[1] = i

                # extend to current point
                mi = i
                xL_mi = x.low.values[mi]
                prev_mi = mi
                prev_vL = xL_mi
                last_grow = None  # already added, reset to avoid duplicates

                mx = i
                xH_mx = x.high.values[mx]

            elif vH > xH_mx:
                mx = i
                xH_mx = x.high.values[mx]
                direction = +1
            else:
                if use_prev_movement_size_for_percentage:
                    l_mv = pcntg * (xH_mx - xL_mi)
                else:
                    l_mv = pcntg * xH_mx

                if (xH_mx - vH >= threshold) or (l_mv < xH_mx - vH):
                    # i_grows.append([mi, mx])

                    # case when HighLow of a one bar are extreme points, to avoid infinite loop
                    if mx == mi:
                        mx += 1
                        # xH_mx = x.high.values[mx]

                    last_grow = [mi, mx]
                    if last_drop:
                        # check if not violate the previous drop
                        max_idx = np.argmax(x.high.values[last_drop[0]: last_drop[1] + 1])
                        if last_drop[1] > (last_drop[0] + 1) and max_idx > 0 and len(i_grows) > 0:
                            # more than 1 bar between points
                            # we have low, which is lower than start of uptrend,
                            # remove the previous drop and replace it with the new one
                            new_grow = [i_grows[-1][0], last_drop[0] + max_idx]
                            i_grows[-1] = new_grow
                            last_drop[0] = last_drop[0] + max_idx

                        i_drops.append(last_drop)
                        last_drop = None

                    prev_vH = xH_mx
                    prev_mx = mx

                    if collect_log:
                        log_rec[timeline[i]] = {'Type': '+', 'Time': timeline[mx], 'Price': xH_mx}

                    # need to move back to the end of last grow
                    i = mx
                    mi = i
                    xL_mi = x.low.values[mi]
                    xH_mx = x.high.values[mx]
                    direction = -1

        i += 1
        if not silent and not (i % n_p_len): print(':', end='')

    if last_grow:
        i_grows.append(last_grow)
        last_grow = None

    if last_drop:
        i_drops.append(last_drop)
        last_drop = None

    if not silent: print(']-')
    i_drops = np.array(i_drops)
    i_grows = np.array(i_grows)

    # Nothing is found 
    if len(i_drops) == 0 or len(i_grows) == 0:
        if not silent:
            print("\n\t[WARNING] find_movements: No trends found for given conditions !")
        return pd.DataFrame({'UpTrends': [], 'DownTrends': []}) if result_as_frame else ([], [], [], [])

    # retain only movements equal or exceed specified threshold
    if not np.isinf(threshold):
        if i_drops.size:
            i_drops = i_drops[abs(x.low[i_drops[:, 1]].values - x.high[i_drops[:, 0]].values) >= threshold, :]
        if i_grows.size:
            i_grows = i_grows[abs(x.high[i_grows[:, 1]].values - x.low[i_grows[:, 0]].values) >= threshold, :]

    # retain only movements which shorter than specified window
    __drops_len = abs(i_drops[:, 1] - i_drops[:, 0])
    __grows_len = abs(i_grows[:, 1] - i_grows[:, 0])
    if i_drops.size: i_drops = i_drops[(__drops_len >= t_window[0]) & (__drops_len <= t_window[1]), :]
    if i_grows.size: i_grows = i_grows[(__grows_len >= t_window[0]) & (__grows_len <= t_window[1]), :]

    # Removed - filter out all movements which cover period from 16:00 till 9:30 next day

    # Removed - drop crossed weekend if required (we would not want to drop them when use daily prices)

    # drops and grows magnitudes
    v_drops = []
    if i_drops.size:
        v_drops = abs(x.low[i_drops[:, 1]].values - x.high[i_drops[:, 0]].values)

    v_grows = []
    if i_grows.size:
        v_grows = abs(x.high[i_grows[:, 1]].values - x.low[i_grows[:, 0]].values)

    # how to return results
    if not result_as_frame:
        # just raw indexes (by default)
        return i_drops, i_grows, v_drops, v_grows
    else:
        indexes = np.array(x.index)
        i_d, i_g = indexes[i_drops], indexes[i_grows]
        x_d = np.array([x.high[i_d[:, 0]].values, x.low[i_d[:, 1]].values]).transpose()
        x_g = np.array([x.low[i_g[:, 0]].values, x.high[i_g[:, 1]].values]).transpose()

        d = pd.DataFrame(OrderedDict({
            'start_price': x_d[:, 0],
            'end_price': x_d[:, 1],
            'delta': v_drops,
            'end': i_d[:, 1]
        }), index=i_d[:, 0])

        g = pd.DataFrame(OrderedDict({
            'start_price': x_g[:, 0],
            'end_price': x_g[:, 1],
            'delta': v_grows,
            'end': i_g[:, 1]
        }), index=i_g[:, 0])

        trends = pd.concat((g, d), axis=1, keys=['UpTrends', 'DownTrends'])
        if collect_log:
            return trends, pd.DataFrame.from_dict(log_rec, orient='index')

        return trends


def find_movements(x, threshold, pcntg=0.75, t_window: Union[List, Tuple, int] = 10,
                   drop_out_of_market=True,
                   drop_weekends_crossings=True,
                   silent=False,
                   use_prev_movement_size_for_percentage=True,
                   result_as_frame=False, collect_log=False):
    """
    Tries to find all movements in timeseies x (should be pandas Series object) which have absolute magnitude >= threshold
    and lasts not more than t_window bars.
    If needed to drop all movements covering out of market time (from 16:00 till 9:30 next day) set drop_out_of_market to True

    # Example:
    # -----------------

    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from pylab import *

    z = 50 + np.random.normal(0, 0.2, 1000).cumsum()
    x = pd.Series(z, index=pd.date_range('1/1/2000 16:00:00', periods=len(z), freq='30s'))

    i_drops, i_grows, _, _ = find_movements(x, threshold=1, t_window=120, pcntg=.75)

    plt.figure(figsize=(15,10))

    # plot series
    plt.plot(x)

    # plot movements
    plt.plot(x.index[i_drops].T, x[i_drops].T, 'r--', lw=1.2);
    plt.plot(x.index[i_grows].T, x[i_grows].T, 'w--', lw=1.2);

    # or new version (after 2018-08-31)
    trends = find_movements(x, threshold=1, t_window=120, pcntg=.75, result_as_indexes=False)
    u, d = trends.UpTrends.dropna(), trends.DownTrends.dropna()
    plt.plot([u.index, u.end], [u.start_price, u.end_price], 'w--', lw=0.7, marker='.', markersize=5);
    plt.plot([d.index, d.end], [d.start_price, d.end_price], 'r--', lw=0.7);

    plt.draw()
    plt.show()

    # -----------------

    :param x: pandas Series object
    :param threshold: movement minimal magnitude threshold
    :param pcntg: percentage of previous movement (if use_prev_movement_size_for_percentage is True) that considered as start of new movement (1 == 100%)
    :param use_prev_movement_size_for_percentage: False if use percentage from previous price extremum (otherwise it uses prev. movement) [True]
    :param t_window: movement's length filter in bars or range: 120 or (0, 100) or (100, np.inf) etc
    :param drop_out_of_market: True if need to drop movements between sessions
    :param drop_weekends_crossings: True if need to drop movemets crossing weekends (for intraday data)
    :param silent: if True it doesn't show progress bar [False by default]
    :param result_as_frame: if False (default) result returned as tuple of indexes otherwise as DataFrame
    :param collect_log: True if need to collect track of tops/bottoms at times when they appeared
    :return: tuple with indexes of (droping movements, growing movements, droping magnitudes, growing magnitudes)
    """

    # check input arguments
    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    # drop nans (not sure about 0 as replacement)
    if x.hasnans:
        x = x.fillna(0)

    mi, mx, direction = 0, 0, 0
    i_drops, i_grows = [], []
    log_rec = OrderedDict()
    timeline = x.index

    # check filter values
    if isscalar(t_window):
        t_window = [0, t_window]
    elif len(t_window) != 2 or t_window[0] >= t_window[1]:
        raise ValueError("t_window must have 2 ascending elements")

    if not silent: print(' -[', end='')
    n_p_len = max(int(len(x) / 100), 1)

    for i in range(1, len(x)):
        v = x.iat[i]

        if direction <= 0:
            if v < x.iat[mi]:
                mi = i
                direction = -1
            else:
                # floating up
                if use_prev_movement_size_for_percentage:
                    l_mv = pcntg * (x.iat[mx] - x.iat[mi])
                else:
                    l_mv = pcntg * x.iat[mi]

                # check condition
                if (v - x.iat[mi] >= threshold) or (l_mv < v - x.iat[mi]):
                    i_drops.append([mx, mi])
                    if collect_log:
                        log_rec[timeline[i]] = {'Type': '-', 'Time': timeline[mi], 'Price': x.iat[mi]}
                    mx = i
                    direction = 1

        if direction >= 0:
            if v > x.iat[mx]:
                mx = i
                direction = +1
            else:
                if use_prev_movement_size_for_percentage:
                    l_mv = pcntg * (x.iat[mx] - x.iat[mi])
                else:
                    l_mv = pcntg * x.iat[mx]

                if (x.iat[mx] - v >= threshold) or (l_mv < x.iat[mx] - v):
                    i_grows.append([mi, mx])
                    if collect_log:
                        log_rec[timeline[i]] = {'Type': '+', 'Time': timeline[mx], 'Price': x.iat[mx]}
                    mi = i
                    direction = -1

        if not silent and not (i % n_p_len): print(':', end='')

    if not silent: print(']-')
    i_drops = np.array(i_drops)
    i_grows = np.array(i_grows)

    # Nothing is found
    if len(i_drops) == 0 or len(i_grows) == 0:
        if not silent:
            print("\n\t[WARNING] find_movements: No trends found for given conditions !")
        return pd.DataFrame({'UpTrends': [], 'DownTrends': []}) if result_as_frame else ([], [], [], [])

    # retain only movements equal or exceed specified threshold
    if not np.isinf(threshold):
        if i_drops.size:
            i_drops = i_drops[abs(x[i_drops[:, 1]].values - x[i_drops[:, 0]].values) >= threshold, :]
        if i_grows.size:
            i_grows = i_grows[abs(x[i_grows[:, 1]].values - x[i_grows[:, 0]].values) >= threshold, :]

    # retain only movements which shorter than specified window
    __drops_len = abs(i_drops[:, 1] - i_drops[:, 0])
    __grows_len = abs(i_grows[:, 1] - i_grows[:, 0])
    if i_drops.size: i_drops = i_drops[(__drops_len >= t_window[0]) & (__drops_len <= t_window[1]), :]
    if i_grows.size: i_grows = i_grows[(__grows_len >= t_window[0]) & (__grows_len <= t_window[1]), :]

    # filter out all movements which cover period from 16:00 till 9:30 next day
    if drop_out_of_market and (isinstance(x, pd.Series) and isinstance(x.index, pd.DatetimeIndex)):
        hours = np.array(x.index.hour)
        if i_drops.size:
            h = hours[i_drops]
            i_drops = i_drops[~((h[:, 0] <= 16) & (h[:, 1] >= 9))]
        if i_grows.size:
            h = hours[i_grows]
            i_grows = i_grows[~((h[:, 0] <= 16) & (h[:, 1] >= 9))]

    # drop crossed weekend if required (we would not want to drop them when use daily prices)
    # drop if start < Sunday and end is Sunday. Drop if start and end are different weeks and start is not Sunday.
    if drop_weekends_crossings:
        dayofweeks = np.array(x.index.dayofweek)
        weeks = np.array(x.index.isocalendar().week)
        if i_drops.size and (isinstance(x, pd.Series) and isinstance(x.index, pd.DatetimeIndex)):
            d = dayofweeks[i_drops]
            w = weeks[i_drops]
            i_drops = i_drops[~(((d[:, 0] < 6) & (d[:, 1] == 6)) | (w[:, 0] != w[:, 1]) & (d[:, 0] != 6))]
        if i_grows.size and (isinstance(x, pd.Series) and isinstance(x.index, pd.DatetimeIndex)):
            d = dayofweeks[i_grows]
            w = weeks[i_grows]
            i_grows = i_grows[~(((d[:, 0] < 6) & (d[:, 1] == 6)) | (w[:, 0] != w[:, 1]) & (d[:, 0] != 6))]

    # drops and grows magnitudes
    v_drops = []
    if i_drops.size:
        v_drops = abs(x[i_drops[:, 1]].values - x[i_drops[:, 0]].values)

    v_grows = []
    if i_grows.size:
        v_grows = abs(x[i_grows[:, 1]].values - x[i_grows[:, 0]].values)

    # how to return results
    if not result_as_frame:
        # just raw indexes (by default)
        return i_drops, i_grows, v_drops, v_grows
    else:
        indexes = np.array(x.index)
        i_d, i_g = indexes[i_drops], indexes[i_grows]
        x_d, x_g = x.values[i_drops], x.values[i_grows]

        d = pd.DataFrame(OrderedDict({
            'start_price': x_d[:, 0],
            'end_price': x_d[:, 1],
            'delta': v_drops,
            'end': i_d[:, 1]
        }), index=i_d[:, 0])

        g = pd.DataFrame(OrderedDict({
            'start_price': x_g[:, 0],
            'end_price': x_g[:, 1],
            'delta': v_grows,
            'end': i_g[:, 1]
        }), index=i_g[:, 0])

        trends = pd.concat((g, d), axis=1, keys=['UpTrends', 'DownTrends'])
        if collect_log:
            return trends, pd.DataFrame.from_dict(log_rec, orient='index')

        return trends
