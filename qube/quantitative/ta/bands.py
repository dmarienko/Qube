"""
 Set of utilities for generating trading signals, handle prices etc
"""

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame

from qube.quantitative.tools import isscalar, nans


def __get_row(r, i):
    return [r.iat[i, j] for j in range(r.shape[1])]


def __set_row(r, i, v):
    for k in range(len(v)): r.iat[i, k] = v[k]


def generate_bands_signals(prices: NDFrame, model: NDFrame, zentry, zexit,
                           back_to_range_cond=True, flat_when_out_again=False,
                           size_func=lambda i, p, d: d * np.zeros(len(p))) -> pd.DataFrame:
    """
    Spread positions generator.

    :param prices: prices dataframe
    :param model: series with model
    :param zentry: level for opening position (scalar or np.array)
    :param zexit: level for closing position (scalar or np.array)
    :param back_to_range_cond: True if needed to check model back inside bands region
    :param flat_when_out_again: if true it flats position when model is get out entry boundaries
    :param size_func: position size generating function
    :return: frame with generated signals
    """
    if isinstance(prices, pd.Series):
        prices = pd.DataFrame(prices)

    signals = pd.DataFrame(data=nans(prices.shape), index=prices.index, columns=prices.columns)
    model = pd.Series(data=model.values, index=model.index).reindex(prices.index)

    if isscalar(zentry): zentry = np.repeat(zentry, len(model))
    if isscalar(zexit): zexit = np.repeat(zexit, len(model))

    # flatting position doesn't have any sense when we do not check crossing the band's boards for entries
    if not back_to_range_cond:
        if flat_when_out_again:
            print('Reseting "flat_when_out_again" to false because "back_to_range_cond" is not set')
        flat_when_out_again = False

    pos = 0
    is_back_to_band = True

    for i in range(1, len(model)):
        mi = model.iat[i]
        if np.isnan(mi): continue

        if pos == 0:
            ix = i

            # if forced check when model is back into bands range
            if back_to_range_cond:
                is_back_to_band = (-zentry[i] < mi < zentry[i])
                ix = i - 1

            # if got signal for long position
            if model.iat[ix] <= -zentry[ix] and is_back_to_band:
                pos = +1
                __set_row(signals, i, size_func(i, __get_row(prices, i), pos))
                continue

            # if got signal for short position
            if model.iat[ix] >= zentry[ix] and is_back_to_band:
                pos = -1
                __set_row(signals, i, size_func(i, __get_row(prices, i), pos))
        else:
            if pos < 0:
                # reversion to long is possible
                if not back_to_range_cond and mi <= -zentry[i]:
                    pos = +1
                    __set_row(signals, i, size_func(i, __get_row(prices, i), pos))
                    continue

                # exit from short position
                if mi <= -zexit[i]:
                    pos = 0
                    signals.iloc[i] = 0
                    continue

                # if model got out of the entry's side - we flat position
                if flat_when_out_again and mi > zentry[i]:
                    pos = 0
                    signals.iloc[i] = 0
                    continue

            if pos > 0:
                # reversion to short is possible
                if not back_to_range_cond and mi >= zentry[i]:
                    pos = -1
                    __set_row(signals, i, size_func(i, __get_row(prices, i), pos))
                    continue

                # exit from long position
                if mi >= zexit[i]:
                    pos = 0
                    signals.iloc[i] = 0
                    continue

                # if model got out of the entry's side - we flat position
                if flat_when_out_again and mi < -zentry[i]:
                    pos = 0
                    signals.iloc[i] = 0

    return signals
