from typing import Union

import numpy as np
import pandas as pd

from qube.quantitative.ta.indicators import atr
from qube.simulator.utils import convert_ohlc_to_ticks


def boundaries(price, up, lw, method='fixed', timeframe=None, period=None):
    """
    Calculate boundaries for triple barrier using different methods.
    Two methods are supported now:
    'fixed' - use fixed upper / lower thresholds
    'atr' - based on ATR volatility

    """
    if method == 'fixed':
        ub, lb = pd.Series(up, price.index), pd.Series(lw, price.index)
    elif method == 'atr':
        if period is None or timeframe is None:
            raise ValueError("Period and timeframe should be specified for ATR volatility")
        ohlc = price.resample(timeframe).agg('ohlc')
        av = atr(ohlc, period, percentage=True).shift(1).reindex(price.index).ffill()
        ub, lb = up * av, lw * av
    elif method == 'std':
        raise ValueError(f"STD vol TODO !!!")
    else:
        raise ValueError(f"Unknown method {method} !")
    return 1 + ub / 100, 1 - lb / 100


def triple_barrier_entries(price_data: Union[pd.Series, pd.DataFrame], entries, ub, lb, period,
                           clip_returns=True):
    """
    Triple Barrier Labeling

        +2 - hit upper boundary
        -2 - hit lower boundary
        +1 - hit time barrier (price >= entry)
        -1 - hit time barrier (price < entry)

    returns datafame with following columns:

        label: label
        hit: time when price hit any label (end)
        ret: actual return of this signal
        price_entry: price when signal started
        price_exit: price when signal ended (hit any label)
        duration: duration of this signal
        U: upper level at the moment of signal's start
        L: lower level at the moment of signal's start

    """
    if isinstance(price_data, pd.DataFrame):
        ticks = convert_ohlc_to_ticks(price_data)
        price = 0.5 * (ticks.ask + ticks.bid)
    elif isinstance(price_data, pd.Series):
        price = price_data
    else:
        raise ValueError(f"Unsupported input data {type(price_data)} !")

    if isinstance(entries, (pd.Series, pd.DataFrame)):
        entries = entries.index

    if isinstance(entries, pd.DatetimeIndex):
        entries = entries.to_list()

    ub = pd.Series(ub, price.index) if isinstance(ub, (int, float)) else ub
    lb = pd.Series(lb, price.index) if isinstance(lb, (int, float)) else lb
    dt = pd.Timedelta(period)
    entries = pd.to_datetime(entries if isinstance(entries, (list, tuple, set)) else [entries])

    labels = {}
    timeline = price.index
    for e in entries:
        ent_idx = timeline.get_indexer([e], method='bfill')[0]
        ent_f_idx = timeline.get_indexer([e + dt], method='bfill')[0]
        # - skip entries that are not in price data
        if ent_idx < 0 or ent_f_idx < 0:
            continue
        xd = price.iloc[ent_idx : ent_f_idx + 1]
        rxd = xd / xd[0]
        U, L = ub[ent_idx], lb[ent_idx]
        where_hit = np.argmax((rxd > U) | (rxd < L)) or -1

        pe = xd[where_hit]
        r = rxd[where_hit]
        te = xd.index[where_hit]

        lbl = 0
        if r >= U:
            lbl = +2
        if r <= L:
            lbl = -2
        if (r >= 1) & (r < U):
            lbl = +1
        if (r < 1) & (r > L):
            lbl = -1

        labels[e] = {'label': lbl,
                     'hit': te,
                     'ret': (np.clip(r, L, U) if clip_returns else r) - 1,
                     'price_entry': xd[0],
                     'price_exit': pe, 'duration': te - e,
                     'U': 100 * (U - 1), 'L': 100 * (1 - L)}

    return pd.DataFrame.from_dict(labels, orient='index')
