from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from qube.quantitative.ta.indicators import atr
from qube.quantitative.tools import scols, infer_series_frequency
from qube.simulator.utils import convert_ohlc_to_ticks, shift_signals


def ___boundaries(price, up, lw, method='fixed', timeframe=None, period=None):
    """
    DEPRECATED

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


def ___triple_barrier_entries(price_data: Union[pd.Series, pd.DataFrame], entries, ub, lb, period,
                           clip_returns=True):
    """

    DEPRECATED

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


class TripleBarrier:
    """
    More complex triple barrier statistics calculations.
    For every signals side (long, short) it returns following labels:

        +2 - hit upper boundary
        -2 - hit lower boundary
        +1 - hit time barrier (price > entry + tolerance)
        -1 - hit time barrier (price < entry - tolerance)
         0 - remains in tolerance interval 

    ```
        :         [Label +2] 
        :++++++++++++++++++++++++++  +upper
        :         [Label +1] 
        :--------------------------  +tolerance
        0 - - - - [Label 0] - - - - 
        :--------------------------  -tolerance
        :         [Label -1] 
        :~~~~~~~~~~~~~~~~~~~~~~~~~~  -lower 
        :         [Label -2] 
    ```
    """

    def __init__(self, price_data: pd.Series | pd.DataFrame) -> None:
        if isinstance(price_data, pd.DataFrame):
            # - emulate quote ticks from OHLC
            ticks = convert_ohlc_to_ticks(price_data)
            self.price = 0.5 * (ticks.ask + ticks.bid)
        elif isinstance(price_data, pd.Series):
            self.price = price_data
        else:
            raise ValueError(f"Unsupported input data {type(price_data)} !")
        self.price_data = price_data
        self.timeframe = pd.Timedelta(infer_series_frequency(price_data))

    def process_signals(self, 
                        signals: pd.Series, delta: str | pd.Timedelta, 
                        take: pd.Series, stop: pd.Series, tol: pd.Series,
                        shift_signals_to_close: Optional[str]='1s',
                        clip_returns=True):
        # - we drop all 0 signals and use sign to indicate entry direction (as our entries can be positions or trading sizes)
        signals = np.sign(signals[signals != 0])
        if shift_signals_to_close is not None:
            signals = shift_signals(signals, self.timeframe - pd.Timedelta(shift_signals_to_close))
        long_signals = signals[signals > 0].index
        short_signals = signals[signals < 0].index

        l_labels = self.tripple_barrier(long_signals, delta, take, stop, tol, clip_returns)
        s_labels = self.tripple_barrier(short_signals, delta, stop, take, tol, clip_returns)
        return scols(l_labels, s_labels, keys=['Longs', 'Shorts']).sort_index()

    def tripple_barrier(self, 
                        entries: pd.DatetimeIndex | List[pd.Timestamp], 
                        delta: str | pd.Timedelta, 
                        upper_b: pd.Series, lower_b: pd.Series, tolerance_b: Optional[pd.Series] = None,
                        clip_returns=True
    ) -> pd.DataFrame:
        if tolerance_b is None:
            tolerance_b = pd.Series(0, index=self.price.index)

        # - tolerance bondaries are symmetrical
        tolerance_b = abs(tolerance_b)

        # - lower is stricly negative, upper - positive
        lower_b = -abs(lower_b)
        upper_b = abs(upper_b)

        # - check some conditions
        if len(lower_b) != len(self.price): raise ValueError(f"Upper boundaries should have the same length of {len(self.price)}")
        if len(upper_b) != len(self.price): raise ValueError(f"Lower boundaries should have the same length of {len(self.price)}")
        if len(tolerance_b) != len(self.price): raise ValueError(f"Tolerance boundaries should have the same length of {len(self.price)}")
        if any(abs(upper_b) <= tolerance_b): raise ValueError(f"Upper boundaries should be above tolerance bounds")
        if any(abs(lower_b) <= tolerance_b): raise ValueError(f"Lower boundaries should be above tolerance bounds")

        if isinstance(entries, (pd.Series, pd.DataFrame)):
            entries = entries.index

        if isinstance(entries, pd.DatetimeIndex):
            entries = entries.to_list()

        entries = pd.to_datetime(entries if isinstance(entries, (list, tuple, set)) else [entries])
        delta = pd.Timedelta(delta) if isinstance(delta, str) else delta

        labels = {}
        timeline = self.price.index

        for e in entries:
            entry_start_idx = timeline.get_indexer([e], method='bfill')[0]
            entry_barrier_idx = timeline.get_indexer([e + delta], method='bfill')[0]

            # - skip entries that are not in price data
            if entry_start_idx < 0 or entry_barrier_idx < 0:
                continue

            xd = self.price.iloc[entry_start_idx : entry_barrier_idx + 1]
            returns_pct = xd / xd.iloc[0] - 1
            U, L, TOL = upper_b.iloc[entry_start_idx], lower_b.iloc[entry_start_idx], tolerance_b.iloc[entry_start_idx]
            where_hit = np.argmax((returns_pct > U) | (returns_pct < L)) or -1

            pe = xd.iloc[where_hit]
            r = returns_pct.iloc[where_hit]
            te = xd.index[where_hit]

            label = 0
            if r >= U: label = +2
            if r <= L: label = -2
            if (r >= +TOL) & (r < U): label = +1
            if (r <= -TOL) & (r > L): label = -1

            labels[e] = {
                'label': label, 'hit': te,
                'returns': (np.clip(r, L, U) if clip_returns else r),
                'price_entry': xd.iloc[0],
                'price_exit': pe, 
                'duration': te - e,
                'U': 100 * (U - 1), 'L': 100 * (1 - L)
            }

        return pd.DataFrame.from_dict(labels, orient='index')

    def get_fixed_ranges(self, up_pct: float, low_pct: float, tol_pct: float=0) -> Tuple[pd.Series, pd.Series]:
        return (
            pd.Series(abs(up_pct) / 100.0, self.price.index), 
            pd.Series(abs(low_pct) / 100.0, self.price.index),
            pd.Series(abs(tol_pct) / 100.0, self.price.index)
        )

    def get_atr_ranges(self, up_factor: float, low_factor: float, tol_factor: float=0, period: int=14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        assert up_factor > low_factor
        assert tol_factor < low_factor
        av = atr(self.price_data, period, percentage=True).shift(1).reindex(self.price.index).ffill() / 100.0
        return (
            abs(up_factor) * av, 
            abs(low_factor) * av, 
            abs(tol_factor) * av
        )