from typing import Dict

import numpy as np
import pandas as pd

from qube.quantitative.tools import scols, srows
from qube.datasource import DataSource
from qube.simulator.core import ExecutionLogger
from qube.simulator.utils import shift_signals
from qube.utils.DateUtils import DateUtils


class MockDataSource(DataSource):

    def __init__(self, start, n_bars, amplitudes=(10, 30, 1), freq="1d"):
        self.start = start
        self.n_bars = n_bars
        self.freq = freq
        self.amplitudes = amplitudes

    def get_name(self):
        return "Fake Testing Data"

    def load_data(self, series=None, start=None, end=None, *args, **kwargs):
        if not isinstance(series, list):
            series = [series]
        s, e, d = self.amplitudes

        blk = np.concatenate((np.arange(s, e + 1), np.arange(e - 1, s, -1)))
        data = np.tile(blk, int(np.ceil(self.n_bars / len(blk))))[: self.n_bars]

        return {
            x: pd.DataFrame(
                np.array([data, d + data]).T,
                columns=["open", "close"],
                index=pd.date_range(start=self.start, periods=self.n_bars, freq=self.freq),
            ).loc[DateUtils.get_datetime(start) : DateUtils.get_datetime(end)]
            for x in series
        }


class TickMockDataSource(DataSource):

    def __init__(self, data):
        self.data = data

    def get_name(self):
        return "Fake Testing Tick Data"

    def load_data(self, series=None, start=None, end=None, *args, **kwargs):
        if not isinstance(series, list):
            series = [series]
        return {x: self.data[x].loc[DateUtils.get_datetime(start) : DateUtils.get_datetime(end)] for x in series}


# Handy positions generator from dictionary
def gen_pos(p_dict):
    sgen = lambda name, sigs: pd.Series(
        data=sigs[1 : len(sigs) : 2],
        index=pd.to_datetime(sigs[0:-1:2], format="mixed"),
        name=name,
    )
    return pd.concat([sgen(s, p) for (s, p) in p_dict.items()], axis=1)


def cumulative_pnl_calcs_eod(positions, prices):
    """
    Simplest method to calculate cumulative PnL by signals/positions for EOD only !
    """
    p_cl = shift_signals(prices["close"], "16:00:00")
    return (p_cl.diff() * positions.shift()).cumsum()


def cumulative_pnl_validation_eod(positions, prices):
    """
    Simplest method to calculate cumulative PnL by positions on close prices (eod)
    """
    p_ch = prices["close"].diff()
    p_ch.index = p_ch.index + pd.DateOffset(hours=16, minutes=0)
    return (p_ch * positions.shift()).cumsum()


def portfolio_from_executions(exec_log: ExecutionLogger, prices: Dict[str, pd.DataFrame]):
    el = exec_log.get_execution_log()
    s2 = {s: el[el.instrument == s].quantity.cumsum() for s in set(el.instrument)}

    for k, v in s2.items():
        v.index = v.index.round("2s")
        s2[k] = v

    prep_prices = lambda s: srows(
        shift_signals(prices[s]["open"], "9:30:00"),
        shift_signals(prices[s]["close"], "16:00:00"),
    )

    d1 = scols(prep_prices("AAPL"), srows(s2["AAPL"], keep="last"), names=["s", "q"]).ffill().fillna(0)
    d2 = scols(prep_prices("XOM"), srows(s2["XOM"], keep="last"), names=["s", "q"]).ffill().fillna(0)
    d3 = scols(prep_prices("MSFT"), srows(s2["MSFT"], keep="last"), names=["s", "q"]).ffill().fillna(0)
    r = (d1.s.diff() * d1.q.shift()) + (d2.s.diff() * d2.q.shift()) + (d3.s.diff() * d3.q.shift())
    return r


def cross_ma(data, instr, amnt=100000, p_fast=10, p_slow=50):
    closes = data[instr]["close"]
    fast = closes.rolling(window=p_fast).mean()
    slow = closes.rolling(window=p_slow).mean()

    long_pos = round(amnt / closes[slow < fast], 0)
    short_pos = -round(amnt / closes[slow >= fast], 0)

    positions = srows(long_pos, short_pos)
    # positions.index = positions.index + pd.DateOffset(hours=15, minutes=59)
    positions.index = positions.index + pd.DateOffset(hours=16, minutes=0)
    positions.name = instr  # setup series name
    return positions


def cross_ma_signals_generator(data, instr, p_fast=10, p_slow=50):
    closes = data[instr]["close"]
    fast = closes.rolling(window=p_fast).mean()
    slow = closes.rolling(window=p_slow).mean()

    long_pos = pd.Series(+1, closes[(fast > slow) & (fast.shift(1) <= slow.shift(1))].index)
    short_pos = pd.Series(-1, closes[(fast < slow) & (fast.shift(1) >= slow.shift(1))].index)

    positions = shift_signals(srows(long_pos, short_pos), "16h")
    positions.name = instr  # setup series name
    return positions
