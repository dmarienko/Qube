from datetime import timedelta

import numpy as np
import pandas as pd

from qube.utils.DateUtils import DateUtils

_SPREAD = np.linspace(0.01, 0.03, 3)
_STEP = np.linspace(-0.03, 0.03, 7)


def generate_feed(start_time, initial_price, series_size, trades=False) -> pd.DataFrame:
    start_time_dt = DateUtils.get_datetime(start_time)
    dt = [start_time_dt + timedelta(seconds=i) for i in range(series_size)]
    np.random.seed(int(initial_price))
    # generating bids
    inc = np.cumsum(np.random.choice(_STEP, series_size))
    bid = np.repeat([initial_price], series_size) + np.insert(inc[:-1], 0, [0])
    # generating spreads
    spreads = np.random.choice(_SPREAD, series_size)
    # ask = bid + spread
    ask = np.round(bid + spreads, 2)

    if trades is False:
        return pd.DataFrame(index=dt, data={'bid': bid, 'ask': ask, 'bidvol': np.repeat([100], series_size),
                                            'askvol': np.repeat([200], series_size)},
                            columns=['bid', 'ask', 'bidvol', 'askvol'])
    else:
        taker_side = np.random.choice([0, 1], series_size)
        trade_price = np.round((bid + ask) / 2, 2)
        return pd.DataFrame(index=dt,
                            data={'price': trade_price, 'size': np.repeat([400], series_size), 'takerside': taker_side},
                            columns=['price', 'size', 'takerside'])
