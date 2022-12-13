import pandas as pd
from os.path import exists


def _read_timeseries_data(symbol, compressed=False, as_dict=False):
    name = f'{symbol}.csv.gz' if compressed else f'{symbol}.csv'
    fpath = f'../data/{name}'
    if not exists(fpath):
        fpath = f'qube/tests/data/{name}'
    data = pd.read_csv(fpath, parse_dates=True, header=0, index_col=['time'])
    return {symbol: data} if as_dict else data