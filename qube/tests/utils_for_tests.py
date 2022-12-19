import pandas as pd
from os.path import exists

from qube.datasource.controllers.MongoController import MongoController


def _read_timeseries_data(symbol, compressed=False, as_dict=False):
    name = f'{symbol}.csv.gz' if compressed else f'{symbol}.csv'
    fpath = f'../data/{name}'
    if not exists(fpath):
        fpath = f'qube/tests/data/{name}'
    data = pd.read_csv(fpath, parse_dates=True, header=0, index_col=['time'])
    return {symbol: data} if as_dict else data


def _init_mongo_db_with_market_data(dbname='md_test'):
    d1 = _read_timeseries_data('solusdt_15min', compressed=True)
    d2 = _read_timeseries_data('ethusdt_15min', compressed=True)

    print('Initializing database ...')
    info = pd.DataFrame({
        'BINANCEF:SOLUSDT':  {
            'Start': d1.index[0],
            'End': d1.index[-1],
        },
        'BINANCEF:ETHUSDT':  {
            'Start': d2.index[0],
            'End': d2.index[-1],
        },
    }).T

    info.index.name = 'Symbol'
    mongo = MongoController(dbname)
    mongo.save_data('m1/BINANCEF:SOLUSDT', d1, is_serialize=True)
    mongo.save_data('m1/BINANCEF:ETHUSDT', d2, is_serialize=True)
    mongo.save_data('INFO/BINANCEF', info, is_serialize=True)
    mongo.close()