import re
from collections import defaultdict
from dataclasses import dataclass
from os import makedirs
from os.path import join, expanduser
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from qube.datasource import DataSource

from qube.learn.core.data_utils import series_period_as_str
from qube.quantitative.tools import drop_duplicated_indexes, ohlc_resample
from qube.quantitative.tools import infer_series_frequency
from qube.utils.nb_functions import z_ld, z_save, z_ls
from qube.utils.ui_utils import green, red, blue

# here is market data database (we will keep market data separately)
_DEFAULT_MARKET_DATA_DB = 'md'


def _md_ld(path):
    return z_ld(path, dbname=_DEFAULT_MARKET_DATA_DB)


def _md_ls(path):
    return z_ls(path, dbname=_DEFAULT_MARKET_DATA_DB)


def _md_save(path, data):
    return z_save(path, data, dbname=_DEFAULT_MARKET_DATA_DB)


def __preprocess(q: pd.DataFrame):
    """
    Remove all repeating consequent quotes to unload backtesting a bit
    """
    q0 = drop_duplicated_indexes(q)
    q1 = q0.shift(1)
    return q0[((q0.ask != q1.ask) | (q0.bid != q1.bid))]


def export_quotes_for_simulator(symbol, quotes, preproc=True, compress=True):
    """
    Small helper method exports quotes to csv for simulator 
    """
    if not all(quotes.columns.isin(['ask', 'bid', 'askvol', 'bidvol'])):
        raise ValueError("data is not quotes dataframe !!!")
    d0 = __preprocess(quotes) if preproc else quotes
    d0['time'] = d0.index.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]
    d0['local_time'] = d0['time']
    d0 = d0.rename(columns={'bidvol': 'bid_size', 'askvol': 'ask_size'})
    d0 = d0[['time', 'local_time', 'bid', 'ask', 'bid_size', 'ask_size']]
    pname = f"{symbol}.csv{'.gz' if compress else ''}"
    print(f" > exporting {len(d0)} quotes to {pname} ...", end='')
    d0.to_csv(pname, index=False)
    print(" [OK]")


@dataclass
class MarketData:
    instrument: str
    symbol: str
    exchange: str
    data: pd.DataFrame

    def ohlc(self, timeframe, tz=None, aux_columns_aggregator='sum'):
        return ohlc_resample(self.data, timeframe, resample_tz=tz, non_ohlc_columns_aggregator=aux_columns_aggregator)

    def ohlcs(self, timeframe, tz=None, aux_columns_aggregator='sum'):
        return {self.symbol: ohlc_resample(self.data, timeframe, resample_tz=tz, non_ohlc_columns_aggregator=aux_columns_aggregator)}

    def datas(self, what, **kwargs):
        return self.ticks() if what == 'ticks' else self.ohlcs(what, **kwargs)

    def data(self, what, **kwargs):
        return self.tick() if what == 'ticks' else self.ohlc(what, **kwargs)

    def ticks(self):
        return {self.symbol: self.data}

    def tick(self):
        return self.data

    def export(self, timeframe=None, tz=None, dest_dir=None, columns=None, prefix='', simulator_tf_format=True,
               include_exchange_name=True, aux_columns_aggregator='sum'):
        """
        Export data to csv file
        """
        if self.data is not None and not self.data.empty:
            c_tf = series_period_as_str(self.data) if timeframe is None else timeframe
            d = self.data
            if timeframe is not None:
                d = ohlc_resample(d, timeframe, resample_tz=tz, non_ohlc_columns_aggregator=aux_columns_aggregator)
            try:
                path = expanduser(join(dest_dir if dest_dir is not None else '',
                                       self.exchange.upper() if include_exchange_name else ''))
                makedirs(path, exist_ok=True)

                # 1Min -> M1 etc
                if simulator_tf_format:
                    _t = re.match('(\d+)(\w+)', c_tf)
                    if _t and len(_t.groups()) > 1:
                        c_tf = f"{_t[2][0].upper()}{_t[1]}"

                f_name = f'{path}/{prefix}{self.symbol}_ohlcv_{c_tf}.csv.gz'

                # fix timestamp format
                d.index = d.index.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]
                print(f" >>> Storing {f_name} ... ", end='')
                d = d[columns] if columns is not None else d
                d.to_csv(f_name, compression='gzip')
                print('[OK]')
            except Exception as e:
                print(f"[Error exporting data for {self.symbol}: ", e, ']')
        else:
            print(f" >>> Data is empty for {self.symbol} nothing to export !")


class MarketMultiSymbolData:
    def __init__(self, *tdata):
        self.tickdata: Dict[str, MarketData] = {t.symbol: t for t in tdata}

    def ohlc(self, timeframe, **kwargs):
        return {s: v.ohlc(timeframe, **kwargs) for s, v in self.tickdata.items()}

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            return MarketMultiSymbolData(*[self.tickdata[i] for i in idx])
        return self.tickdata[idx]

    def __add__(self, other):
        return MarketMultiSymbolData(*(list(self.tickdata.values()) + list(other.tickdata.values())))

    def ticks(self, *symbols):
        if symbols:
            return {s: v.tick() for s, v in self.tickdata.items() if s in symbols}
        return {s: v.tick() for s, v in self.tickdata.items()}

    def symbols(self):
        return set(self.tickdata.keys())

    def export(self, timeframe=None, tz=None, dest_dir=None, columns=None, prefix='', simulator_tf_format=True,
               include_exchange_name=True):
        for k, v in self.tickdata.items():
            v.export(timeframe, tz, dest_dir=dest_dir, columns=columns, prefix=prefix,
                     simulator_tf_format=simulator_tf_format, include_exchange_name=include_exchange_name)

    def __repr__(self):
        return '\n'.join([f'{n} ({v.tick().index[0]} / {v.tick().index[-1]} [{len(v.tick())}] records)' for n, v in
                          self.tickdata.items()])


def load_instrument_data(instrument, start, end, timeframe, dbtype, path) -> Optional[MarketData]:
    """
    Try to load insrument's data
    """
    instr, exch, symbol = (lambda x: (x, *x.split(':')))(instrument)
    data = None

    # transform hinted timeframe
    if timeframe is not None:
        _t = re.match('(\d+)(\w+)', timeframe)
        timeframe = f"{_t[2][0].lower()}{_t[1]}" if _t and len(_t.groups()) > 1 else timeframe

    if dbtype == 'mongo':
        _path = f'{timeframe}/{instr}'
        if timeframe is not None and _md_ls(_path):
            data = _md_ld(_path)
        else:
            # try to find optimal timeframe
            last_available = None
            search_next = False
            for pfx in ['ticks', 'm1', 'm5', 'm15', 'h1', 'd', 'w', 'm']:
                _path = f'{pfx}/{instr}'
                
                if _md_ls(_path):
                    last_available = _path
                    
                if pfx == timeframe or not search_next:
                    if last_available is not None: 
                        data = _md_ld(last_available)
                        break
                    else:
                        search_next = False
    elif dbtype=='csv':
        from os.path import join, exists
        def _find_exts(f: str, exts=['csv', 'gz']):
            n = f
            for e in exts:
                n += '.' + e
                if exists(n):
                    return n
            return None
        file = _find_exts(join(path if path else '', f"{exch}/{symbol}_ohlcv_{timeframe.upper()}"))
        if file:
            data = pd.read_csv(file, parse_dates=True, index_col='time')
    else:
        raise ValueError(f"Unupported database '{dbtype}'")

    if data is not None and start is not None:
        data = data[start:] 

    if data is not None and end is not None:
        data = data[:end] 

    if data is None:
        raise ValueError(f"Can't find stored data in '{dbtype}' for {instrument} @ {exch} | {timeframe} in {start} - {end} !")

    return MarketData(instrument, symbol, exch, data)


def load_data(*instrument, start='1900-01-01', end='2200-01-01', timeframe='1Min', dbtype='mongo',
              path=None) -> MarketMultiSymbolData:
    """
    Loads data from database for instruments from list
    """
    in_list = instrument if isinstance(instrument, (tuple, list)) else list(instrument)
    data = []
    for l in in_list:
        try:
            data.append(load_instrument_data(l, start, end, timeframe, dbtype, path))
        except Exception as e:
            print(f" > Issue during loading: {str(e)}")
    return MarketMultiSymbolData(*data)


def write_data(instrument: str, data: pd.DataFrame, prefix: str = 'm1'):
    """
    Overwrite cached data for instrument
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception(f"Data for {instrument} must be pandas DataFrame !")

    if ':' not in instrument:
        raise Exception(f"Incorrect instrument name for {instrument}. Must be in form EXCHANGE:SYMBOL")

    _md_save(f'{prefix}/{instrument}', data)


def ls_data(exchange='.*', n_symbols=6, as_list=False, full_symbol_name=False) -> Union[None, Dict[str, List[str]]]:
    """
    Show all available data
    
    :param exchange: show only data from exchange (can bw used as regex pattern)
    :param n_symbols: number of symbols to display in one row
    :param as_list: if True returns dictionary of symbols for every exchange matches pattern
    :param full_symbol_name: if True adds exchange prefix to symbol name (i.e. 'BITMEX:XBTUSD')
    """
    exn = defaultdict(lambda: defaultdict(list))
    m1 = _md_ls('m1/')
    t0 = _md_ls('ticks/')
    _l_ret = defaultdict(list)
    for d in [*t0, *m1]:
        tp, path = d.split('/')
        exc, symbol = path.split(':')
        if exchange and not re.match(exchange.lower(), exc.lower()):
            continue
        exn[exc][tp].append(symbol)
        _l_ret[exc].append(f'{exc}:{symbol}' if full_symbol_name else symbol)

    if as_list:
        return {k: list(set(v)) for k, v in _l_ret.items()}

    for e, vs in exn.items():
        print(red(f"[{e}]"))
        for t, sl in vs.items():
            print(f'\t{green(t)}:', end='')
            for k, ll in enumerate(np.array_split(sorted(sl), len(sl) // n_symbols + 1)):
                print(('\t' if k == 0 else '\t\t') + blue(','.join(ll)))


def update_info_data(exchange: str):
    """
    Update info about data availability in DB
    """
    print(f">>> Updating info for {exchange} ... ", end='')
    s_info = {}
    for s in tqdm(_md_ls(f'm1/{exchange}:')):
        symb = s.split('/')[-1]
        d = _md_ld(s)
        if d is not None:
            s_info[symb] = {'Start': d.index[0], 'End': d.index[-1]}

    data = pd.DataFrame.from_dict(s_info, orient='index')
    data.index = data.index.rename('Symbol')
    _md_save(f"INFO/{exchange}", data)

    print("[OK]")


def get_data_time_range(instrument: str, ds: Union[DataSource, callable]) -> Tuple[str]:
    """
    Find data ranges for given instrument
    :parameter instrument: instrument spec (exchange:symbol)
    """
    if isinstance(ds, DataSource):
        ranges = ds.get_range(instrument)
        if ranges and ranges[0] is not None:
            data_start_date = str(ranges[0].date() + pd.Timedelta('1d'))
            data_end_date = str(ranges[0].date())
        else:
            raise ValueError(f"Can't find start/stop dates for {instrument}")
    else:
        # - TODO: to be removed in future !
        exchange, symbol = instrument.split(':')

        t_dates = _md_ld(f'INFO/{exchange}')
        if t_dates is None or instrument not in t_dates.index:
            update_info_data(exchange)

        t_dates = _md_ld(f'INFO/{exchange}')
        if instrument not in t_dates.index:
            raise ValueError(f"Can't find information in INFO/{exchange} table for {symbol}")

        s_info = t_dates.loc[instrument]
        data_start_date = str(s_info.Start.date() + pd.Timedelta('1d'))
        data_end_date = str(s_info.End.date())

    return data_start_date, data_end_date


def check_data_consistency(exchange, min_incosistent=10):
    """
    Check if data is consistent for all symbols from given exchange
    
    check_data_consistency('BINANCEF')
    
    :param exchange: exchange name
    :param min_incosistent: minimal number of wrong time deltas (10)
    :return: list of inconsistent symbols
    """
    inconsist = []
    for ex in tqdm(ls_data(exchange, as_list=1, full_symbol_name=1)[exchange]):
        xd = load_data(ex)
        for s in xd.symbols():
            xl = xd.ticks()[s]
            if xl is None:
                print(f" >>> {s} has no data !!!")
                continue
            f = pd.Timedelta(infer_series_frequency(xl[:100]))
            si = pd.Series(xl.index, index=xl.index)
            dsi = si.diff().dropna()
            where_diffs = si[1:][dsi != f]
            if not where_diffs.empty:
                print(f"{s} has gaps at {where_diffs.index}")
                if len(where_diffs) > min_incosistent:
                    print(f"{s} has too much gaps !")
                    inconsist.append(s)

    return inconsist
