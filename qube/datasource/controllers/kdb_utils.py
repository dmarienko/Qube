import os

import pandas as pd

from qube.datasource import DataSource
from qube.datasource.controllers.KdbServerController import KdbServerController


def encode_instrument(instr: str):
    instr = 'UNDER' + instr if instr[0] == '_' else instr
    instr = 'DOT_' + instr[1:] if instr[0] == '.' else instr
    instr = instr.replace('.', '_DOT_')
    instr = instr.replace('-', '_DASH_')

    return instr


def decode_instrument(instr: str):
    instr = instr.replace('UNDER_', '_')
    instr = '.' + instr[4:] if instr[:4] == 'DOT_' else instr
    instr = instr.replace('_DOT_', '.')
    instr = instr.replace('_DASH_', '-')
    return instr


def kdb_write_ohlcv(data: pd.DataFrame, symbol: str, port=5579, ds_name: str = None, ds: DataSource = None,
                    kdbc: KdbServerController = None):
    _q_write_func = """writeData:{[dbpath; dbname; baseId; data]
    writepath:.Q.par[hsym dbpath; baseId; dbname];
    .[upsert;(writepath; data);{ -1 "ERROR failed to save table: ", x }];
    }"""

    _q_last_date_func = 'select last time from `%s'
    if not ds and not kdbc:
        ds = DataSource(ds_name, overload_props={'port': port})
        kdbc = KdbServerController.create(ds.get_properties())

    series_list = ds.series_list()
    encoded_symbol = encode_instrument(symbol)

    if encoded_symbol in series_list:
        last_time = kdbc.exec(_q_last_date_func % encoded_symbol)
        kdb_last_time = pd.to_datetime(last_time.values[0][0])
        data = data[kdb_last_time:]

    database_path = os.path.abspath(ds.get_properties()['db_path']).replace("\\", "/")

    if len(data) > 1:
        as_str = data.to_csv(sep=',', header=False, date_format="%Y.%m.%dD%H:%M:%S")
        data = '(' + ';'.join(['"%s"' % s for s in as_str.split('\n') if s]) + ')'
        kdbc.exec(
            'parsed:flip `time`open`high`low`close`volume!{$[(count x)=1; enlist x; x]} each (\"%s\";\",\") 0: ' % 'PFFFFF' + data)
        kdbc.exec(_q_write_func)
        kdbc.exec('writeData[`$"%s"; `%s; `%s; parsed]' % (database_path, encoded_symbol, "."))
        kdbc.exec('delete parsed from `.')


def kdb_write_data_by_day(data: pd.DataFrame, symbol: str, port=5579,
                          type='ohlcv', ds_name: str = None, ds: DataSource = None, kdbc: KdbServerController = None):
    _q_write_func = """
                        writeData:{[dbpath; dbname; datePart; data] writepath:.Q.par[hsym dbpath; datePart; `$(dbname,"/")];
                        .[upsert;(writepath; data);{ L"ERROR failed to save table: ", x }];  }
                        """
    _q_unset_parsed = 'delete parsed from `.'
    _q_write_table = 'writeData[`$"."; "%s"; `%s; parsed]'

    _write_block_size = 1000000

    if type == 'ohlcv':
        _fields = '`time`open`high`low`close`volume`id'
        _format = 'PFFFFFJ'
        _q_list_to_table = 'parsed:flip %s!((); (); (); (); (); (); ())' % _fields
        _df_fields = ['open', 'high', 'low', 'close', 'volume', 'id']
    elif type == 'trades':
        _fields = '`time`localtime`price`size`takerSide`tradeId`id'
        _format = 'PPFFX*J'
        _q_list_to_table = 'parsed:flip %s!((); (); (); (); (); (); ())' % _fields
        _df_fields = ['localtime', 'price', 'size', 'takerSide', 'tradeId', 'id']
    elif type == 'quotes':
        _fields = '`time`localtime`bid`ask`bidSize`askSize`id'
        _format = 'PPFFFFJ'
        _q_list_to_table = 'parsed:flip %s!((); (); (); (); (); (); ())' % _fields
        _df_fields = ['localtime', 'bid', 'ask', 'bidSize', 'askSize', 'id']
    else:
        raise ValueError("Type param must be ohlcv, trades or quotes!")

    data = data[_df_fields]

    _q_last_date_func = 'select last time from `%s'
    if not ds and not kdbc:
        ds = DataSource(ds_name, overload_props={'port': port})
        kdbc = KdbServerController.create(ds.get_properties())

    series_list = ds.series_list()
    encoded_symbol = encode_instrument(symbol)
    dirs = kdbc.exec("dirs:key `$(\":\",system[\"cd\"]);\ndirs@:where dirs like\"[0-9]*\";\ndirs")

    kdbc.exec(_q_write_func)

    if len(dirs):
        kdb_dirs = [d.decode("utf-8") for d in dirs]
    else:
        kdb_dirs = []

    def _write_empty_data(symbols, kdb_day):
        for symbol in symbols:
            kdbc.exec(_q_list_to_table)
            kdbc.exec(_q_write_table % (encode_instrument(symbol), kdb_day))

        kdbc.exec(_q_unset_parsed)

    if encoded_symbol in series_list:
        last_time = kdbc.exec(_q_last_date_func % encoded_symbol)
        kdb_last_time = pd.to_datetime(last_time.values[0][0])
        data = data[kdb_last_time:]

    if len(data) > 1:
        for group in data.groupby(data.index.date):
            day_data = group[1]
            kdb_day = day_data.index[0].strftime("%Y.%m.%d")

            if not series_list and not kdb_dirs:  # if first write in db
                _write_empty_data([symbol], kdb_day)
                series_list.append(symbol)
                kdb_dirs.append(kdb_day)
            if kdb_day not in kdb_dirs:
                write_instruments = series_list if series_list else [symbol]
                _write_empty_data(write_instruments, kdb_day)
                kdb_dirs.append(kdb_day)
            if symbol not in series_list:
                write_dirs = kdb_dirs if kdb_dirs else [kdb_day]
                for kdb_dir in write_dirs:
                    _write_empty_data([symbol], kdb_dir)
                series_list.append(symbol)

            start = 0
            end = min(len(day_data), _write_block_size)

            while start < end:
                block_data = day_data.iloc[start:end]
                if len(block_data) <= 1:
                    # not possible to save one element with this method
                    break
                start = end
                end = min(end + _write_block_size, len(day_data))

                as_str = block_data.to_csv(sep=',', header=False, date_format="%Y.%m.%dD%H:%M:%S.%f")
                block_data = '(' + ';'.join(['"%s"' % s for s in as_str.split('\n') if s]) + ')'
                kdbc.exec('parsed:flip %s!{$[(count x)=1; enlist x; x]} each (\"%s\";\",\") 0: ' % (
                _fields, _format) + block_data)
                kdbc.exec('writeData[`$"."; "%s"; `%s; parsed]' % (encoded_symbol, kdb_day))
                kdbc.exec('delete parsed from `.')
