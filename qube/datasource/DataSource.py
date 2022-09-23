import importlib
import logging
import re
import sys
import time
from os.path import dirname

import pandas as pd

from qube.utils import QubeLogger
from ..configs import Properties
from . import _CONNECTORS_LOOKUP

IN_MEMORY_DATASOURCE_NAME = 'in_memory_datasource'
MULTI_EXCHANGE_DATASOURCE_NAME = 'multi_exchange_datasource'


class BasicConnector:
    """
    Basic class for any data connector. Connector should override
    load_series method of base class.
    Connector must be defined in _CONNECTORS_LOOKUP table at __init__.py
    """

    def __init__(self, _dir, _cfg, _name):
        self.config = _cfg
        self.cwd = _dir
        self.__type = _cfg['type']
        self.__name = _name
        self.__logger = QubeLogger.getLogger('.'.join(['ira.datasource', _name]))

    def check_mandatory_props(self, props):
        for p in props:
            if not p in self.config:
                raise ValueError("Mandatory property '%s' is not found !" % p)

    def peek_or(self, key, defval=None):
        return self.config.get(key, defval)

    def peek_bool(self, key, defval=False):
        val = self.peek_or(key)
        if val is None:
            return defval
        elif isinstance(val, bool):
            return val
        elif isinstance(val, str):
            return 'true' == val.lower()
        elif isinstance(val, int):
            return val == 1
        else:
            raise ValueError('Value for property %s has unexpected type %s' % (key, type(val).__name__))

    def peek_float(self, key, defval=None):
        val = self.peek_or(key)
        if val is None:
            return defval
        elif isinstance(val, (float, int)):
            return val
        elif isinstance(val, str):
            return float(val)
        else:
            raise ValueError('Value for property %s has unexpected type %s' % (key, type(val).__name__))

    def warn(self, msg):
        self.__logger.warning(msg)

    def info(self, msg):
        self.__logger.info(msg)

    def error(self, msg):
        self.__logger.error(msg)

    def get_name(self):
        return self.__name

    def get_type(self):
        return self.__type

    def load_data(self, series, start, end, *args, **kwargs):
        raise NotImplementedError("load_data() is not implemented for %s" % self.__class__.__name__)

    def series_list(self, pattern=r".*"):
        raise NotImplementedError("series_list() is not implemented for %s" % self.__class__.__name__)

    def reload(self):
        pass

    def close(self):
        pass


class DataSource:
    """
    Data source class is decorator for BasicConnector.
    """
    VERBOSE_MODE = 'verbose'

    def __init__(self, data_source_name, config_path=None, overload_props=None, **kwargs):
        self.__logger = QubeLogger.getLogger('ira.datasource')
        self.__logger.setLevel(logging.INFO) if kwargs.get(self.VERBOSE_MODE, False) else self.__logger.setLevel(
            logging.WARN)

        self.data_source_name = data_source_name
        self.config_path = config_path
        self.__connector = None
        self._cfg = {}

        if self.is_derived_datasource():
            return

        # base directory for this datasource. All relative paths starts from here
        self._wrk_dir = None
        config = self.__load_ds_cfg(config_path)
        if data_source_name in config:
            self._info('Loading %s ...' % data_source_name)
            self._cfg = config[data_source_name]

            if overload_props is not None:
                self._cfg.update(overload_props)
            self.__lookup_ds_connector(data_source_name)
        else:
            raise ValueError(
                "Configuration for '%s' datasource not found. Check your datasources config." % data_source_name)

    def __lookup_ds_connector(self, data_source_name):

        if not 'type' in self._cfg:
            raise ValueError("Check configuration for '%s' : can't found 'type'")

        _type = self._cfg['type'].lower()
        if not _type in _CONNECTORS_LOOKUP:
            raise ValueError("Connector '%s' not found" % _type)

        c_name = _CONNECTORS_LOOKUP[_type]
        # clazz = getattr(importlib.import_module(__name__), c_name)
        self._info('%s, %s' % (sys.modules[__name__], sys.modules[__name__].__name__))
        module_name = sys.modules[__name__].__name__.split('.')[:-1]
        if isinstance(module_name, list):
            module_name = '.'.join(module_name)
        full_cl_name = '.'.join([module_name, c_name])
        self._info('trying import %s from %s' % (c_name, full_cl_name))
        basic_class = getattr(importlib.import_module(full_cl_name), c_name)

        # instantite connector class
        self.__connector = basic_class(self._wrk_dir, self._cfg, data_source_name)

    def __load_ds_cfg(self, config_path):
        self._wrk_dir = dirname(Properties.get_formatted_path(config_path)) if config_path else dirname(
            Properties.get_config_path('datasource.json'))
        return Properties.get_properties(config_path, is_refresh=True) if config_path else Properties.get_config_properties(
            'datasource.json', is_refresh=True)

    def _warn(self, msg):
        self.__logger.warning(msg)

    def _info(self, msg):
        self.__logger.info(msg)

    def get_name(self):
        return self.__connector.get_name()  # or self.data_source_name (actually same values)

    def get_type(self):
        return self.__connector.get_type()

    def get_prop(self, key, defval=None):
        return self._cfg[key] if key in self._cfg else defval

    def get_properties(self):
        return self._cfg

    def load_data(self, series=None, start=None, end=None, *args, **kwargs):
        if series is None:
            series = []
        elif not isinstance(series, list):
            series = [series]
        series = [symb.upper() for symb in series]
        data_shift_minutes = self._cfg.get('delayed_data_shift_minutes', 0)
        if data_shift_minutes:
            start = pd.to_datetime(start) - pd.Timedelta(minutes=data_shift_minutes) if start else None
            end = pd.to_datetime(end) - pd.Timedelta(minutes=data_shift_minutes) if end else None

        data = self.__connector.load_data(series, start, end, *args, **kwargs)

        # lower case all columns
        lowercase_columns = lambda df: df.rename(
            columns={col_name: col_name.lower() if isinstance(col_name, str) else col_name for col_name in df.columns},
            inplace=True)
        if isinstance(data, dict):
            [lowercase_columns(df) for (_, df) in data.items()]
        elif isinstance(data, pd.DataFrame):
            lowercase_columns(data)

        # index to datetime if it's str and shift data if needed
        for i, df in data.items():
            if len(df.index) > 0 and isinstance(df.index[0], str):
                is_string_date = False
                try:  # let's check if index isn't a date like finviz, so we don't need to to cast index strs to datetimes
                    pd.to_datetime(df.index[0])
                    is_string_date = True
                except:
                    pass
                if is_string_date:
                    df.index = pd.to_datetime(df.index, infer_datetime_format=True)

            # shift data
            if data_shift_minutes:
                data[i] = df.shift(data_shift_minutes, 'Min')

        return data

    def series_list(self, pattern=r".*"):
        series = self.__connector.series_list(pattern)
        rc = re.compile(pattern)
        return [i.upper() for i in series if rc.match(i.upper())]

    def is_derived_datasource(self):
        return self.data_source_name == IN_MEMORY_DATASOURCE_NAME or self.data_source_name == MULTI_EXCHANGE_DATASOURCE_NAME

    def reload(self):
        self.__connector.reload()

    def close(self, sleep_period=0):
        # connector may be empty for IN_MEMORY_DATASOURCE_NAME for example
        if self.__connector:
            self.__connector.close()
        if sleep_period > 0:
            time.sleep(sleep_period)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_t):
        self.close()
