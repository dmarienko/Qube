import datetime
import socket
from collections import OrderedDict
from typing import Tuple

import numpy as np
import pandas as pd

from qube.utils.utils import is_localhost

from os.path import join, isfile, abspath, isabs
from .DataSource import BasicConnector
from qube.utils.DateUtils import DateUtils
from qube.datasource.controllers.KdbServerController import KdbServerController
from qube.datasource.controllers.kdb_utils import decode_instrument, encode_instrument


class KdbConnector(BasicConnector):
    # following functions must be declared at kdb server
    _API_FUNCTIONS = '`i_fetch`i_timeframe`i_series'

    _TIME_FRAMES = {
        '1w': 7 * 86400, 'weekly': 7 * 86400,
        '1d': 86400, 'daily': 86400,
        '1h': 3600,
        '30m': 1800,
        '15m': 900,
        '10m': 600,
        '5m': 300,
        '2m': 120,
        '1m': 60,
        'tick': 0, 'ticks': 0
    }

    _WEEKS_SPLIT = 24

    def __init__(self, _dir, _cfg, _name):
        super(KdbConnector, self).__init__(_dir, _cfg, _name)
        self.check_mandatory_props(['host'])
        self.is_shutdown = False

        # load parameters
        init_script = ''
        init = self.peek_or('init')
        if init:  # if init specified let's get abs path to it
            if not isabs(init):
                init_script = abspath(join(self.cwd, self.peek_or('init'))).replace("\\", "/")
            else:
                init_script = init
            # check if init script is presented
            if is_localhost(self.peek_or('host')) and not isfile(
                    init_script):  # if init specified but file is missing raise an exception
                raise ValueError("File not found at path defined in 'init' property: '%s'" % init_script)

        # get kdb port
        if self.peek_or('port'):
            # initiate kdb with single port
            self.__create_kdb_controller(self.peek_or('port'), init_script)
        else:
            # initiate kdb with range ports
            if not self.peek_or('ports_from') or not self.peek_or('ports_to'):
                raise ValueError(
                    "Either 'port' or pair 'ports_from', 'ports_to' must be specified in a config %s" % self.get_name())
            retries = 0
            MAX_UNSUCCESSFUL_RETRIES = 50
            while retries < MAX_UNSUCCESSFUL_RETRIES:
                kdb_port = self._get_free_port(self.peek_or('ports_from'), self.peek_or('ports_to'),
                                               self.peek_or('host'))
                try:
                    self.__create_kdb_controller(kdb_port, init_script)

                    if retries > MAX_UNSUCCESSFUL_RETRIES / 2:
                        self.warn('Created KdbServerController object for %s on %d retry' % (self.get_name(), retries))

                # KdbServerController constructor throws ConnectionError if client tries connect to occupied port
                except ConnectionError:
                    retries = retries + 1
                else:
                    break

            if not self._controller:
                raise ValueError('Could not connect to kdb after %s attempts' % MAX_UNSUCCESSFUL_RETRIES)

    def __create_kdb_controller(self, port, init_script=None):
        self._controller = KdbServerController(self.peek_or('host'), port,
                                               self.peek_or('user'), self.peek_or('pass', self.peek_or('password')),
                                               self.peek_float('timeout'),
                                               self.peek_or('db_path'), init_script)
        # check if database contains all needed functions defined
        if not self._controller.exec('all %s in system("f")' % self._API_FUNCTIONS):
            raise ConnectionError(
                "Not all functions from the list (%s) are declared at KDB server" % self._API_FUNCTIONS)

    def load_data(self, series, start, end=None, *args, **kwargs):
        if self.is_shutdown:
            raise ConnectionError('KDB Connection has already shut down')

        # which timeframe (in seconds)
        time_frame = kwargs.get('nbars', kwargs.get('bars', kwargs.get('timeframe', None)))

        if not start:
            raise ValueError('\'start\' is a required parameter')

        # check if shortcut string passed
        if isinstance(time_frame, str):
            time_frame = self._TIME_FRAMES.get(time_frame, None)

        # use database storage timeframe
        if time_frame is None:
            time_frame = self._controller.exec('i_timeframe[]', skip_connection_check=True)[0]

        self.info('Timeframe to loading %d sec' % time_frame)

        # if empty series list specified try to load all instruments (?)
        if not series:
            series = self.series_list()

        # resulting data dictionary
        data = OrderedDict()

        # check and transform date ranges
        s0 = self.__fmt_date(time_frame, start)
        e0 = self.__fmt_date(time_frame, end if end else DateUtils.get_now())

        k_idx = None
        for ser_name in series:
            self.info('loading %s [%s:%s]... ' % (ser_name, s0, e0))
            if not self.peek_or('load_by_days') and time_frame >= self._TIME_FRAMES['1d']:
                d_s0 = DateUtils.get_datetime(s0)
                d_e0 = DateUtils.get_datetime(e0)
                if d_e0 - d_s0 > datetime.timedelta(weeks=self._WEEKS_SPLIT):
                    # split range on 24 weeks
                    l_series = pd.DataFrame()
                    start_chunk_date = d_s0
                    while start_chunk_date < d_e0:
                        end_chunk_date = start_chunk_date + datetime.timedelta(weeks=self._WEEKS_SPLIT)
                        if end_chunk_date > d_e0:
                            end_chunk_date = d_e0
                        x_data, k_idx = self.__load_data(ser_name, time_frame,
                                                         DateUtils.format_kdb_datetime_usec(start_chunk_date),
                                                         DateUtils.format_kdb_datetime_usec(end_chunk_date), k_idx)
                        l_series = l_series.append(x_data)
                        start_chunk_date += datetime.timedelta(weeks=self._WEEKS_SPLIT, microseconds=1)
                    data[ser_name.upper()] = l_series
                else:
                    data[ser_name.upper()], k_idx = self.__load_data(ser_name, time_frame, s0, e0, k_idx)

            else:
                # for timeframe lower than 1d we will split on smaller date ranges to avoid
                # kdb memory limitation (for free 32 bit version)
                d_s0 = DateUtils.get_datetime(s0)
                d_e0 = DateUtils.get_datetime(e0)
                if d_e0.day != d_s0.day or d_e0 - d_s0 > datetime.timedelta(days=1):
                    # start loading every day separately
                    l_series = pd.DataFrame()
                    for x in DateUtils.daterange(d_s0, d_e0):
                        self.info("Fetching data for {} ...".format(x))
                        x_s0 = max(datetime.datetime.combine(x, datetime.time(0, 0, 0)), d_s0)
                        x_e0 = min(datetime.datetime.combine(x, datetime.time(23, 59, 59, 999999)), d_e0)
                        x_data, k_idx = self.__load_data(ser_name, time_frame,
                                                         DateUtils.format_kdb_datetime_usec(x_s0),
                                                         DateUtils.format_kdb_datetime_usec(x_e0),
                                                         k_idx)
                        # and join everything
                        l_series = l_series.append(x_data)
                    data[ser_name.upper()] = l_series
                else:
                    data[ser_name.upper()], k_idx = self.__load_data(ser_name, time_frame, s0, e0, k_idx)
        return data

    def __load_data(self, ser_name, time_frame, s0, e0, k_idx):
        """
        Load bunch of data from kdb and also do indexing if needs
        :param ser_name: name of instrument to load
        :param time_frame: timeframe in seconds (0 for ticks)
        :param s0: start time
        :param e0: end time
        :param k_idx: indexing column name (if none it tries to find)
        :return: loaded series and indexing column
        """

        date_param = ''
        if self.peek_or('load_by_days'):
            date_param = ';%s' % DateUtils.get_as_string(DateUtils.get_datetime(s0), DateUtils.KDB_DATE_FORMAT)

        s_data = self._controller.exec(
            'i_fetch[`%s;%d;%s;%s%s]' % (encode_instrument(ser_name.upper()), time_frame, s0, e0, date_param),
            skip_connection_check=True)

        # try to find indexing column
        if not k_idx:
            k_idx = self.__find_indexing_column(s_data)
            if not k_idx:
                self.warn("Can't find indexing column. It should be named as 'time', 'date' or 'datetime'")

        # use index column if presented
        if k_idx:
            s_data.set_index(k_idx, inplace=True)

        # converting object columns to string (qpython automatically converts kdb string to object by some reason)
        datacols = s_data.dtypes.to_dict()
        for datacol in datacols:
            if datacols[datacol] == np.dtype('O') and len(s_data[datacol]) > 0 and isinstance(s_data[datacol].iat[0],
                                                                                              bytes):
                s_data[datacol] = s_data[datacol].str.decode('utf-8')
        return s_data, k_idx

    def __find_indexing_column(self, dframe):
        cols = dframe.columns.map(lambda x: x.lower())
        idx_c = dframe.columns[cols == 'time'].any()
        if not idx_c:
            idx_c = dframe.columns[cols == 'date'].any()
        if not idx_c:
            idx_c = dframe.columns[cols == 'datetime'].any()
        return idx_c

    def __fmt_date(self, tframe, time):
        if isinstance(time, str): time = DateUtils.get_datetime(time)

        # find start of the week for weekly dateframe
        if tframe == self._TIME_FRAMES['1w']:
            while not time.weekday() == 0:
                time = time - datetime.timedelta(1)

        if tframe >= self._TIME_FRAMES['1d']:
            s0 = DateUtils.format_kdb_date(time)
        else:
            s0 = DateUtils.format_kdb_datetime_usec(time)
        return s0

    def _get_free_port(self, ports_from: int, ports_to: int, host: str):
        kdb_port = None
        for port in range(ports_from, ports_to + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, port))
            if result != 0:
                kdb_port = port
                break

        if not kdb_port:
            raise ValueError(
                'It looks like all ports are occupied at moment or some kdb connection leak issue has occurred for %s.' % self.get_name())

        return kdb_port

    def get_range(self, symbol: str) -> Tuple:
        """
        Get start / end for given symbol data 
        """
        try:
            ranges = self._controller.exec(f'select start:first time, stop:last time from {symbol.upper()}')
            return (ranges.iloc[0].start, ranges.iloc[0].stop)
        except:
            return (None, None)

    def series_list(self, pattern=r".*"):
        if self.is_shutdown:
            raise ConnectionError('KDB Connection has already shut down')
        """
        Returns list of available series for this connector
        It's possible to use wildcard for series list selection.
        """
        as_byte_lists = self._controller.exec('i_series[]', skip_connection_check=True)
        return [decode_instrument(as_bytes.decode()) for as_bytes in as_byte_lists]

    def reload(self):
        self._controller.reload()

    def close(self):
        if not self.is_shutdown:
            if hasattr(self, '_controller'):
                self._controller.shutdown()
            self.is_shutdown = True

    def __del__(self):
        self.close()
