import glob
import os.path as op
from collections import OrderedDict

import pandas as p

from qube.utils.DateUtils import DateUtils
from .DataSource import BasicConnector


class CsvConnector(BasicConnector):
    """
    Load series from csv file or list of csv files.
    """

    def __init__(self, _dir, _cfg, _name):
        super(CsvConnector, self).__init__(_dir, _cfg, _name)

        # check if there are all properties in config
        self.check_mandatory_props(['file'])

        # where to read data from
        self.path_to_csv = op.join(self.cwd, self.peek_or('file'))

        # csv parsing parameters
        self.is_header = self.peek_or('header')
        self.delimiter = self.peek_or('delimiter', ',')
        self.index_col = self.peek_or('index')
        self.parse_dates = self.peek_or('parse_dates', True)  # Accepts not only booleans but int and [] of ints
        self.need_sort = self.peek_bool('need_sort')

    def load_data(self, series, start=None, end=None, *args, **kwargs):
        self.info('reading data from %s' % self.path_to_csv)

        # resulting dictionary
        r = OrderedDict()

        # when plain file just read data from it
        if op.isfile(self.path_to_csv):
            s_name = op.splitext(op.basename(self.path_to_csv))[0].lower()
            r[s_name.upper()] = self.__select_time_range(self.path_to_csv, start, end)

        # tries to load all *.csv files from dest. folder
        elif op.isdir(self.path_to_csv):
            f_list = glob.glob(op.join(self.path_to_csv, '*.csv'))
            self.info("Found %d series in '%s'" % (len(f_list), self.path_to_csv))
            file_dict = {}
            for f in f_list:
                file_name = op.splitext(op.basename(f))[0].lower()
                file_dict[file_name] = f

            for s_name in [x.lower() for x in series]:
                if s_name in file_dict.keys():
                    r[s_name.upper()] = self.__select_time_range(file_dict[s_name], start, end)

        return r

    def __select_time_range(self, f, start, end):
        all_rows = p.read_csv(f, delimiter=self.delimiter, index_col=self.index_col, header=self.is_header,
                              parse_dates=self.parse_dates)
        if self.need_sort:
            all_rows = all_rows.sort_index()

        # todo adjust timezone to every index value
        return all_rows.loc[DateUtils.get_datetime(start):DateUtils.get_datetime(end)]

    """
    Returns list of available series for this connector
    It's possible to use wildcard for series list selection.
    """

    def series_list(self, pattern=r".*"):
        path_to_csv = op.join(self.cwd, self.peek_or('file'))

        f_list = []
        if op.isfile(path_to_csv):
            f_list = [op.splitext(op.basename(path_to_csv))[0]]
        elif op.isdir(path_to_csv):
            f_list = [op.splitext(op.basename(s))[0] for s in glob.glob(op.join(path_to_csv, '*.csv'))]

        # select only matching to pattern ones
        return [x for x in f_list]
