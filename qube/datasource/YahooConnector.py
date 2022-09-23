from collections import OrderedDict

import numpy as np
from pandas import pandas as pd

from qube.utils.DateUtils import DateUtils
from .DataSource import BasicConnector


class YahooConnector(BasicConnector):
    """
    Load series from Yahoo finance.
    """

    def __init__(self, _dir, _cfg, _name):
        super(YahooConnector, self).__init__(_dir, _cfg, _name)
        self.check_mandatory_props(['adjusted'])

    def load_data(self, series, start, end=None, *args, **kwargs):

        is_adjusted = self.peek_bool('adjusted')

        self.info(
            "Getting yahoo {0} data from {1} to {2} for list of {3} symbols {4}".format(
                'adjusted' if is_adjusted else 'not_adjusted',
                DateUtils.get_as_string(start),
                'end' if end is None else end,
                len(series), series if len(series) <= 5 else "['" + series[0] + "' ... '" + series[-1] + "']"))

        r = OrderedDict()

        from pandas_datareader.data import DataReader

        for ticker in series:
            try:
                read_result = DataReader(ticker, "yahoo", DateUtils.get_datetime(start), DateUtils.get_datetime(end))
                if is_adjusted:
                    mult_ser = read_result['Adj Close'] / read_result['Close']
                    ones = np.ones(len(mult_ser))
                    mult_matrix = pd.DataFrame(
                        {'Open': mult_ser, 'High': mult_ser, 'Low': mult_ser, 'Close': ones,
                         'Adj Close': ones, 'Volume': ones})
                    adjusted_matrix_all = read_result * mult_matrix
                    adjusted_matrix = adjusted_matrix_all[['Open', 'High', 'Low', 'Adj Close', 'Volume']]. \
                        rename(columns={"Adj Close": "Close"})
                    r[ticker.upper()] = adjusted_matrix
                else:
                    r[ticker.upper()] = read_result[['Open', 'High', 'Low', 'Close', 'Volume']]
            except:
                self.warn("Can't get data for %s" % ticker)

        return r
