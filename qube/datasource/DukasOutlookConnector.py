import pandas as pd

from ..configs import Properties
from .DataSource import BasicConnector


class DukasOutlookConnector(BasicConnector):

    def __init__(self, _dir, _cfg, _name):
        super(DukasOutlookConnector, self).__init__(_dir, _cfg, _name)
        self.check_mandatory_props(['dukas_outlook_file'])
        self.dukas_outlook = None

    def load_data(self, series, start=None, end=None, *args, **kwargs):
        if not self.dukas_outlook or kwargs.get("refresh", False):
            self.__read_csv()
        return self.dukas_outlook

    def __read_csv(self):
        outlook_csv = Properties.get_config_path(self.peek_or('dukas_outlook_file'))
        self.dukas_outlook = pd.read_csv(outlook_csv, index_col=0)
