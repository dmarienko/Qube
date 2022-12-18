from typing import Tuple
import pandas as pd

from qube.datasource import DataSource
from qube.datasource.DataSource import IN_MEMORY_DATASOURCE_NAME


class InMemoryDataSource(DataSource):

    def __init__(self, data):
        super().__init__(IN_MEMORY_DATASOURCE_NAME)
        if not isinstance(data, (pd.DataFrame, dict)):
            raise ValueError('data must be either DataFrame or dict')
        self.data = data

    def load_data(self, series=None, start=None, end=None, *args, **kwargs):
        if series is None:
            series = []
        elif isinstance(series, str):
            series = [series]
        series = [i.upper() for i in series]
        if isinstance(self.data, pd.DataFrame):
            return {series[0]: self.data[start:end]}
        elif isinstance(self.data, dict):
            result = {}
            for instr in self.data:
                if instr.upper() in series:
                    result.update({instr.upper(): self.data[instr][start:end]})
            return result

    def series_list(self, pattern=r".*"):
        raise NotImplementedError("series_list() is not implemented for %s" % self.__class__.__name__)

    def get_range(self, symbol: str) -> Tuple:
        """
        Get start / end for given symbol data 
        """
        if isinstance(self.data, pd.DataFrame):
            return self.data.index[0], self.data.index[-1]
        elif isinstance(self.data, dict) and symbol.upper() in self.data:
            data = self.data[symbol.upper()]
            return data.index[0], data.index[-1]
        return None, None

