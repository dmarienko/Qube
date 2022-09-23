import pandas as pd

from qube.datasource import DataSource
from qube.datasource.DataSource import MULTI_EXCHANGE_DATASOURCE_NAME

EXCHANGE_SEPARATOR = "_"


class MultiExchangeDataSource(DataSource):

    def __init__(self, ds_dict: dict):
        super().__init__(MULTI_EXCHANGE_DATASOURCE_NAME)
        if not isinstance(ds_dict, dict):
            raise ValueError('ds_dict must be Dict[str:DataSource]')
        for ds in ds_dict:
            if isinstance(ds_dict[ds], str):
                ds_dict[ds] = DataSource(ds_dict[ds])
        self.ds_dict = ds_dict  # Dict[str:DataSource] or Dict[str:Dict[str : pd.DataFrame] or Dict[str:str]]

    def load_data(self, series=None, start=None, end=None, *args, **kwargs):
        if series is None:
            series = []
        elif isinstance(series, str):
            series = [series]
        series = [i.upper() for i in series]

        result = {}
        for ex_i in series:
            for exchange_id in self.ds_dict:
                if ex_i.startswith(exchange_id.upper() + EXCHANGE_SEPARATOR):
                    pure_i = ex_i[len(exchange_id) + len(EXCHANGE_SEPARATOR):].upper()

                    ds = self.ds_dict[exchange_id]
                    if isinstance(ds, DataSource):
                        res = ds.load_data(pure_i, start, end, *args, **kwargs)
                    elif isinstance(ds, dict):
                        res = {pure_i: ds[pure_i][start:end]}
                    else:
                        raise ValueError(
                            'ds_dict must be either Dict[str:DataSource] or Dict[str:Dict[str : pd.DataFrame]]')

                    result.update({ex_i: res.get(pure_i, pd.DataFrame())})

                    break
        return result

    def series_list(self, pattern=r".*"):
        raise NotImplementedError("series_list() is not implemented for %s" % self.__class__.__name__)

    def close(self):
        [ds.close() for ds in self.ds_dict.values() if isinstance(ds, DataSource)]
