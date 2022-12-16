
from typing import List, Union

from qube.datasource.controllers.MongoController import MongoController
from .DataSource import BasicConnector


class MongoConnector(BasicConnector):
    """
    Load series from mongo storage
    """

    def __init__(self, _dir, _cfg, _name):
        super(MongoConnector, self).__init__(_dir, _cfg, _name)
        self.check_mandatory_props(['database', 'path', 'exchange'])
        self._dbase = _cfg['database'] 
        self._path = _cfg['path'] 
        self._exchange = _cfg['exchange'] 
        self._drop_exch_name = _cfg.get('drop_exchange_name', True)
        self._mctrl = MongoController(self._dbase)

    def _process_name(self, name):
        return name if self._drop_exch_name else f'{self._exchange}:{name}'

    def load_data(self, series: Union[List[str], str], start, end=None, *args, **kwargs):
        data = {}
        for s in series:
            key = f'{self._path}/{self._exchange}:{s}' if self._path is not None else s
            ds = self._mctrl.load_data(key)['data']
            if start is not None:
                ds = ds[start:]
            if end is not None:
                ds = ds[:end]
            data[self._process_name(s)] = ds 
        return data

    def close(self):
        self._mctrl.close()

    def series_list(self, pattern=r".*"):
        recs = self._mctrl.ls_data(f'{self._path}/{self._exchange}:{pattern}' if self._path is not None else pattern)
        pat = f'/{self._exchange}:' if self._drop_exch_name else '/' 
        return [ r.split(pat)[1] for r in recs if r is not None ]
