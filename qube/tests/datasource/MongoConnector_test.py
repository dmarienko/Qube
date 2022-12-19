import unittest
import mongomock
from mongomock.gridfs import enable_gridfs_integration

enable_gridfs_integration()

from qube.datasource.controllers.MongoController import MongoController
from qube.datasource.DataSource import DataSource
from qube.tests.utils_for_tests import _read_timeseries_data
from qube.utils.nb_functions import z_ls, z_ld


class MongoConnectorTest(unittest.TestCase):
    DS_CFG_PATH = 'qube/tests/ds_test_cfg.json'

    @mongomock.patch()
    def setUp(self):
        self.d1 = _read_timeseries_data('solusdt_15min', compressed=True)
        self.d2 = _read_timeseries_data('ethusdt_15min', compressed=True)

    def _initalize(self):
        print('Initializing database ...')
        self.mongo = MongoController('md_test')
        self.mongo.save_data('m1/BINANCEF:SOLUSDT', self.d1, is_serialize=True)
        self.mongo.save_data('m1/BINANCEF:ETHUSDT', self.d2, is_serialize=True)
        self.mongo.close()

    @mongomock.patch()
    def test_data_loading(self):
        self._initalize()

        with DataSource("test::mongo-market-data-1min", self.DS_CFG_PATH) as ds:
            data = ds.load_data(['ETHUSDT', 'SOLUSDT'], '2021-01-01 00:00', '2021-01-01 01:00')
            symbs = ds.series_list()
        
        # print(symbs)
        # print(data)
        self.assertListEqual(sorted(['ETHUSDT', 'SOLUSDT']), sorted(symbs))
        self.assertEquals(len(data['ETHUSDT']), len(data['SOLUSDT']))


from pytest import main
if __name__ == '__main__':
    main()