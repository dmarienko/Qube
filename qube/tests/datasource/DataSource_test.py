from datetime import datetime
from unittest import TestCase

import pandas as pd

from qube import datasource
from qube.datasource.InMemoryDataSource import InMemoryDataSource
from qube.utils.DateUtils import DateUtils


class TestDataSource(TestCase):
    DS_CFG_PATH = 'qube/tests/ds_test_cfg.json'

    def setUp(self):
        pass

    def testDataSourceInitCustomConfig(self):
        ds = datasource.DataSource('test::test_local', self.DS_CFG_PATH)
        self.assertEqual('test::test_local', ds.get_name())

    def testDataSourceNotExisted(self):
        self.assertRaises(ValueError, lambda: datasource.DataSource('test::not_exists', self.DS_CFG_PATH))

    def testCsvConnector_single_file(self):
        ds = datasource.DataSource('test::csv_single', self.DS_CFG_PATH)
        self.assertEqual('test::csv_single', ds.get_name())
        self.assertEqual('csv', ds.get_type())
        sers = ds.load_data(['test'])
        self.__assert_series(sers['TEST'], '2016-01-01', 12.0, '2016-01-07', 18.60)
        print(sers['TEST'])

    def testCsvConnector_single_file_range(self):
        ds = datasource.DataSource('test::csv_single', self.DS_CFG_PATH)
        self.assertEqual('test::csv_single', ds.get_name())
        self.assertEqual('csv', ds.get_type())
        sers = ds.load_data(['TEST'], start='2016-01-03 00:00:00', end='2016-01-05')
        self.__assert_series(sers['TEST'], '2016-01-03', 14.20, '2016-01-05', 16.40)

        sers2 = ds.load_data(['test'], start=datetime(2016, 1, 3), end='2016-01-05')
        self.__assert_series(sers2['TEST'], '2016-01-03', 14.20, '2016-01-05', 16.40)

    def testCsvConnector_single_file_range_dt(self):
        ds = datasource.DataSource('test::csv_single', self.DS_CFG_PATH)
        self.assertEqual('test::csv_single', ds.get_name())
        self.assertEqual('csv', ds.get_type())
        sers = ds.load_data(['TEST'], start=datetime(2016, 1, 3), end=datetime(2016, 1, 5))
        self.__assert_series(sers['TEST'], '2016-01-03', 14.20, '2016-01-05', 16.40)

    def testCsvConnector_multi_file(self):
        ds = datasource.DataSource('test::csv_dir', self.DS_CFG_PATH)
        self.assertEqual('test::csv_dir', ds.get_name())
        self.assertTrue(len(ds.series_list()) > 0)
        self.assertListEqual(['MSFT'], ds.series_list('MS*'))
        sers = ds.load_data(['msft', 'aapl'], start='2016-12-01', end='now')

        self.__assert_series(sers['MSFT'], '2016-12-01', 59.20, '2016-12-02', 59.25)
        self.__assert_series(sers['AAPL'], '2016-12-01', 109.49, '2016-12-02', 109.90)

        # also check if resulting dict has same order
        self.assertEqual(len(sers), 2)
        self.assertEqual(list(sers.keys())[0], 'MSFT')
        self.assertEqual(list(sers.keys())[1], 'AAPL')

    def testInMemoryDataSource(self):
        ds = InMemoryDataSource({'msft': pd.DataFrame()})
        self.assertTrue(isinstance(ds, datasource.DataSource))
        self.assertTrue(isinstance(ds.load_data('MSFT')['MSFT'], pd.DataFrame))

        ds = InMemoryDataSource(pd.DataFrame())
        self.assertTrue(isinstance(ds.load_data('msft')['MSFT'], pd.DataFrame))

        self.assertRaises(ValueError, lambda: InMemoryDataSource(['stupid param']))

    def testTimezone(self):
        ds = datasource.DataSource('test::csv_single.timezone', self.DS_CFG_PATH)
        ser = ds.load_data(['IBB'])
        self.assertEqual(ser['IBB'].shape, (267, 5))
        # todo assert dates adjusted with timezone

    def test_ds_data_nback(self):
        ds_cfg_file = 'qube/tests/data/simulator/sim_test_datasource_cfg.json'
        ds = datasource.DataSource('simulator::quotes_data', ds_cfg_file)
        ds_data = ds.load_data(['GBPUSD'])
        data = ds.load_data_nbars(10, ['GBPUSD'], date_from='2017-03-21 20:38:02')
        self.assertEqual(len(data['GBPUSD']), 10)
        self.assertTrue(data['GBPUSD'].index[-1] < DateUtils.get_datetime('2017-03-21 20:38:02'))
        print(data)
        print(ds_data)

    def __assert_series(self, ser, date_beg, val_beg, date_end, val_end):
        self.__assert_close(ser, 0, date_beg, val_beg)
        self.__assert_close(ser, -1, date_end, val_end)

    def __assert_close(self, ser, idx, date_str, val_close):
        self.assertEqual(
            [DateUtils.get_as_string(ser.index[idx], DateUtils.DEFAULT_DATE_FORMAT), ser['close'][idx]],
            [date_str, val_close])


from pytest import main
if __name__ == '__main__':
    main()