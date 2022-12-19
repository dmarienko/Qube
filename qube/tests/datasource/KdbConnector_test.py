from unittest import TestCase

import pandas as pd

from qube import datasource


class KdbConnector_test(TestCase):
    TEST_DATASOURCE_CONFIG_PATH = 'qube/tests/ds_test_cfg.json'

    def setUp(self):
        pass

    def testKdbConnectorDaily(self):
        ds = datasource.DataSource('kdb::daily', self.TEST_DATASOURCE_CONFIG_PATH)
        self.assertEquals(ds.series_list(), ['AAL', 'AAL.TEST', 'AAPL', 'GE', 'MSFT', 'SPY'])

        data = ds.load_data(['aal', 'ge'], start='2016-01-01', end='2016-04-01')
        self.assertEqual(len(data.keys()), 2)
        self.assertGreater(data['AAL'].shape[0], 30)
        self.assertGreater(data['GE'].shape[0], 30)

        data = ds.load_data('AAL', timeframe='1w', start='2016-02-01', end='2016-03-01')
        self.assertGreater(data['AAL'].shape[0], 4)  # 4 weeks or more as rows

        data = ds.load_data('AAL.TEST', timeframe='1w', start='2016-02-01', end='2016-03-01')
        self.assertGreater(data['AAL.TEST'].shape[0], 4)  # 4 weeks or more as rows
        # print(data['AAL'])
        ds.close(0.1)

    def testKdbConnectorDailySplit(self):
        ds = datasource.DataSource('kdb::daily', self.TEST_DATASOURCE_CONFIG_PATH)
        data_part_1 = ds.load_data(['AAL'], timeframe='1d', start='2016-01-01', end='2016-03-31')
        data_part_2 = ds.load_data(['AAL'], timeframe='1d', start='2016-04-01', end='2016-08-01')
        data_full = ds.load_data(['AAL'], timeframe='1d', start='2016-01-01', end='2016-08-01')
        data_parts = pd.concat([data_part_1['AAL'], data_part_2['AAL']])
        ds.close(0.1)
        self.assertTrue(data_full['AAL'].equals(data_parts))

    def testKdbConnectorQuotes(self):
        ds = datasource.DataSource('kdb::quotes', self.TEST_DATASOURCE_CONFIG_PATH)
        self.assertEquals(ds.series_list(), ['MSFT', 'SPY'])

        # load raw tick data
        data = ds.load_data('spy', timeframe='0', start='2016-01-01 9:30', end='2016-01-01 10:00')
        self.assertGreater(data['SPY'].shape[0], 1000)  # should be much greater than 1000 records

        # load 5 min integral data
        data = ds.load_data('spy', timeframe='5m', start='2016-01-01 9:30', end='2016-01-02 16:00')
        self.assertEqual(data['SPY'].shape[0], 158)  # 2 days of 5 min
        # print(data['spy'])

        # loading integrated daily data
        data = ds.load_data('SPY', timeframe='1d', start='2016-01-01', end='2016-01-11')
        # print(data['spy'])
        self.assertEquals(10, data['SPY'].shape[0])
        ds.close()


from pytest import main
if __name__ == '__main__':
    main()