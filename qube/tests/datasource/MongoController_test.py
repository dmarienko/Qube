import unittest
import mongomock
from mongomock.gridfs import enable_gridfs_integration

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from qube.datasource import DataSource
from qube.utils.DateUtils import DateUtils
from qube.datasource.controllers.MongoController import MongoController

enable_gridfs_integration()


class MongoControllerTest(unittest.TestCase):

    @mongomock.patch()
    def setUp(self):
        self.mongo = MongoController()
        self.table_name = '_test_name_'

    def tearDown(self):
        self.mongo.del_data(self.table_name)

    def testBasic(self):
        ds = DataSource('test::csv_single', 'qube/tests/ds_test_cfg.json')
        sers = ds.load_data(['test'])
        df = sers["TEST"]
        df = df.reindex(sorted(df.columns), axis=1)
        meta = {"author": "222", "param": "9999"}
        self.mongo.save_data(self.table_name, df, meta)
        data_from_mongo = self.mongo.load_data(self.table_name)
        # compare DataFrames
        assert_frame_equal(df, data_from_mongo['data'])
        # compare meta
        self.assertEqual(meta, data_from_mongo['meta'])

        # find collection
        collections = self.mongo.ls_data("_test.*na.*")
        # Is found collection
        self.assertTrue(self.table_name in collections)

    def testSaveDict(self):
        # save dict instead df
        dictonary = {"name_key": "value", "name_key2": "value2"}
        self.mongo.save_data(self.table_name, dictonary)
        data_from_mongo = self.mongo.load_data(self.table_name)
        result = data_from_mongo['data']
        self.assertEqual(dictonary, result)

    def testSaveObject(self):
        # save object to mongo
        obj = TestObject()
        self.mongo.save_data(self.table_name, obj)
        data_from_mongo = self.mongo.load_data(self.table_name)
        result = data_from_mongo['data']
        self.assertEqual(result.test_method(), obj.test_method())

    def testSaveString(self):
        # save string to mongo
        string = "test_string"
        self.mongo.save_data(self.table_name, string)
        data_from_mongo = self.mongo.load_data(self.table_name)
        result = data_from_mongo['data']

        self.assertEqual(string, result)

    def testSavelistofDicts(self):
        # save list of dicts to mongo
        dictonary = {"name_key": "value", "name_key2": "value2"}
        dictonary2 = {"name_key": "value3", "name_key2": "value4"}
        test_list = [dictonary, dictonary2]
        copy_list = list(test_list)
        self.mongo.save_data(self.table_name, test_list)
        data_from_mongo = self.mongo.load_data(self.table_name)
        result = data_from_mongo['data']
        self.assertEqual(copy_list, result)

    def testSavelist(self):
        # save list to mongo
        test_list = [1, 2, 3, 5, '6', 7.20]
        copy_list = list(test_list)
        self.mongo.save_data(self.table_name, test_list)
        data_from_mongo = self.mongo.load_data(self.table_name)
        result = data_from_mongo['data']
        self.assertEqual(copy_list, result)

    def testSaveOneElemetInList(self):
        # save one elemet with list to mongo
        test_list = [1]
        copy_list = list(test_list)
        self.mongo.save_data(self.table_name, test_list)
        data_from_mongo = self.mongo.load_data(self.table_name)
        result = data_from_mongo['data']
        self.assertEqual(copy_list, result)

    def testDeleteRecords(self):
        data = {"name_key": "value"}
        self.mongo.save_data(self.table_name, data)
        data = {"name_key2": "value2"}
        self.mongo.save_data(self.table_name, data)
        self.mongo.delete_records(self.table_name, {"name_key": "value"})

        data_from_mongo = self.mongo.load_data(self.table_name)
        result = data_from_mongo['data']

        self.assertEqual(1, len(result))
        self.assertEqual(data, result)

    def testLoadRecords(self):
        data = {"name_key": "value"}
        self.mongo.save_data(self.table_name, pd.DataFrame([data], columns=data.keys()),
                             meta={'meta_key': 'meta_value'}, is_serialize=False)

        data_from_mongo = self.mongo.load_records(self.table_name, {"name_key": "value"})
        self.remove_id_col(data_from_mongo['data'])

        self.assertEqual(1, len(data_from_mongo['data']))
        self.assertEqual(data, data_from_mongo['data'][0])
        self.assertEqual(data_from_mongo['meta'], {'meta_key': 'meta_value'})

    def testSaveDictsWithoutSerialize(self):
        dicts = [{"name_key": "value"}, {"name_key2": "value2"}]
        dicts2 = [{"name_key3": "value3"}, {"name_key4": "value4"}]
        self.mongo.append_data(self.table_name, dicts, {'meta_key': 'meta_value'})
        self.mongo.append_data(self.table_name, dicts2)

        data_from_mongo = self.mongo.load_data(self.table_name)
        result = data_from_mongo['data']
        meta = data_from_mongo['meta']
        self.remove_id_col(result)

        self.assertEqual(
            [{"name_key": "value"}, {"name_key2": "value2"}, {"name_key3": "value3"}, {"name_key4": "value4"}], result)
        self.assertEqual({'meta_key': 'meta_value'}, meta)

    def testBigData(self):
        bigdf = pd.DataFrame([i for i in range(800000)],
                             index=pd.date_range('1970-01-01 01:00:00', periods=800000, freq='5min'), columns=['A'])
        self.mongo.save_data(self.table_name, bigdf)
        data_from_mongo = self.mongo.load_data(self.table_name)
        assert_frame_equal(bigdf, data_from_mongo['data'])

    def testMongoDf(self):
        df = pd.DataFrame([i for i in range(80)], index=pd.date_range('1970-01-01 01:00:00', periods=80, freq='5min'),
                          columns=['A'])
        self.mongo.save_data(self.table_name, df.iloc[:40], is_serialize=False)
        self.mongo.append_data(self.table_name, df.iloc[40:79])
        self.mongo.append_data(self.table_name, df.iloc[-1])
        data_from_mongo = self.mongo.load_data(self.table_name)['data']
        self.remove_id_col(data_from_mongo)
        assert_frame_equal(df, data_from_mongo, check_freq=False)

    def testEmpyData(self):
        self.mongo.save_data(self.table_name, [], is_serialize=False)
        self.mongo.save_data(self.table_name, [{'test_key': 'test_value'}], is_serialize=False)
        self.mongo.append_data(self.table_name, [])
        data_from_mongo = self.mongo.load_data(self.table_name)['data']
        self.remove_id_col(data_from_mongo)
        self.assertEqual(data_from_mongo, [{'test_key': 'test_value'}])

    def testLoadRecordsDf(self):
        df = pd.DataFrame([i for i in range(5)], index=pd.date_range('1970-01-01 01:00:00', periods=5, freq='5min'),
                          columns=['A'])
        self.mongo.append_data(self.table_name, df.iloc[1])
        self.mongo.append_data(self.table_name, df.iloc[2])
        self.mongo.append_data(self.table_name, df.iloc[3])
        data_from_mongo = \
        self.mongo.load_records(self.table_name, {'__index': {'$gte': DateUtils.get_datetime('1970-01-01 01:05:00')}},
                                True)['data']
        self.assertEqual(data_from_mongo.index.dtype.type, np.datetime64)
        self.assertTrue(isinstance(data_from_mongo, pd.DataFrame))
        self.assertEqual(data_from_mongo.index[0], DateUtils.get_datetime('1970-01-01 01:05:00'))

    def test_aggregate(self):
        df = pd.DataFrame([i for i in range(5)], index=pd.date_range('1970-01-01 01:00:00', periods=5, freq='5min'),
                          columns=['A'])
        self.mongo.save_data(self.table_name, df, is_serialize=False)
        aggr_query = [{'$group': {'_id': 0, 'min_date': {'$min': "$__index"}, 'max_date': {'$max': "$__index"}}}]
        result = self.mongo.load_aggregate(self.table_name, aggr_query)[0]
        self.assertEqual(DateUtils.get_as_string(result['min_date']), '1970-01-01 01:00:00')
        self.assertEqual(DateUtils.get_as_string(result['max_date']), '1970-01-01 01:20:00')

    def test_load_is_sorted(self):
        df = pd.DataFrame(pd.date_range('1970-01-01 01:00:00', periods=10, freq='5min'), index=[i for i in range(10)],
                          columns=['time'])
        # shuffle df
        df = df.sample(frac=1)
        self.mongo.save_data(self.table_name, df, is_serialize=False, mongo_index='time')
        result = self.mongo.load_data(self.table_name, [('time', 1)])['data']
        self.assertEqual(DateUtils.get_as_string(result['time'].iloc[0]), '1970-01-01 01:00:00')
        result = self.mongo.load_data(self.table_name, [('time', -1)])['data']
        self.assertEqual(DateUtils.get_as_string(result['time'].iloc[0]), '1970-01-01 01:45:00')
        indexes = self.mongo.db[self.table_name].index_information()
        self.assertTrue('time_1' in indexes)

    def test_count(self):
        df = pd.DataFrame([i for i in range(5)], index=pd.date_range('1970-01-01 01:00:00', periods=5, freq='5min'),
                          columns=['A'])
        self.mongo.save_data(self.table_name, df, is_serialize=False)
        count_all = self.mongo.get_count(self.table_name)
        count_query = self.mongo.get_count(self.table_name,
                                           {'__index': {'$gte': DateUtils.get_datetime('1970-01-01 01:10:00')}})
        self.assertEqual(count_all, 5)
        self.assertEqual(count_query, 3)

    def test_skip_limit(self):
        df = pd.DataFrame([i for i in range(5)], index=pd.date_range('1970-01-01 01:00:00', periods=5, freq='5min'),
                          columns=['A'])
        self.mongo.save_data(self.table_name, df, is_serialize=False)
        data = self.mongo.load_records(self.table_name, {}, as_dataframe=True, skip=2, limit=1)['data']
        self.assertEqual(data.index[0], DateUtils.get_datetime('1970-01-01 01:10:00'))
        self.assertEqual(len(data), 1)
        data = self.mongo.load_records(self.table_name, {}, as_dataframe=True, skip=3, limit=2)['data']
        self.assertEqual(data.index[0], DateUtils.get_datetime('1970-01-01 01:15:00'))
        self.assertEqual(len(data), 2)

    def remove_id_col(self, data):
        # remove _id col for comparison of tests
        [d.pop('_id') for d in data] if isinstance(data, list) else data.pop('_id')


class TestObject:
    def test_method(self):
        return 'test_string'


from pytest import main
if __name__ == '__main__':
    main()