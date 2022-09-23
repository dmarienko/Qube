import os
import pickle
import platform
import re
import subprocess
from typing import Union, List

import gridfs
import pandas as pd
import pymongo

from qube.utils import QubeLogger
from qube.configs import Properties


class MongoController:
    TIME_COLS = ['__index', 'time', 'date', 'Time', 'Date']

    def __init__(self, dbname=None, mongourl=None, username="", password=""):
        self.__logger = QubeLogger.getLogger(self.__module__)

        if not mongourl:
            mongourl = Properties.get_config_properties('main-props.json')['mongo']['default_mongourl']
        if not dbname:
            dbname = Properties.get_config_properties('main-props.json')['mongo']['default_db_name']
        selection_timeout = Properties.get_config_properties('main-props.json')['mongo']['selection_timeout_ms']

        self.mongoclient = pymongo.MongoClient(mongourl, serverSelectionTimeoutMS=selection_timeout,
                                               username=username, password=password, authSource=dbname)
        self.db = self.mongoclient[dbname]
        self.gfs = gridfs.GridFS(self.db)

        self._start_mongo()

    def save_data(self, name: str, data, meta: dict = None, is_serialize=True, mongo_index=None):
        if meta is None:
            meta = {}
        collection = self.db[name]

        self.del_data(name)
        self.gfs.delete(name)

        if not is_serialize:
            if isinstance(data, (pd.DataFrame, pd.Series)):
                mongo_data, meta = self._prepare_df_to_save(data, meta)
            elif isinstance(data, list):
                mongo_data = data
                meta['__type'] = 'list'
            else:
                raise ValueError('is_serialize=False supported only for DataFrame, Series and list objects')
            if mongo_data:
                if isinstance(mongo_data, list):
                    collection.insert_many(mongo_data)
                    if mongo_index:
                        collection.create_index(mongo_index)
                    for d in mongo_data:
                        d.pop('_id', None)
                        d.pop('__index', None)
                else:
                    collection.insert_one(mongo_data)
        else:
            meta['__type'] = 'object'
            mongo_data = pickle.dumps(data)
            self.gfs.put(mongo_data, _id=name)

        meta['__meta'] = 1
        # insertion meta
        collection.insert_one(meta)
        meta.pop('_id', None)
        meta.pop('__index_name', None)
        meta.pop('__meta', None)
        meta.pop('__type', None)

        return

    def append_data(self, name, data: Union[pd.DataFrame, pd.Series, list], meta: dict = None, mongo_index=None):
        if meta is None:
            meta = {}
        is_gfs_data = self.gfs.exists(name)
        if is_gfs_data:
            raise ValueError('append_data supported only for saved not serialized DataFrame, Series and list objects')

        collection = self.db[name]
        if not collection.estimated_document_count():
            self.save_data(name, data, meta, is_serialize=False, mongo_index=mongo_index)
            return

        if isinstance(data, (pd.DataFrame, pd.Series)):
            mongo_data, _ = self._prepare_df_to_save(data, {})
        elif isinstance(data, list):
            mongo_data = data
        if not mongo_data:
            return
        collection.insert_many(mongo_data)
        for d in mongo_data:
            d.pop('_id', None)
            d.pop('__index', None)

    def load_data(self, name: str, sort: List[tuple] = None):
        collection = self.db[name]
        # get records where meta != 1
        db_cursor = collection.find({'__meta': {'$ne': 1}})
        if sort:
            db_cursor.sort(sort)
        db_data = list(db_cursor)
        is_gfs_data = self.gfs.exists(name)
        if not db_data and not is_gfs_data:
            return {'data': None, 'meta': None}

        # get records where meta = 1
        metarecord = list(collection.find({'__meta': 1}, {'_id': False, '__meta': False}))
        meta = metarecord[0] if metarecord else {
            '__type': 'DataFrame'}  # if meta is not present, loads data as DataFrame
        if meta:
            if meta['__type'] == 'DataFrame':
                db_data, meta = self._prepare_df_to_load(db_data, meta)
                if isinstance(db_data, list) and len(db_data) == 1:
                    db_data = db_data[0]
            elif meta['__type'] != 'list':
                # it is serialize object
                db_data = pickle.loads(self.gfs.get(name).read())
            meta.pop('__type', None)

        result = {'data': db_data, 'meta': meta}

        return result

    def ls_data(self, query=r'.*'):
        if not isinstance(query, str):
            raise TypeError('input data is not str')
        collections = self.db.list_collection_names()
        reg = re.compile(query)
        # filter collections by regex
        result_collections = list(filter(reg.match, collections))
        return result_collections

    def del_data(self, name):
        if not isinstance(name, str):
            raise TypeError('input data is not str')
        self.db.drop_collection(name)
        self.gfs.delete(name)

    def delete_records(self, name, query):
        if not isinstance(name, str):
            raise TypeError('input data is not str')

        if not isinstance(query, dict):
            raise TypeError('input data is not dict')

        collection = self.db[name]
        return collection.delete_many(query)

    def load_records(self, name, query, as_dataframe=False, sort: List[tuple] = None, skip=0, limit=0):
        if not isinstance(name, str):
            raise TypeError('input data is not str')

        if not isinstance(query, dict):
            raise TypeError('input data is not dict')

        collection = self.db[name]
        query.update({'__meta': {'$ne': 1}})
        if not as_dataframe:
            db_cursor = collection.find(query, {'__index': False}).skip(skip).limit(limit)
            if sort:
                db_cursor.sort(sort)
            db_data = list(db_cursor)
            meta = list(collection.find({'__meta': 1}, {'_id': False, '__meta': False,
                                                        '__type': False, '__index_name': False}))
            meta = meta[0] if meta else {}
        else:
            meta = list(collection.find({'__meta': 1}, {'_id': False, '__meta': False}))
            meta = meta[0] if meta else {}
            db_cursor = collection.find(query).skip(skip).limit(limit)
            if sort:
                db_cursor.sort(sort)
            db_data = list(db_cursor)
            db_data, meta = self._prepare_df_to_load(db_data, meta)
            if isinstance(db_data, list) and len(db_data) == 1:
                db_data = db_data[0]
        return {'data': db_data, 'meta': meta}

    def get_count(self, name, query: dict = None):
        if query is None:
            query = {}
        collection = self.db[name]
        query.update({'__meta': {'$ne': 1}})
        return collection.count_documents(query)

    def load_aggregate(self, name, query):
        if not isinstance(query, list):
            raise TypeError('aggregate query must be a list')
        collection = self.db[name]
        result = list(collection.aggregate(query))
        return result

    def close(self):
        return self.mongoclient.close()

    def _start_mongo(self):
        self.__logger.debug('try connect to mongo')
        try:
            self.mongoclient.server_info()
        except:
            run_os = platform.system().lower()
            if run_os == "windows":
                win_mongo_db_path = r'C:\data\db'
                if not os.path.isdir(win_mongo_db_path):
                    os.makedirs(win_mongo_db_path)
                subprocess.Popen(['mongod', '--dbpath', win_mongo_db_path])
            elif run_os == "linux":
                self.__logger.info('try start service mongod')
                subprocess.Popen(['sudo', 'service', 'mongod', 'start'])
        else:
            self.__logger.debug('mongodb already running')
        try:
            self.mongoclient.server_info()
        except:
            raise ConnectionError("Can not start Mongo Server")
        return

    def _prepare_df_to_save(self, data, meta):
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data.to_dict(), index=[data.name])
        data['__index'] = data.index
        meta['__index_name'] = data.index.name  # storing DataFrame's index name
        meta['__type'] = 'DataFrame'
        mongo_data = data.to_dict('records')
        del data['__index']
        return mongo_data, meta

    def _prepare_df_to_load(self, data, meta):
        if not data:
            return pd.DataFrame(), meta
        df_data = pd.DataFrame.from_dict(data)
        df_index = self.__find_df_index(df_data)
        if df_index:
            df_data.index = df_data[df_index]
            del df_data[df_index]
            df_data.index.name = meta.get('__index_name')
            meta.pop('__index_name', None)
        return df_data, meta

    def __find_df_index(self, data):
        for c in self.TIME_COLS:
            if c in data:
                return c
