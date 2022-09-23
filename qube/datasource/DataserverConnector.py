import json
from urllib.parse import urlparse, urlencode

import requests

from .DataSource import BasicConnector
from .KdbConnector import KdbConnector


class DataserverConnector(BasicConnector):
    DEFAULT_API_URL = ''
    DEFAULT_API_KEY = ''

    OPEN_CONNECTION_PATH = '/open_connection/%s/%s'
    CLOSE_CONNECTION_PATH = '/close_connection/%s/%s'
    IS_FEED_SUPPORTED = '/is_feed_supported/%s/%s'

    def __init__(self, _dir, _cfg, _name):
        super(DataserverConnector, self).__init__(_dir, _cfg, _name)

        self._source = self.peek_or('source')
        self._api_key = self.peek_or('api_key', self.DEFAULT_API_KEY)
        self._api_url = self.peek_or('api_url', self.DEFAULT_API_URL)
        self._feed_type = self.peek_or('feed_type')
        self._client_id = _name

        r = requests.get(self._build_request_url(self.IS_FEED_SUPPORTED))

        if r.status_code != 200 or r.content.decode("utf-8") != 'true':
            raise ValueError(
                'IS FEED SUPPORTED request failed. Feed:%s, received status code:%s, response:\n%s' %
                (self._feed_type, r.status_code, r.content))

        r = requests.get(self._build_request_url(self.OPEN_CONNECTION_PATH))
        if r.status_code != 200:
            raise RuntimeError(
                'Failed to open dataserver connection. Status code:%s, response:\n%s' % (r.status_code, r.content))

        dataserver_config = json.loads(r.content)
        self._port = dataserver_config['port']

        kdb_datasource_cfg = {'host': urlparse(self._api_url).hostname,
                              'port': self._port,
                              'db_path': dataserver_config['kdbDbPath'],
                              'type': 'kdb',
                              'init': self.peek_or('init'),
                              'load_by_days': self.peek_or('load_by_days')}

        self.kdb_connector = KdbConnector(_dir, kdb_datasource_cfg, _name)

    def load_data(self, series, start, end=None, *args, **kwargs):
        return self.kdb_connector.load_data(series, start, end, *args, **kwargs)

    def series_list(self, pattern=r".*"):
        return self.kdb_connector.series_list(pattern)

    def reload(self):
        self.kdb_connector.reload()

    def close(self):
        self.kdb_connector.close()
        r = requests.get(self._build_request_url(self.CLOSE_CONNECTION_PATH), {'port': self._port})
        if r.status_code != 200:
            raise RuntimeError(
                'Failed to close dataserver connection. Status code:%s, response:\n%s' % (r.status_code, r.content))

    def __del__(self):
        self.close()

    def _build_request_url(self, path, params=None):
        if not params:
            params = {}
        params['api_key'] = self._api_key
        params['client_id'] = self._client_id
        url = self._api_url + path % (self._source, self._feed_type)
        url += "?" + urlencode(params)
        return url
