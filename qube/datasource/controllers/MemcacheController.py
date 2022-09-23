import json
import os
import platform
import subprocess

from pymemcache.client.base import Client

from qube.utils import QubeLogger
from qube.configs import Properties


class MemcacheController:

    def __init__(self):
        memcache_props = Properties.get_config_properties('main-props.json')['memcache']
        port = memcache_props['memcached_port']
        key_prefix = memcache_props['key_prefix']
        self.__logger = QubeLogger.getLogger(self.__module__)
        self.client = Client(('localhost', port), key_prefix=key_prefix)
        self._check_connection()

    def write_data(self, key, data):
        encoded_data = json.dumps(data, ensure_ascii=False)
        self.client.set(key, encoded_data)

    def append_data(self, key, data, check_exists=True):
        if not check_exists or self.client.get(key):
            encoded_data = json.dumps(data, ensure_ascii=False)
            self.client.append(key, ',' + encoded_data)
        else:
            self.write_data(key, data)

    def get_data(self, key):
        data = self.client.get(key)

        if not data:
            return None

        try:
            decoded_data = json.loads(data.decode("utf-8"))
        except json.decoder.JSONDecodeError:
            encoded_data = data.decode("utf-8")
            decoded_data = json.loads('[' + encoded_data + ']')
        return decoded_data

    def delete_data(self, key):
        self.client.delete(key)

    def close(self):
        """
        Close connection to memcached
        """
        self.client.close()

    def _check_connection(self):
        """
        Check if the memcached is starting
        """
        try:
            self.client.stats()
        except ConnectionRefusedError:
            self.__logger.info('Memcached is not starting. Attempt to run')
            self._run_memcached()
            # Check again
            try:
                self.client.stats()
            except ConnectionRefusedError:
                raise RuntimeError('Failed to start memcached')
            else:
                self.__logger.info('Memcached is running now')

    def _run_memcached(self):
        win_mc_path = r'c:\memcached\memcached.exe'
        run_os = platform.system().lower()

        if run_os == "windows":
            if not os.path.isfile(win_mc_path):
                raise FileNotFoundError('Memcached is not found in %s' % win_mc_path)
            subprocess.Popen(['memcached.exe', 'start'], executable=win_mc_path)

        elif run_os == "linux":
            subprocess.Popen(['sudo', 'service', 'memcached', 'start'])
