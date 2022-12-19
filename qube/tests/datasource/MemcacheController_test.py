import unittest

from qube.datasource.controllers.MemcacheController import MemcacheController


class MemcacheControllerTest(unittest.TestCase):

    def setUp(self):
        self.memcache = MemcacheController()

    def tearDown(self):
        self.memcache.delete_data('progress_1')
        self.memcache.delete_data('intval')
        self.memcache.delete_data('testappend')
        self.memcache.close()

    def test_memcache_dict(self):
        # test dict
        self.memcache.write_data('progress_1', {'chunk_1': '20%', 'chunk_2': '40%'})
        self.assertEqual({'chunk_1': '20%', 'chunk_2': '40%'}, self.memcache.get_data('progress_1'))

        # test str
        self.memcache.write_data('intval', 'string')
        self.assertEqual('string', self.memcache.get_data('intval'))

    def test_append_memcache(self):
        self.memcache.append_data('testappend', {'a': 1})
        self.memcache.append_data('testappend', {'a': 2})
        self.assertEqual([{'a': 1}, {'a': 2}], self.memcache.get_data('testappend'))


from pytest import main
if __name__ == '__main__':
    main()