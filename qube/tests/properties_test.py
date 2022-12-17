import os
import unittest

from qube.configs import Properties


class PropertiesTest(unittest.TestCase):
    def test_verify_running_from_tests(self):
        self.assertEqual(Properties.get_env(), 'TEST')

    def test(self):
        self.assertEqual(Properties.get_properties('qube/tests/qube_props_test.json')['test1'], 'hello')
        self.assertEqual(Properties.get_properties('qube/tests/qube_props_test.json')['test1'], 'hello')
        self.assertEqual(Properties.get_properties('qube/tests/qube_props_test.json')['test1'], 'hello')
        self.assertEqual(Properties.get_properties('qube/tests/qube_props_test')['test2'], 'world')
        self.assertEqual(Properties.get_properties("qube/tests/qube_props_test.json"),
                         {"test1": "hello", "test2": "world"})

        if os.name == 'nt':
            self.assertEqual(
                Properties.get_properties(os.path.join(Properties.get_root_dir(), 'tests\\qube_props_test.json')),
                {"test1": "hello", "test2": "world"})


from pytest import main
if __name__ == '__main__':
    main()