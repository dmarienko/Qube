import unittest
from datetime import datetime
from time import monotonic
import mongomock
from mongomock.gridfs import enable_gridfs_integration
enable_gridfs_integration()
from qube.datasource.controllers.MongoController import MongoController

import numpy as np
from qube.datasource import DataSource

from qube.simulator.multiproc import Task, RunningInfoManager, run_tasks, ls_running_tasks
from qube.tests.utils_for_tests import _read_timeseries_data


class TestCalcTask(Task):
    """
    Example of task
    """

    def run(self, task_obj, run_name, run_id, t_id, task_name, ri: RunningInfoManager):
        np.random.seed(int(monotonic() * 1e25) % 100000)

        v = 0
        for i in range(20):
            ri.update_task_info(run_id, t_id, {'id': t_id,
                                               'update_time': str(datetime.now()),
                                               'progress': i + 1,
                                               })
            v += np.random.randn(1000000, 1).std()

            # if np.random.randint(3) == 0:
            #     raise ValueError('Boom !!!')

        return v


class TestWeirdCaseTask(Task):
    DS_CFG_PATH = 'qube/tests/ds_test_cfg.json'

    """
    Example of task
    """
    def __init__(self, ctor, *args, **kwargs):
        super().__init__(ctor, *args, **kwargs)
        # - Mongo has issues with pickling/unpickling in mp !
        self.datasource = DataSource('mongo::binance-perpetual-1min')

    def run(self, task_obj, run_name, run_id, t_id, task_name, ri: RunningInfoManager):
        np.random.seed(int(monotonic() * 1e25) % 100000)
        v = 0
        for i in range(20):
            ri.update_task_info(run_id, t_id, {'id': t_id,
                                               'update_time': str(datetime.now()),
                                               'progress': i + 1,
                                               })
            v += np.random.randn(1000000, 1).std()
        return v


class MultiProcTest(unittest.TestCase):
    def __initialize_mongo_db(self):
        print('Initializing database ...')
        d1 = _read_timeseries_data('solusdt_15min', compressed=True)

        self.mongo = MongoController('md')
        self.mongo.save_data('m1/BINANCEF:SOLUSDT', d1, is_serialize=True)
        self.mongo.close()

    def test_run_tasks(self):
        # - this requires runnig memcached
        run_id, res = run_tasks('Just test',
                                {
                                    '1': TestCalcTask(str, 'some test stub 1').save(False),
                                    '2': TestCalcTask(str, 'some test stub 2').save(False),
                                }, max_cpus=1, collect_results=True)

        # just debug
        ls_running_tasks()

        ri = RunningInfoManager()

        self.assertListEqual([1, 2], ri.list_tasks(run_id))
        ri.del_run_id(run_id)
        for r in res:
            print(r.run_id, r.task_id, r.result)
        ri.close()

    @mongomock.patch()
    def test_weird_case_tasks(self):
        self.__initialize_mongo_db()

        run_id, res = run_tasks('Just test',
                                {
                                    '1': TestWeirdCaseTask(str, 'some test stub 1').save(False),
                                    '2': TestWeirdCaseTask(str, 'some test stub 2').save(False), 
                                }, max_cpus=1, collect_results=True)
        ls_running_tasks()
        ri = RunningInfoManager()
        self.assertListEqual([1, 2], ri.list_tasks(run_id))
        ri.del_run_id(run_id)
        for r in res:
            print(r.run_id, r.task_id, r.result)
        ri.close()



from pytest import main
if __name__ == '__main__':
    main()