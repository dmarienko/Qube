import unittest
from datetime import datetime
from time import monotonic

import numpy as np

from qube.simulator.multiproc import Task, RunningInfoManager, run_tasks, ls_running_tasks


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


class MultiProcTest(unittest.TestCase):

    def test_run_tasks(self):
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
