"""
  Multiprocess tasks handling module
"""
import copy
import hashlib
import itertools
import multiprocessing as mp
from datetime import datetime
from multiprocessing import Manager
from time import gmtime, strftime, sleep, monotonic, perf_counter
from typing import Union, Dict, List, Tuple

import numpy as np

from qube.datasource.controllers.MemcacheController import MemcacheController
from qube.simulator.core import DB_SIMULATION_RESULTS
from qube.utils.nb_functions import z_save
from qube.utils.ui_utils import red, yellow, blue, ui_progress_bar
from qube.utils.utils import mstruct, runtime_env


def generate_id(salt):
    """
    Unique ID generator
    """
    return hashlib.sha256((salt + f"{(monotonic() + perf_counter()):.25f}").encode('utf-8')).hexdigest()[:12].upper()


class RunningInfoManager:
    """
    Running tasks informartion manager
    """
    LIST_ID = 'runningtasks'

    def __init__(self):
        self._mc = MemcacheController()

    def list_runs(self):
        return (lambda y: [] if y is None else y)(self._mc.get_data(self.LIST_ID))

    def list_tasks(self, run_id):
        return (lambda y: [] if y is None else y)(self._mc.get_data(f'run/tasks/{run_id}'))

    def add_run_id(self, run_id):
        runs = self.list_runs()
        if run_id not in runs:
            runs.append(run_id)
            self._mc.write_data(self.LIST_ID, runs)

    def add_task_id(self, run_id, t_id):
        tasks = self.list_tasks(run_id)
        if t_id not in tasks:
            tasks.append(t_id)
            self._mc.write_data(f'run/tasks/{run_id}', tasks)

    def update_task_info(self, run_id, t_id, data):
        if t_id in self.list_tasks(run_id):
            self._mc.write_data(f'run/tasks/{run_id}/{t_id}', data)

    def get_task_info(self, run_id, t_id):
        return self._mc.get_data(f'run/tasks/{run_id}/{t_id}')

    def del_run_id(self, run_id):
        lst = self.list_runs()
        if run_id in lst:
            lst.remove(run_id)
            self._mc.write_data(self.LIST_ID, lst)
            self._mc.delete_data(f'run/{run_id}')
            self._mc.delete_data(f'run/err/{run_id}')

            # clear tasks info data
            [self._mc.delete_data(f'run/tasks/{run_id}/{t}') for t in self.list_tasks(run_id)]

            # clear tasks id data
            self._mc.delete_data(f'run/tasks/{run_id}')

    def update_id_info(self, run_id, data):
        self._mc.write_data(f'run/{run_id}', data)

    def update_id_error(self, run_id, error):
        self._mc.write_data(f'run/err/{run_id}', error)

    def get_id_info(self, run_id):
        return self._mc.get_data(f'run/{run_id}')

    def get_id_error(self, run_id):
        return self._mc.get_data(f'run/err/{run_id}')

    def close(self):
        self._mc.client.close()

    def cleanup(self):
        for r in self.list_runs():
            self.del_run_id(r)


class Task:
    """
    Abstract task class
    """

    def __init__(self, ctor, *args, **kwargs):
        """
        :param ctor: constructor of class
        :param args: constructor's aruments
        """
        self.ctor = ctor
        self.args = args
        self.kwargs = kwargs
        self.save_to_storage = True
        self.storage_db = DB_SIMULATION_RESULTS

    def save(self, save: bool, storage=DB_SIMULATION_RESULTS):
        self.save_to_storage = save
        self.storage_db = storage
        return self

    def run(self, task_obj, run_name: str, run_id: str, t_id: str, task_name: str, ri: RunningInfoManager):
        """
        Method should be implemented in child class
        """
        pass

    def _run(self, run_name: str, run_id: str, t_id: str, task_name: str, ri: RunningInfoManager) -> mstruct:
        started_time = datetime.now()
        err = None
        result = None

        try:
            # create instance
            obj = self.ctor(*self.args, **self.kwargs)

            # run task
            result = self.run(obj, run_name, run_id, t_id, task_name, ri)
        except Exception as exc:
            # let's show more info about exception here
            import traceback
            stack_trace = traceback.format_exc()
            err = f"{run_name}/{task_name}: {str(exc)} | {stack_trace}"
            print(f'ERROR: {err}')

        finish_time = datetime.now()
        # avoid of using class stored into DB !
        task_ctor_class_name = '.'.join([self.ctor.__module__, self.ctor.__name__])
        result_to_return = mstruct(name=run_name, run_id=run_id, task=task_name, task_id=t_id,
                                   started=started_time, finished=finish_time,
                                   execution_time=finish_time - started_time,
                                   task_args=[self.args, self.kwargs],
                                   task_class=task_ctor_class_name,
                                   error=err,
                                   result=result)

        if self.save_to_storage:
            z_save(f'runs/{run_name}/{task_name}/{run_id}', result_to_return, dbname=self.storage_db)
        else:
            print(f' >> runs/{run_name}/{task_name}/{run_id}: {task_ctor_class_name} | NOT SAVED |')

        return result_to_return


def __run_task(args) -> mstruct:
    run_name, run_id, t_id, task_name, task, lock = copy.copy(args)

    # adding new task in guarded section
    lock.acquire(True)
    ri = RunningInfoManager()
    ri.add_task_id(run_id, t_id)
    lock.release()
    return task._run(run_name, run_id, t_id, task_name, ri)


def __wait_all_tasks(name, run_id, results_iterator, total,
                     rinf: RunningInfoManager, ui_progress, poll_timeout=0.5,
                     collect_results=False) -> List:
    completed, failed = 0, 0
    started_time = datetime.now()

    def elapsed_seconds():
        return (datetime.now() - started_time).total_seconds()

    def info_(res):
        elapsed_time = strftime("%Hh %Mm %Ssec", gmtime(elapsed_seconds()))
        infrm = ''
        if res:
            infrm = f" Execution time: {res.execution_time} s Task: <font color='#40ff30'>{res.task}</font>"

        infrm += f" Elapsed time: {elapsed_time}" \
                 f" Completed {completed} from {total} <font color='#ff0505'> | Failed: {failed}</font>"

        if res is not None and res.error:
            infrm += f" | Error in <font color='#40ff30'>{res.task}</font> [{res.task_id}]: <font color='#ff0505'>{res.error}</font>"

        return infrm

    # here we will gather all results
    results = []
    for result in results_iterator:
        if result is not None:  # new result is ready
            completed += 1
            if result.error:
                failed += 1

            if ui_progress is not None:
                ui_progress.progress.value = completed / total
                ui_progress.info.value = info_(result)

            # update active run status
            rinf.update_id_info(run_id, {'name': name, 'progress': completed, 'total': total, 'failed': failed,
                                         'elapsed': elapsed_seconds()})

            # on big number of simualtion to avoid memory consumption
            # results collecting may be turned off
            if collect_results:
                results.append(result)

        # small timeout before next iteration
        sleep(poll_timeout)

    # after all task finished
    if ui_progress is not None:
        ui_progress.progress.style.bar_color = 'green'
        ui_progress.info.value = info_(None)

    return results


def run_tasks(name: str, tasks: Union[Dict, List], max_cpus=np.inf, max_tasks_per_proc=10, cleanup=False,
              superseded_run_id=None, task_id_start=0, collect_results=False) -> Tuple[str, List]:
    """
    Run tasks in parallel processes
    
    :return: run_id, list of results or empty list if collect_results is False (default)
    """
    n_cpu, n_tasks = max(min(mp.cpu_count(), max_cpus), 1), len(tasks)
    run_id = generate_id(name) if superseded_run_id is None else superseded_run_id
    ui_progress = ui_progress_bar(f"{name} [{run_id}]")
    results = []

    # turn into dict if needed
    if isinstance(tasks, List):
        tasks = {f"task_{n}": t for n, t in enumerate(tasks, task_id_start)}

    # running tasks manager
    rinf = RunningInfoManager()
    rinf.add_run_id(run_id)

    # Creating pool. Pool will restart process after "max_tasks_per_proc" simulations to avoid memory leak
    with mp.Pool(n_cpu, maxtasksperchild=max_tasks_per_proc) as pool, Manager() as manager:
        # run info lock
        lock = manager.Lock()

        tasks_arguments = zip(itertools.repeat(name), itertools.repeat(run_id),
                              range(1, n_tasks + 2),
                              tasks.keys(), tasks.values(),
                              itertools.repeat(lock))

        # run tasks in the pool
        results_iterator = pool.imap_unordered(__run_task, tasks_arguments)

        if runtime_env() == 'notebook':
            # ui for jupyter
            from IPython.display import display
            display(ui_progress.panel)
            results = __wait_all_tasks(name, run_id, results_iterator, n_tasks, rinf, ui_progress,
                                       collect_results=collect_results)
        else:
            # progress in console
            from tqdm import tqdm
            results = __wait_all_tasks(name, run_id, tqdm(results_iterator, desc=str(name), total=n_tasks), n_tasks,
                                       rinf, ui_progress, collect_results=collect_results)

        # remove from active list if needed
        if cleanup:
            rinf.del_run_id(run_id)

        # close run info manager
        rinf.close()

    return run_id, results


def ls_running_tasks(cleanup=False, only_finished=True, details=False):
    """
    List all running tasks (processes)
    """
    rinf = RunningInfoManager()
    runs = rinf.list_runs()
    for r in runs:
        r_info = rinf.get_id_info(r)
        if r_info is None:
            print(f'{blue(r)} -> {red("None info yet")}')
            continue
        _p = r_info.get('progress', '-1')
        _t = r_info.get('total', '-1')
        _pct = 100 * int(_p) / int(_t)
        s_info = f"{r_info.get('name', '???')} : {_p} ({_pct:.2f}%) from {_t} [FAILED: {red(r_info.get('failed', '0'))}] "
        print(f"{blue(r)} -> {s_info}")
        if details:
            for t in rinf.list_tasks(r):
                task_data = rinf.get_task_info(r, t)

                if only_finished and isinstance(task_data, dict):
                    if task_data.get('progress', 0) >= 100:
                        continue

                print(f"\t{yellow(t)} -> {task_data}")

    if cleanup:
        rinf.cleanup()

    rinf.close()
