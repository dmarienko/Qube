import copy
import itertools
import multiprocessing as mp
import time
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from qube.booster.utils import class_import, short_performace_report, b_del, b_ld, b_save, b_ls
from qube.datasource.loaders import load_data
from qube.portfolio.reports import tearsheet
from qube.simulator import simulation
from qube.utils.ui_utils import red, green, yellow, blue, ui_progress_bar
from qube.utils.utils import runtime_env


def model_short_report(rpt):
    """
    Short one-line colored string report on model's performance
    """
    if rpt:
        try:
            _mx, _km = -np.inf, ''
            for k, ri in rpt.items():
                if ri is not None and ri['sharpe'] > _mx:
                    _mx, _km = ri['sharpe'], k
            best = rpt[_km]
            short_rep = yellow(' | '.join([f'{n[:1]}: {x:.2f}' for n, x in best.items() if
                                           n in ['sharpe', 'gain', 'cagr', 'dd_usd', 'dd_pct', 'qr', 'nexecs']]))
        except Exception as e:
            short_rep = f'[{str(e)}]'
    else:
        short_rep = red('no report data')
    return short_rep


class Operation:
    """
    Abstract operation
    """

    def __init__(self, op, *args, **kwargs):
        self.op = op
        self.args = args
        self.kwargs = kwargs


class IReport:
    def __floordiv__(self, operation: Operation):
        if isinstance(operation, Operation):
            if operation.op == 'report':
                return self.report(*operation.args, **operation.kwargs)
            if operation.op == 'stats':
                return self.stats(*operation.args, **operation.kwargs)
            if operation.op == 'delete':
                if 'delete' in dir(self):
                    return self.delete(*operation.args, **operation.kwargs)
            if operation.op == 'run':
                return self.run(*operation.args, **operation.kwargs)
        return self


class SimView(IReport):
    def __init__(self, data, stats):
        self.data = data
        self.stats = stats

    def tearsheet(self, init_cash=0, params={}):
        """
        Wrapper for bit convenient tearsheet method
        """
        if params is None:
            params = dict()

        r = self.data
        t_params = r.task_args[1]
        i_cash = t_params.get('capital', init_cash)

        try:
            insmpl = (r.result.start, r.result.end)
        except:
            insmpl = None

        return tearsheet(r.result, i_cash, insample=insmpl, meta={'parameters': {**t_params, **params}})

    def ts(self, init_cash=0, params={}):
        return self.tearsheet(init_cash=init_cash, params=params)

    def report(self, init_cash=0, params={}):
        return self.tearsheet(init_cash=init_cash, params=params)

    def run(self, start, broker, exchage, stop=None):
        if isinstance(self.data.task_class, str):
            task_clazz = class_import(self.data.task_class)
        else:
            task_clazz = self.data.task_class
        t_obj = task_clazz(**self.data.task_args[1])
        mktdata = load_data(*[f"{exchage}:{s}" for s in self.data.result.instruments])
        r = simulation({self.data.task: t_obj}, mktdata.ticks(), broker, self.data.name, start=start, stop=stop)
        return r.results[0]

    def __repr__(self):
        s = self.stats
        d = 'none'
        if s is not None:
            d = f"gain: {s.gain:.3f} cagr: {s.cagr:.2f} sharpe: {s.sharpe:.2f} qr: {s.qr:.3f} execs: {s.n_execs}"
        return f"{self.data.task} | {d}"


class ModelView(IReport):
    def __init__(self, project, model, sims):
        self.project = project
        self.model = model
        self.sims = sims
        self.reports_path = f"reports/{self.project}/{self.model}"

    def run(self, *args, **kwargs):
        print("Not sure what to run here")
        return self

    def spath(self, sim):
        return f"stats/{self.project}/{sim}/{self.model}"

    def dpath(self, sim):
        return f"runs/{self.project}/{sim}/{self.model}"

    def delete(self):
        """
        Dangerous !!!
        """
        if self.sims:
            print(f'>>> Deleting {len(self.sims)} simulations for {self.project} : {self.model}')
            for s in tqdm(self.sims):
                b_del(self.spath(s))
                b_del(self.dpath(s))
            self.sims = []

        # delete stored report
        b_del(self.reports_path)

    def clean(self, min_sharpe, min_gain=-np.inf, min_qr=-np.inf, min_execs=0):
        """
        Dangerous
        """
        if self.sims:
            retained = []
            for s in tqdm(self.sims):
                s_path = self.spath(s)
                d_path = self.dpath(s)
                perf = b_ld(s_path)
                if perf:
                    try:
                        if (perf.sharpe < min_sharpe or
                                perf.gain < min_gain or
                                perf.qr < min_qr or
                                perf.n_execs < min_execs):
                            b_del(s_path)
                            b_del(d_path)
                        else:
                            retained.append(s)
                    except Exception as exc:
                        print(f'>>> Exception in processing {s}: {str(exc)}')
            self.sims = retained

    def make_report(self, p):
        if p is not None:
            try:
                return {
                    'sharpe': p.sharpe, 'gain': p.gain,
                    'cagr': 100 * p.cagr, 'dd_usd': p.mdd_usd,
                    'dd_pct': p.drawdown_pct, 'nexecs': p.n_execs,
                    # Average trades per month
                    'atpm': p.avg_trades_month if hasattr(p, 'avg_trades_month') else -1,
                    'qr': p.qr, 'comm': p.commissions, **p.args
                }
            except KeyError as k:
                print(f" >> error getting performance metric: '{k}'")
        return None

    def report(self, sharpe=-np.inf, qr=-np.inf, execs=0, init_cash=None, refresh=False):
        report = b_ld(self.reports_path)

        if report is None or refresh:
            report = {}
            for s in tqdm(self.sims):
                s_path = self.spath(s)
                p = b_ld(s_path)

                if p is None and init_cash is not None and init_cash < 0:
                    p = self.update_stats(self.dpath(s), s_path, init_cash=init_cash)
                    if p is None:
                        continue

                _s_rep = self.make_report(p)
                if _s_rep is not None:
                    report[s] = _s_rep
            # save  
            b_save(self.reports_path, report)

        rpt = pd.DataFrame.from_dict(report, orient='index')
        if not rpt.empty:
            rpt = rpt[(rpt.sharpe > sharpe) & (rpt.qr > qr) & (rpt.nexecs > execs)]
            rpt = rpt.sort_values('sharpe', ascending=False) if not rpt.empty else rpt

        return rpt

    def __getitem__(self, sim: Union[str, int]):
        if isinstance(sim, int):
            search = f'sim.{sim}.'
            for s in self.sims:
                if s.startswith(search):
                    sim = s
                    break

        sim = self.sims[sim] if isinstance(sim, int) else sim
        data = b_ld(self.dpath(sim))
        stats = b_ld(self.spath(sim))
        return SimView(data, stats)

    def __truediv__(self, p):
        return self.__getitem__(p)

    def update_stats(
            self, data_path, stats_path, init_cash, account_transactions=True, performance_statistics_period=252
    ):
        sd = b_ld(data_path)
        performance = None
        if sd:
            res = sd.result
            try:
                performance = short_performace_report(res, init_cash, account_transactions=account_transactions,
                                                      performance_statistics_period=performance_statistics_period)
                task_params = dict()
                if "task_args" in dir(sd):
                    task_params = sd.task_args[1] if isinstance(sd.task_args, list) else sd.task_args
                performance.args = task_params

                # save report to DB
                b_save(stats_path, performance)
            except Exception as exc:
                print(f'>>> Exception during processing {stats_path}: {str(exc)}')

        return performance

    def do_stats(self, init_cash, force_calc=False, performance_statistics_period=365):
        """
        Calculate performance metrics for simulations (if no perf data)
        """
        for s in tqdm(self.sims):
            data_path = self.dpath(s)
            stats_path = self.spath(s)
            stats = b_ld(stats_path)
            if stats is None or force_calc:
                self.update_stats(data_path, stats_path, init_cash, account_transactions=True,
                                  performance_statistics_period=performance_statistics_period)
        return self

    def chunks(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _stat_calcs(self, args):
        _, sims, init_cash, force_calc, performance_statistics_period = copy.copy(args)
        # sys.stdout.write(f' >>> processing : {".".join([str(a) for a in args])}')
        reports = {}
        for s in sims:
            data_path = self.dpath(s)
            stats_path = self.spath(s)
            stats = b_ld(stats_path)
            if stats is None or force_calc:
                stats = self.update_stats(data_path, stats_path, init_cash, account_transactions=True,
                                          performance_statistics_period=performance_statistics_period)
            reports[s] = self.make_report(stats)
        return reports

    def stats(self, init_cash, force_calc=False, max_n_procs=100, min_per_proc=20, performance_statistics_period=365):
        data = self.sims
        name = f" >>> Processing {len(data)} simulations ... "

        n_procs = min(mp.cpu_count() - 1, max_n_procs)
        n_tasks = max(len(data) // min_per_proc, 1)
        n_points_per_task = len(data) // n_tasks

        pool = mp.Pool(n_procs)
        res_iter = pool.imap_unordered(self._stat_calcs, zip(itertools.repeat(self),  # self
                                                             self.chunks(data, n_points_per_task),
                                                             # bunch of simulations
                                                             itertools.repeat(init_cash),  # initial cash
                                                             itertools.repeat(force_calc),  # force stats calculations
                                                             itertools.repeat(performance_statistics_period)
                                                             # performance statistics period (252 or 365)
                                                             ))
        ui_progress = ui_progress_bar(name)
        if runtime_env() == 'notebook':
            # ui for jupyter
            from IPython.display import display
            display(ui_progress.panel)
        else:
            # progress in console
            from tqdm import tqdm
            res_iter = tqdm(res_iter, desc=name, total=n_tasks)

        completed = 0
        reports = []
        for result in res_iter:
            if result is not None:
                completed += 1
                reports.append(result)

                if ui_progress is not None:
                    ui_progress.progress.value = completed / n_tasks
                    ui_progress.info.value = f" processed <font color='green'>{completed}</font> from <b>{n_tasks}</b>"

            # small timeout before next iteration
            time.sleep(0.05)

        # collect all reports into single dict and store into DB
        all_reps = dict()

        # collect reports
        [all_reps.update(a) for a in reports]

        # clean reports
        cleaned = {k: v for k, v in all_reps.items() if v is not None}
        b_save(self.reports_path, cleaned)

        print('OK')
        return self

    def __repr__(self):
        short_rep = f'{len(self.sims)} ' + model_short_report(
            b_ld(f"reports/{self.project}/{self.model}")
        )
        return f'{self.project} | {self.model} : {short_rep}'


class ProjectModelView:
    def __init__(self, project, paths):
        self.paths = paths
        data = [p.split('/')[1:] for p in paths]
        self.models = defaultdict(lambda: list())
        self.project = project
        for _, sim, mdl in data:
            self.models[mdl].append(sim)

    def models_list(self):
        return sorted(self.models.keys())

    def __getitem__(self, model: Union[str, int]):
        k = self.models_list()[model] if isinstance(model, int) else model
        return ModelView(self.project, k, self.models[k]) if k in self.models else None

    def __truediv__(self, p):
        return self.__getitem__(p)

    def run(self, *args, **kwargs):
        print("Not sure what to run here")
        return self

    def __repr__(self):
        if not self.models:
            return '-[no data]-'

        ml = max(map(len, self.models.keys()))
        reprs = ''
        for i, m in enumerate(self.models_list()):
            v = self.models[m]
            short_rep = model_short_report(b_ld(f"reports/{self.project}/{m}"))
            _n = green(f'[{i}]'.ljust(7))
            reprs += f'{_n} {(blue(m) + ":").ljust(ml + 2)}\t{len(v)} {short_rep}\n'
        return reprs


class OCtrl:
    def projects(self):
        return list(set([p.split('/')[1] for p in b_ls('runs/.*')]))

    def select(self, project: str, model: str = '.*'):
        if any([p in project for p in ['$', '*', '?', '/', "'", '^', '&', '\\']]):
            raise ValueError("Project name must not contain any regular expressions")

        search_str = f'runs/{project}/.*' + model if model else ''
        recs = b_ls(search_str)
        return ProjectModelView(project, recs) if len(recs) > 0 else None

    def report(self, *args, **kwargs):
        return Operation('report', *args, **kwargs)

    def delete(self, *args, **kwargs):
        return Operation('delete')

    def run(self, start, broker, exchage, **kwargs):
        return Operation('run', start, broker, exchage, **kwargs)

    def stats(self, init_cash, force_calc=False, max_n_procs=100, min_per_proc=20, performance_statistics_period=365):
        return Operation('stats', init_cash, force_calc=force_calc, max_n_procs=max_n_procs, min_per_proc=min_per_proc,
                         performance_statistics_period=performance_statistics_period)

    def __truediv__(self, p):
        return self.select(p)

    def __repr__(self):
        return str(self.projects())
