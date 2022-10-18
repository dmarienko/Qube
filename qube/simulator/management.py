"""
   Simulation management utils
"""
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd

from qube.simulator.core import DB_SIMULATION_RESULTS
from qube.simulator.multisim import MultiResults
from qube.utils.nb_functions import z_load, z_save, z_ls, z_del, z_ld
from qube.utils.ui_utils import red, green
from qube.utils.utils import mstruct, runtime_env

if runtime_env() == 'notebook':
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class SimulationRunData:
    """
    Represents simulation's runs data
    """

    def __init__(self, prj: str, run_id: str, prj_data: dict, host: str = None):
        self.p = prj_data
        self.prj = prj
        self.run_id = run_id
        self.host = host

    def results(self) -> MultiResults:
        """
        Return results as MultiResult class
        """
        simres = []
        for k, r in tqdm(self.p.items()):
            sd = z_ld(r.path, host=self.host, dbname=DB_SIMULATION_RESULTS)
            if sd:
                simres.append(sd.result)
        return MultiResults(simres, self.prj, '', '', '', '')

    def calc_performance(self, init_cash, account_transactions=True, force_calc=False):
        """
        Calculate performance metrics for simulations (if no perf data)
        """
        for k, r in tqdm(self.p.items()):
            sd = z_load(r.path, host=self.host)['data']
            if sd:
                try:
                    if force_calc or 'performance' not in dir(sd):
                        prf = sd.result.performance(init_cash, account_transactions=account_transactions)
                        eqt = sd.result.equity(account_transactions=account_transactions)
                        sd.performance = mstruct(
                            gain=eqt[-1] - eqt[0],
                            cagr=prf.cagr,
                            sharpe=prf.sharpe,
                            qr=prf.qr,
                            sortino=prf.sortino,
                            calmar=prf.calmar,
                            drawdown_pct=prf.drawdown_pct,
                            drawdown_pct_on_init_bp=prf.drawdown_pct_on_init_bp,
                            mdd_usd=prf.mdd_usd,
                            mdd_start=prf.mdd_start,
                            mdd_peak=prf.mdd_peak,
                            mdd_recover=prf.mdd_recover,
                            annual_volatility=prf.annual_volatility,
                            dd_stat=prf.dd_stat,
                            tail_ratio=prf.tail_ratio,
                            stability=prf.stability,
                            var=prf.var,
                            n_execs=len(sd.result.executions) if sd.result.executions is not None else 0,
                            mean_return=prf.mean_return,
                            commissions=prf.broker_commissions,
                        )
                        z_save(r.path, sd, host=self.host, dbname=DB_SIMULATION_RESULTS)
                except Exception as exc:
                    print(f'>>> Exception in processing {r.path}: {str(exc)}')

    def load(self, t_id) -> mstruct:
        return z_ld(f'runs/{self.prj}/{t_id}/{self.run_id}', host=self.host, dbname=DB_SIMULATION_RESULTS)

    def __getitem__(self, t_id):
        return self.load(t_id)

    def delete_by_sharpe_gain_threshold(self, min_sharpe, min_gain=-np.inf) -> Tuple[int, List[str]]:
        """
        Remove all stored simulations with sharpe and PnL gain less then provided thresholds from database

        :param min_sharpe: minimal sharpe ratio for keep this simulation in DB
        :param min_gain: minimal PnL gain (in USD) for keep this simulation in DB (-inf by default)
        :return n, list: number of deleted and list of retained
        """
        n_deleted = 0
        retained = []
        for k, r in tqdm(self.p.items()):
            sd = z_load(r.path, host=self.host)['data']
            if sd:
                try:
                    if 'performance' in dir(sd) and \
                            (sd.performance.sharpe < min_sharpe or sd.performance.gain < min_gain):
                        n_deleted += 1
                        z_del(r.path)
                    else:
                        retained.append(r.path)
                except Exception as exc:
                    print(f'>>> Exception in processing {r.path}: {str(exc)}')
        return n_deleted, retained

    def comparison_report(self) -> pd.DataFrame:
        """
        Returns comparison report for all simulations in this project/run_id
        """
        report = {}
        for k, r in tqdm(self.p.items()):
            sd = z_ld(r.path, host=self.host, dbname=DB_SIMULATION_RESULTS)
            if sd is None or 'performance' not in dir(sd): continue
            p = sd.performance

            # params = ','.join([f'{k}={repr(v)}' for k, v in sd.task_args[1].items()])
            pps = sd.task_args[1] if isinstance(sd.task_args, list) else sd.task_args
            try:
                report[sd.task] = {
                    'sharpe': p.sharpe,
                    'gain': p.gain,
                    'cagr': 100 * p.cagr,
                    'dd_usd': p.mdd_usd,
                    'dd_pct': p.drawdown_pct,
                    'nexecs': p.n_execs,
                    'qr': p.qr,
                    'comm': p.commissions,
                    **pps
                }
            except KeyError as k:
                print(f"  [{sd.task}] >> error getting performance metric: '{k}'")

        return pd.DataFrame.from_dict(report, orient='index').sort_values('sharpe', ascending=False)

    def delete(self):
        """
        Deletes all simulationa data
        """
        for k, r in tqdm(self.p.items()):
            z_del(r.path)
        self.p = []


class SimulationsManager:
    """
    Simulation runs management
    """

    def __init__(self, host=None):
        self.host = host
        self.p = self._collect_data(host)

    def _collect_data(self, host=None):
        prj_data = defaultdict(lambda: defaultdict(dict))

        recs = z_ls('runs/', host=host)
        for p, s, r, path in [s.split('/')[1:] + [s, ] for s in recs]:
            sinf = s.split('.')
            ra = prj_data[p][r]
            k = int(sinf[1])
            ra[k] = mstruct(runid=r, symbol=sinf[2], path=path)
            ra = sorted(ra.items(), key=lambda kv: kv[0])
            prj_data[p][r] = dict(ra)

        return prj_data

    def projects(self) -> List[str]:
        """
        Returns list of all projects
        """
        return list(self.p.keys())

    def runs_for(self, prj) -> List[str]:
        """
        Returns all runs for project
        """
        return list(self.p[prj].keys())

    def run_data(self, prj, run_id) -> SimulationRunData:
        """
        Returns runs data for project
        """
        if prj not in self.p:
            raise ValueError(f"Can't find project {prj} !")

        if run_id not in self.p[prj]:
            raise ValueError(f"Can't find run id {run_id} in project {prj} !")

        return SimulationRunData(prj, run_id, self.p[prj].get(run_id))

    def ls(self):
        ml = 0
        for p in self.projects():
            ml = max(max(map(len, self.runs_for(p))), ml)

        for p in self.projects():
            print(red(p))
            for r in self.runs_for(p):
                print(f'  {(green(r) + ":").ljust(ml + 10)}\t{len(self.p[p][r])}')


def ls_simulations(host=None):
    """
    List all stored simulations on host
    """
    return SimulationsManager(host)
