import getpass
import logging
import os

import sys
import time
from collections import defaultdict
from typing import Union, List, Dict

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

if os.name == 'nt':
    try:
        import multiprocess as mp
    except ImportError:
        print(" >>> For support multiprocessing under Windows 'multiprocess' module should be installed")
else:
    import multiprocessing as mp

from qube.booster.simctrl import OCtrl
from qube.booster.utils import (
    restore_configs_from_records, rm_sim_data, rm_blend_data, check_model_already_exists, class_import, calculate_weights, short_performace_report,
    average_trades_per_period, b_ld, b_ls, b_del, b_save, BOOSTER_DB
)
from qube.datasource.loaders import load_data, get_data_time_range
from qube.datasource.DataSource import DataSource
from qube.quantitative.tools import srows, scols
from qube.simulator import simulation, Market
from qube.simulator.multiproc import run_tasks
from qube.simulator.utils import permutate_params
from qube.utils.ui_utils import red, green, yellow, blue
from qube.utils.utils import mstruct, dict2struct

_CONFIG_KEYS = ['instrument', 'broker', 'capital', 'project']
_OPTIMIZER_KEYS = ['task', 'spreads']
_MODEL_KEYS = ['optimizer', 'parameters']
_BLENDER_KEYS = ['blending_model', 'weighting', 'models']
_PORTFOLIO_KEYS = ['task', 'parameters', ]
_TUNING_KEYS = ['method', 'risk_parameter_name']
DEFAULT_STATS_PERIOD = 365

LOGGER_FOLDER = '/var/log/booster/'
LOG_FORMAT = "%(asctime)s [%(levelname)s] - %(message)s"


# some helpers on mstruct
def _get(self, n, default=None):
    return getattr(self, n) if hasattr(self, n) else default


mstruct._get_ = _get
_amend = lambda x, **kwargs: {**x, **kwargs}

del _get


class _NoLog:
    def info(self, *args):
        pass

    def error(self, *args):
        pass

    def warning(self, *args):
        pass

    def warn(self, *args):
        pass


class Booster:
    """
    = Optimizations tasks runner class
    """
    _cfg: dict = None

    def __init__(self, config_file, reload_config=True, log=True):
        # configure logger
        self._logger = _NoLog()
        if log:
            # check if folder exists
            if not os.path.exists(LOGGER_FOLDER):
                os.makedirs(LOGGER_FOLDER)

            self._logger = logging.getLogger('Booster')
            self._logger.setLevel(logging.INFO)

            log_out = logging.StreamHandler(sys.stdout)
            log_out.setFormatter(logging.Formatter(LOG_FORMAT))
            self._logger.addHandler(log_out)

            log_file = logging.FileHandler(f"{LOGGER_FOLDER}/booster-{getpass.getuser()}.log")
            log_file.setFormatter(logging.Formatter(LOG_FORMAT))
            self._logger.addHandler(log_file)

        self._reload_config = reload_config
        self._config_file = config_file
        self._load_config_from_file()

    def _ld_data(self, entry_id):
        cfg = self.load_entry_config(entry_id)
        # todo: use another DB (dbname='booster') !!!
        return b_ld(f"blends/{cfg.project}/{entry_id}")

    def _ld_simulation_stats_data(self, project, sim_id, entry_id):
        # todo: use another DB (dbname='booster') !!!
        return b_ld(f"stats/{project}/{sim_id}/{entry_id}")

    def _ld_simulation_run_data(self, project, sim_id, entry_id):
        # todo: use another DB (dbname='booster') !!!
        return b_ld(f"runs/{project}/{sim_id}/{entry_id}")

    def _save_data(self, entry_id, data):
        cfg = self.load_entry_config(entry_id)
        return b_save(f"blends/{cfg.project}/{entry_id}", data)

    def _save_portfolio_task_report(self, entry_id: str, data: dict, 
                                    config: mstruct, portfolio_config: dict):
        path = f"portfolios/{config.project}/{entry_id}"

        cfg_to_save = {
            'config': config.to_dict(),
            'portfolio': portfolio_config
        }
        
        b_save(f'portfolios/_index/{config.project}/{entry_id}', {
            'path': path,
            'experiment': entry_id,
            'description': data.get('description', ''),
            'timestamp': data.get('timestamp', ''),
            'status': data.get('status', '???'),
            # 2023-03-05: store experiment configuration
            entry_id: cfg_to_save,
        })
        return b_save(path, data)

    def _preproc(self, xs):
        """
        Preprocessing config because some values may be np.nan, None etc
        """
        _repl = {'np.nan': np.nan, 'None': None, 'np.inf': np.inf, '-np.inf': -np.inf}
        if isinstance(xs, list):
            return [self._preproc(x) for x in xs]
        elif isinstance(xs, dict):
            return {k: self._preproc(v) for k, v in xs.items()}
        return _repl.get(xs, xs)

    def _load_config_from_file(self):
        # try to restore config from existing records
        if self._config_file is None:
            self._cfg = restore_configs_from_records()
        else:
            # read config yaml file
            with open(self._config_file, 'r') as stream:
                try:
                    self._cfg = self._preproc(yaml.safe_load(stream))
                except yaml.YAMLError as exc:
                    raise ValueError(exc)

    def _check_mandatory_keys(self, c: dict, keys: list):
        for k in keys:
            if k not in c:
                raise ValueError(f"Mandatory key '{k}' not found in config {c}")
        return c

    def load_entry_config(self, entry_id, skip_reload_config=False) -> mstruct:
        """
        Load configuration. It reloads data from file if _reload_config is set and skip_reload_config
        is not set.
        """
        if self._reload_config and not skip_reload_config:
            self._load_config_from_file()

        if entry_id not in self._cfg:
            raise ValueError(f"Can't find entry '{entry_id}' in config")

        entry = self._cfg[entry_id]
        if 'config' not in entry:
            raise ValueError(f"Entry '{entry_id}' doesn't contain any config section !")

        return dict2struct(self._check_mandatory_keys(entry['config'], _CONFIG_KEYS))

    def get_optimizations(self, entry_id):
        """
        Get optimizations section
        """
        config = self.load_entry_config(entry_id)
        opts = self._cfg[entry_id].get('optimizations')
        if opts is None or not opts:
            raise ValueError(f"Can't find optimizations section in '{entry_id}'")
        return opts

    def task_optimizations(self, entry_id):
        """
        Run optimization task
        """
        self._logger.info(f" - - - - - - -< Run optimization for {entry_id} >- - - - - - -")

        # general config
        config = self.load_entry_config(entry_id)

        # optimizations
        opts = self.get_optimizations(entry_id)

        # check models
        for m_name, m_cfg in opts.items():
            self._run_optimization(entry_id, config, m_name, self._check_mandatory_keys(m_cfg, _MODEL_KEYS))

    def task_clean(self, entry_id, **clean_conds):
        self._logger.info(f" - - - - - - -< Cleaning {entry_id} >- - - - - - -")

        config = self.load_entry_config(entry_id)
        opts = self._cfg[entry_id].get('optimizations')
        if opts is None or not opts:
            raise ValueError(f"Can't find 'optimizations' section in '{entry_id}'")

        projects_simulations = OCtrl() / config.project
        for m_name, _ in opts.items():
            sims = projects_simulations / f"{entry_id}_{m_name}"
            print('cleaning: ', sims, ' ... ')
            sims.clean(**clean_conds)

            # delete report
            if hasattr(sims, 'reports_path') and sims.reports_path:
                b_del(sims.reports_path)

            print('done')

    def task_delete(self, entry_id, target=None):
        """
        Run clean task
        """
        self._logger.info(f" - - - - - - -< Deleting {entry_id} >- - - - - - -")

        # general config
        config = self.load_entry_config(entry_id)

        # - check if this is portfolio task
        if 'portfolio' in self._cfg[entry_id]:
            self.delete_previous_portfolio_runs(config.project, entry_id, drop_index=True)
            return

        what_to_clean = target if target is not None else self._cfg[entry_id].get('clean')
        if what_to_clean is None:
            raise ValueError(f"Can't find 'clean' section in '{entry_id}'")

        opts = self._cfg[entry_id].get('optimizations')
        if opts is None or not opts:
            raise ValueError(f"Can't find 'optimizations' section in '{entry_id}'")

        # check models
        for m_name, m_cfg in opts.items():
            if m_name in what_to_clean or what_to_clean == 'all':
                rm_sim_data(config.project, f'{entry_id}_{m_name}')

        # delete bleneded models
        if what_to_clean == 'all':
            rm_blend_data(config.project, entry_id)

    def _get_best_run(self, simulations, criterion, ordering=None) -> Union[mstruct, None]:
        best_runs = simulations.report().query(criterion)

        if len(best_runs.index) == 0:
            return None

        # sort by ordering if required
        if ordering:
            best_runs = best_runs.sort_values(ordering[0], ascending=ordering[1])

        sim_name = best_runs.index[0]
        entry = simulations / sim_name
        return mstruct(name=best_runs.index[0],
                       parameters=entry.data.task_args[1],
                       result=entry.data.result,
                       performance=entry.stats)

    def show(self, entry_id: str):
        """
        Displays info about entry
        """
        config = self.load_entry_config(entry_id)
        b_cfg = self._cfg[entry_id].get('blender')
        opts = self._cfg[entry_id].get('optimizations')
        portfolio = self._cfg[entry_id].get('portfolio')
        print(f"-- ({green(entry_id)}) --")
        for k, v in config.to_dict().items():
            print(f"  {blue(k)}:\t{v}")

        # show portfolio config if presented
        if portfolio and isinstance(portfolio, dict):
            print(f"\n - Portfolio config -")
            for k, v in portfolio.items():
                print(f"  {green(k)}:\t{v}")

        # more info about this project
        projects_simulations = OCtrl() / config.project

        if opts:
            print(f"\n - Optimizations -")
            for k, v in opts.items():
                ocfg = v['optimizer']
                conditions = eval(v['conditions']) if 'conditions' in v else None
                parameters = permutate_params(v["parameters"], conditions=conditions)
                print(f"  {green(k)}: {len(parameters)} runs of {ocfg['task']}")
                if projects_simulations:
                    print("\t" + str(projects_simulations / f"{entry_id}_{k}"))

        if b_cfg:
            print(f"\n - Blender -")
            for k, v in b_cfg.items():
                vs = v
                if isinstance(v, dict):
                    vs = '\n\t\t'.join([f"{green(_n)}:{_v}" for _n, _v in v.items()])
                print(f"  {blue(k)}:\t{vs}")

            # blended model report 
            b_rep = self.get_blend_report(entry_id)
            if b_rep is not None:
                print(yellow('\n == Blended model parameters =='))
                reps = '\n'.join([f'\t{x}' for x in b_rep.T.to_string(header=True).split('\n')])
                print(green(reps))

    def get_blend_report(self, entry_id, weight_capital=True):
        """
        Show blending report
        """
        cfg = self.load_entry_config(entry_id)
        b_report = self._ld_data(entry_id)
        if b_report is None:
            print(red(f"No blended report for {cfg.project}/{entry_id}"))
            return None
        else:
            mdls = {}
            for i, (m, v) in enumerate(b_report['models'].items()):
                w = b_report['weights'].get(m, 1)
                cap = {}

                # if we need to see 'weighted' capital 
                if weight_capital:
                    weighted_cap_value = w * cfg.capital if hasattr(cfg, 'capital') else v.parameters.get('capital', 0)
                    cap = {'capital': round(weighted_cap_value, 2)}

                mdls[m] = {**{'Weight': w}, **v.parameters, **cap}

        return pd.DataFrame.from_dict(mdls, orient='index')

    def get_all_entries(self) -> List[str]:
        """
        Returns all entries
        """
        if self._reload_config:
            self._load_config_from_file()

        entrs = []
        for e, v in self._cfg.items():
            if isinstance(v, dict) and 'config' in v and ('optimizations' in v or 'portfolio' in v):
                entrs.append(e)

        return entrs

    def load_blended_results(self, entry_id):
        b_data = self._ld_data(entry_id)
        return b_data

    def ls(self):
        """
        List all entries
        """
        for i, e in enumerate(sorted(self.get_all_entries())):
            print(f"[{green(str(i))}]\t{red(e)}")

    def task_run_oos(self, entry_id: str, capital=None,
                     parameters=None, insample_only=True, blender_parameters=None, replace=False):
        """
        Run blended model out of sample
        """
        self._logger.info(f" - - - - - - -< Running OOS for {entry_id}  >- - - - - - -")

        if blender_parameters is None:
            blender_parameters = dict()

        if parameters is None:
            parameters = dict()

        # general config
        config = self.load_entry_config(entry_id)
        capital = capital if capital is not None else config.capital

        b_cfg = self._cfg[entry_id].get('blender')
        if b_cfg is None or not b_cfg:
            raise ValueError(f"Can't find 'blender' section in '{entry_id}'")

        # load blended information from DB
        blends = b_ld(f"blends/{config.project}/{entry_id}")
        if blends is None:
            raise ValueError(f"Can't find blended results for '{entry_id}' probably you need to run 'blend' task first")

        # models weights
        optimal_weights = blends['weights']

        # instance of blending class
        blender_model_class = class_import(blends.get('blending_model'))

        # blender parameters
        blender_parameters = b_cfg.get('blender_parameters', blender_parameters)

        # make blended model
        _mdls = []
        for m, v in blends['models'].items():
            # import task's class if needs
            _model_class = class_import(v.task_class)

            # model parameters with adjusted capital
            _model_params = _amend(v.parameters, capital=optimal_weights[m] * capital, **parameters.get(m, dict()))
            try:
                _mdls.append(_model_class(**_model_params))
            except Exception as err:
                import traceback
                trace = traceback.format_exc()
                self._logger.error(f"error on instantiating object of '{_model_class}': {str(err)}\n{trace}")
                raise err

        # blended model for final backtest
        final_blended_model = blender_model_class(*_mdls, **blender_parameters)
        self._logger.info('>>> Blended model:\n' + green(repr(final_blended_model)))

        # finally run backtest
        start_date = blends.get('insample', config._get_('end_date')) if insample_only else None
        market_data = load_data(config.instrument)
        self._logger.info(f'>>> Running blended model from {start_date} ... ')
        backtest = simulation({f'Blended {entry_id}': final_blended_model}, market_data.ticks(),
                              config.broker, config.project, start=start_date)

        # replace information in database
        if replace:
            self._logger.info(f'>>> Updating recent backtest results for {entry_id} ... ')

            # recalculating results
            blended_result = backtest.results[0]
            performance = blended_result.performance(capital, account_transactions=True,
                                                     performance_statistics_period=DEFAULT_STATS_PERIOD)

            # update in DB
            blends['timestamp'] = pd.Timestamp.now()
            blends['backtest'] = blended_result
            blends['performance'] = performance.to_dict()
            self._save_data(entry_id, blends)

        return backtest

    def task_blender(self, entry_id: str, capital=None, criterions=dict(), replace=True,
                     force_performace_calculations=False, skip_tuning=False):
        """
        Blending task
        """
        self._logger.info(f" - - - - - - -< Start blending for {entry_id}  >- - - - - - -")

        # general config
        config = self.load_entry_config(entry_id)
        capital = capital if capital is not None else config.capital

        b_cfg = self._cfg[entry_id].get('blender')
        if b_cfg is None or not b_cfg:
            raise ValueError(f"Can't find 'blender' section in '{entry_id}'")

        # check if all mandatory fields are here
        self._check_mandatory_keys(b_cfg, _BLENDER_KEYS)

        # we need optimizations config
        opts = self._cfg[entry_id].get('optimizations')
        if opts is None or not opts:
            raise ValueError(f"Can't find optimizations section in '{entry_id}' needed for blended model")

        # find best models for each model
        best_models_parameters = self.select_models(entry_id, criterions=criterions, capital=capital,
                                                    force_performace_calculations=force_performace_calculations)

        # calculate optimal weights 
        weight_method = b_cfg['weighting']
        optimal_weights = calculate_weights(best_models_parameters, capital, weight_method)

        # make blended model
        _mdls = []

        # here configs for models for OOS range
        _oos_mdls = {}
        _mdls_weighted_cap = {}

        for m, v in best_models_parameters.items():
            # import task's class if needs
            _model_class = class_import(v.task_class)

            # model parameters with adjusted capital (model should exploite 'capital' parameter)
            _model_params = _amend(v.parameters, capital=optimal_weights[m] * capital)
            _mdls.append(_model_class(**_model_params))
            _oos_mdls[m] = _model_class(**_model_params)
            _mdls_weighted_cap[m] = optimal_weights[m] * capital

        # instance of blending class
        blender_model_class = class_import(b_cfg.get('blending_model'))

        # additional parameters for blender
        blender_parameters = b_cfg.get('blender_parameters', dict())

        # blended model for final backtest
        final_blended_model = blender_model_class(*_mdls, **blender_parameters)

        _bp_str = f"\n\tparameters: {blue(str(blender_parameters))}" if blender_parameters else ""
        self._logger.info(f'>>> Blended {entry_id} model:\n' + green(repr(final_blended_model)) + _bp_str)

        # market data for instrument
        market_data = load_data(config.instrument)

        # 05-Feb-2022: check if tuning step is configured for this blending step
        if not skip_tuning and 'tuning' in b_cfg:
            tuned_params = self._check_mandatory_keys(b_cfg['tuning'], _TUNING_KEYS)
            self._logger.info(f'>>> Run tuning of {entry_id} blended model insample: {str(tuned_params)}')

            # run blended model insample
            perf_ins = self._run_model_insample(capital, config, final_blended_model, market_data)

            # insample dd of blended model
            ins_dd = perf_ins.drawdown_pct

            # tune risk parameter  
            risk_param = tuned_params['risk_parameter_name']
            factor = min(tuned_params['desired_dd_pct'] / ins_dd, 100)
            self._logger.info(f'>>> Tuned {risk_param} parameter factor: {factor:.3f} | insample DD: {ins_dd: .3f}')

            _mdls1 = []
            _oos_mdls = {}
            _mdls_weighted_cap = {}
            for m, v in best_models_parameters.items():
                # import task's class if needs
                _model_class = class_import(v.task_class)

                # model parameters with adjusted capital (model should exploite 'capital' parameter)
                v.parameters[risk_param] = factor * v.parameters[risk_param]
                _model_params = _amend(v.parameters, capital=optimal_weights[m] * capital)

                # blended model constituents
                _mdls1.append(_model_class(**_model_params))

                # this is for out of sample run (to see separated models report)
                _oos_mdls[m] = _model_class(**_model_params)
                _mdls_weighted_cap[m] = optimal_weights[m] * capital

            # updated blended model for final backtest
            final_blended_model = blender_model_class(*_mdls1, **blender_parameters)
            self._logger.info(f'>>> Tuned {entry_id} model:\n' + green(repr(final_blended_model)))

        # backtest blended model
        start_date = config._get_('start_date')
        self._logger.info(f'>>> Running {entry_id} blended model from {start_date} ... ')
        backtest = simulation(
            {
                # final blended model
                f'Blended {entry_id}': final_blended_model,

                # constituents with adjusted capital
                **_oos_mdls,
            },
            market_data.ticks(),
            config.broker, config.project, start=start_date
        )

        # blended model result for in_sample + out_of_sample
        blended_result = backtest.results[0]

        # constituents model results for in_sample + out_of_sample
        self._logger.info(f'>>> Recalculating performance for constituents for {entry_id} ...')
        for i, (m, md) in enumerate(best_models_parameters.items()):
            best_models_parameters[m].result = backtest.results[i + 1]
            best_models_parameters[m].performance = short_performace_report(
                backtest.results[i + 1], _mdls_weighted_cap[m], account_transactions=True,
                performance_statistics_period=DEFAULT_STATS_PERIOD
            )

        # one temporary cleanup: we need to drop signals stats for some old runs (it contains class sensitive data !!!)
        for m, md in best_models_parameters.items():
            for _, ts in md.result.trackers_stat.items():
                ts['signals'] = {}

        # put all results together
        performance = blended_result.performance(capital, account_transactions=True,
                                                 performance_statistics_period=DEFAULT_STATS_PERIOD)

        # add average number of trades per month
        performance.avg_trades_month = average_trades_per_period(blended_result.executions, 'BM')

        output = mstruct(
            id=entry_id,
            timestamp=pd.Timestamp.now(),
            config=config,
            blending_model=b_cfg.get('blending_model'),
            insample=config._get_('end_date'),
            models=best_models_parameters,
            start_date=start_date,
            weights=optimal_weights,
            weight_method=weight_method,
            backtest=blended_result,
            performance=performance
        )

        # store information into database
        if replace:
            self._save_data(entry_id, output.to_dict())

        return output

    def _run_model_insample(self, capital, config, model, market_data) -> mstruct:
        """
        Runs model on insample period of data.
        
        :param capital: capital for blended model testing
        :param config: entry config 
        :param model: model's 
        """
        start_date, end_date = config._get_('start_date'), config._get_('end_date')
        self._logger.info(f'>>> Backtest {green(repr(model))} on {start_date} - {end_date} for tuning')
        backtest = simulation({f'Tuning': model},
                              market_data.ticks(), config.broker, config.project,
                              start=start_date, stop=end_date)
        perf = backtest.results[0].performance(capital, account_transactions=True,
                                               performance_statistics_period=DEFAULT_STATS_PERIOD)
        return perf

    def _run_optimization(self, entry_id: str, config: mstruct, model_name: str, model_cfg: dict,
                          run=True, sync_mode=True) -> bool:
        """
        Main optimizations running method
        """
        cfg_key = f'{entry_id}_{model_name}'
        exchange, symbol = config.instrument.split(':')
        simulator_timeframe = config._get_('simulator_timeframe', None)
        estimator_composer = config._get_('estimator_composer', 'single')
        estimator_used_data = config._get_('estimator_data', 'close')
        m_optimizer = model_cfg['optimizer']
        broker = m_optimizer.get('broker', config.broker)
        max_cpus = min(m_optimizer.get('max_cpus', mp.cpu_count() - 1), mp.cpu_count() - 1)
        start_date = m_optimizer.get('start_date', config._get_('start_date'))
        end_date = m_optimizer.get('end_date', config._get_('end_date'))
        fit_end_date = m_optimizer.get('fit_end_date', config._get_('fit_end_date'))
        task_clazz = class_import(m_optimizer["task"])

        # number of simulation to start from 
        start_sim_number = m_optimizer.get('start_sim_number', 0)

        if start_sim_number == 0:
            exist_models_len = check_model_already_exists(config.project, cfg_key)
            if exist_models_len > 0:
                if config._get_('delete_previous_results'):
                    rm_sim_data(config.project, cfg_key)
                else:
                    self._logger.warning(
                        f"{config.project} / {cfg_key} already has {exist_models_len} stored simulations, remove them first if need to restart !")
                    return False

        # get available data ranges
        data_start_date, data_end_date = get_data_time_range(config.instrument, None)
        self._logger.info(f"Available data for {symbol} : {data_start_date} / {data_end_date}")

        # start/end dates
        start_date = start_date if start_date is not None else data_start_date
        end_date = end_date if end_date is not None else data_end_date
        if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
            print(
                red("ERROR:") + f" {config.project} / {cfg_key} start date {green(start_date)} is after {green(end_date)} !")
            self._logger.error(
                red("ERROR:") + f" {config.project} / {cfg_key} start date {green(start_date)} is after {green(end_date)} !")
            return False

            # check conditions section
        conditions = None
        if 'conditions' in model_cfg:
            conditions = eval(model_cfg['conditions'])

        # check parameters permutations
        parameters = permutate_params(model_cfg["parameters"], conditions=conditions)

        def run_fn():
            market = Market(
                broker, start_date, end_date, fit_end_date, m_optimizer.get('spreads', 0),
                data_loader=load_data,
                test_timeframe=simulator_timeframe,
                estimator_portfolio_composer=estimator_composer,
                estimator_used_data=estimator_used_data
            )
            run_tasks(config.project,
                      market.new_simulations_set(
                          config.instrument, task_clazz, parameters[start_sim_number:],
                          simulation_id_start=start_sim_number, storage_db=BOOSTER_DB
                      ),
                      max_cpus=max_cpus,
                      max_tasks_per_proc=m_optimizer.get('max_tasks_per_proc', 10),
                      cleanup=1,
                      superseded_run_id=cfg_key)

        self._logger.info(
            f'start {len(parameters)} runs for {red(entry_id)}/{green(model_name)} {broker} | {config.instrument} from {start_date} to {end_date}')

        if run:
            f_id = f"proc_{str(time.time()).replace('.', '_')}"
            task = mp.Process(target=run_fn, name=f_id)
            # run_fn()
            self._logger.info(f">>> {green(cfg_key)} -> {red(f_id)}")
            task.start()

            if sync_mode:
                self._logger.info(f">>> waiting for finishing {red(f_id)}")
                task.join()
                self._logger.info(f'{green(cfg_key)} / process {red(f_id)}  finished')

        return True

    def task_stats(self, entry_id: str, capital=None):
        """
        Calculate statistics report for entry and store it to DB
        """
        # general config
        config = self.load_entry_config(entry_id)
        capital = capital if capital is not None else config.capital

        # we need optimizations config
        opts = self._cfg[entry_id].get('optimizations')
        if opts is None or not opts:
            raise ValueError(f"Can't find optimizations section in '{entry_id}' needed for blended model")

        # optimization data
        pobj = OCtrl()
        projects_simulations = pobj / config.project
        if projects_simulations is None:
            self._logger.error(
                f"Can't find any optimizations for '{entry_id}' in {config.project} project. Try to start 'run' command first !")
            return

        # calculate statistics for every entry in optimization section
        for m in opts:
            cfg_key = f'{entry_id}_{m}'
            # simulations list
            sims = projects_simulations / cfg_key

            self._logger.info(f"Calculating performance stats for {config.project} | {entry_id} / {m} ...")
            sims // pobj.stats(capital, force_calc=True, performance_statistics_period=DEFAULT_STATS_PERIOD)
        self._logger.info("Done")

    def select_models(self, entry_id: str,
                      criterions: Dict[str, Dict[str, str]],
                      capital=None,
                      force_performace_calculations=False) -> Dict[str, mstruct]:
        """
        Select models based on defined criterions
        
        Example:
        
        b = Booster('/var/appliedalpha/booster/GPTR/binance_blended.yml')
        
        mdls = b.select_models('DOT15M', criterions={
            'DYNFBREAK': { 'criterion': 'dd_pct < 25', 'ordering': [['sharpe', 'nexecs'], [False, False]] },
            'DYNTOUCH': { 'criterion': 'dd_pct < 25', 'ordering': [['sharpe', 'nexecs'], [False, False]] }
            }
        )
        
        """
        # general config
        config = self.load_entry_config(entry_id)
        capital = capital if capital is not None else config.capital

        # we need optimizations config
        opts = self._cfg[entry_id].get('optimizations')
        if opts is None or not opts:
            raise ValueError(f"Can't find optimizations section in '{entry_id}' needed for blended model")

        # blender config
        b_cfg = self._cfg[entry_id].get('blender')

        # optimization data
        pobj = OCtrl()
        projects_simulations = pobj / config.project

        best_models_parameters = {}
        for m in opts:
            cfg_key = f'{entry_id}_{m}'
            criterion = criterions[m] if m in criterions else b_cfg['models'].get(m)
            if criterion is None:
                err = f" >>> Selection criterion for '{m}' / '{entry_id}' is not defined !"
                self._logger.error(err)
                raise ValueError(err)

            self._logger.info(f"{cfg_key} / {criterion}")

            # here we need only optimizer config from model
            m_optimizer = self._check_mandatory_keys(opts[m], _MODEL_KEYS)['optimizer']

            # simulations list
            sims = projects_simulations / cfg_key

            # --- run statistics calculations if requested or needed
            if force_performace_calculations or b_cfg.get('recalculate', False) or not hasattr(
                    sims, 'reports_path'
            ) or not b_ld(sims.reports_path):
                self._logger.info(f"Calculating performance stats for {m} ...")
                sims // pobj.stats(capital, force_calc=True, performance_statistics_period=DEFAULT_STATS_PERIOD)

                # --- get best run using criterion
            best_run = self._get_best_run(sims, **criterion)
            if best_run is not None:
                # re calculate performance for specified capital if no stats found
                if best_run.performance is None:
                    self._logger.warning(f"Performance stats is not found for {m} trying to recalculate ...")
                    best_run.performance = best_run.result.performance(capital, account_transactions=True,
                                                                       performance_statistics_period=DEFAULT_STATS_PERIOD)

                # we need task's class here for running final backtest
                best_run.task_class = m_optimizer['task']

                # store best run for this model
                best_models_parameters[m] = best_run
                self._logger.info(
                    f"Best result for {red(m)} [{yellow(best_run.name)}]: + {green(str(best_run.parameters))} -> {yellow(best_run.task_class)}\n")

            else:
                self._logger.warning(f"Can't find best model for {m} based on these criterion: {criterion}, skip it !")

        return best_models_parameters

    def delete_previous_portfolio_runs(self, project, entry, drop_index=False):
        # drop runs
        prev_runs = b_ls(f'runs/{project}/.*/{entry}_PORTFOLIO')
        if prev_runs:
            for s in tqdm(prev_runs):
                b_del(s)

        # drop stats
        prev_stats = b_ls(f'stats/{project}/.*/{entry}_PORTFOLIO')
        if prev_stats:
            for s in tqdm(prev_stats):
                b_del(s)

        # drop final combined stats
        b_del(f'portfolios/{project}/{entry}')

        # drop index if requested
        if drop_index:
            b_del(f'portfolios/_index/{project}/{entry}')

    def _create_data_source(self, dsinfo: Union[mstruct, None]) -> Union[DataSource, callable]:
        # TODO: to be removed !!!
        if dsinfo is None or dsinfo == '(load-data)':
            return load_data

        if isinstance(dsinfo, mstruct):
            return DataSource(dsinfo._get_('name'), dsinfo._get_('path'))

    def task_portfolio(self, entry_id: str, run=True, stats=True, capital=None, save_to_storage=True):
        """
        Portfolio estimator task
        """

        # general config
        config = self.load_entry_config(entry_id)
        descr = config._get_('description', '')
        capital = capital if capital is not None else config.capital
        broker = config.broker
        max_cpus = min(config._get_('max_cpus', mp.cpu_count() - 1), mp.cpu_count() - 1)
        start_date = config._get_('start_date')
        sprds = config._get_('spreads', {})

        # end date
        now_time_str = pd.Timestamp('now').strftime('%Y-%m-%d %H:%M:00')
        end_date = config._get_('end_date')
        if not end_date or end_date.lower() == 'now':
            end_date = now_time_str

        # end date
        fit_end_date = config._get_('fit_end_date')
        if not fit_end_date or fit_end_date.lower() == 'now':
            fit_end_date = now_time_str

        symbols = [config.instrument] if isinstance(config.instrument, str) else config.instrument
        total_cap = capital * len(symbols)
        simulator_timeframe = config._get_('simulator_timeframe', None)
        estimator_composer = config._get_('estimator_composer', 'single')
        estimator_used_data = config._get_('estimator_data', 'close')

        # - mode / datasource config
        mode = config._get_('mode', 'each')
        datasource = config._get_('datasource', '(load-data)')

        self._logger.info(f" - - - - - - -< Portfolio backtest for {red(entry_id)} in [{mode}] mode >- - - - - - -")
        self._logger.info(f" - Datasource {datasource} | {start_date} - {end_date}")

        # default key for portfolio task
        cfg_key = f"{entry_id}_PORTFOLIO"

        # - instantiate data loader
        ds = self._create_data_source(datasource)

        # we need portfolio config
        portfolio_config = self._cfg[entry_id].get('portfolio')
        if portfolio_config is None or not portfolio_config:
            raise ValueError(f"Can't find portfolio section in '{entry_id}'")

        # create new entry in db
        experiment_result = {
            'report': None,
            'description': descr,
            'portfolio': dict(),
            'executions': dict(),
            'simulations': dict(),
            'parameters': dict(),  # {set_name : {param: value}}
            'capital_per_run': capital,
            'total_capital': total_cap,
            'timestamp': pd.Timestamp.now().isoformat(),
            'symbols': symbols,
            'status': 'STARTED',
        }
        self._save_portfolio_task_report(entry_id, experiment_result, config, portfolio_config)

        # - check if all required fields are there  
        self._check_mandatory_keys(portfolio_config, _PORTFOLIO_KEYS)

        # - instantiate model
        task_class = class_import(portfolio_config.get('task'))

        # - check instruments
        self._logger.info(f"\tportfolio: {','.join([green(s) for s in symbols])}")

        # - generate permutation for parameters
        parameters = permutate_params(
            portfolio_config["parameters"],
            conditions=eval(portfolio_config['conditions']) if 'conditions' in portfolio_config else None
        )

        # - collect all simulations configs -
        sims = dict()

        # - new 'portfolio' mode backtest
        if mode == 'portfolio':
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            #   PORTFOLIO mode
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            if start_date is None or end_date is None:
                raise ValueError("Both start_date and end_date must be defined in mode='portfolio' !")

            market = Market(
                broker,
                start=start_date, stop=end_date, fit_stop=fit_end_date,
                spreads=sprds if sprds else 0,
                data_loader=ds,
                test_timeframe=simulator_timeframe,
                estimator_portfolio_composer=estimator_composer,
                estimator_used_data=estimator_used_data
            )

            sims = market.new_simulations_set(symbols, task_class, parameters, simulation_id_start=0,
                                              save_to_storage=save_to_storage, storage_db=BOOSTER_DB)
            self._logger.info(f" > {yellow('PORTFOLIO')} : {start_date} / {end_date} -> {len(sims)} runs")

        else:
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            #  EACH (one by one) mode 
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            for symbol in symbols:
                data_start_date, data_end_date = get_data_time_range(symbol, ds)
                market = Market(
                    broker,
                    start=start_date, stop=end_date, fit_stop=fit_end_date,
                    spreads=sprds.get(symbol, 0),
                    data_loader=ds,
                    test_timeframe=simulator_timeframe,
                    estimator_portfolio_composer=estimator_composer,
                    estimator_used_data=estimator_used_data
                )
                simulations = market.new_simulations_set(symbol, task_class, parameters, simulation_id_start=0,
                                                         save_to_storage=save_to_storage, storage_db=BOOSTER_DB)
                self._logger.info(f" > {green(symbol)} : {data_start_date} / {data_end_date} -> {len(simulations)} runs")
                sims = {**sims, **simulations}

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        #  Run backtests
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        if run:
            # cleanup previous runs results if exist
            self.delete_previous_portfolio_runs(config.project, entry_id)

            def run_fn():
                run_tasks(config.project, sims, max_cpus=max_cpus,
                          max_tasks_per_proc=config._get_('max_tasks_per_proc', 10), cleanup=1, superseded_run_id=cfg_key)

            self._logger.info(f'start {len(sims)} runs for {red(entry_id)} @ {broker}')

            # - start it in separate process
            f_id = f"proc_{str(time.time()).replace('.', '_')}"
            task = mp.Process(target=run_fn, name=f_id)
            self._logger.info(f">>> {green(cfg_key)} -> {red(f_id)}")
            task.start()
            self._logger.info(f">>> waiting for finishing {red(f_id)} ... ")
            self._save_portfolio_task_report(entry_id, _amend(experiment_result, status="RUNNING"), config, portfolio_config)
            task.join()
            self._logger.info(f'{green(cfg_key)} / process {red(f_id)}  finished')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        #  Calculate statistics
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        if stats:
            pobj = OCtrl()
            projects_simulations = pobj / config.project
            if projects_simulations is None:
                self._logger.error(
                    f"Can't find any simulations for '{entry_id}' in {config.project} project. Try to start 'run' command first !")
                self._save_portfolio_task_report(
                    entry_id, 
                    _amend(experiment_result, status="Error: Can't find any simulations"),
                    config, portfolio_config
                )
                return

            # simulations results list
            self._logger.info(f"Calculating performance stats for {config.project} | {entry_id} ...")
            projects_simulations / cfg_key // pobj.stats(capital, force_calc=True, performance_statistics_period=DEFAULT_STATS_PERIOD)
            self._logger.info("Done")

        # - after stats is ready let's get final report
        report = {}
        basic = {}
        combined_portfolio = {}
        combined_executions = {}
        simulations_infos = defaultdict(dict)
        sets_parameters = {}
        for i, p in enumerate(parameters):
            p_name = [f'Set{i}']
            if not basic:
                basic = p
            else:
                p_name.extend([f'{_n}={p.get(_n)}' for _n, _v in basic.items() if _v != p.get(_n)])

            # set_name = '|'.join(p_name)
            set_name = f'Set{i}'
            sets_parameters[set_name] = dict(p)

            # add records
            symbol_records = {}
            set_portfolio, set_execs = None, None

            iterated_names = symbols if mode != 'portfolio' else ['(PORTFOLIO)']
            for s in iterated_names:
                # sims_names_by_symbol[s]
                _run = self._ld_simulation_run_data(config.project, f'sim.{i}.{s}', cfg_key)

                if _run is not None and _run.result:
                    set_portfolio = pd.concat((set_portfolio, _run.result.portfolio), axis=1)
                    set_execs = pd.concat((set_execs, _run.result.executions), axis=0)

                _sim_path = f'sim.{i}.{s}'
                simulations_infos[set_name][s] = _sim_path
                _stats = self._ld_simulation_stats_data(config.project, _sim_path, cfg_key)
                _g, _dd, _sh, _nexecs, _qr = np.nan, np.nan, np.nan, np.nan, np.nan
                if _stats:
                    _g, _dd, _sh, _nexecs, _qr = _stats.gain, _stats.drawdown_pct, _stats.sharpe, _stats.n_execs, _stats.qr

                symbol_records[s] = {
                    'Gain': _g,
                    'GainPct': 100 * _g / capital,
                    'Sharpe': _sh,
                    'MaxDD': _dd,
                    'Execs': _nexecs,
                    'QR': _qr,
                }

            # combined portfolio / execs
            # probably it's not worth to save it - huge size in DB !!!
            # - need to make it bit lightweight
            combined_portfolio[set_name] = None  # set_portfolio
            combined_executions[set_name] = None  # set_execs

            # - attach summary
            set_report = pd.DataFrame.from_dict(symbol_records, orient='index')
            total_cap = len(symbols) * capital
            tot_gain = set_report.Gain.sum()
            tot_gain_pct = 100 * tot_gain / total_cap
            tot_execs = set_report.Execs.sum()

            # combine totals
            set_report = srows(
                set_report,

                pd.DataFrame.from_dict({
                    'Gain': {'Total': tot_gain},
                    'GainPct': {'Total': tot_gain_pct},
                    'Sharpe': {'Total': np.nan},
                    'MaxDD': {'Total': np.nan},
                    'Execs': {'Total': tot_execs},
                    'QR': {'Total': np.nan},
                }),

                pd.DataFrame.from_dict({
                    'Gain': {'Average per trade': tot_gain / tot_execs},
                    'GainPct': {'Average per trade': tot_gain_pct / tot_execs},
                    'Sharpe': {'Average per trade': np.nan},
                    'MaxDD': {'Average per trade': np.nan},
                    'Execs': {'Average per trade': np.nan},
                    'QR': {'Average per trade': np.nan},
                }),
                sort=0
            )
            # return set_report

            report[set_name] = set_report
        # return report 
        final_report = scols(*report.values(), keys=report.keys())
        experiment_result = _amend(
            experiment_result,
            report=final_report,
            portfolio=combined_portfolio,
            executions=combined_executions,
            simulations=dict(simulations_infos),
            timestamp=pd.Timestamp.now().isoformat(),
            parameters=sets_parameters,
            status='FINISHED'
        )
        self._save_portfolio_task_report(entry_id, experiment_result, config, portfolio_config)
        self._logger.info(f"Portfolio task for {config.project} | {entry_id} is finished")
        return experiment_result
