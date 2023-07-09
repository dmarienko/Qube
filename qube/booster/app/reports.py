import airspeed
import numpy as np
import pandas as pd

from qube.booster.core import Booster
from qube.booster.utils import b_ld
from qube.portfolio.drawdown import absmaxdd
from qube.portfolio.performance import (
    portfolio_stats, combine_portfolios, sharpe_ratio, cagr, sortino_ratio, calmar_ratio, qr
)
from qube.portfolio.reports import tearsheet
from qube.quantitative.tools import scols, srows
from qube.simulator.multiproc import RunningInfoManager
from qube.simulator.multisim import MultiResults
from qube.simulator.utils import __instantiate_simulated_broker as inst_sim_brok
from qube.utils.utils import mstruct

PERFORMANCE_PERIOD = 365


def get_progress_for(experiment: str):
    """
    Tries to get progress for experiment by it's name
    """
    rinf = RunningInfoManager()
    runs = rinf.list_runs()
    for r in runs:
        r_info = rinf.get_id_info(r)
        if r_info:
            name = r_info.get('name', '???')
            if name == experiment:
                _p = int(r_info.get('progress', '-1'))
                _t = int(r_info.get('total', '-1'))
                return 100.0 * _p / _t, _t, r_info.get('failed', '0')

    return 0, 0


class BoosterProgressReport:
    """
    Optimization progress reporter
    """

    def __init__(self, config: str,
                 details_url_generator=lambda project, symbol, entry: f"<a href=''>{symbol}</a>",
                 signals_url_generator=lambda project, entry, symbol, model_name, nexecs: f"<a href=''>{nexecs}</a>"
                 ):
        self.boo = Booster(config, reload_config=True)
        self.rinf = RunningInfoManager()
        self.details_url_generator = details_url_generator
        self.signals_url_generator = signals_url_generator

    def check_running_status(self, entry_id):
        ns = {}
        status = None
        try:
            for _m, _c in self.boo.get_optimizations(entry_id).items():
                for sim_runs in self.rinf.list_runs():
                    if _m in sim_runs and entry_id in sim_runs:
                        status = 'running'
                        ri = self.rinf.get_id_info(sim_runs)
                        _p = ri.get('progress', '-1')
                        _t = ri.get('total', '-1')
                        _pct = 100 * int(_p) / int(_t)
                        s_info = f"{_p} ({_pct:.2f}%) from {_t} [FAILED: {ri.get('failed', '0')}] "
                        ns[_m] = {'Status': s_info}
                    else:
                        ns[_m] = {'Status': 'no data'}
        except Exception as e:
            print('>>> ', e)

        return ns, status

    def get_report(self, sorted_by='Gain', asc=False):
        """
        Return report as HTML
        """
        tmpl = {}
        hdrs = []

        for _n in self.boo.get_all_entries():
            # speed up loading by disabling config file read
            cfg = self.boo.load_entry_config(_n, skip_reload_config=True)
            model = cfg.project
            instr = cfg.instrument
            insample = cfg.end_date
            symbol = instr.split(':')[1]
            cap = cfg.capital
            brk = inst_sim_brok(cfg.broker, 0)
            comms = f"{10000 * brk.tcc.taker:.2f}/{10000 * brk.tcc.maker:.2f}"
            sharpe, dd_pct, cagr, eqty, gain, qr = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            atpm = np.nan
            nexecs = 0

            r = b_ld(f'blends/{model}/{_n}')
            ns = {}
            if r is None:
                ns, status = self.check_running_status(_n)
                status = status if status else 'not started'
            else:
                # get backtest info
                bt = r['backtest']
                bt_perf = r['performance']
                bt_perf = bt_perf.to_dict() if isinstance(bt_perf, mstruct) else bt_perf
                sharpe, dd_pct, cagr, eqty, qr = bt_perf['sharpe'], bt_perf['drawdown_pct'], bt_perf['cagr'], bt_perf[
                    'equity'], bt_perf['qr']
                nexecs = len(bt.executions)
                gain = eqty[-1] - eqty[0]
                atpm = bt_perf.get('avg_trades_month', -1)
                status = 'finished'

                for m, v in r['models'].items():
                    perf = v.performance
                    ns[m] = {'Weight': round(r['weights'][m], 2),
                             'Gain': round(perf.gain, 2),
                             'Sharpe': round(perf.sharpe, 2),
                             'Max DD': round(perf.drawdown_pct, 2),
                             'CAGR': round(perf.cagr, 2),
                             }

            # submodels mini report embedded into common report
            # ckey = f"sm_{instr.replace(':','')}_{model}"
            ckey = f"sm_{instr.replace(':', '')}_{model}_{_n}"

            # link to details viewer
            link = 'linked_details_' + symbol + "_" + _n

            # current status label
            _stat = f'{ckey}_status' + symbol + "_" + _n

            # link to signals viewer for blended model
            signal_viewer_link = 'signal_viewer_' + symbol + "_" + _n
            hdr = {
                'Product': f'${link}',
                'Gain': round(gain, 2), 'Gain %': round(100 * gain / cap, 2),
                'Sharpe': round(sharpe, 2),
                'Max DD': round(dd_pct, 2),
                'CAGR': round(100 * cagr, 2),
                'QR': round(qr, 2),
                'Execs': f'${signal_viewer_link}',
                'ATPM': round(atpm, 2),
                'Insample': insample,
                'Comm. (bps)': comms,
                'Status': f'${_stat}',
                'Sub-models': f'${ckey}'
            }

            tmpl[ckey] = pd.DataFrame.from_dict(ns).T.to_html()
            tmpl[link] = self.details_url_generator(model, symbol, _n)
            # here we want to take a look at blended model's signals here
            # todo: we would need to see signals for models - constituents
            tmpl[signal_viewer_link] = self.signals_url_generator(model, _n, symbol, 'backtest', nexecs)
            tmpl[_stat] = f'<p class="status-{status.replace(" ", "_")}">{status.upper()}</p>'
            hdrs.append(hdr)

        h_table = pd.DataFrame(hdrs).sort_values([sorted_by, 'Product'], ascending=[asc, True]).fillna(
            '---').reset_index(drop=True).to_html()
        return airspeed.Template(h_table).merge(tmpl)

    def show_detailed_report(self, project, symbol, entry):
        b_report = b_ld(f'blends/{project}/{entry}')
        if b_report is None:
            return "<h2>Data is not ready yet</h2>"

        mdls = {}
        for i, (m, v) in enumerate(b_report['models'].items()):
            mdls[f"{m}:sharpe"] = round(v.performance.sharpe, 2)
            mdls[f"{m}:dd"] = round(v.performance.drawdown_pct, 2)
            mdls[f"{m}:cagr"] = round(100 * v.performance.cagr, 2)

        capital = b_report['config']['capital']
        tsh = tearsheet(b_report['backtest'], capital, insample=b_report['insample'],
                        meta={'parameters': {**{'capital': capital}, **mdls}},
                        performance_statistics_period=PERFORMANCE_PERIOD,
                        plain_html=True)

        return tsh.data if hasattr(tsh, 'data') else tsh

    def show_parameters_report(self, project, entry, weight_capital=True) -> mstruct:
        """
        Table with models parameters as html
        """
        m_perfs = {}

        b_report = b_ld(f'blends/{project}/{entry}')
        if b_report is None:
            return mstruct(models_parameters="---", models_performance=m_perfs, models_names=['none'])

        # load config
        cfg = self.boo.load_entry_config(entry)

        mdls = {}
        for i, (m, v) in enumerate(b_report['models'].items()):
            w = b_report['weights'].get(m, 1)
            cap = {}

            # if we need to see 'weighted' capital 
            if weight_capital:
                weighted_cap_value = w * cfg.capital if hasattr(cfg, 'capital') else v.parameters.get('capital', 0)
                cap = {'capital': round(weighted_cap_value, 2)}

                # collect  
                m_perfs[m] = tearsheet(v.result,
                                       weighted_cap_value, insample=b_report['insample'],
                                       meta={'parameters': v.parameters},
                                       performance_statistics_period=PERFORMANCE_PERIOD,
                                       plain_html=True)

            mdls[m] = {**{'Weight': w}, **v.parameters, **cap}

        return mstruct(
            models_parameters=pd.DataFrame.from_dict(mdls, orient='index').to_html(),
            models_performance=m_perfs,
            models_names=list(m_perfs.keys())
        )


def portfolios_parameters_sets(data: dict, full_parameters_list):
    """
    Get portfolios parameters named sets
    
    Example:
    
    portfolios_parameters_sets({'parameters': {
        'set0': {'a': 1, 'b': 2},
        'set1': {'a': 2, 'b': 2},
        'set2': {'a': 3, 'b': 2},
    }}, False)
    
        set0  set1  set2
    a      1     2     3
    
    :param data: dictionary with parameters
    :param full_parameters_list: true if we need full list (false - only changed)
    :return: dataframe with sets as columns 
    """
    sets_params = data.get('parameters')
    if sets_params:
        param_report = pd.DataFrame.from_dict(sets_params).fillna('')

        # remains only ones were changed
        if not full_parameters_list:
            param_report = param_report[~param_report.eq(param_report.iloc[:, 0], axis='index').all(axis=1)]
    return param_report


def get_portfolio_run_info(project, entry):
    """
    Returns info about portfolio run
    """
    return b_ld(f'portfolios/{project}/{entry}')


def get_combined_portfolio(project, entry, set_name) -> pd.DataFrame:
    """
    Returns combined potfolio for experiment at project
    """
    data = None
    path = f'portfolios/{project}/{entry}'
    p_data = b_ld(path)

    # - combine from runs
    if p_data is not None:
        comb_portfolio = []
        for sp in p_data['simulations'][set_name].values():
            data = b_ld(f'runs/{project}/{sp}/{entry}_PORTFOLIO')

            # - some simulations may be empty -so skiping it
            if data is not None and hasattr(data, 'result'):
                if data.result is not None and hasattr(data.result, 'portfolio'):
                    comb_portfolio.append(data.result.portfolio)

        data = scols(*comb_portfolio)
    return data


def get_combined_executions(project, entry, set_name) -> pd.DataFrame:
    """
    Returns combined executions for experiment at project
    """
    data = None
    path = f'portfolios/{project}/{entry}'
    p_data = b_ld(path)

    # - combine from runs
    if p_data is not None:
        comb_portfolio = []
        for sp in p_data['simulations'][set_name].values():
            data = b_ld(f'runs/{project}/{sp}/{entry}_PORTFOLIO')

            # - some simulations may be empty -so skiping it
            if data is not None and hasattr(data, 'result'):
                if data.result is not None and hasattr(data.result, 'executions'):
                    comb_portfolio.append(data.result.executions)

        data = srows(*comb_portfolio)
    return data


def get_combined_results(project, entry, set_name) -> MultiResults:
    """
    Returns combined results as MultiResult for experiment at project
    """
    data = None
    path = f'portfolios/{project}/{entry}'
    p_data = b_ld(path)
    # return p_data

    # - combine from runs
    if p_data is not None:
        results = []
        for sp in p_data['simulations'][set_name].values():
            data = b_ld(f'runs/{project}/{sp}/{entry}_PORTFOLIO')

            # - some simulations may be empty -so skiping it
            if data is not None and hasattr(data, 'result'):
                if data.result is not None and hasattr(data.result, 'portfolio'):
                    results.append(data.result)

    return MultiResults(results, project, None, None, None, None)


def get_runs_portfolio(project, entry, sim_path):
    """
    Returns simulation portfolio and parameters for particular simulation
    """
    data = None
    m_params = {}
    path = f'runs/{project}/{sim_path}/{entry}'
    p_data = b_ld(path)
    if p_data is not None and hasattr(p_data, 'result'):
        data = p_data.result
        m_params = p_data.task_args[1] if hasattr(p_data, 'task_args') else {}
    return data, m_params


def get_experiment_trend_report(project, experiment):
    """
    Portfolio experiments as chart
    """
    data = b_ld(f'portfolios/{project}/{experiment}')
    p_sets = portfolios_parameters_sets(data, False)
    chart = {}
    for sn in p_sets.columns:
        k = sn + ' : ' + ','.join(map(str, p_sets[sn].values))
        set_data = data['report'][sn]
        # combined_portfolio = get_combined_portfolio(project, experiment, sn)
        chart[k] = {
            'Gain': set_data.loc['Total', 'Gain'],
            'GainPct': set_data.loc['Total', 'GainPct'],
            'PctPerTrade': set_data.loc['Average per trade', 'GainPct'],
            'Loosers': len(set_data[set_data['Gain'] < 0])
        }
    return mstruct(
        chart=pd.DataFrame(chart).T,
        data=data,
        parameters=p_sets,
    )


def tearsheet_report(where, project, sim_path, entry, capital, insample=None):
    """
    Generate tearsheet report
    """
    data = None
    m_params = {}
    path = None
    if where == 'runs':
        data, m_params = get_runs_portfolio(project, entry, sim_path)

    elif where == 'portfolios':
        data = get_combined_portfolio(project, entry, sim_path)

    elif where == 'blends':
        # TODO: !!!
        path = f'{where}/{project}/{entry}'

    elif where == 'results':
        # TODO: !!!
        path = f'{where}/{project}/{entry}'

    if data is not None:
        tsh = tearsheet(data, capital, insample=insample,
                        meta={'parameters': {**{'capital': capital}, **m_params}},
                        performance_statistics_period=PERFORMANCE_PERIOD,
                        plain_html=True)

        return tsh.data if hasattr(tsh, 'data') else tsh
    else:
        return f"<H1> No data found at '{path}' !!! </H1>"


def portfolio_stats_report(portfolio: pd.DataFrame, cash: float):
    """
    Get portfolio statistics prepared for exporting / storing
    """
    ps0 = portfolio_stats(portfolio, cash, account_transactions=True, performance_statistics_period=PERFORMANCE_PERIOD)

    equity = ps0['equity']
    equity1d = equity[0] + equity.diff().resample('1D').sum().cumsum()
    month = ps0['monthly_returns'] * 100

    # for json 
    equity1d.index = equity1d.index.map(lambda x: x.strftime('%Y-%m-%d'))
    month.index = month.index.map(lambda x: x.strftime('%Y-%m-%d'))
    stats = {k: v for k, v in ps0.items() if k in ['cagr', 'drawdown_pct',
                                                   'mdd_usd', 'sharpe',
                                                   'qr', 'sortino',
                                                   'calmar', 'broker_commissions',
                                                   'mean_return', 'annual_volatility', 'var']}
    stats['cagr'] *= 100
    stats['dd_duration_till_recovered'] = str(ps0['mdd_recover'] - ps0['mdd_start'])
    stats['dd_duration_to_peak'] = str(ps0['mdd_peak'] - ps0['mdd_start'])
    # cumulative pct 
    cumulative_pct = (equity1d.pct_change() + 1).cumprod().dropna()

    final_report = {
        'equity': equity1d.to_dict(),
        'cumulative_pct': cumulative_pct.to_dict(),
        'monthly_returns': month.to_dict(),
        'stats': stats,
    }
    return final_report


def export_portfolio_performance(project, entry, set_name='Set0'):
    """
    Export portfolio performance as dict
    """
    data = get_portfolio_run_info(project, entry)
    cash = data['total_capital']
    s0 = get_combined_portfolio(project, entry, set_name)
    return mstruct(report=portfolio_stats_report(s0, cash), portfolio=s0, cash=cash)


def get_multiportolio_combined(*ids):
    """
    Combine multiple portfolios from it's id into single one 
    """
    total_cash = 0
    total_pfl = pd.DataFrame()
    for eid in ids:
        sids = eid.split('/')
        project, entry, set_name = [*sids, 'Set0'] if len(sids) == 2 else sids[:3]
        data = get_portfolio_run_info(project, entry)
        total_cash += data['total_capital']
        cpfl = get_combined_portfolio(project, entry, set_name)
        total_pfl = combine_portfolios(total_pfl, cpfl)
    return mstruct(portfolio=total_pfl, total_cash=total_cash)


def get_multiportolio_combined_report(*ids, cash=None):
    """
    Get report for combined multiportfolio
    """
    c = get_multiportolio_combined(*ids)
    return portfolio_stats_report(c.portfolio, c.total_cash if cash is None else cash)


def cumulative_growth_cashflow(abs_rets, flows):
    R_t_cf = (abs_rets / (abs_rets + flows).cumsum().shift(1)).dropna()
    return mstruct(returns=R_t_cf, cumulative=(R_t_cf + 1).cumprod())


def cumulative_growth(abs_rets, cash):
    E_t = abs_rets.cumsum() + cash
    R_t = (abs_rets / E_t.shift(1)).dropna()
    Rc_t = (R_t + 1).cumprod()
    return mstruct(returns=R_t, cumulative=(R_t + 1).cumprod())


def cashflow(portfolio, cash_per_instrument):
    nactive = portfolio.filter(regex='.*_Price')
    nactive = (~np.isnan(nactive)).sum(axis=1)
    bp = cash_per_instrument * nactive
    F_t = bp.diff()
    F_t[0] = bp[0]
    return F_t


def portfolio_stats_cashflow_adjusted(portfolio: pd.DataFrame, cash_per_instrument: float):
    """
    Calculate portfolio performance based on dynamic cash allocation 
    We allocate fixed 'cash_per_instrument' per market if market is active
    """
    # - commissions
    comms = portfolio.filter(regex='.*_Commissions').sum(axis=1)

    # - absolute returns with commissions included
    A_t = portfolio.filter(regex='.*_PnL').sum(axis=1) - comms

    # - cash flow (actual markets x cash per market)
    F_t = cashflow(portfolio, cash_per_instrument)

    # - aggregated returns / cumulative growth rate
    A_t_d1, A_t_mon1 = A_t.resample('1D').sum(), A_t.resample('1M').sum()
    F_t_d1, F_t_mon1 = F_t.resample('1D').sum(), F_t.resample('1M').sum()

    # - equity 1d / do not take in account withdrawals
    equity1d = (A_t_d1 + F_t_d1[F_t_d1 >= 0]).ffill().cumsum()
    mdd_usd, _, _, _, _ = absmaxdd(equity1d)

    # - daily / monthly returns
    grD1 = cumulative_growth_cashflow(A_t_d1, F_t_d1)
    grM1 = cumulative_growth_cashflow(A_t_mon1, F_t_mon1)

    # - stats
    mdd, ddstart, ddpeak, ddrecover, dd_data = absmaxdd(grD1.cumulative)
    dd_pct = 100 * mdd / grD1.cumulative.iloc[ddstart]
    dd_duration_till_recovered = grD1.cumulative.index[ddrecover] - grD1.cumulative.index[ddstart]
    dd_duration_to_peak = grD1.cumulative.index[ddpeak] - grD1.cumulative.index[ddstart]
    sh = sharpe_ratio(grD1.returns, 0.0, PERFORMANCE_PERIOD)
    agr = 100 * cagr(grD1.returns, PERFORMANCE_PERIOD)
    mean_return = np.mean(grD1.returns)
    sortino = sortino_ratio(grD1.returns, 0.0, PERFORMANCE_PERIOD, None)
    calmar = calmar_ratio(grD1.returns, PERFORMANCE_PERIOD)
    stats = {
        'cagr': agr,
        'dd_duration_till_recovered': str(dd_duration_till_recovered),
        'dd_duration_to_peak': str(dd_duration_to_peak),
        'drawdown_pct': dd_pct,
        'mdd_usd': mdd_usd,
        'sharpe': sh,
        'qr': qr(equity1d),
        'sortino': sortino,
        'calmar': calmar,
        'broker_commissions': sum(comms),
        'mean_return': mean_return,
        'annual_volatility': 0,
        'var': 0,
    }

    def idx_as_str(xs):
        xs.index = xs.index.map(lambda x: x.strftime('%Y-%m-%d'))
        return xs

    return {
        'equity': idx_as_str(equity1d).to_dict(),
        'used_buying_power': idx_as_str(F_t_d1.cumsum()).to_dict(),
        'cumulative_pct': idx_as_str(grD1.cumulative).to_dict(),
        'monthly_returns': idx_as_str(100 * grM1.returns).to_dict(),
        'stats': stats,
    }
