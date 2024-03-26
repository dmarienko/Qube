import base64
import os
from io import BytesIO
from typing import Union

import airspeed
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt

from qube.charting.highcharts.core import serialize, union_charts
from qube.charting.plot_helpers import subplot
from qube.configs.Properties import get_root_dir
from qube.portfolio.performance import *
from qube.portfolio.performance import calculate_total_pnl
from qube.simulator.core import SimulationResult
from qube.utils.utils import runtime_env

TPL_EXTENSION = '.tpl'
TPL_DIR = 'portfolio/report_templates/'


def report_templator(sheet, template, handlers=None):
    namespace = sheet.copy()
    if handlers:
        if any(k in sheet for k in handlers.keys()):
            raise ValueError('Keys in Handlers should not be the same as keys in the Sheet')
        # namespace = {**sheet, **handlers} # not yet provided in 3.5
        namespace.update(handlers)

    airspeed_tpl = airspeed.Template(template)
    return airspeed_tpl.merge(namespace)


def optimizer_tearsheet(results, meta=None):
    template_body = _get_template_by_name('optimizer_report')
    if meta:
        results['meta'] = meta
    return report_templator(results, template_body)


def comparison_report_tearsheet(pfl_list, meta=None):
    results = {}
    template_body = _get_template_by_name('comparison_report')
    if meta:
        results['meta'] = meta
    results['pfl_list'] = pfl_list
    handlers = {'comparison_pnl_chart': comparison_pnl_chart}
    return report_templator(results, template_body, handlers)


def tearsheet(portfolio: Union[pd.DataFrame, SimulationResult], init_cash, risk_free=0.0,
              rolling_sharpe_window=DAILY,
              account_transactions=True,
              performance_statistics_period=DAILY_365,
              figsize=(9, 8),
              template='simple_report', handlers=None,
              meta=None,
              highcharts=False, insample=None, **kwargs):
    """
    Tearsheet report for simulated portfolio

    :param portfolio: portfolio dataframe or result of simulation
    :param init_cash: initial cash invested
    :param risk_free: risk free rate (0.0)
    :param rolling_sharpe_window: window size for rolling sharpe (252 days by default)
    :param account_transactions: if True and portfolio contains _Commissions column commissions included in equity
    :param performance_statistics_period: annualization period for performance statistics (default 252)
    :param figsize: default figure size for equity plot
    :param template: report's template
    :param handlers:
    :param meta: additional info
    :param insample: in sample period to show on chart [ex. as tuple ('2020-01-01', '2020-10-01') | default none]
    :param plain_html: if true returns html presentation (False by default)
    :return:
    """
    strategy_name = portfolio.name if hasattr(portfolio, 'name') else None
    simulation_info = portfolio.description if hasattr(portfolio, 'description') else ''
    number_signals = portfolio.number_processed_signals if hasattr(portfolio, 'number_processed_signals') else '???'

    if hasattr(portfolio, 'portfolio'):
        portfolio = portfolio.portfolio

    sheet = portfolio_stats(portfolio, init_cash, risk_free=risk_free,
                            rolling_sharpe_window=rolling_sharpe_window,
                            account_transactions=account_transactions,
                            performance_statistics_period=performance_statistics_period, **kwargs)

    if meta is not None:
        sheet['meta'] = meta

    sheet['gain'] = sheet['equity'].iloc[-1] - sheet['equity'].iloc[0]
    sheet['max_dd_days'] = sheet['mdd_recover'] - sheet['mdd_start']
    sheet['figsize'] = figsize
    sheet['fontsize'] = kwargs.get('fontsize', 9)
    sheet['highcharts'] = highcharts
    sheet['insample'] = insample
    sheet['strategy_name'] = strategy_name
    sheet['simulation_info'] = simulation_info
    sheet['number_signals'] = number_signals

    if handlers is None:
        if highcharts:
            handlers = {'chart': _highcharts_chart}
        else:
            handlers = {'chart': _mpl_chart}

    template_body = _get_template_by_name(template)
    report_string = report_templator(sheet, template_body, handlers)

    return _display_report(report_string, highcharts, **kwargs)


def signal_statistics_report(portfolio: pd.DataFrame, figsize=(16, 4), template="signal_statistics_report", **kwargs):
    sig_stats = collect_entries_data(portfolio)
    sE = pd.concat(sig_stats.values())

    _grp_h = sE.groupby(sE.index.hour)
    pl_sum_by_hrs = _grp_h['SignalPL'].apply(lambda x: np.sum(x))
    execs_counts_by_hrs = _grp_h['SignalPL'].count()

    avg_pl_by_hrs = _grp_h['SignalPL'].apply(np.nanmean)
    median_pl_by_hrs = _grp_h['SignalPL'].apply(np.median)

    plt.figure(figsize=figsize)
    subplot(13, 1)
    sns.kdeplot(sE.SignalPL, cut=0, label='Long and Short PnL')
    sns.kdeplot(sE[sE.BuySell == 1].SignalPL, cut=0, label='Long PnL')
    sns.kdeplot(sE[sE.BuySell == -1].SignalPL, cut=0, label='Short PnL')
    plt.legend(loc='upper left')
    plt.title('PnLs Distributions')

    subplot(13, 2)
    plt.hist(sE.Duration.apply(lambda x: x.seconds / 3600), bins=100)
    plt.title('Position duration in hours')
    subplot(13, 3)
    sns.kdeplot([s.seconds / 3600 for s in sE.TimeToMax], cut=0, label='Time to DrawUp in hours')
    sns.kdeplot([s.seconds / 3600 for s in sE.TimeToMin], cut=0, label='Time to DrawDown in hours')
    plt.title('Times to nearest Draw(Up/Down)')

    l_sE = sE[sE.BuySell == 1]
    s_sE = sE[sE.BuySell == -1]
    _gh_l = l_sE.groupby(l_sE.index.hour)
    _gh_s = s_sE.groupby(s_sE.index.hour)

    l_pl_sum, s_pl_sum = _gh_l['SignalPL'].apply(np.sum), _gh_s['SignalPL'].apply(np.sum)
    l_cnts, s_cnts = _gh_l['SignalPL'].count(), _gh_s['SignalPL'].count()

    rprt = pd.concat((execs_counts_by_hrs, l_cnts, s_cnts,
                      pl_sum_by_hrs, l_pl_sum, s_pl_sum,
                      avg_pl_by_hrs, median_pl_by_hrs),
                     axis=1,
                     keys=['Positions', 'Longs', 'Shorts', 'Sum PL', 'Long PL', 'Short PL', 'Avg PL', 'Median PL'])

    report_table = pd.concat((rprt, rprt.drop(columns=['Avg PL', 'Median PL']).sum().to_frame().T.rename({0: 'Total'})),
                             axis=0)[
        rprt.columns].fillna('---')

    fig = plt.gcf()
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png', transparent=True)
    imgdata.seek(0)
    uri = 'data:image/png;base64,' + base64.b64encode(imgdata.getvalue()).decode('utf8')
    plt.clf()

    template_body = _get_template_by_name(template)

    report_string = report_templator(
        {'report_table': report_table.to_html(float_format=lambda x: '%.2f' % x), 'image': uri}, template_body)

    return _display_report(report_string, False, **kwargs)


def tearsheets(*args, **kwargs):
    """
    Method for displaying multiple portfolios at the same output (handy when we need to compare two or more ones)
    This method will work only in Jupyter notebook interface.

    >>> tearsheets(pfl1, pfl2, pfl3, init_cash=1000)
    >>> tearsheets(pfl1, pfl2, pfl3, init_cash=[1000, 2000, 5000])

    :param args: portfolios logs
    :param kwargs: tearsheet arguments (init_cash is required)
    :return:
    """
    if runtime_env() == 'notebook':
        from IPython.core.display import display, HTML
        s0 = """<style>
                .report_table td, .report_table th {
                    text-align:left !important;
                    font-size:9px !important;
                    border: 1px solid #252525 !important;
                }
                .rendered_html tbody tr:nth-child(2n+1) {
                    background: transparent !important;
                }
                </style>"""
        s0 += '<table><tr>'

        # check for initial cashs
        i_cash = kwargs.pop('init_cash', None)
        if i_cash is not None:
            if isinstance(i_cash, (tuple, list)):
                if len(i_cash) != len(args):
                    raise ValueError('init_cash array must have same number elements as portfolios')
            else:
                i_cash = [i_cash] * len(args)

        ic_i = 0
        for a in args:
            if isinstance(a, (pd.DataFrame, SimulationResult)):
                s0 += '<td>' + tearsheet(a, init_cash=i_cash[ic_i], plain_html=True, **kwargs).data + '</td>'
                ic_i += 1
        s0 += '</tr></table>'

        display(HTML(s0))
    else:
        raise ValueError('tearsheets may be called only in Jupyter Notebook environment !!!')


def _display_report(report_string, highcharts=False, **kwargs):
    # if it's being called from Jupyter Notebook we want to display it
    if runtime_env() == 'notebook':
        if highcharts:
            from qube.charting.highcharts import load_highcharts
            load_highcharts()
        from IPython.core.display import HTML, display
        if kwargs.get('plain_html', False):
            return HTML(report_string)
        else:
            display(HTML(report_string))
    else:
        return report_string


def _get_template_by_name(template_name):
    tpl_path = os.path.join(get_root_dir(), TPL_DIR, template_name + TPL_EXTENSION)
    if not os.path.isfile(tpl_path):
        raise ValueError("template %s is not found" % tpl_path)
    with open(tpl_path, 'r') as tpl:
        return tpl.read()


def _mpl_chart(strategy_name, equity, mdd_start, mdd_recover, compound_returns, drawdown_usd, long_value,
               short_value, insample=None, behcmark_compound=None, figsize=(16, 8), fontsize=9, vertical=False):
    # insample period information pre-processing
    insample_xmin, insample_xmax = None, None
    if insample is not None:
        if isinstance(insample, str):
            insample_xmin, insample_xmax = equity.index[0], pd.Timestamp(insample)
        elif len(insample) > 1:
            insample_xmin, insample_xmax = pd.Timestamp(insample[0]), pd.Timestamp(insample[-1])
        else:
            # didn't recognize in sample period
            insample = None

    if vertical:
        if behcmark_compound is not None:
            shape = (6, 1)
            loc = [(0, 0), (2, 0), (4, 0), (5, 0)]
        else:
            shape = (4, 1)
            loc = [(0, 0), (2, 0), (3, 0), (4, 0)]

    else:
        shape = (3, 2)
        loc = [(0, 0), (0, 1), (2, 0), (2, 1)]

    matplotlib.rcParams.update({'font.size': fontsize})
    plt.figure(figsize=figsize)
    ax = plt.subplot2grid(shape, loc.pop(0), rowspan=2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: str(y / 1000) + ' K'))
    plt.plot(equity, 'g')
    plt.title('Portfolio Equity (USD)')
    plt.plot(equity[mdd_start: mdd_recover], 'r')
    plt.ylabel('Model Equity (USD)')
    if insample is not None:
        plt.axvline(insample_xmin, ls=':', lw=1, c='#ffae62')
        plt.axvline(insample_xmax, ls=':', lw=1, c='#ffae62')
        # plt.hlines(min(equity.values), xmin=equity.index[0], xmax=equity.index[-1], linewidth=4, color='#ffae62')
        # plt.hlines(min(equity.values), xmin=insample_xmin, xmax=insample_xmax, linewidth=4, color='#639fff')
    plt.grid()

    if not vertical or behcmark_compound is not None:
        plt.subplot2grid(shape, loc.pop(0), rowspan=2)
        plt.plot(compound_returns)
        plt.title('Portfolio vs Benchmark')
        plt.plot(compound_returns[mdd_start: mdd_recover], 'r')
        if behcmark_compound is not None:
            plt.plot(behcmark_compound, '#505050')
            plt.ylabel('Compound Return')
        if insample:
            plt.hlines(min(compound_returns.values), xmin=compound_returns.index[0], xmax=compound_returns.index[-1],
                       linewidth=4, color='#ffae62')
            plt.hlines(min(compound_returns.values), xmin=insample_xmin, xmax=insample_xmax, linewidth=4,
                       color='#639fff')
        plt.grid()

    ax = plt.subplot2grid(shape, loc.pop(0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: str(y / 1000) + ' K'))
    dd_data = drawdown_usd
    plt.plot(-dd_data, 'r')
    plt.fill_between(dd_data.index, -dd_data.values, 0, color='r')
    plt.ylabel('Drawdown (USD)')
    plt.grid()

    ax = plt.subplot2grid(shape, loc.pop(0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: str(y / 1000) + ' K'))
    plt.plot(long_value + abs(short_value), 'g')
    plt.ylabel('Gross Value (USD)')
    plt.grid()
    plt.tight_layout()

    fig = plt.gcf()
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png', transparent=True)
    imgdata.seek(0)
    uri = 'data:image/png;base64,' + base64.b64encode(imgdata.getvalue()).decode('utf8')
    plt.clf()
    return uri


def _highcharts_chart(strategy_name, equity, mdd_start, mdd_recover, compound_returns, drawdown_usd, long_value,
                      short_value, behcmark_compound=None, figsize=(16, 8), fontsize=9):
    result = []
    equity_df = equity.to_frame('equity')
    equity_df['max dd'] = equity[mdd_start: mdd_recover]
    output_eq = serialize(equity_df, render_to='tearsheet', title="Portfolio Equity (USD)",
                          colors=['#03510b', '#e51919'], output_type='dict')

    pfl_bench_df = compound_returns.to_frame('compound_returns')
    pfl_bench_df['compound_returns_max_dd'] = compound_returns[mdd_start: mdd_recover]

    if behcmark_compound is not None:
        pfl_bench_df['behcmark_compound'] = behcmark_compound

    output_bench = serialize(pfl_bench_df, render_to='tearsheet', title="Portfolio vs Benchmark",
                             colors=['#474747', '#62bef7', '#e51919'], output_type='dict')

    dd_data_df = -drawdown_usd.to_frame('drawdown_usd')
    output_dd = serialize(dd_data_df, kind='area', render_to='tearsheet', title="Drawdown (USD)",
                          colors=['#e51919'], output_type='dict')

    gross_val = long_value + abs(short_value)
    gross_val_df = gross_val.to_frame('gross_value')
    output_gross = serialize(gross_val_df, render_to='tearsheet', title="Gross Value (USD)",
                             colors=['#03510b'], output_type='dict')

    chart = union_charts([output_eq, output_bench, output_dd, output_gross], [360, 80, 80, 80])
    result.append({'chart': chart, 'name': 'tearsheet'})
    return result


def comparison_pnl_chart(pfl_list, figsize=(16, 8)):
    plt.figure(figsize=figsize)
    for pfl in pfl_list:
        pfl_frame = calculate_total_pnl(pfl['data'], split_cumulative=False)
        label = pfl['label']
        plt.plot(pfl_frame['Total_PnL'], label=label)
        plt.ylabel('Total_PnL')
    plt.legend(loc='upper left', bbox_to_anchor=(0, -0.05))

    fig = plt.gcf()
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png', transparent=True, bbox_inches="tight")
    imgdata.seek(0)
    uri = 'data:image/png;base64,' + base64.b64encode(imgdata.getvalue()).decode('utf8')
    plt.clf()
    return uri
