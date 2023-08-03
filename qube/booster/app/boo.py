import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from os.path import join

import numpy as np
import pandas as pd
import plotly
import yaml
from dotenv import load_dotenv

from qube import __version__ as QVER
from qube.booster import __version__ as BVER
from qube import DARK_MATLPLOT_THEME
from qube.booster.utils import b_ls, b_ld

from flask import Flask, render_template, request
from flask_basicauth import BasicAuth

from qube.booster.core import Booster
from qube.utils.utils import mstruct, dict2struct
from qube.booster.app.reports import (
    BoosterProgressReport, tearsheet_report, get_progress_for, portfolios_parameters_sets, get_experiment_trend_report
)
from qube.booster.app.signal_viewer import show_signals
from qube.charting.lookinglass import LookingGlass

DEFAULT_TIMEFRAME = '15Min'
DEBUG = True
BOOSTER_CONFIG_PATH = '/var/appliedalpha/booster/'
BOOSTER_PROJECT_CONFIGS_FILE = f'{BOOSTER_CONFIG_PATH}/booster.yml'
BOOSTER_APP_CONFIG_FILE = join(BOOSTER_CONFIG_PATH, 'config.cfg')
VERSION = QVER + " / " + BVER

sys.stdout = sys.stderr
__dark_theme_installed = False
app = Flask(__name__)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Just to prevent reporting server from access by any unauthorized person
load_dotenv(BOOSTER_APP_CONFIG_FILE)
app.config['BASIC_AUTH_USERNAME'] = os.getenv('app_username', 'user')
app.config['BASIC_AUTH_PASSWORD'] = os.getenv('app_passwd', 'User1User1')
app.config['BASIC_AUTH_FORCE'] = True
basic_auth = BasicAuth(app)


@dataclass
class ViewerData:
    symbol: str
    entry: str
    model: str
    start: pd.Timestamp
    stop: pd.Timestamp
    ohlc: pd.DataFrame
    signals: list
    timeframe: str = DEFAULT_TIMEFRAME


def __install_charts_theme():
    global __dark_theme_installed

    if not __dark_theme_installed:
        import matplotlib
        import plotly.io as pio
        import plotly.express as px
        pio.templates.default = "plotly_dark"
        px.defaults.template = "plotly_dark"
        for (k, v) in DARK_MATLPLOT_THEME:
            matplotlib.rcParams[k] = v
        __dark_theme_installed = True


# just handy helper
def aslist(o):
    return o if isinstance(o, (list, tuple)) else [o]


def collect_all_projects():
    """
    Collect all configurations
    """
    projects = {}
    with open(BOOSTER_PROJECT_CONFIGS_FILE, 'r') as stream:
        try:
            # config entries
            entries = yaml.safe_load(stream)['entries']
        except yaml.YAMLError as exc:
            raise ValueError(exc)

    for p, rec in entries.items():
        if isinstance(rec, dict):
            r = mstruct(path=join(BOOSTER_CONFIG_PATH, rec.get('path')), desc=rec.get('description'))
        else:
            r = mstruct(path=join(BOOSTER_CONFIG_PATH, rec), desc='')

        projects[p.upper()] = r
    return projects


def get_config_for_project(project):
    p = project.upper()
    projects = collect_all_projects()
    if p in projects:
        return projects[p].path
    else:
        raise ValueError(f"Can't find entry record in configuration for {p} !")


def collect_all_portfolio_experiments(sort_by_time=True):
    """
    Collect all portfolios experiments
    """
    projects = defaultdict(dict)
    for p in b_ls('portfolios/_index/.*/.*'):
        ps = p.split('/')
        if len(ps) == 4:
            project = ps[2]
            entry = ps[3]
            dps = projects[project]
            dps[entry] = dict2struct(b_ld(p))

    if sort_by_time:
        sorted_projects = {}
        for ke in sorted(projects.keys()):
            for e, vs in projects[ke].items():
                vs.timestamp = pd.Timestamp(vs.timestamp)
            sorted_projects[ke] = dict(sorted(projects[ke].items(), key=lambda item: item[1].timestamp, reverse=True))
        projects = sorted_projects

    return projects


@app.route('/')
def main_index():
    report = '<table><tr><th class="index-page">Project</th><th class="index-page">Description</th></tr>'
    try:
        projects = collect_all_projects()
        for p, pdata in projects.items():
            report += f"<tr><td class='index-page'> <a href='report/{p.upper()}'>{p.upper()} </a> </td><td class='index-page'>{pdata.desc if pdata.desc is not None else ''}</td><tr>"
        report += '</table>'
    except Exception as e:
        trace = ''
        if DEBUG:
            import traceback
            trace = traceback.format_exc().replace('\n', '<br/>')
        report = f"<h2><font color='red'>Exception: {str(e)} / </font> </h2><br/> {trace}"

    return render_template('index.html', main_report=report)


@app.route('/experiments/')
def portfolios_index():
    try:
        td = lambda x: f"<td class='index-page'> {x} </td>"
        tr = lambda *xs: f"<tr> {''.join(xs)} </tr>"
        th = lambda x: f"<th class='index-page'> {x} </th>"
        bold = lambda x: f"<b>{x}</b>"
        red = lambda x: f"<font color='red'>{x}</font>"
        gray = lambda x: f"<font color='#606040'>{x}</font>"
        report = "<table>" + tr(
            th('Project'),
            th('Experiment'),
            th('Description'),
            th('Updated'),
            th('Status'),
        )
        projects = collect_all_portfolio_experiments()
        report += "<tr>" + "<td class='empty-bar'/>" * 5 + "</tr>"
        for p, prj in projects.items():
            _is_first = True
            for entry, pdata in prj.items():

                stat = gray(pdata.status)
                if 'RUN' in pdata.status:
                    progress = get_progress_for(p)
                    stat = red('RUN') + f'&nbsp; {progress[0]:.2f}%/{progress[1]}'

                # detailed reports links
                links = f"<a href='report/{p}/{pdata.experiment}?full=false' target='_blank' >{pdata.experiment}</a>"
                links += f"<a href='report_chart/{p}/{pdata.experiment}' target='_blank'> | <font color='#405030'>Chart</font></a>"

                report += tr(
                    td(bold(p) if _is_first else '&nbsp;' * 2 + '&nbsp;.' * 5),
                    td(links),
                    td(pdata.description),
                    td(pd.Timestamp(pdata.timestamp).strftime('%Y-%m-%d %H:%M:%S')),
                    td(stat),
                )
                _is_first = False
            report += "<tr>" + "<td class='empty-bar'/>" * 5 + "</tr>"
        report += '</table>'
    except Exception as e:
        trace = ''
        if DEBUG:
            import traceback
            trace = traceback.format_exc().replace('\n', '<br/>')
        report = f"<h2><font color='red'>Exception: {str(e)} / </font> </h2><br/> {trace}"

    return render_template('experiments_index.html', main_report=report, version=VERSION)


@app.route('/experiments/report_chart/<project>/<experiment>')
def experiment_chart_report(project, experiment):
    __install_charts_theme()

    report = ''
    descr = ''
    try:
        rc = get_experiment_trend_report(project, experiment)
        _G = lambda c, t, color='green': ['dots', color, c[t], 'line', color, c[t]]
        descr = rc.data['description']

        g = LookingGlass(_G(rc.chart, 'Gain'), {
            'Gain %': _G(rc.chart, 'GainPct', 'orange'),
            '% per trade': _G(rc.chart, 'PctPerTrade', 'orange'),
            'Loosers': _G(rc.chart, 'Loosers', 'white'),
        }).look(
            title=experiment
        ).update_xaxes(
            showline=True, linewidth=2, linecolor='#104040',
            showgrid=True, gridwidth=1, gridcolor='#202020',

            showspikes=True, spikemode='across',
            spikesnap='cursor', spikecolor='#306020',
            spikethickness=1, spikedash='dot'

        ).update_yaxes(
            showline=True, linewidth=2, linecolor='#104040',
            showgrid=True, gridwidth=1, gridcolor='#202020',
            tickformat=f".3f",
            zeroline=True, zerolinewidth=2, zerolinecolor='#401010',

            spikesnap='cursor', spikecolor='#306020', spikethickness=1,
        ).update_layout(
            # height=600, 
            hovermode="x unified",
            # showlegend=False, 
            hoverdistance=0,
            # xaxis={'hoverformat': '%d-%b-%y %H:%M'}, 
            # yaxis={'hoverformat': f'.{2}f'},
            dragmode='zoom',
            # newshape=dict(line_color='#f090ff', line_width=1.5, fillcolor='#101010', opacity=0.75),
            # modebar_remove=['lasso', 'select'],
            # modebar_add=['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape'],
        )
        report = json.dumps(g, cls=plotly.utils.PlotlyJSONEncoder)

    except Exception as e:
        trace = ''
        if DEBUG:
            import traceback
            trace = traceback.format_exc().replace('\n', '<br/>')
        report = f"<h2><font color='red'>Exception: {str(e)} / </font> </h2><br/> {trace}"

    return render_template(
        'experiment_chart_report.html',
        main_report=report,
        # param_report=param_report,
        experiment_id=experiment,
        project_id=project,
        description=descr,
        table_report=f'/booster/experiments/report/{project}/{experiment}?full=true'
    )


@app.route('/experiments/report/<project>/<experiment>')
def experiment_report(project, experiment):
    try:
        # if true we want to see full parameters set
        full_parameters_list = request.args.get('full', 'false').lower() in ['true', '1', 'yes']
        vert_parameters_list = request.args.get('vert', 'false').lower() in ['true', '1', 'yes']

        data = b_ld(f'portfolios/{project}/{experiment}')
        param_report = ''

        def highlight_neg(s):
            if s.dtype == object:
                is_neg = [False for _ in range(s.shape[0])]
                is_nan = [False for _ in range(s.shape[0])]
            else:
                is_neg = s < 0
                is_nan = np.isnan(s)
            # return ['color: red !important;' if cell else None for cell in is_neg]
            return ['color: red !important;' if m else 'color: #00000000 !important;' if n else None for m, n in
                    zip(is_neg, is_nan)]

        if data is not None and data.get('report') is not None:
            total_capital = data.get('total_capital', np.nan)
            capital_per_run = data.get('capital_per_run', np.nan)
            with pd.option_context('display.precision', 2):
                idx = pd.IndexSlice
                df = data['report']
                sims = data['simulations']
                n_sets = len(df.columns.levels[0])
                n_cols = len(df.columns.levels[1])

                # fix columns a bit
                # df.columns = pd.MultiIndex.from_product(
                # [df.columns.levels[0].map(lambda x: x.split('|')[0]).values, df.columns.levels[1].values])

                # make links to results
                if sims:
                    for set_name, s_info in sims.items():
                        x_set_name = set_name.split('|')[0]
                        for symbol, spath in s_info.items():
                            ex_val = df.loc[symbol, (x_set_name, 'Execs')]
                            df.loc[symbol, (x_set_name,
                                            'Execs')] = f'<a href="/booster/tearsheet?project={project}&entry={experiment}_PORTFOLIO&path={spath}&capital={capital_per_run}&type=runs" onclick="return show_modal(this);">{ex_val:.0f}</a>'

                        # total cell is for whole portfolio
                        ex_val = df.loc['Total', (x_set_name, 'Execs')]
                        df.loc['Total', (x_set_name,
                                         'Execs')] = f'<a href="/booster/tearsheet?project={project}&entry={experiment}&set_name={set_name}&capital={total_capital}&type=portfolios" onclick="return show_modal(this);">{ex_val:.0f}</a>'
                        # remove data from averge cell
                        df.loc['Average per trade', (x_set_name, 'Execs')] = ''

                report = df.style.apply(highlight_neg).set_properties(
                    subset=df.columns[n_cols - 1::n_cols].values,
                    **{
                        'border-right-width': '10px !important;',
                        'border-right-color': '#305030;'
                    }
                ).apply(
                    lambda x: ['border-top-color: #305030 !important; border-top-width: 2px !important;' for _ in
                               range(x.shape[0])],
                    subset=idx['Total', :]
                ).to_html()

                # - sets parameters
                p_report = portfolios_parameters_sets(data, full_parameters_list)
                if p_report is not None:
                    def highlight_changed(s):
                        if isinstance(s[0], (list, tuple, np.ndarray)):
                            is_changed = pd.Series([x != s[0] for x in s], index=s.index)
                        else:
                            is_changed = s != s[0]
                        return ['color: red !important; font-weight: bolder !important;' if m else '' for m in
                                is_changed]

                    if len(sims) > 10:
                        vert_parameters_list = True

                    param_report = (p_report.T if vert_parameters_list else p_report).style.apply(highlight_changed, axis=1).to_html()

        else:
            report = f'<h1>No data for {project}/{experiment} !</h1>'
    except Exception as e:
        trace = ''
        if DEBUG:
            import traceback
            trace = traceback.format_exc().replace('\n', '<br/>')
        report = f"<h2><font color='red'>Exception: {str(e)} / </font> </h2><br/> {trace}"
        param_report = ''
        data = {}

    return render_template(
        'experiment_report.html',
        main_report=report,
        param_report=param_report,
        experiment_id=experiment,
        project_id=project,
        description=data.get('description', '- - -'),
        chart_report=f'/booster/experiments/report_chart/{project}/{experiment}'
    )


@app.route('/tearsheet')
def tsheet_report():
    __install_charts_theme()
    try:
        project = request.args.get('project')
        entry = request.args.get('entry')
        path = request.args.get('path')
        r_type = request.args.get('type', 'runs')
        set_name = request.args.get('set_name')
        capital = float(request.args.get('capital', 0.0))
        insample = request.args.get('insample')

        if r_type.lower() == 'portfolios':
            path = set_name

        sheet = tearsheet_report(r_type, project, path, entry, capital, insample=insample)
        report = render_template(
            'tearsheet_report.html',
            title_id=entry,
            tearsheet_report=sheet,
            signals_report="<h2><font color='red'> Not implemented yet ... </font> </h2>",
            drawdown_report="<h2><font color='red'> Not implemented yet ... </font> </h2>",
        )

    except Exception as e:
        trace = ''
        if DEBUG:
            import traceback
            trace = traceback.format_exc().replace('\n', '<br/>')
        report = f"<h2><font color='red'>Exception: {str(e)} / </font> </h2><br/> <font color='green'>{trace}</font>"

    return report


@app.route('/report/<project>')
def main_report(project):
    sorted_by = request.args.get('sorted', 'Gain')
    asc = request.args.get('asc', False)
    t_frame = request.args.get('timeframe', DEFAULT_TIMEFRAME)

    try:
        bpr = BoosterProgressReport(
            get_config_for_project(project),
            lambda prj, symbol,
                   entry: f"<a href='/booster/details/{prj}/{symbol}/{entry}' onclick='return show_modal(this);'>{symbol} <font color='#606060'> {entry} </font></a>",
            # for open it in dialog: test its usability
            # lambda prj, entry, symbol, model_name, nexecs: f"<a href='/booster/signals/{prj}/{symbol}/{entry}/{model_name}' onclick='return show_modal(this);'>{nexecs}</a>"
            lambda prj, entry, symbol, model_name,
                   nexecs: f"<a href='/booster/signals/{prj}/{symbol}/{entry}/{model_name}?timeframe={t_frame}' target='_x1'>{nexecs}</a>"
        )
        report = bpr.get_report(sorted_by, asc)
    except Exception as e:
        trace = ''
        if DEBUG:
            import traceback
            trace = traceback.format_exc().replace('\n', '<br/>')
        report = f"<h2><font color='red'>Exception: {str(e)} / </font> </h2><br/> {trace}"

    return render_template(
        'main_report.html',
        project_id=project,
        main_report=report
    )


@app.route('/details/<project>/<symbol>/<entry>')
def details_report(project, symbol, entry):
    try:
        bpr = BoosterProgressReport(get_config_for_project(project))
        report = bpr.show_detailed_report(project, symbol.upper(), entry)
        params = bpr.show_parameters_report(project, entry, weight_capital=True)

        report = render_template(
            'detailed_report.html',
            symbol_id=symbol,
            detailed_report=report,
            parameters_report=params.models_parameters,
            detailed_report_1=params.models_performance.get(params.models_names[0], ''),
            # KLR-111: fix for single model
            detailed_report_2=params.models_performance.get(params.models_names[1], '') if len(
                params.models_names) > 1 else '',
        )

    except Exception as e:
        trace = ''
        if DEBUG:
            import traceback
            trace = traceback.format_exc().replace('\n', '<br/>')
        report = f"<h2><font color='red'>Exception: {str(e)} / </font> </h2><br/> <font color='green'>{trace}</font>"

    return report


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# global viewer data
_g: ViewerData = None


@app.route('/callback', methods=['POST', 'GET'])
def viewer_callbacks():
    """
    Callback on parameters
    """
    try:
        global _g
        if _g is None:
            return {}
        _g.start = pd.Timestamp(request.args.get('start_date'))
        _g.stop = pd.Timestamp(request.args.get('end_date'))
        n_tf = request.args.get('timeframe')
        if n_tf and n_tf != _g.timeframe:
            _g.timeframe = n_tf

        return get_view_json(_g)
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"{str(e)} / {trace}")


@app.route('/signals/<project>/<symbol>/<entry>/<modelname>')
def signals_viewer(project, symbol, entry, modelname):
    # timeframe to view data
    t_frame = request.args.get('timeframe', DEFAULT_TIMEFRAME)
    _start_date = request.args.get('start_date')
    _end_date = request.args.get('end_date')

    try:
        from qube.utils.nb_functions import z_ld
        global _g

        # loading blended report
        boo = Booster(get_config_for_project(project), reload_config=True)
        b_data = boo.load_blended_results(entry)

        if b_data is None:
            _g = {}
            return

        # loading OHLC data
        ohlc = z_ld(f'm1/{b_data["config"]["instrument"]}', dbname='md')

        # fetching signals
        signals = [json.loads(s) for s in b_data['backtest'].trackers_stat[symbol]['signals']]

        # min / max period for viewer
        s_tF = pd.Timestamp(signals[0]['time'])
        e_time = aslist(signals[-1]['risk_hit_time'])[-1]

        if not e_time or e_time is None:
            e_time = signals[-1]['time']
        s_tL = pd.Timestamp(e_time)

        # default view period is 1w from 1'st signal
        s_t0 = pd.Timestamp(_start_date) if _start_date else s_tF
        s_t1 = pd.Timestamp(_end_date) if _end_date else s_tF + pd.Timedelta('1w')

        # create global object for viewing signals
        _g = ViewerData(symbol, entry, modelname, s_t0, s_t1, ohlc, signals, timeframe=t_frame)

        graphJSON = get_view_json(_g)
        return render_template('signals_viewer.html',
                               graphJSON=graphJSON,
                               start_date=_g.start.strftime('%Y-%m-%d'),
                               end_date=_g.stop.strftime('%Y-%m-%d'),
                               min_date=s_tF.strftime('%Y-%m-%d'),
                               max_date=s_tL.strftime('%Y-%m-%d'),
                               header='Test', description='Test')
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"{str(e)} / {trace}")


def get_view_json(gv: ViewerData):
    """
    Signals viewer as json
    """
    sview = show_signals(gv.ohlc, gv.signals, gv.start, gv.stop, gv.timeframe,
                         title=f"{gv.entry}/{gv.model}/{gv.symbol}")
    return json.dumps(sview, cls=plotly.utils.PlotlyJSONEncoder)


if __name__ == '__main__':
    print(' %%% Starting debug application ....')
    app.run(host='0.0.0.0', port=59567, debug=False, ssl_context=('certs/cert.pem', 'certs/key.pem'))
