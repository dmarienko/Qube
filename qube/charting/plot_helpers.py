"""
   Misc graphics handy utilitites to be used in interactive analysis
"""
import itertools as it
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as st

try:
    import matplotlib
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
except:
    print(" >>> Can't import matplotlib modules in qube charting modlue")

from qube.utils.DateUtils import DateUtils
from qube.quantitative.tools import isscalar


def install_plotly_helpers():
    try:
        from plotly.graph_objs.graph_objs import FigureWidget
        import plotly.graph_objects as go

        def rline_(look, x, y, c='red', lw=1):
            """
            Ray line
            """
            return look.update_layout(shapes=(
                dict(type="line", xref="x1", yref="y1",
                     x0=pd.Timestamp(x), y0=y, x1=look.data[0]['x'][-1], y1=y,
                     fillcolor=c, opacity=1, line=dict(color=c, width=lw))), overwrite=False)

        def rline(look, x, y, c='red', lw=1, ls=None):
            return look.add_shape(go.layout.Shape(type="line",
                                                  x0=pd.Timestamp(x), x1=look.data[0]['x'][-1], y0=y, y1=y,
                                                  xref='x1', yref='y1', line=dict(width=lw, color=c, dash=ls)))

        def rlinex(look, x0, x1, y, c='red', lw=1, ls=None):
            return look.add_shape(go.layout.Shape(type="line",
                                                  x0=pd.Timestamp(x0), x1=pd.Timestamp(x1), y0=y, y1=y,
                                                  xref='x1', yref='y1', line=dict(width=lw, color=c, dash=ls)))

        def dliney(look, x0, y0, y1, c='red', lw=1, ls=None):
            return look.add_shape(go.layout.Shape(type="line",
                                                  x0=pd.Timestamp(x0), x1=pd.Timestamp(x0), y0=y0, y1=y1,
                                                  xref='x1', yref='y1', line=dict(width=lw, color=c, dash=ls)))

        def vline(look, x, c='yellow', lw=1, ls='dot'):
            return look.add_shape(go.layout.Shape(type="line",
                                                  x0=pd.Timestamp(x), x1=pd.Timestamp(x), y0=0, y1=1,
                                                  xref='x1', yref='paper', line=dict(width=lw, color=c, dash=ls)))

        def arrow(look, x2, y2, x1, y1, c='red', text='', lw=1, font=dict(size=8)):
            return look.add_annotation(x=x1, y=y1, ax=x2, ay=y2, xref='x', yref='y', axref='x', ayref='y', text=text,
                                       font=font,
                                       showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=lw, arrowcolor=c)

        def custom_hover(v, h=600, n=2, legend=False, show_info=True):
            return v.update_traces(
                xaxis="x1"
            ).update_layout(
                height=h, hovermode="x unified",
                showlegend=legend,
                hoverdistance=1000 if show_info else 0,
                xaxis={'hoverformat': '%d-%b-%y %H:%M'},
                yaxis={'hoverformat': f'.{n}f'},
                dragmode='zoom',
                newshape=dict(line_color='yellow', line_width=1.),
                modebar_add=['drawline', 'drawopenpath', 'drawrect', 'eraseshape'],
            ).update_xaxes(
                showspikes=True, spikemode='across',
                spikesnap='cursor', spikecolor='#306020',
                spikethickness=1, spikedash='dot'
            ).update_yaxes(
                spikesnap='cursor', spikecolor='#306020',
                tickformat=f".{n}f", spikethickness=1,
            )

        FigureWidget.hover = custom_hover
        FigureWidget.rline = rline
        FigureWidget.rlinex = rlinex
        FigureWidget.rline_ = rline_
        FigureWidget.vline = vline
        FigureWidget.dliney = dliney
        FigureWidget.arrow = arrow
    except:
        print(" >>> Cant attach helpers to plotly::FigureWidget - probably it isn't installed !")

# - install plotly helpers
# install_plotly_helpers()


def multiplot(frame: pd.DataFrame, names: Union[List, Tuple, str] = None, pos=None, figsize=None,
              title=None, colors=('#40a0ff', '#40c040', '#ff3030', 'aquamarine', 'w', 'y', 'm'),
              loc=0, x_format='%d-%b %H:%M', y_format='%.1f', lw=(1,), ls=('-',),
              tz=DateUtils.DEFAULT_TIME_ZONE):
    """
    Small helping function for drawing multiple series each in separate scale (axis) but on same plot

    Example:
    ---------
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    from qube.charting.plot_helpers import miltiplot

    data = pd.DataFrame(data=np.random.randn(1000,4).cumsum(axis=0),
                        index=pd.date_range('1/1/2000 00:00:00', periods=1000, freq='30s'),
                        columns=['Ser1', 'Ser2', 'Ser3', 'Ser4'])

    fig = plt.figure(figsize=(16,8))
    fig.subplots_adjust(right=1.5)

    multiplot(data, ['Ser1', 'Ser2', 'Ser3'],
              pos=(1,2,1), loc=3, x_format='%H:%M:%S', lw=[1,2], ls=['-', '-', '--', '--'])
    multiplot(data, ['Ser3', 'Ser4', 'Ser1'],
              pos=(1,2,2), loc=3, x_format='%H:%M:%S', lw=[1,2], ls=['-', '--', '-', '--'])

    plt.draw()
    plt.show()
    ---------

    :param frame: data source pandas dataframe
    :param names: column names to be drawn
    :param pos: position at subplot area
    :param figsize: figure size (if set it creates new figure)
    :param title: plot title
    :param colors: line colors
    :param loc: legend location
    :param x_format: x-axis format (usually for time index)
    :param y_format: y-axis format
    :param lw: lines width array
    :param ls: lines styles array
    :param tz: timezone for matplotlib
    """

    if not pos: pos = (1, 1, 1)

    # here temporary hack - mpl doesn't use pandas timezone correctly
    matplotlib.rcParams['timezone'] = tz

    __fig = None
    if figsize:
        __fig = plt.figure(figsize=figsize)

    ax1 = plt.subplot(*pos)
    if figsize and __fig is not None:
        __fig.subplots_adjust(right=0.75)

    if names is None:
        names = frame.columns.tolist()

    if not isinstance(names, (list, tuple)): names = [names]

    colors_lst = list(it.islice(it.cycle(colors), len(names)))
    line_widths = list(it.islice(it.cycle(lw), len(names)))
    line_styles = list(it.islice(it.cycle(ls), len(names)))

    def _plt_decor(axn, data, clr, label, lw=1, ls='-'):
        p, = axn.plot(data, label=label, color=clr, lw=lw, ls=ls)
        axn.set_ylabel(n)
        axn.yaxis.label.set_color(clr)
        axn.tick_params(axis='y', color=clr, labelcolor=clr)
        axn.spines['right'].set_color(clr)
        axn.xaxis.set_major_formatter(mdates.DateFormatter(x_format))
        axn.yaxis.set_major_formatter(mticker.FormatStrFormatter(y_format))
        axn.xaxis_date(tz)
        axn.tick_params(axis='x', which='major', labelsize=8)
        axn.tick_params(axis='x', which='minor', labelsize=8)
        axn.grid(False, axis='y')
        return p

    plts = []
    for i, n in enumerate(names):
        axn = []
        if i == 0:
            axn = ax1
        else:
            axn = ax1.twinx()
            axn.spines["right"].set_position(("axes", 1 + (i - 1) * 0.08))

        plts.append(_plt_decor(axn, frame[n], colors_lst[i], n, lw=line_widths[i], ls=line_styles[i]))

    ax1.legend(plts, [l.get_label() for l in plts], loc=loc)
    ax1.grid(True, axis='y')
    ax1.spines['bottom'].set_color('w')

    if title:
        ax1.set_title(title)

    plt.draw()

    return ax1


def smultiplot(*args, **kwargs):
    """
    Multiplot helper (may combine multiple series before using of multiplot function)

    >>> idx = pd.date_range('2000-01-01', periods=500)
    >>> smultiplot(
    >>>    pd.Series(np.random.randn(len(idx)), index=idx, name='Series1').cumsum(),
    >>>    pd.Series(np.random.randn(len(idx)), index=idx).cumsum(),
    >>>    zoom=('2000-01', '2000-02'))
    """
    r = pd.DataFrame()
    for i, s in enumerate(args):
        if isinstance(s, (pd.Series, pd.DataFrame)):
            if not s.name:
                s = s.rename('Argument_%d' % i)
            r = pd.concat((r, s), axis=1)
    _zmx = kwargs.pop('zoom', None)
    if _zmx:
        if isinstance(_zmx, (tuple, list)):
            _zmx = slice(_zmx[0], _zmx[1] if len(_zmx) > 1 else _zmx[0])
        r = r[_zmx]
    return multiplot(r, **kwargs)


def fig(w=16, h=5, dpi=96, facecolor=None, edgecolor=None, num=None):
    """
    Simple helper for creating figure
    """
    return plt.figure(num=num, figsize=(w, h), dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)


def subplot(shape, loc, rowspan=2, colspan=1):
    """
    Some handy grid splitting for plots. Example for 2x2:
    
    >>> subplot(22, 1); plt.plot([-1,2,-3])
    >>> subplot(22, 2); plt.plot([1,2,3])
    >>> subplot(22, 3); plt.plot([1,2,3])
    >>> subplot(22, 4); plt.plot([3,-2,1])

    same as following

    >>> subplot((2,2), (0,0)); plt.plot([-1,2,-3])
    >>> subplot((2,2), (0,1)); plt.plot([1,2,3])
    >>> subplot((2,2), (1,0)); plt.plot([1,2,3])
    >>> subplot((2,2), (1,1)); plt.plot([3,-2,1])

    :param shape: scalar (like matlab subplot) or tuple
    :param loc: scalar (like matlab subplot) or tuple
    :param rowspan: rows spanned
    :param colspan: columns spanned
    """
    if isscalar(shape):
        if 0 < shape < 100:
            shape = (max(shape // 10, 1), max(shape % 10, 1))
        else:
            raise ValueError("Wrong scalar value for shape. It should be in range (1...99)")

    if isscalar(loc):
        nm = max(shape[0], 1) * max(shape[1], 1)
        if 0 < loc <= nm:
            x = (loc - 1) // shape[1]
            y = loc - 1 - shape[1] * x
            loc = (x, y)
        else:
            raise ValueError("Wrong scalar value for location. It should be in range (1...%d)" % nm)

    return plt.subplot2grid(shape, loc=loc, rowspan=rowspan, colspan=colspan)


def plot_pacf(x, lags: int = 30, alpha=0.05):
    """
    Draw partial autocorrelation function within confidence intervals (if alpha is passed)

    :param x: series
    :param lags: number of lags
    :param alpha: confident level (0.0 ... 1.0)
    """
    # clean up na
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.dropna()

    if alpha is not None:
        pcf, ci = st.pacf(x, nlags=lags, alpha=alpha)
    else:
        pcf = st.pacf(x, nlags=lags)

    n_lags = range(1, len(pcf))
    pcf_x = pcf[1:]
    plt.stem(n_lags, pcf_x, use_line_collection=True)

    if alpha is not None:
        plt.fill_between(n_lags, ci[1:, 0] - pcf_x, ci[1:, 1] - pcf_x, alpha=.20)

    plt.xticks(n_lags)
    plt.title('Partial Autocorrelation')


def plot_acf(x, lags: int = 30, alpha=0.05):
    """
    Draw autocorrelation function within confidence intervals (if alpha is passed)

    :param x: series
    :param lags: number of lags
    :param alpha: confident level (0.0 ... 1.0)
    """
    # clean up na
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.dropna()

    if alpha is not None:
        pcf, ci = st.acf(x, nlags=lags, alpha=alpha)
    else:
        pcf = st.acf(x, nlags=lags)

    n_lags = range(1, len(pcf))
    pcf_x = pcf[1:]
    plt.stem(n_lags, pcf_x, use_line_collection=True)

    if alpha is not None:
        plt.fill_between(n_lags, ci[1:, 0] - pcf_x, ci[1:, 1] - pcf_x, alpha=.20)

    plt.xticks(n_lags)
    plt.title('Autocorrelation')


def autoscale_y(ax, margin=0.1):
    """
    This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims
    """

    def __to_ordinal_time(dt):
        dtt = pd.Timestamp(dt).to_pydatetime()
        ts = (dtt.hour * 60 + dtt.minute) * 60 + dtt.second + dtt.microsecond * 10 ** (-6)
        return dtt.toordinal() + ts / (24 * 3600)

    def __get_bottom_top(line):
        xd = line.get_xdata()
        if isinstance(xd[0], np.datetime64):
            xd = [__to_ordinal_time(dt) for dt in xd]
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        y_displayed = yd[((xd > lo) & (xd < hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed) - margin * h
        top = np.max(y_displayed) + margin * h
        return bot, top

    lines = ax.get_lines()
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = __get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot, top)


def zoomx(x1, x2, autoscale=True, margin=0.1, figure=None):
    """
    Zoom active figure on X axis

    :param x1:
    :param x2:
    :param autoscale:
    :param margin:
    :param figure: affected figure (if None it will affect all active figures)
    :return:
    """
    xset = lambda x1, x2, arr: [x1, x2, arr[2], arr[3]]
    if figure is None:
        for i in plt.get_fignums():
            for axi in plt.figure(i).axes:
                axi.axis(xset(x1, x2, axi.axis()))
                if autoscale: autoscale_y(axi, margin)
    else:
        for axi in figure.axes:
            axi.axis(xset(x1, x2, axi.axis()))
            if autoscale: autoscale_y(axi, margin)


def plot_trends(trends, uc='w--', dc='c--', lw=0.7, ms=5, fmt='%H:%M'):
    """
    Plot find_movements function output as trend lines on chart

    >>> from qube.quantitative.ta.swings.swings_splitter import find_movements
    >>>
    >>> tx = pd.Series(np.random.randn(500).cumsum() + 100, index=pd.date_range('2000-01-01', periods=500))
    >>> trends = find_movements(tx, np.inf, use_prev_movement_size_for_percentage=False,
    >>>                    pcntg=0.02,
    >>>                    t_window=np.inf, drop_weekends_crossings=False,
    >>>                    drop_out_of_market=False, result_as_frame=True, silent=True)
    >>> plot_trends(trends)

    :param trends: find_movements output
    :param uc: up trends line spec ('w--')
    :param dc: down trends line spec ('c--')
    :param lw: line weight (0.7)
    :param ms: trends reversals marker size (5)
    :param fmt: time format (default is '%H:%M')
    """
    if not trends.empty:
        u, d = trends.UpTrends.dropna(), trends.DownTrends.dropna()
        plt.plot([u.index, u.end], [u.start_price, u.end_price], uc, lw=lw, marker='.', markersize=ms);
        plt.plot([d.index, d.end], [d.start_price, d.end_price], dc, lw=lw, marker='.', markersize=ms);

        from matplotlib.dates import num2date
        import datetime
        ax = plt.gca()
        ax.set_xticklabels([datetime.date.strftime(num2date(x), fmt) for x in ax.get_xticks()])


def sbp(shape, loc, r=1, c=1):
    """
    Just shortcut for subplot(...) function

    :param shape: scalar (like matlab subplot) or tuple
    :param loc: scalar (like matlab subplot) or tuple
    :param r: rows spanned
    :param c: columns spanned
    :return:
    """
    return subplot(shape, loc, rowspan=r, colspan=c)


def vline(ax, x, c, lw=1, ls='--'):
    x = pd.to_datetime(x) if isinstance(x, str) else x
    if not isinstance(ax, (list, tuple)):
        ax = [ax]
    for a in ax:
        a.axvline(x, 0, 1, c=c, lw=1, linestyle=ls)


def hline(*zs, mirror=True):
    [plt.axhline(z, ls='--', c='r', lw=0.5) for z in zs]
    if mirror:
        [plt.axhline(-z, ls='--', c='r', lw=0.5) for z in zs]


def ellips(ax, x, y, c='r', r=2.5, lw=2, ls='-'):
    """
    Draw ellips annotation on specified plot at (x,y) point
    """
    from matplotlib.patches import Ellipse
    x = pd.to_datetime(x) if isinstance(x, str) else x
    w, h = (r, r) if np.isscalar(r) else (r[0], r[1])
    ax.add_artist(Ellipse(xy=[x, y], width=w, height=h, angle=0, fill=False, color=c, lw=lw, ls=ls))


def plot_fractals(frs: pd.DataFrame, uc='y', lc='w'):
    """
    Plot fractals indicator
    """
    if not (isinstance(frs, pd.DataFrame) and all(frs.columns.isin(['L', 'U']))):
        raise ValueError('Wrong input data: should be pd.DataFrame amd contains "L" and "U" columns !')
    [plt.plot(t, v, marker=6, c=uc) for t, v in frs.L.dropna().iteritems()];
    [plt.plot(t, v, marker=7, c=lc) for t, v in frs.U.dropna().iteritems()];


def glow_effects(ax: Optional[plt.Axes] = None) -> None:
    """Add a glow effect to the lines in an axis object and an 'underglow' effect below the line."""
    make_lines_glow(ax=ax)


def make_lines_glow(ax: Optional[plt.Axes] = None,
                    n_glow_lines: int = 10, diff_linewidth: float = 1.05,
                    alpha_line: float = 0.3, lines: Union[Line2D, List[Line2D]] = None) -> None:
    """Add a glow effect to the lines in an axis object.
    Each existing line is redrawn several times with increasing width and low alpha to create the glow effect.
    """
    if not ax:
        ax = plt.gca()

    lines = ax.get_lines() if lines is None else lines
    lines = [lines] if isinstance(lines, Line2D) else lines
    alpha_value = alpha_line / n_glow_lines

    for line in lines:
        data = line.get_data(orig=False)
        linewidth = line.get_linewidth()

        try:
            step_type = line.get_drawstyle().split('-')[1]
        except:
            step_type = None

        for n in range(1, n_glow_lines + 1):
            if step_type:
                glow_line, = ax.step(*data)
            else:
                glow_line, = ax.plot(*data)
            # line properties are copied as seen in this solution: https://stackoverflow.com/a/54688412/3240855
            glow_line.update_from(line)

            glow_line.set_alpha(alpha_value)
            glow_line.set_linewidth(linewidth + (diff_linewidth * n))
            glow_line.is_glow_line = True  # mark the glow lines, to disregard them in the underglow function.
