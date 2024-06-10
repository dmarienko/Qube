import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qube.charting.mpl_finance import ohlc_plot
from qube.charting.plot_helpers import plot_trends
from qube.charting.plot_helpers import subplot
from qube.utils.utils import mstruct


class AbstractLookingGlass:
    """
    Handy utility for plotting data
    """

    def __init__(self, master, studies: dict, title=""):
        self.m = master
        self.s = {} if studies is None else studies
        self._title = title

    def look(self, *args, title=None, **kwargs):
        _vert_bar = None
        zoom = None

        if len(args) == 1:
            zoom = args[0]
        elif len(args) > 1:
            zoom = args

        if zoom and not isinstance(zoom, slice):
            zoom = zoom if isinstance(zoom, (list, tuple)) else [zoom]

            if len(zoom) == 2:
                z0, is_d_0 = self.__as_time(zoom[0])
                z1, is_d_1 = self.__as_time(zoom[1])

                if is_d_0:
                    if is_d_1:
                        raise ValueError(
                            "At least one of zoom values must be timestamp !"
                        )
                    zoom = slice(z1 - z0, z1)
                else:
                    zoom = slice(z0, (z0 + z1) if is_d_1 else z1)
            elif len(zoom) > 2:
                z0, is_d_0 = self.__as_time(zoom[0])
                z1, is_d_1 = self.__as_time(zoom[1])
                z2, is_d_2 = self.__as_time(zoom[2])

                if is_d_1:
                    raise ValueError("Second argument must be timestamp !")
                if not is_d_0:
                    raise ValueError("First argument must be timedelta !")
                if not is_d_2:
                    raise ValueError("Third argument must be timedelta !")

                zoom = slice(z1 - z0, z1 + z2)
                _vert_bar = z1
            elif len(zoom) == 1:
                import datetime

                z1, is_d = self.__as_time(zoom[0])
                if is_d:
                    raise ValueError("Argument must be date time not timedelta !")

                if z1.time() == datetime.time(0, 0):
                    shift = pd.Timedelta(kwargs.get("shift", "24h"))
                    zoom = slice(z1, z1 + shift)
                else:
                    shift = pd.Timedelta(kwargs.get("shift", "12h"))
                    zoom = slice(z1 - shift, z1 + shift)
                    _vert_bar = z1
            else:
                raise ValueError("Don't know how to interpret '%s'" % str(zoom))

        return self._show_plot(_vert_bar, title, zoom)

    def _frame_has_cols(self, df, cols):
        return isinstance(df, pd.DataFrame) and all(x in df.columns for x in cols)

    def __as_time(self, z):
        _is_delta = False
        if isinstance(z, str):
            try:
                z = pd.Timedelta(z)
                _is_delta = True
            except:
                try:
                    z = pd.Timestamp(z)
                except:
                    raise ValueError("Value '%s' can't be recognized" % z)
        else:
            _is_delta = isinstance(z, pd.Timedelta)
        return z, _is_delta

    def _show_plot(self, _vert_bar, title, zoom):
        raise NotImplementedError(
            "Must be implemented in child class %s", self.__class__.__name__
        )


class LookingGlass:

    def __init__(
        self, master, studies: dict = None, title="", backend="plotly", **kwargs
    ):
        if backend in ["matplotlib", "mpl"]:
            self.__instance = LookingGlassMatplotLib(
                master=master, studies=studies, title=title, **kwargs
            )
        elif backend in ["plotly", "ply", "plt"]:
            self.__instance = LookingGlassPlotly(
                master=master, studies=studies, title=title, **kwargs
            )
        else:
            raise ValueError("Backend %s is not recognized" % backend)

    def look(self, *args, **kwargs):
        return self.__instance.look(*args, **kwargs)


class LookingGlassMatplotLib(AbstractLookingGlass):

    def __init__(
        self,
        master,
        studies: dict = None,
        master_size=3,
        study_size=1,
        title="",
        legend_loc="upper left",
        fmt="%H:%M",
        ohlc_width=0,
    ):
        super().__init__(master, studies, title)
        self.s_size = study_size
        self.m_size = master_size
        self.legend_loc = legend_loc
        self._fmt = fmt
        self._ohlc_width = ohlc_width
        self._n_style = "-"

    def __plt_series(self, y, zoom, study_name, k, plot_style="line"):
        _forced_limits = []
        if isinstance(y, (int, float)):
            try:
                plt.axhline(y, lw=0.5, ls=self._n_style)
            except:
                plt.axhline(y, lw=0.5, ls="--")
        else:
            _lbl = (
                y.name if hasattr(y, "name") and y.name else ("%s_%d" % (study_name, k))
            )

            if isinstance(y, pd.DataFrame):

                yy = y[zoom] if zoom else y

                # reversal points
                if self._frame_has_cols(y, ["start_price", "delta"]):
                    _lo_pts = yy[yy.delta > 0].rename(columns={"start_price": "Bottom"})
                    _hi_pts = yy[yy.delta < 0].rename(columns={"start_price": "Top"})
                    plt.plot(
                        _lo_pts.index,
                        _lo_pts.Bottom,
                        marker=6,
                        markersize=5,
                        ls="",
                        c="#10aa10",
                    )
                    plt.plot(
                        _hi_pts.index, _hi_pts.Top, marker=7, markersize=5, ls="", c="r"
                    )

                # trends
                elif self._frame_has_cols(y, ["UpTrends", "DownTrends"]):
                    plot_trends(yy, fmt=self._fmt)

                # tracks
                elif self._frame_has_cols(y, ["Type", "Time", "Price", "PriceOccured"]):
                    _bot = yy[yy.Type == "-"]
                    _top = yy[yy.Type == "+"]
                    plt.plot(
                        _bot.index,
                        _bot.PriceOccured.rename("B"),
                        marker=5,
                        markersize=8,
                        ls="",
                        c="#1080ff",
                    )
                    plt.plot(
                        _top.index,
                        _top.PriceOccured.rename("T"),
                        marker=5,
                        markersize=8,
                        ls="",
                        c="#909010",
                    )

                # executed signals from signal tester
                elif self._frame_has_cols(y, ["exec_price", "quantity"]):
                    _b_ords = yy[yy.quantity > 0]
                    _s_ords = yy[yy.quantity < 0]
                    plt.plot(
                        _b_ords.index,
                        _b_ords.exec_price.rename("BOT"),
                        marker="2",
                        markersize=10,
                        ls="",
                        c="#1080ff",
                    )
                    plt.plot(
                        _s_ords.index,
                        _s_ords.exec_price.rename("SLD"),
                        marker="1",
                        markersize=10,
                        ls="",
                        c="#909010",
                    )

                # order executions from simulator
                elif self._frame_has_cols(
                    y, ["side", "fill_avg_price", "quantity", "status"]
                ):
                    _b_ords = yy[yy.side == "BUY"]
                    _s_ords = yy[yy.side == "SELL"]
                    plt.plot(
                        _b_ords.index,
                        _b_ords.fill_avg_price.rename("BOT"),
                        marker="2",
                        markersize=10,
                        ls="",
                        c="#1080ff",
                    )
                    plt.plot(
                        _s_ords.index,
                        _s_ords.fill_avg_price.rename("SLD"),
                        marker="1",
                        markersize=10,
                        ls="",
                        c="#909010",
                    )

                # experimental tester view of trading log
                elif self._frame_has_cols(y, ["Action", "Price", "Info", "PnL_ticks"]):
                    _b_ords = yy[yy.Action == "Long"]
                    _s_ords = yy[yy.Action == "Short"]
                    _t_ords = yy[yy.Action == "Take"]
                    _l_ords = yy[yy.Action == "Stop"]
                    _e_ords = yy[(yy.Action == "Expired") | (yy.Action == "Flat")]
                    plt.plot(
                        _b_ords.index,
                        _b_ords.Price.rename("BOT"),
                        marker=6,
                        markersize=10,
                        ls="",
                        c="#3cfa00",
                    )
                    plt.plot(
                        _s_ords.index,
                        _s_ords.Price.rename("SLD"),
                        marker=7,
                        markersize=10,
                        ls="",
                        c="#20ffff",
                    )
                    plt.plot(
                        _t_ords.index,
                        _t_ords.CurrentPrice.rename("Take"),
                        marker="P",
                        markersize=10,
                        ls="",
                        c="#fffb00",
                    )
                    plt.plot(
                        _l_ords.index,
                        _l_ords.CurrentPrice.rename("Stop"),
                        marker="8",
                        markersize=10,
                        ls="",
                        c="#fffb00",
                    )
                    plt.plot(
                        _e_ords.index,
                        _e_ords.CurrentPrice.rename("Exp"),
                        marker="X",
                        markersize=10,
                        ls="",
                        c="#b0b0b0",
                    )

                elif self._frame_has_cols(y, ["open", "high", "low", "close"]):
                    ohlc_plot(yy, width=self._ohlc_width, fmt=self._fmt)
                    _forced_limits = 0.999 * min(yy["low"]), 1.001 * max(yy["high"])
                    # temp hack to aling scales
                    plt.plot(yy["close"], lw=0, label=_lbl)
                else:
                    for _col in yy.columns:
                        self.__plot_as_type(yy[_col], plot_style, self._n_style, _col)
            else:
                yy = y[zoom] if zoom else y
                self.__plot_as_type(yy, plot_style, self._n_style, _lbl)

            # we want to see OHLC at maximal scale
            return _forced_limits

    def __plot_as_type(self, y, plot_style, line_style: str, label):
        __clr = (
            line_style[0] if len(line_style) > 0 and line_style[0].isalpha() else None
        )
        if plot_style == "line":
            plt.plot(y, line_style, label=label)
        elif plot_style == "area":
            plt.fill_between(y.index, y, color=__clr, label=label)
        elif plot_style.startswith("step"):
            _where = "post" if "post" in plot_style else "pre"
            plt.step(y.index, y, color=__clr, where=_where, label=label)
        elif plot_style.startswith("bar"):
            _bw = pd.Series(y.index).diff().mean().total_seconds() / 24 / 60 / 60
            plt.bar(
                y.index, y, lw=0.4, width=_bw, edgecolor=__clr, color=__clr, label=label
            )

    def _show_plot(self, vert_bar, title, zoom):

        # plot all master series
        shape = (self.s_size * len(self.s) + self.m_size, 1)
        subplot(shape, 1, rowspan=self.m_size)
        ms = self.m if isinstance(self.m, (tuple, list)) else [self.m]
        _limits_to_set = None

        for j, m in enumerate(ms):
            # if style description
            if isinstance(m, str):
                self._n_style = m
            else:
                _lims = self.__plt_series(m, zoom, "Master", j)
                self._n_style = "-"
                if _limits_to_set is None and _lims:
                    _limits_to_set = _lims

        # special case
        if _limits_to_set:
            plt.ylim(*_limits_to_set)

        if self.legend_loc:
            plt.legend(loc=self.legend_loc)

        if vert_bar:
            plt.axvline(vert_bar, ls="-.", lw=0.5)
            if title is None:
                plt.title("%s %s" % (self._title, str(vert_bar)))

        if title is not None:
            plt.title("%s %s" % (self._title, str(title)))

        # plot studies
        i = 1 + self.m_size
        for k, vs in self.s.items():
            subplot(shape, i, rowspan=self.s_size)
            vs = vs if isinstance(vs, (tuple, list)) else [vs]
            self._n_style = "-"
            plot_style = "line"

            wait_for_limits = False
            for j, v in enumerate(vs):

                # if we need to read limits
                if wait_for_limits:
                    if isinstance(v, (list, tuple)) and len(v) > 1:
                        plt.ylim(*v)
                    wait_for_limits = False
                    continue

                # if style description
                if isinstance(v, str):
                    vl = v.lower()
                    if vl.startswith("lim"):
                        wait_for_limits = True
                    elif any(
                        [
                            vl.startswith(x)
                            for x in ["line", "bar", "step", "stem", "area"]
                        ]
                    ):
                        plot_style = vl
                    else:
                        self._n_style = v
                else:
                    self.__plt_series(v, zoom, k, j, plot_style=plot_style)

            if vert_bar:
                plt.axvline(vert_bar, ls="-.", lw=0.5)

            i += self.s_size
            plt.legend(loc=self.legend_loc)

        self._n_style = "-"


class LookingGlassPlotly(AbstractLookingGlass):
    TREND_COLORS = mstruct(
        uline="#ffffff",
        dline="#ffffff",
        udot="rgb(10,168,10)",
        ddot="rgb(168,10,10)",
    )

    def __init__(
        self,
        master,
        studies: dict = None,
        master_plot_height=400,
        study_plot_height=100,
        title="",
    ):
        super().__init__(master, studies, title)
        self.mph = master_plot_height
        self.sph = study_plot_height
        self._n_style = "-"

    def __plt_series(self, y, zoom, study_name, k, row, col, plot_style="line"):
        _lbl = y.name if hasattr(y, "name") and y.name else ("%s_%d" % (study_name, k))

        if isinstance(y, pd.DataFrame):
            yy = y[zoom] if zoom else y

            # candlesticks
            if self._frame_has_cols(y, ["open", "high", "low", "close"]):
                self.fig.add_trace(
                    go.Candlestick(
                        x=yy.index,
                        open=yy["open"],
                        high=yy["high"],
                        low=yy["low"],
                        close=yy["close"],
                        name=_lbl,
                        line={"width": 1},
                    ),
                    row=row,
                    col=col,
                )

            # trends
            elif self._frame_has_cols(y, ["UpTrends", "DownTrends"]):
                u, d = yy.UpTrends.dropna(), yy.DownTrends.dropna()
                for i, r in enumerate(u.iterrows()):
                    self.fig.add_trace(
                        go.Scatter(
                            x=[r[0], r[1].end],
                            y=[r[1].start_price, r[1].end_price],
                            mode="lines+markers",
                            name="UpTrends",
                            line={
                                "color": LookingGlassPlotly.TREND_COLORS.uline,
                                "width": 1,
                                "dash": "dot",
                            },
                            marker={"color": LookingGlassPlotly.TREND_COLORS.udot},
                            showlegend=i == 0,
                            legendgroup="trends_UpTrends",
                        ),
                        row=row,
                        col=col,
                    )

                for i, r in enumerate(d.iterrows()):
                    self.fig.add_trace(
                        go.Scatter(
                            x=[r[0], r[1].end],
                            y=[r[1].start_price, r[1].end_price],
                            mode="lines+markers",
                            name="DownTrends",
                            line={
                                "color": LookingGlassPlotly.TREND_COLORS.dline,
                                "width": 1,
                                "dash": "dot",
                            },
                            marker={"color": LookingGlassPlotly.TREND_COLORS.ddot},
                            showlegend=i == 0,
                            legendgroup="trends_DownTrends",
                        ),
                        row=row,
                        col=col,
                    )

            # order executions from simulator
            elif self._frame_has_cols(
                y, ["side", "fill_avg_price", "quantity", "status"]
            ):
                yy = yy[yy.status == "FILLED"]
                _b_ords = yy[yy.side == "BUY"]
                _s_ords = yy[yy.side == "SELL"]
                _info_b = "INFO : " + _b_ords["user_description"]
                _info_s = "INFO : " + _s_ords["user_description"]

                self.fig.add_trace(
                    go.Scatter(
                        x=_b_ords.index,
                        y=_b_ords.fill_avg_price,
                        mode="markers",
                        name="BOT",
                        text=_info_b,
                        marker={
                            "symbol": "triangle-up",
                            "size": 13,
                            "color": "#3cfa00",
                        },
                    ),
                    row=row,
                    col=col,
                )
                self.fig.add_trace(
                    go.Scatter(
                        x=_s_ords.index,
                        y=_s_ords.fill_avg_price,
                        mode="markers",
                        name="SLD",
                        text=_info_s,
                        marker={
                            "symbol": "triangle-down",
                            "size": 13,
                            "color": "#20ffff",
                        },
                    ),
                    row=row,
                    col=col,
                )

            # experimental tester view of trading log
            elif self._frame_has_cols(y, ["Action", "Price", "Info", "PnL_ticks"]):
                _b_ords = yy[yy.Action == "Long"]
                _s_ords = yy[yy.Action == "Short"]
                _t_ords = yy[yy.Action == "Take"]
                _l_ords = yy[yy.Action == "Stop"]
                _e_ords = yy[(yy.Action == "Expired") | (yy.Action == "Flat")]
                _info = (
                    "INFO : "
                    + yy["Info"]
                    + "<br>PnL Ticks: "
                    + yy["PnL_ticks"].astype(str)
                )

                self.fig.add_trace(
                    go.Scatter(
                        x=_b_ords.index,
                        y=_b_ords.Price,
                        mode="markers",
                        name="BOT",
                        text=_info,
                        marker={
                            "symbol": "triangle-up",
                            "size": 13,
                            "color": "#3cfa00",
                        },
                    ),
                    row=row,
                    col=col,
                )

                self.fig.add_trace(
                    go.Scatter(
                        x=_s_ords.index,
                        y=_s_ords.Price,
                        mode="markers",
                        name="SLD",
                        text=_info,
                        marker={
                            "symbol": "triangle-down",
                            "size": 13,
                            "color": "#20ffff",
                        },
                    ),
                    row=row,
                    col=col,
                )

                self.fig.add_trace(
                    go.Scatter(
                        x=_t_ords.index,
                        y=_t_ords.CurrentPrice,
                        mode="markers",
                        name="Take",
                        text=_info,
                        marker={"symbol": "cross", "size": 13},
                    ),
                    row=row,
                    col=col,
                )

                self.fig.add_trace(
                    go.Scatter(
                        x=_l_ords.index,
                        y=_l_ords.CurrentPrice,
                        mode="markers",
                        name="Stop",
                        text=_info,
                        marker={"symbol": "circle", "size": 13},
                    ),
                    row=row,
                    col=col,
                )

                self.fig.add_trace(
                    go.Scatter(
                        x=_e_ords.index,
                        y=_e_ords.CurrentPrice,
                        mode="markers",
                        name="Exp",
                        text=_info,
                        marker={"symbol": "x", "size": 13},
                    ),
                    row=row,
                    col=col,
                )

            # reversal points
            elif self._frame_has_cols(y, ["start_price", "delta"]):
                _lo_pts = yy[yy.delta > 0]
                _hi_pts = yy[yy.delta < 0]
                self.fig.add_trace(
                    go.Scatter(
                        x=_lo_pts.index,
                        y=_lo_pts.start_price,
                        mode="markers",
                        name="Bottom",
                        marker={"symbol": "triangle-up"},
                    ),
                    row=row,
                    col=col,
                )
                self.fig.add_trace(
                    go.Scatter(
                        x=_hi_pts.index,
                        y=_hi_pts.start_price,
                        mode="markers",
                        name="Top",
                        marker={"symbol": "triangle-down"},
                    ),
                    row=row,
                    col=col,
                )

            # tracks
            elif self._frame_has_cols(y, ["Type", "Time", "Price", "PriceOccured"]):
                _bot = yy[yy.Type == "-"]
                _top = yy[yy.Type == "+"]
                self.fig.add_trace(
                    go.Scatter(
                        x=_bot.index,
                        y=_bot.PriceOccured,
                        mode="markers",
                        name="B",
                        marker={"symbol": "triangle-right"},
                    ),
                    row=row,
                    col=col,
                )

                self.fig.add_trace(
                    go.Scatter(
                        x=_top.index,
                        y=_top.PriceOccured,
                        mode="markers",
                        name="T",
                        marker={"symbol": "triangle-right"},
                    ),
                    row=row,
                    col=col,
                )

            # executed signals from signal tester
            elif self._frame_has_cols(y, ["exec_price", "quantity"]):
                _b_ords = yy[yy.quantity > 0]
                _s_ords = yy[yy.quantity < 0]
                self.fig.add_trace(
                    go.Scatter(
                        x=_b_ords.index,
                        y=_b_ords.exec_price,
                        mode="markers",
                        name="BOT",
                        marker={
                            "symbol": "triangle-up",
                            "size": 13,
                            "color": "#3cfa00",
                        },
                    ),
                    row=row,
                    col=col,
                )

                self.fig.add_trace(
                    go.Scatter(
                        x=_s_ords.index,
                        y=_s_ords.exec_price,
                        mode="markers",
                        name="SLD",
                        marker={
                            "symbol": "triangle-down",
                            "size": 13,
                            "color": "#20ffff",
                        },
                    ),
                    row=row,
                    col=col,
                )

            else:
                for _col in yy.columns:
                    self.__plot_as_type(yy[_col], row, col, plot_style, _col)
        else:
            yy = y[zoom] if zoom else y
            self.__plot_as_type(yy, row, col, plot_style, _lbl)

    def __plot_as_type(self, y, row, col, plot_style, label):
        style, color = self.__line_style_to_color_dash(self._n_style)
        if plot_style == "line":
            self.fig.add_trace(
                go.Scatter(
                    x=y.index,
                    y=y,
                    mode="lines",
                    line={"width": 0.5, "dash": style, "color": color},
                    name=label,
                ),
                row=row,
                col=col,
            )
        elif plot_style == "area":
            self.fig.add_trace(
                go.Scatter(
                    x=y.index,
                    y=y,
                    mode="lines",
                    line={"width": 1, "color": color},
                    fill="tozeroy",
                    name=label,
                ),
                row=row,
                col=col,
            )
        elif plot_style.startswith("step"):
            self.fig.add_trace(
                go.Scatter(
                    x=y.index,
                    y=y,
                    mode="lines",
                    line={"shape": "hv", "color": color},
                    name=label,
                ),
                row=row,
                col=col,
            )
        elif plot_style.startswith("bar"):
            self.fig.add_trace(
                go.Bar(x=y.index, y=y, name=label, marker_color=color), row=row, col=col
            )
        elif plot_style.startswith("dots") or plot_style.startswith("point"):
            self.fig.add_trace(
                go.Scatter(
                    x=y.index,
                    y=y,
                    mode="markers",
                    name=label,
                    marker_color=color,
                    marker={"symbol": "circle", "size": 4},
                ),
                row=row,
                col=col,
            )
        elif plot_style.startswith("arrow"):
            self.fig.add_trace(
                go.Scatter(
                    x=y.index,
                    y=y,
                    mode="markers",
                    name=label,
                    marker_color=color,
                    marker={"symbol": plot_style, "size": 12},
                ),
                row=row,
                col=col,
            )

    def __line_style_to_color_dash(self, style):
        """
        Convert mpl format color-style to plotly format
        :param style: style in format 'color style'. Example: 'r --' means red color and dash style line.
        :return style, color:
        """
        plotly_styles = ["dash", "dot", "dashdot", "solid"]
        splitted = style.split(" ")
        if len(splitted) > 1:
            mpl_color, mpl_style = splitted[0], splitted[1]
        else:
            mpl_color, mpl_style = None, splitted[0]

        plotly_style, plotly_color = None, None

        if mpl_style in plotly_styles:
            # specified plotly style line
            plotly_style = mpl_style
        elif mpl_style == ":":
            plotly_style = "dot"
        elif mpl_style == "--":
            plotly_style = "dash"
        elif mpl_style == "-.":
            plotly_style = "dashdot"
        elif mpl_style == "-":
            plotly_style = "solid"

        if plotly_style is None and mpl_color is None:
            # it looks only color is specified
            mpl_color = mpl_style

        if mpl_color == "r":
            plotly_color = "red"
        elif mpl_color == "g":
            plotly_color = "green"
        elif mpl_color == "w":
            plotly_color = "white"
        else:
            plotly_color = mpl_color

        return plotly_style, plotly_color

    def __mpl_color_to_plotly(self, color):
        if color == "r":
            color = "red"
        elif color == "g":
            color = "green"
        elif color == "w":
            color = "white"
        return color

    def _show_plot(self, vert_bar, title, zoom):
        # plot all master series
        master_fraction = (
            self.mph / (self.mph + self.sph * len(self.s)) if len(self.s) else 1.0
        )
        row_heights = [master_fraction]
        axis_rules = {}
        if len(self.s):
            row_heights.extend([(1 - master_fraction) / len(self.s)] * len(self.s))
        self.fig = make_subplots(
            rows=len(self.s) + 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            row_heights=row_heights,
        )

        ms = self.m if isinstance(self.m, (tuple, list)) else [self.m]
        plot_style = "line"
        for j, m in enumerate(ms):
            if isinstance(m, str):
                if any(
                    [
                        m.startswith(x)
                        for x in [
                            "line",
                            "bar",
                            "step",
                            "stem",
                            "area",
                            "dots",
                            "point",
                            "arrow-up",
                            "arrow-down",
                        ]
                    ]
                ):
                    plot_style = m
                else:
                    self._n_style = m
            else:
                self.__plt_series(m, zoom, "Master", j, 1, 1, plot_style=plot_style)
                self._n_style = "-"

        if vert_bar:
            self.__add_vline("x", vert_bar)
        i = 1
        for k, vs in self.s.items():
            wait_for_limits = False
            i += 1
            vs = vs if isinstance(vs, (tuple, list)) else [vs]
            plot_style = "line"
            self._n_style = "-"
            for j, v in enumerate(vs):
                if wait_for_limits and isinstance(v, (list, tuple)) and len(v) > 1:
                    axis_rules["yaxis%d" % i] = {"range": v}
                    wait_for_limits = False

                elif isinstance(v, str):
                    vl = v.lower()
                    if vl.startswith("lim"):
                        wait_for_limits = True
                        continue
                    elif any(
                        [
                            vl.startswith(x)
                            for x in [
                                "line",
                                "bar",
                                "step",
                                "stem",
                                "area",
                                "dots",
                                "point",
                                "arrow-up",
                                "arrow-down",
                            ]
                        ]
                    ):
                        plot_style = vl
                    else:
                        self._n_style = v
                elif isinstance(v, (float, int)):
                    self.__add_hline("y%d" % i, v)
                else:
                    self.__plt_series(v, zoom, k, j, i, 1, plot_style=plot_style)

        if title is not None:
            plot_title = "%s %s" % (self._title, str(title))
        else:
            plot_title = "%s %s" % (self._title, str(vert_bar))

        if plot_title is not None:
            self.fig.update_layout(
                title={
                    "text": plot_title,
                    "y": 0.9,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            )

        self.fig.update_layout(axis_rules)
        self.fig.update_layout(
            xaxis_rangeslider_visible=False,
            margin=dict(l=5, r=5, t=35, b=5),
            height=self.mph + self.sph * len(self.s),
        )

        return go.FigureWidget(self.fig)

    def __add_vline(self, xref, x):
        # Line Vertical
        self.fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=x,
                x1=x,
                xref=xref,
                yref="paper",
                y0=0,
                y1=1,
                line=dict(width=1, dash="dot"),
            )
        )

    def __add_hline(self, yref, y):
        # Line Horizontal
        style, color = self.__line_style_to_color_dash(self._n_style)
        self.fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=0,
                x1=1,
                xref="paper",
                yref=yref,
                y0=y,
                y1=y,
                line=dict(width=1, dash=style, color=color),
            )
        )
