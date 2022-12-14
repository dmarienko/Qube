import qube
from qube.utils.utils import version, runtime_env

if runtime_env() in ['notebook', 'shell']:

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # -- all imports below will appear in notebook after calling %%alphalab magic ---
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - Common stuff - - - -
    import numpy as np
    import pandas as pd
    import datetime
    from datetime import time, timedelta
    from tqdm.auto import tqdm

    from qube.datasource import DataSource
    from qube.datasource.controllers.MongoController import MongoController

    # - - - - TA stuff and indicators - - - -
    from qube.quantitative.ta.indicators import (
        ema, dema, tema, kama, zlema, sma, jma, wma, ema_time, pivot_point, lrsi, rsi,
        adx, atr, rolling_atr, rolling_rank, rolling_series_slope,
        holt_winters_second_order_ewma, series_halflife, running_view, smooth,
        bollinger, bollinger_atr, detrend, moving_detrend, moving_ols,
        rolling_std_with_mean, macd, trend_detector, fractals, denoised_trend,
        stochastic, laguerre_filter, psar
    )
    from qube.quantitative.ta.swings.swings_splitter import (
        find_movements, find_movements_hilo
    )
    from qube.series.BarSeries import BarSeries
    from qube.series.Quote import Quote

    # - - - - Portfolio analysis - - - -
    from qube.portfolio.reports import tearsheet, tearsheets
    from qube.portfolio.performance import split_cumulative_pnl, portfolio_stats
    from qube.portfolio.allocating import (
        runnig_portfolio_allocator, tang_portfolio, gmv_portfolio, effective_portfolio
    )

    # - - - - Simulator stuff - - - -
    from qube.simulator.utils import (
        shift_signals, rolling_forward_test_split, permutate_params
    )
    from qube.portfolio.Position import Position, ForexPosition, CryptoPosition, CryptoFuturesPosition
    from qube.simulator.multisim import simulation, Market
    from qube.simulator.multiproc import ls_running_tasks, run_tasks
    from qube.simulator.management import ls_simulations
    from qube.simulator.utils import ls_brokers
    from qube.simulator.backtester import backtest_signals_inplace, backtest
    from qube.simulator.tracking.trackers import (
        FixedRiskTrader, DispatchTracker, PipelineTracker, TakeStopTracker,
        MultiTakeStopTracker, SignalBarTracker,
        TimeExpirationTracker, ATRTracker, TriggeredOrdersTracker
    )
    from qube.simulator.tracking.sizers import (IPositionSizer, FixedSizer, FixedRiskSizer)

    # - - - - Learn stuff - - - -
    from qube.learn.core.pickers import SingleInstrumentPicker, PortfolioPicker
    from qube.learn.core.operations import Imply, And, Or, Neg
    from qube.learn.core.metrics import (
        ForwardDirectionScoring, ForwardReturnsSharpeScoring, ReverseSignalsSharpeScoring, ForwardReturnsCalculator
    )
    from qube.learn.core.utils import ls_params, debug_output
    from qube.learn.core.base import signal_generator, SingleInstrumentComposer, PortfolioComposer
    from qube.learn.core.mlhelpers import gridsearch
    from qube.examples.learn.generators import WalkForwardTest

    # - - - - Charting stuff - - - -
    from matplotlib import pyplot as plt
    from qube.charting.plot_helpers import (
        fig, multiplot, smultiplot, subplot, sbp, plot_pacf, plot_acf, zoomx, glow_effects,
        plot_trends, plot_fractals
    )
    from qube.charting.mpl_finance import ohlc_plot
    from qube.charting.lookinglass import LookingGlass

    # - - - - Utils - - - -
    from qube.quantitative.tools import (
        isscalar, nans, apply_to_frame, ohlc_resample, scols, srows, drop_duplicated_indexes,
        retain_columns_and_join
    )
    from qube.utils.nb_functions import *
    from qube.utils.utils import (terminal, mstruct, add_project_to_system_path, dict2struct, urange)
    from qube.utils.ui_utils import (green, yellow, cyan, magenta, white, blue, red)

    # setup short numpy output format
    np_fmt_short()
    
    # add project home to system path
    add_project_to_system_path()

    # some new logo
    if not hasattr(qube.QubeMagics, '__already_initialized__'):
        print(f'''
           {green("   .+-------+")} 
           {green(" .' :     .'|")}   {yellow("QUBE")} | {cyan("Quantitative Backtesting Environment")}
           {green("+-------+'  |")}  
           {green("|   : ") + red("*") + green(" |   |")}   (c) 2022,  ver. {magenta(version().rstrip())}
           {green("|  ,+---|---+")} 
           {green("|.'     | .' ")}   
           {green("+-------+'   ")} ''')
        qube.QubeMagics.__already_initialized__ = True

    # some fancy outputs if enabled
    import os
    if not os.path.exists(os.path.expanduser('~/.config/alphalab/.norich')):
        try:
            from IPython.core.interactiveshell import InteractiveShell
            import rich
            from rich import print
            from rich.traceback import Traceback
            
            rich.get_console().push_theme(rich.theme.Theme({
                "repr.number": "bold green blink",
                "repr.none": "red",
                "progress.percentage": "bold green",
                "progress.description": "bold yellow",
                "bar.pulse": rich.theme.Style(color="rgb(255,0,0)"),
                "bar.back": rich.theme.Style(color="rgb(20,20,20)"),
                "bar.finished": rich.theme.Style(color="cyan"),
                "bar.complete": rich.theme.Style(color="rgb(200,10,10)"),
            }))

            def __fancy_exc(self, exc_tuple=None, filename=None, tb_offset=None, exception_only=False, running_compiled_code=False):
                etype, value, tb = self._get_exc_info(exc_tuple)
                noff = tb_offset if tb_offset is not None else self.InteractiveTB.tb_offset
                for _ in range(noff):
                    tb = tb.tb_next
                print(rich.traceback.Traceback.from_exception(etype, value, tb, extra_lines=3, show_locals=False))

            InteractiveShell.showtraceback = __fancy_exc
        except:
            pass
