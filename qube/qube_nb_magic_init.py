import qube
from qube.utils.utils import version, runtime_env

if runtime_env() in ['notebook', 'shell']:

    # -- all imports below will appear in notebook after calling %%alphalab magic ---
    import numpy as np
    import pandas as pd

    from qube.simulator import SignalTester
    from qube.simulator.Brokerage import (
        GenericStockBrokerInfo, GenericForexBrokerInfo, GenericCryptoBrokerInfo, GenericCryptoFuturesBrokerInfo
    )

    from qube.utils.utils import add_project_to_system_path
    from qube.datasource.controllers.MongoController import MongoController
    from qube.utils.nb_functions import *
    from qube.utils.ui_utils import green, yellow, cyan, magenta

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
           {green("|   : * |   |")}   (c) 2022,  ver. {magenta(version().rstrip())}
           {green("|  ,+---+---+")} 
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
