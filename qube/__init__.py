from . import quantitative
from . import charting
from . import datasource
from . import utils

__version__ = '0.3.43'

from qube.utils.utils import runtime_env
from cycler import cycler

# reload cython modules
try:
    from qube.utils.utils import reload_pyx_module

    reload_pyx_module()
except Exception as exception:
    print(f" >>> Exception during reloading cython pyx modules {str(exception)}")

DARK_MATLPLOT_THEME = [
    ('backend', 'module://matplotlib_inline.backend_inline'),
    ('interactive', True),
    ('lines.color', '#5050f0'),
    ('text.color', '#d0d0d0'),
    ('axes.facecolor', '#000000'),
    ('axes.edgecolor', '#404040'),
    ('axes.grid', True),
    ('axes.labelsize', 'large'),
    ('axes.labelcolor', 'green'),
    ('axes.prop_cycle', cycler('color', ['#08F7FE', '#00ff41', '#FE53BB', '#F5D300', '#449AcD', 'g',
                                         '#f62841', 'y', '#088487', '#E24A33', '#f01010'])),
    ('legend.fontsize', 'small'),
    ('legend.fancybox', False),
    ('legend.edgecolor', '#305030'),
    ('legend.shadow', False),
    ('lines.antialiased', True),
    ('lines.linewidth', 0.8),  # reduced line width
    ('patch.linewidth', 0.5),
    ('patch.antialiased', True),
    ('xtick.color', '#909090'),
    ('ytick.color', '#909090'),
    ('xtick.labelsize', 'large'),
    ('ytick.labelsize', 'large'),
    ('grid.color', '#404040'),
    ('grid.linestyle', '--'),
    ('grid.linewidth', 0.5),
    ('grid.alpha', 0.8),
    ('figure.figsize', [12.0, 5.0]),
    ('figure.dpi', 80.0),
    ('figure.facecolor', '#050505'),
    ('figure.edgecolor', (1, 1, 1, 0)),
    ('figure.subplot.bottom', 0.125),
    ('savefig.facecolor', '#000000'),
]

LIGHT_MATPLOT_THEME = [
    ('backend', 'module://matplotlib_inline.backend_inline'),
    ('interactive', True),
    ('lines.color', '#101010'),
    ('text.color', '#303030'),
    ('lines.antialiased', True),
    ('lines.linewidth', 1),
    ('patch.linewidth', 0.5),
    ('patch.facecolor', '#348ABD'),
    ('patch.edgecolor', '#eeeeee'),
    ('patch.antialiased', True),
    ('axes.facecolor', '#fafafa'),
    ('axes.edgecolor', '#d0d0d0'),
    ('axes.linewidth', 1),
    ('axes.titlesize', 'x-large'),
    ('axes.labelsize', 'large'),
    ('axes.labelcolor', '#555555'),
    ('axes.axisbelow', True),
    ('axes.grid', True),
    ('axes.prop_cycle', cycler('color', ['#6792E0', '#27ae60', '#c44e52', '#975CC3', '#ff914d', '#77BEDB',
                                         '#303030', '#4168B7', '#93B851', '#e74c3c', '#bc89e0', '#ff711a',
                                         '#3498db', '#6C7A89'])),
    ('legend.fontsize', 'small'),
    ('legend.fancybox', False),
    ('xtick.color', '#707070'),
    ('ytick.color', '#707070'),
    ('grid.color', '#606060'),
    ('grid.linestyle', '--'),
    ('grid.linewidth', 0.5),
    ('grid.alpha', 0.3),
    ('figure.figsize', [8.0, 5.0]),
    ('figure.dpi', 80.0),
    ('figure.facecolor', '#ffffff'),
    ('figure.edgecolor', '#ffffff'),
    ('figure.subplot.bottom', 0.1)
]

# registering magic for jupyter notebook
if runtime_env() in ['notebook', 'shell']:
    from IPython.core.magic import (Magics, magics_class, line_magic, line_cell_magic)
    from IPython import get_ipython

    @magics_class
    class QubeMagics(Magics):
        # process data manager
        __manager = None

        @line_magic
        def qubed(self, line: str):
            self.alphalab('dark')

        @line_magic
        def qubel(self, line: str):
            self.alphalab('light')

        @line_magic
        def alphalab(self, line: str):
            """
            QUBE framework initialization
            """
            import os
            from qube.configs.Properties import get_root_dir
            import matplotlib
            import plotly.io as pio

            tpl_path = os.path.join(get_root_dir(), "qube_nb_magic_init.py")
            with open(tpl_path, 'r') as myfile:
                s = myfile.read()

            exec(s, self.shell.user_ns)

            # setup more funcy mpl theme instead of ugly default
            if line:
                if 'dark' in line.lower():
                    pio.templates.default = "plotly_dark"
                    for (k, v) in DARK_MATLPLOT_THEME:
                        matplotlib.rcParams[k] = v

                elif 'light' in line.lower():
                    pio.templates.default = "plotly_white"
                    for (k, v) in LIGHT_MATPLOT_THEME:
                        matplotlib.rcParams[k] = v

            # install additional plotly helpers
            from qube.charting.plot_helpers import install_plotly_helpers
            install_plotly_helpers()

        def _get_manager(self):
            if self.__manager is None:
                import multiprocessing as m
                self.__manager = m.Manager()
            return self.__manager

        @line_cell_magic
        def proc(self, line, cell=None):
            """
            Run cell in separate process

            >>> %%proc x, y as MyProc1
            >>> x.set('Hello')
            >>> y.set([1,2,3,4])
            
            """
            import multiprocessing as m
            import time, re

            # create ext args
            name = None
            if line:
                # check if custom process name was provided
                if ' as ' in line:
                    line, name = line.split('as')
                    if not name.isspace():
                        name = name.strip()
                    else:
                        print('>>> Process name must be specified afer "as" keyword !')
                        return

                ipy = get_ipython()
                for a in [x for x in re.split('[\ ,;]', line.strip()) if x]:
                    ipy.push({a: self._get_manager().Value(None, None)})

            # code to run
            lines = '\n'.join(['    %s' % x for x in cell.split('\n')])

            def fn():
                result = get_ipython().run_cell(lines)

                # send errors to parent
                if result.error_before_exec:
                    raise result.error_before_exec

                if result.error_in_exec:
                    raise result.error_in_exec

            t_start = str(time.time()).replace('.', '_')
            f_id = f'proc_{t_start}' if name is None else name
            if self._is_task_name_already_used(f_id):
                f_id = f"{f_id}_{t_start}"

            task = m.Process(target=fn, name=f_id)
            task.start()
            print(' -> Task %s started' % f_id)

        def _is_task_name_already_used(self, name):
            import multiprocessing as m
            for p in m.active_children():
                if p.name == name:
                    return True
            return False

        @line_magic
        def list_proc(self, line):
            import multiprocessing as m
            for p in m.active_children():
                print(p.name)

        @line_magic
        def kill_proc(self, line):
            import multiprocessing as m
            for p in m.active_children():
                if line and p.name.startswith(line):
                    p.terminate()


    # registering magic here
    get_ipython().register_magics(QubeMagics)
