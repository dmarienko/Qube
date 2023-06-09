import codecs
import glob
import os
import pickle
import urllib.parse
from collections import OrderedDict, namedtuple
from functools import partial, wraps
from os.path import basename, exists, dirname, join, expanduser

import pandas as pd
import numpy as np
import requests

from qube.configs import Properties
from qube.datasource.controllers.MongoController import MongoController
from qube.utils import QubeLogger


__numba_is_available = False
try:
    from numba import njit, jit
    njit_optional = njit
    __numba_is_available = True
except:
    def njit_optional(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return args[0]
        else:
            return lambda f: f
    print('WARNING: Numba package is not found')

__logger = QubeLogger.getLogger(__name__)


def pyx_reload(path: str):
    """
    Reload specified cython module
    path must have .pyx extension
    """
    if exists(path):
        f_name, f_ext = basename(path).split('.')
        if f_ext == 'pyx':
            import numpy as np
            import pyximport
            pyximport.install(setup_args={'include_dirs': np.get_include()}, reload_support=True, language_level=3)
            pyximport.load_module(f_name, path, language_level=3, pyxbuild_dir=expanduser("~/.pyxbld"))
            print("Reloaded %s" % path)
    else:
        raise ValueError("Path '%s' not found !" % path)


def reload_pyx_module():
    _module_dir = dirname(__file__)
    for _m in glob.glob(join(_module_dir, '*.pyx')):
        pyx_reload(_m)


def version():
    """
    Get current version of framework.
    """
    build_file = Properties.get_formatted_path("qube/build.txt")
    if os.path.exists(build_file):
        with open(build_file) as f:
            return f.read()
    else:
        from qube import __version__
        return __version__


def runtime_env():
    """
    Check what environment this script is being run under
    :return: environment name, possible values:
             - 'notebook' jupyter notebook
             - 'shell' any interactive shell (ipython, PyCharm's console etc)
             - 'python' standard python interpreter
             - 'unknown' can't recognize environment
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__

        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            return 'notebook'
        elif shell.endswith('TerminalInteractiveShell'):  # Terminal running IPython
            return 'shell'
        else:
            return 'unknown'  # Other type (?)
    except (NameError, ImportError):
        return 'python'  # Probably standard Python interpreter


def terminal(id=1, height=500, weight=910):
    """
    Attempt to connect to remote terminal and open it embedded in notebook's cell
    """
    if runtime_env() == "notebook":
        try:
            import re
            from IPython.core.display import HTML, display
            from notebook import notebookapp
            servers = list(notebookapp.list_running_servers())
            ex = re.compile('http[s]?://(.+:?\d+)(/.+)')
            b_url = ex.match(servers[0]['url']).groups()[1]
            display(
                HTML(
                    "<h4><font color='#c03030'>Terminal :: %d</font></h4><br>"
                    "<div align='left'><iframe height='%d' width='%d' src='%s/terminals/%d'/></div>" %
                    (id, height, weight, b_url, id)
                )
            )
        except:
            print("Error opening remote terminal")
    else:
        print("Remote terminal is supported only for Jupyter Notebooks")


def remote_copy_mongo_table(host: str, table_name: str, destination_table_name: str = None, is_serialize=True):
    remote_mongo_data = remote_load_mongo_table(host, table_name)
    mongo_table = destination_table_name if destination_table_name else table_name
    mongo = MongoController()
    mongo.save_data(mongo_table, remote_mongo_data['data'], remote_mongo_data['meta'], is_serialize=is_serialize)


def remote_load_mongo_table(host: str, table_name: str, dbname: str = "", query="", username: str = "",
                            password: str = ""):
    remote_host, remote_port = _get_remote_host_info(host)
    dbname = dbname if isinstance(dbname, str) else ""
    request_url = 'http://%s:%s/get_binary_mongo_table?table_name=%s&dbname=%s&username=%s&password=%s&query=%s' % (
        remote_host, remote_port,
        urllib.parse.quote(table_name), dbname,
        username, password, codecs.encode(pickle.dumps(query), "base64").decode())
    response = requests.get(request_url)
    if response.status_code != 200:
        raise RuntimeError('Error while getting data from %s' % request_url)
    return pickle.loads(codecs.decode(response.text.encode(), "base64"))


def remote_ls_mongo_table(host: str, ls_query: str, dbname: str = None, username: str = "", password: str = ""):
    remote_host, remote_port = _get_remote_host_info(host)
    dbname = dbname if isinstance(dbname, str) else ""
    request_url = 'http://%s:%s/get_binary_mongo_ls?ls_query=%s&dbname=%s&username=%s&password=%s' % (
        remote_host, remote_port, urllib.parse.quote(ls_query),
        dbname, username, password)
    response = requests.get(request_url)
    if response.status_code != 200:
        raise RuntimeError('Error while getting data from %s' % request_url)
    return pickle.loads(codecs.decode(response.text.encode(), "base64"))


def remote_del_mongo_table(host: str, table_name: str, dbname: str = "", username: str = "", password: str = ""):
    remote_host, remote_port = _get_remote_host_info(host)
    dbname = dbname if isinstance(dbname, str) else ""
    request_url = 'http://%s:%s/delete_mongo_table?table_name=%s&dbname=%s&username=%s&password=%s' % (
        remote_host, remote_port,
        urllib.parse.quote(table_name), dbname,
        username, password)
    response = requests.get(request_url)
    if response.status_code != 200:
        raise RuntimeError('Error while getting data from %s' % request_url)


def remote_save_mongo_table(host: str, table_name: str, data, is_serialize=True, dbname: str = None, username: str = "",
                            password: str = ""):
    remote_host, remote_port = _get_remote_host_info(host)
    dbname = dbname if isinstance(dbname, str) else ""
    request_url = 'http://%s:%s/save_mongo_table?table_name=%s&is_serialize=%s&dbname=%s&username=%s&password=%s' % (
        remote_host, remote_port,
        urllib.parse.quote(table_name), is_serialize,
        dbname, username, password)
    pickled_data = codecs.encode(pickle.dumps(data), "base64").decode()
    response = requests.post(request_url, data={'data': pickled_data})
    if response.status_code != 200:
        raise RuntimeError('Error while saving data in %s' % request_url)
    return response


def add_project_to_system_path():
    """
    Add path to projects folder to system python path to be able importing any modules from project
    from test.Models.handy_utils import some_module
    """
    import sys
    from os.path import expanduser, relpath
    from pathlib import Path
    
    # we want to track folders with these files as separate paths
    toml = Path('pyproject.toml')
    src = Path('src')
    
    try:
        prj = Path(relpath(expanduser('~/projects')))
    except ValueError as e:
        # This error can occur on Windows if user folder and python file are on different drives
        print(f"Qube> Error during get path to projects folder:\n{e}")
    else:
        insert_path_iff = lambda p: sys.path.insert(0, p.as_posix()) if p.as_posix() not in sys.path else None
        if prj.exists():
            insert_path_iff(prj)
            
            for di in prj.iterdir():
                _src = di / src
                if (di / toml).exists():
                    # when we have src/
                    if _src.exists() and _src.is_dir():
                        insert_path_iff(_src)
                    else:
                        insert_path_iff(di)
        else:
            print(f'Qube> Cant find "projects/" folder for adding to python path !')


def is_localhost(host):
    return host.lower() == 'localhost' or host == '127.0.0.1'


def _get_remote_host_info(host: str):
    sc_props = Properties.get_main_properties()['services-controller']
    host = Properties.get_main_properties()['host_aliases'].get(host.lower(), host)
    return host, sc_props['port']


def urange(start, stop, step=1, units=None, none=False):
    """
    Range list generator helper. Units can be specified.

    urange(1, 10, 1) -> [1,2,3,4.....10]
    urange(1, 10, 0.5) -> [1.0,1.5,...]
    urange(1, 10, 1, 'Min') -> ['1Min', '2Min', ..., '10Min']
    urange(1, 3, 1, none=True) -> [None,1,2,3]

    """
    r = list(np.arange(start, stop + step, step))
    if units:
        r = [f'{i}{units}' for i in r]
    return ([None, ] + r) if none else r


class IProgressListener:
    def on_change(self, **kwargs):
        pass

    def on_finish(self, **kwargs):
        pass


class mstruct:
    """
    Dynamic structure (similar to matlab's struct it allows to add new properties dynamically)

    >>> a = mstruct(x=1, y=2)
    >>> a.z = 'Hello'
    >>> print(a)

    mstruct(x=1, y=2, z='Hello')
    
    >>> mstruct(a=234, b=mstruct(c=222)).to_dict()
    
    {'a': 234, 'b': {'c': 222}}

    """

    def __init__(self, **kwargs):
        _odw = OrderedDict(**kwargs)
        self.__initialize(_odw.keys(), _odw.values())

    def __initialize(self, fields, values):
        self._fields = list(fields)
        self._meta = namedtuple('mstruct', ' '.join(fields))
        self._inst = self._meta(*values)

    def __getattr__(self, k):
        return getattr(self._inst, k)

    def __dir__(self):
        return self._fields

    def __repr__(self):
        return self._inst.__repr__()

    def __setattr__(self, k, v):
        if k not in ['_inst', '_meta', '_fields']:
            new_vals = {**self._inst._asdict(), **{k: v}}
            self.__initialize(new_vals.keys(), new_vals.values())
        else:
            super().__setattr__(k, v)

    def __getstate__(self):
        return self._inst._asdict()

    def __setstate__(self, state):
        self.__init__(**state)

    def __ms2d(self, m) -> dict:
        r = {}
        for f in m._fields:
            v = m.__getattr__(f)
            r[f] = self.__ms2d(v) if isinstance(v, mstruct) else v
        return r

    def to_dict(self) -> dict:
        """
        Return this structure as dictionary
        """
        return self.__ms2d(self)

    def copy(self) -> 'mstruct':
        """
        Returns copy of this structure
        """
        return dict2struct(self.to_dict())


def dict2struct(d: dict) -> mstruct:
    """
    Convert dictionary to structure
    >>> s = dict2struct({'f_1_0': 1, 'z': {'x': 1, 'y': 2}})
    >>> print(s.z.x)
    1
    
    """
    m = mstruct()
    for k, v in d.items():
        # skip if key is not valid identifier
        if not k.isidentifier():
            continue
        if isinstance(v, dict):
            v = dict2struct(v)
        m.__setattr__(k, v)
    return m


def dict_to_frame(x: dict, index_type=None, orient='index', columns=None, column_types=dict()):
    """
    Utility for convert dictionary to indexed DataFrame
    It's possible to pass columns names and type of index
    """
    y = pd.DataFrame().from_dict(x, orient=orient)
    if index_type:
        if index_type in ['ns', 'nano']:
            index_type = 'M8[ns]'

        y.index = y.index.astype(index_type)
    # rename if needed
    if columns:
        columns = [columns] if not isinstance(columns, (list, tuple, set)) else columns
        if len(columns) == len(y.columns):
            y.rename(columns=dict(zip(y.columns, columns)), inplace=True)
        else:
            raise ValueError('dict_to_frame> columns argument must contain %d elements' % len(y.columns))

    # if additional conversion is required
    if column_types:
        _existing_cols_conversion = {c: v for c, v in column_types.items() if c in y.columns}
        y = y.astype(_existing_cols_conversion)

    return y


def jit_optional(func=None, *args, **kwargs):
    """
    Decorator used instead of jit from numba for resolving situation when numba is not presented
    """
    if func is None:
        return partial(njit_optional, *args, **kwargs)

    @wraps(func)
    def inner(*i_args, **i_kwargs):
        if __numba_is_available:
            return jit(*args, **kwargs)(func)(*i_args, **i_kwargs)
        else:
            return func(*i_args, **i_kwargs)

    return inner
