import yaml
import re
import inspect
import hashlib
import string 
import tempfile
from itertools import cycle
from os.path import exists, expanduser, abspath, dirname, join
from subprocess import Popen

from qube.booster.app.reports import get_combined_portfolio, get_combined_executions
from qube.booster.core import Booster
from qube.simulator.utils import ls_brokers
from qube.utils.ui_utils import red, green

DEFAULT_TEMP_CONFIGS = '/var/qube/booster/configs/'


def _dive(d, tags):
    for k, v in d.items():
        if k in tags:
            if isinstance(v, dict):
                return _dive(v, tags)
            else:
                return v
    return None


def _select_market(query: str, file: str):
    if not exists(file):
        file = join(dirname(abspath(inspect.getfile(Do))), file)
        
    if not exists(file):
        raise ValueError(f"Can't find market description file at {file} !")

    with open(file) as f:
        known_sets = yaml.safe_load(f)
        
    tags = query.split(' ')
    instrs = _dive(known_sets, tags)
    
    if instrs is None:
        raise ValueError(f"Can't find instruments for specified references !")

    if 'noprefix' in tags:
        instrs = [s.split(':')[1] if ':' in s else s for s in instrs]

    pm = re.findall('([+-])([a-zA-Z\._:0-9]+)', query)
    for s, w in pm:
        if s == '+': instrs.append(w.upper())
        if s == '-':
            wu = w.upper()
            to_rm = [x for x in instrs if x.startswith(wu) or (':' + wu) in x] 
            [instrs.remove(x) for x in to_rm]       

    # - very dirty lookup for broker's name - may produce wrong matches !
    ranks = {}
    aliases = {'futures': 'um'}
    for b in ls_brokers():
        b_name = b.split('(')[0]
        ranks[b_name] = sum([-z*(t in b_name or aliases.get(t, '^') in b_name) for z, t in enumerate(tags, -len(tags))]) 
        
    broker = max(ranks, key=ranks.get)
    
    return dict(broker=broker, instrument=instrs)
    

def _strategy_params(clz, **kwargs):
    sgn = inspect.signature(clz.__init__)
    params = {}
    for k, v in sgn.parameters.items():
        if k != 'self':
            dflt = v.default
            if dflt != inspect._empty:
                params[k] = kwargs.get(k, dflt)
            else:
                if k in kwargs:
                    params[k] = kwargs.get(k)
                else:
                    print(f'WARN: Not initialized parameter {k} !')
    task = f"{clz.__module__}.{clz.__name__}"
    cpus = kwargs.get('max_cpus', 12)
    max_tasks = max(kwargs.get('tasks', cpus - 1), 1)
        
    c = {
        'spreads': 0,
        'max_cpus': cpus,
        'max_tasks_per_proc': max_tasks,
        'task': task,
        'parameters': params
    }
    if 'conditions' in kwargs:
        cond = kwargs.get('conditions').strip()
        cond = '\\' + cond if cond.startswith('[') else cond
        c['conditions'] = cond
    return {'portfolio': c}


def _get_all_key_vals(query):
    m = re.findall('\ ?([\w,_]+)\ *?[:,=,]\ *?([\w,_,-]+)', query)
    return dict(m)


def _get_ver(clz):
    for _v in ['__version__', '_version_', '__ver__', 'VERSION']:
        if hasattr(clz, _v):
            return str(getattr(clz, _v)).strip()
    return None


def _config(clz, notes, where, datasource=None, markets_description_file='data/markets.yml'):
    c = {}
    def _put(alias, *keys):
        for k in keys:
            v = kv.get(k)
            if v: 
                c[alias] = v
                break
                
    where = ' '.join(where.split())
    kv = _get_all_key_vals(where) 
    docs = inspect.cleandoc(inspect.getdoc(clz))
    ver = _get_ver(clz)
    info = _select_market(where, file=markets_description_file)

    # try to extract backtesting intervals
    m = re.match('.*?\ ?(\w+-\w+-\w+)\ *?->\ *?(\w+-\w+-\w+).*', where)

    # check if 'now' specified
    if not m:
        m = re.match('.*?\ ?(\w+-\w+-\w+)\ *?->\ *?(now).*', where)

    if m:
        start, stop = m.groups()
    else:
        print(red("ERR: Backtest interval not specified. Use '2020-01-01 -> 2021-01-01' in description or '2020-01-01 -> now'"))
        return
        
    prj = kv.get('project')
    if not prj:
        print(red('ERR: project is not specified !'))
        return 
    else:
        if '/' in prj:
            _p0 = prj.split('/')
            prj = _p0[0].strip()
            notes = _p0[1].strip()
    cap = kv.get('capital', 0)
    if cap == 0:
        print("ERR: Capital is not specified !!!")
        return
    c['project'] = prj
    c['capital'] = float(cap)
    c['description'] = f"{docs} {('| v. ' + ver + ' ') if ver else '' }| {notes} | on {len(info['instrument'])} symbols"
    c['instrument'] = info['instrument']
    c['broker'] = info['broker'] 
    _put('mode', 'mode')
    _put('estimator_composer', 'estimator')
    _put('simulator_timeframe', 'timeframe', 'tf')
    
    c['start_date'] = start
    c['end_date'] = stop
    if datasource:
        c['datasource'] = {'name': datasource} 
    return {'config': c}


__VOWS = "aeiou"
__CONS = "".join(set(string.ascii_lowercase) - set(__VOWS))

def _generate_name_like_id(content, n1, ns=0):
    __NV, __NC = len(__VOWS), len(__CONS)
    hdg = hashlib.sha256(content.encode('utf-8')).hexdigest().upper()
    w = ''
    for i, x in enumerate(hdg[ns:n1+ns]):
        if i % 2 == 0:
            w += __CONS[int(x, 16) % __NC]
        else:
            w += __VOWS[int(x, 16) % __NV]
    return w[0].upper() + w[1:]


def _generate_experiment_id(clz, descr, ver, symbols, human_name_like=True):
    symbs = ''.join(sorted(symbols)) if symbols else ''
    ver = inspect.cleandoc(inspect.getdoc(clz))
    content = '%s/%s/%s/%s' % (clz.__name__, descr, ver, symbs)
    if human_name_like:
        id1 = _generate_name_like_id(content, 8)
    else:
        id1 = hashlib.sha256(content.encode('utf-8')).hexdigest()[:5].upper()
    return f"{clz.__name__}-{id1}"


def _booster_cfg(clz, where, notes, datasource=None, markets_description_file='data/markets.yml', **kwargs):
    if not notes:
        raise ValueError(" > Notes must be non-empty !")
    cfg = _config(clz, notes, where, datasource=datasource, markets_description_file=markets_description_file)
    cc = cfg['config']
    capital = cc['capital']
    scfg = _strategy_params(clz, capital=capital, **kwargs)
    return { **cfg, **scfg }

class Do:
    '''
    Prepare and run booster experiments on strategy implemented in StrategyClass
    # - Example - - - 
    
    Boo.do(StrategyClass, 
        """
                project: ProjectName

                capital = 1000
    
                on binance futures test +INDEX noprefix 

                2020-01-01 -> 2023-03-01 
                
                mode: portfolio and estimator: portfolio
                
                timeframe = 1H
                
         """,  'First run of strategy',
        max_capital_in_risk=[0.1, 0.5],
        vol_ratio=[2, 3, 4, 5, 6],
        filter_period=[24, 48, 72, 96],
        datasource="mongo::binance-perpetual-1min",
        #markets='custom_markets.yml'
    
    ) >> '~/experiment_config' #& 'run'
    

    # - - - - - 
    ''' 

    class _MyDumper(yaml.SafeDumper):
        def write_line_break(self, data=None):
            super().write_line_break(data)
            if len(self.indents) == 2:
                super().write_line_break()
                
    def __init__(self, clz, where, notes, datasource=None, markets='data/markets.yml', **kwargs):
        config = _booster_cfg(clz, where, notes, datasource=datasource, 
                             markets_description_file=markets, **kwargs)
        cc = config['config']
        self.experiment = _generate_experiment_id(
            clz,  cc['description'],  cc.get('version', ''), cc['instrument'])
        self.cfg = config
        self.filename = None
            
    def __rshift__(self, dest):
        d0 = {}
        self.filename = expanduser((dest + '.yml') if not dest.endswith('.yml') else dest)
        self._write_config()
        return self

    def _write_config(self):
        d0 = {}
        if exists(self.filename):
            with open(self.filename, 'r') as fs:
                d0 = yaml.safe_load(fs)
                d0 = {} if d0 is None else d0
                
        with open(self.filename, 'w') as fs:
            self.dump_yaml({**d0, **{self.experiment: self.cfg}}, fs)
    
    def __and__(self, cmd):
        if self.filename is None:
            if not exists(DEFAULT_TEMP_CONFIGS):
                raise ValueError(f" > Directory for temporary files '{DEFAULT_TEMP_CONFIGS}' not exists !")
            self.filename = tempfile.mktemp(prefix=DEFAULT_TEMP_CONFIGS, suffix='.yml')
            print(f" > Generating temporary config file: {green(self.filename)}")
            self._write_config()
            
        with open(f"{self.experiment}.log", 'ab') as logfile:
            if cmd.startswith('run'):
                p = Popen(['booster', 'runx', self.experiment, '-c', self.filename], 
                          stdout=logfile, stderr=logfile, bufsize=1)
                return p
            
            if cmd.startswith('del'):
                p = Popen(['booster', 'del', self.experiment, '-c', self.filename], 
                          stdout=logfile, stderr=logfile, bufsize=1)
                return p
    
    def dump_yaml(self, cfg, stream):
        return yaml.dump(cfg, stream=stream, sort_keys=False, default_flow_style=None, indent=4, Dumper=self._MyDumper)           

    def __repr__(self):
        return self.dump_yaml({self.experiment: self.cfg}, None)


# - - - - temporary helpers - - - -
class Boo:
    pass

def __delete_backtest(entry_id: str):
    Booster(None, reload_config=False).task_delete(entry_id)


def __list_backtest():
    Booster(None, reload_config=False).ls()


def __show_backtest(entry_id):
    Booster(None, reload_config=False).show(entry_id)


def __show_config(entry_id):
    b = Booster(None, reload_config=False)
    b.load_entry_config(entry_id)
    print(yaml.dump(b._cfg[entry_id], sort_keys=False, default_flow_style=None, indent=4, Dumper=Do._MyDumper))


def __selector_helper(query, file='data/markets.yml'):
    """
    Use query for get symbols: 'binance futures all '
    """
    try:
        return _select_market(query, file).get('instrument', '')
    except Exception as e:
        print(f"Error> {str(e)}")
    return None


def __ls_parameters(clz):
    pls = _strategy_params(clz)
    args = ''
    if pls and 'portfolio' in pls:
        args = '\t\n' + ',\n'.join([f"\t{k} = {repr(v)}" for k,v in pls['portfolio'].get('parameters', {}).items()])
        args += '\n'
    print(f"\n\n{clz.__name__}({args})")


Boo.do = Do
Boo.portfolio = get_combined_portfolio
Boo.executions = get_combined_executions
Boo.delete = __delete_backtest
Boo.ls = __list_backtest
Boo.config = __show_config
Boo.show = __show_backtest
Boo.symbols = __selector_helper
Boo.parameters = __ls_parameters

