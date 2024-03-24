import numpy as np
from itertools import zip_longest
from collections import defaultdict
import glob, ast, os

from qube.datasource.controllers.MongoController import MongoController
from qube.utils.ui_utils import red, yellow, blue, green, magenta, cyan


def z_load(data_id: str, host: str = None, query: str = None, dbname: str = None, username="", password=""):
    """
    Loads data stored in mongo DB
    :param data_id: record's name (not mongo __id)
    :param host: server's host (default localhost)
    :param query: additional query to mongo if needed
    :param dbname: database name
    :param username: user
    :param password: pass
    :return: data record
    """
    if host:
        from qube.utils.utils import remote_load_mongo_table
        return remote_load_mongo_table(host, data_id, dbname, query, username, password)
    else:
        mongo = MongoController(dbname, username=username, password=password)
        if query:
            data = mongo.load_records(data_id, query, True)
        else:
            data = mongo.load_data(data_id)
        mongo.close()
        return data


def z_ld(data_id, host=None, dbname=None, username="", password=""):
    """
    Just shortcuted version of z_load
    """
    return z_load(data_id, host=host, dbname=dbname, username=username, password=password)['data']


def z_save(data_id, data, host=None, dbname=None, is_serialize=True, username="", password=""):
    """
    Store data to mongo db at given id
    """
    if host:
        from qube.utils.utils import remote_save_mongo_table
        remote_save_mongo_table(host, data_id, data, dbname=dbname, is_serialize=is_serialize, username=username,
                                password=password)
    else:
        mongo = MongoController(dbname, username=username, password=password)
        mongo.save_data(data_id, data, is_serialize=is_serialize)
        mongo.close()


def z_ls(query=r'.*', host=None, dbname=None, username="", password=""):
    if host:
        from qube.utils.utils import remote_ls_mongo_table
        return remote_ls_mongo_table(host, query, dbname, username=username, password=password)
    else:
        mongo = MongoController(dbname, username=username, password=password)
        data = mongo.ls_data(query)
        mongo.close()
        return data


def z_del(name, host=None, dbname=None, username="", password=""):
    """
    Delete object from operational database by name

    :param name: object's name
    """
    if host:
        from qube.utils.utils import remote_del_mongo_table
        return remote_del_mongo_table(host, name, dbname, username, password)
    else:
        mongo = MongoController(dbname, username=username, password=password)
        result = mongo.del_data(name)
        mongo.close()
        return result


def np_fmt_short():
    # default np output is 75 columns so extend it a bit and suppress scientific fmt for small floats
    np.set_printoptions(linewidth=240, suppress=True)


def np_fmt_reset():
    # reset default np printing options
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8,
                        suppress=False, threshold=1000, formatter=None)


def scan_directory(root_path: str):
    def _iter_funcs(class_node):
        decls = {}
        for n in ast.walk(class_node): 
            if isinstance(n, ast.FunctionDef) and n.name == '__init__':
                args = []
                for x in n.args.args:
                    args.append(x.arg)
                for x, dx in zip_longest(reversed(args), reversed(n.args.defaults)):
                    if x != 'self':
                        # print(f"{x} = {dx.value if dx else None}")
                        decls[x] = dx.value if dx and isinstance(dx, ast.Constant) else None
        # return dict(reversed(decls))
        return dict(reversed(list(decls.items())))

    def _flatten(decl_list):
        for d in decl_list:
            try:
                yield d.id
            except AttributeError:
                try:
                    yield d.func.id
                except AttributeError:
                    yield None

    def _is_decorated(decorator_list, decorator_name):
        for x in _flatten(decorator_list):
            if x == decorator_name:
                return True
        return False


    strats = defaultdict(list)
    for file in glob.glob(os.path.join(root_path, '**/*.py'), recursive=True):
        name = os.path.splitext(os.path.basename(file))[0]
        # Ignore __ files
        if name.startswith("__"):
            continue

        with open(file, 'r') as f:
            src = f.read()

        try:
            class_node = ast.parse(src)
        except:
            continue

        nodes = [node for node in ast.walk(class_node) if isinstance(node, ast.ClassDef)]
        for n in nodes:
            if _is_decorated(n.decorator_list, 'signal_generator'):
                strats[file].append((n.name, ast.get_docstring(n), _iter_funcs(n)))
    return strats


def ls_strats(direcory=os.path.relpath(os.path.expanduser('~/projects'))):
    """
    List all available Qube1 based strategies in directory
    """
    strats = scan_directory(direcory)
    for f, sd in strats.items():
        strs = ""
        for sn, descr, pars in sd:
            descr = (': ' + green(descr.replace('\n', ' ').strip('" '))) if descr else ''
            strs += f"   |-- {cyan(sn)} {descr} \n   |   {blue(str(pars))}\n   |\n"

        rst = f""" - {magenta(f)} -
    {strs}"""
        print(rst)
