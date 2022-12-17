import json
import os

import sys
from os.path import expanduser, split, join, exists, isfile
from pathlib import Path

QUBE_ENV_VAR = 'QUBE_ENV'
QUBE_CONF_FOLDER_VAR = 'QUBE_CONF_FOLDER'
DEFAULT_ENV = 'DEFAULT'
TEST_ENV = 'TEST'

__json_to_props = dict()


def get_root_dir():
    return str(Path(__file__).absolute().parent.parent)
    # return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_properties(json_path, is_refresh=False):
    json_path = get_formatted_path(json_path)
    json_path = json_path if json_path.endswith(".json") else json_path + ".json"

    if is_refresh or json_path not in __json_to_props:
        with open(json_path) as f: __json_to_props[json_path] = json.load(f)
    return __json_to_props[json_path]


# get formatted path.
def get_formatted_path(file_path, relative_folder: str = None):
    if os.path.isabs(file_path):  # absolute path linux or windows
        return file_path
    # relative path from qube sources context
    if any(file_path.replace('\\', '/').startswith(x) for x in ['qube/']):
        return join(get_root_dir(), *Path(file_path).parts[1:])

    elif relative_folder:  # relative from specified folder
        return join(expanduser(relative_folder), file_path)

    elif QUBE_CONF_FOLDER_VAR in os.environ:  # relative from server config folder
        return join(os.environ[QUBE_CONF_FOLDER_VAR], file_path)

    else:  # relative path from running context
        return file_path


def get_env():
    if QUBE_ENV_VAR in os.environ:
        return os.environ[QUBE_ENV_VAR]
    else:
        sa0 = sys.argv[0]
        # check if it's a test running from nose2
        if sa0.endswith('nose2') or sa0.endswith('_testlauncher.py') or 'pytest' in sa0:
            return TEST_ENV
        # check if it's a test running from PyCharm 2016
        elif len(sys.argv) > 1 and 'test.py' in sys.argv[1].lower():
            return TEST_ENV
        # check if it's a test running from PyCharm 2017
        elif len(sys.argv) == 1 and '_jb_unittest_runner.py' in sa0.lower():
            return TEST_ENV
        else:
            return DEFAULT_ENV


def is_sc_integration_test():
    # it's always False. We overload it in sc_test_int
    return False


def get_config_path(prop_file_name):
    # returning absolute path to config file or config folder path if prop_file_name not specified
    return get_config_path_env(prop_file_name, get_env())


def get_config_path_env(file_name, env, is_properties=False):
    if env not in (TEST_ENV, DEFAULT_ENV,) and QUBE_CONF_FOLDER_VAR in os.environ:
        config_folder = os.environ[QUBE_CONF_FOLDER_VAR]
    else:
        config_folder = join(get_root_dir(), 'configs', 'config-' + env.lower())
    result = join(config_folder, file_name)
    result = result if not is_properties or result.endswith(".json") else result + ".json"
    if env not in (TEST_ENV, DEFAULT_ENV,):  # env var is set
        if exists(result) and isfile(result):  # config exists for env
            return result
        else:  # otherwise getting default
            return get_config_path_env(file_name, DEFAULT_ENV)
    else:
        return result


def get_config_properties(prop_file_name, is_refresh=False):
    return get_config_properties_env(prop_file_name, get_env(), is_refresh)


def get_main_properties(is_refresh=False):
    return get_config_properties('main-props.json', is_refresh)


def get_config_properties_env(prop_file_name, env, is_refresh=False):
    result = get_properties(get_config_path_env(prop_file_name, env, is_properties=True), is_refresh=is_refresh)
    if env not in (TEST_ENV, DEFAULT_ENV,):
        merged_with_default_props = get_properties(get_config_path_env(prop_file_name, DEFAULT_ENV, is_properties=True),
                                                   is_refresh=is_refresh)
        merged_with_default_props.update(result)
        return merged_with_default_props
    else:
        return result
