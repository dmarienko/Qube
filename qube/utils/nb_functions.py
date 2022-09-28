import numpy as np

from qube.datasource.controllers.MongoController import MongoController


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
