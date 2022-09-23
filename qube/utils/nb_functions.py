from typing import Union, Dict

import numpy as np
import pandas as pd

from qube import runtime_env
from qube.portfolio.commissions import (
    TransactionCostsCalculator, ZeroTCC, ForexTCC, FxcmTCC, StockTCC, BitmexTCC,
    GenericUSDTM_TCC_5_2, GenericUSDTM_TCC_5_0,
    BinanceCOINM_TCC_VIP0,
    BinanceUSDTM_TCC_VIP0,
    BinanceUSDTM_TCC_VIP1,
    BinanceUSDTM_TCC_VIP2,
    BinanceUSDTM_TCC_VIP3,
    BinanceUSDTM_TCC_VIP4,
    BinanceUSDTM_TCC_VIP5,
    BinanceUSDTM_TCC_VIP9, BinanceRatesCommon,
    WooXRatesCommon
)
from qube.simulator import SignalTester
from qube.simulator.Brokerage import (
    GenericStockBrokerInfo, GenericForexBrokerInfo, GenericCryptoBrokerInfo,
    GenericCryptoFuturesBrokerInfo, BrokerInfo
)
from qube.simulator.SignalTester import Tracker, ExecutionLogger, SimulationResult
from qube.datasource.controllers.MongoController import MongoController


def _progress_bar(description='[Backtest]'):
    """
    Default progress bar (based on tqdm)
    """
    if runtime_env() == 'notebook':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    class __MyProgress:
        def __init__(self, descr):
            self.p = tqdm(total=100, unit_divisor=1, unit_scale=1, unit=' quotes', desc=descr)

        def __call__(self, i, label=None):
            d = i - self.p.n
            if d > 0:
                self.p.update(d)

    return __MyProgress(description)


def z_load(data_id, host=None, query=None, dbname=None, username="", password=""):
    """
    Loads data stored in mongo DB
    :param data_id: record's name (not mongo __id)
    :param host: server's host (default localhost)
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


def z_backtest(signals, datasource, broker,
               spread: Union[float, Dict[str, float]] = None,
               execution_logger: bool = False,
               trackers: Union[Tracker, Dict[str, Tracker]] = None,
               single_tracker_for_all=False,
               progress=None,
               name: str = None,
               tcc: TransactionCostsCalculator = ZeroTCC(),
               verbose=False,
               **kwargs) -> SimulationResult:
    """
    Shortcut for test trading signals in current notebook process.
    It's good for fast testing on small set of instruments and history.

    :param signals: set of signals
    :param datasource: datasource to be used for testing
    :param broker: brokerage details ('forex' or 'stock' are supported now)
    :param spread: default spread size (only for OHLC cases)
    :param execution_logger: execution logger
    :param trackers: custom positions trackers
    :param single_tracker_for_all: if true and tracker is object do not make copy for each symbol
    :param progress: custom progress indicator
    :param name: name of simualtion
    :param tcc: transaction costs calculator (default is None i.e. zero commissions)
    :param verbose: true if need more information
    :return: simulation results structure
    """
    if isinstance(datasource, (pd.DataFrame, dict)):
        # not needed to import this class to notebook
        from qube.datasource.InMemoryDataSource import InMemoryDataSource
        datasource = InMemoryDataSource(datasource)

    tester = SignalTester(__instantiate_simulated_broker(broker, spread, tcc), datasource)
    r = tester.run_signals(signals,
                           jupyter_progress_listener=_progress_bar() if progress is None else progress,
                           tracker=trackers,
                           single_tracker_for_all=single_tracker_for_all,
                           execution_logger=ExecutionLogger() if execution_logger else None,
                           verbose=verbose, name=name, **kwargs)
    return r


def z_test_signals_inplace(signals, datasource, broker, spread: Union[float, Dict[str, float]] = None,
                           execution_logger: ExecutionLogger = None,
                           trackers: Union[Tracker, Dict[str, Tracker]] = None,
                           verbose=False,
                           **kwargs) -> SimulationResult:
    """
    Shortcut for test trading signals in current notebook process.
    It's good for fast testing on small set of instruments and history.
    Just for compatibility

    :return: portfolio log dataframe(PnL is splitted)
    """
    r = z_backtest(signals, datasource, broker, spread, execution_logger is not None, trackers, verbose, **kwargs)
    return r.portfolio


def np_fmt_short():
    # default np output is 75 columns so extend it a bit and suppress scientific fmt for small floats
    np.set_printoptions(linewidth=240, suppress=True)


def np_fmt_reset():
    # reset default np printing options
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8,
                        suppress=False, threshold=1000, formatter=None)


def __create_brokerage_instances(spread: Union[dict, float], tcc: TransactionCostsCalculator = None) -> dict:
    """
    Some predefined list of broerages
    """

    def _binance_generator(broker_class, mtype, currencies):
        def _f(_cl, _t, _i, _c):
            return lambda: _cl(spread=spread, tcc=BinanceRatesCommon(_t, _i, _c))

        return {f'binance_{mtype}_vip{i}_{c.lower()}': _f(broker_class, mtype, f'vip{i}', c.upper()) for i in range(10)
                for c in currencies}

    def _woox_generator(broker_class, mtype, currencies):
        def _f(_cl, _t, _i, _c):
            return lambda: _cl(spread=spread, tcc=WooXRatesCommon(_t, _i, _c))

        return {f'woox_{mtype}_t{i}_{c.lower()}': _f(broker_class, mtype, f't{i}', c.upper()) for i in range(7)
                for c in currencies}

    return {
        'stock': lambda: GenericStockBrokerInfo(spread=spread, tcc=StockTCC(0.05 / 100) if tcc is None else tcc),
        'forex': lambda: GenericForexBrokerInfo(spread=spread,
                                                tcc=ForexTCC(35 / 1e6, 35 / 1e6) if tcc is None else tcc),
        'crypto': lambda: GenericCryptoBrokerInfo(spread=spread, tcc=ZeroTCC() if tcc is None else tcc),
        'crypto_futures': lambda: GenericCryptoFuturesBrokerInfo(spread=spread, tcc=ZeroTCC() if tcc is None else tcc),

        # --- some predefined ---
        'bitmex': lambda: GenericCryptoFuturesBrokerInfo(spread=spread, tcc=BitmexTCC()),
        'bitmex_vip': lambda: GenericCryptoFuturesBrokerInfo(spread=spread, tcc=BitmexTCC(0.03 / 100, -0.01 / 100)),
        'binance_cm_vip0': lambda: GenericCryptoFuturesBrokerInfo(spread=spread, tcc=BinanceCOINM_TCC_VIP0()),

        # - these are remained for compatibility
        'binance_um_vip0': lambda: GenericCryptoFuturesBrokerInfo(spread=spread, tcc=BinanceUSDTM_TCC_VIP0()),
        'binance_um_vip1': lambda: GenericCryptoFuturesBrokerInfo(spread=spread, tcc=BinanceUSDTM_TCC_VIP1()),
        'binance_um_vip2': lambda: GenericCryptoFuturesBrokerInfo(spread=spread, tcc=BinanceUSDTM_TCC_VIP2()),
        'binance_um_vip3': lambda: GenericCryptoFuturesBrokerInfo(spread=spread, tcc=BinanceUSDTM_TCC_VIP3()),
        'binance_um_vip4': lambda: GenericCryptoFuturesBrokerInfo(spread=spread, tcc=BinanceUSDTM_TCC_VIP4()),
        'binance_um_vip5': lambda: GenericCryptoFuturesBrokerInfo(spread=spread, tcc=BinanceUSDTM_TCC_VIP5()),
        'binance_um_vip9': lambda: GenericCryptoFuturesBrokerInfo(spread=spread, tcc=BinanceUSDTM_TCC_VIP9()),

        # - Binance spot
        **_binance_generator(GenericCryptoBrokerInfo, 'spot', ['USDT', 'BNB']),

        # - Binance um
        **_binance_generator(GenericCryptoFuturesBrokerInfo, 'um', ['USDT', 'USDT_BNB', 'BUSD', 'BUSD_BNB']),

        # - WooX spot
        **_woox_generator(GenericCryptoBrokerInfo, 'spot', ['USDT']),

        # - WooX spot
        **_woox_generator(GenericCryptoFuturesBrokerInfo, 'futures', ['USDT']),

        'crypto_futures_52': lambda: GenericCryptoFuturesBrokerInfo(spread=spread, tcc=GenericUSDTM_TCC_5_2()),
        'crypto_futures_50': lambda: GenericCryptoFuturesBrokerInfo(spread=spread, tcc=GenericUSDTM_TCC_5_0()),
        'dukas': lambda: GenericForexBrokerInfo(spread=spread, tcc=ForexTCC(35 / 1e6, 35 / 1e6)),
        'dukas_premium': lambda: GenericForexBrokerInfo(spread=spread, tcc=ForexTCC(17.5 / 1e6, 17.5 / 1e6)),
        'fxcm': lambda: GenericForexBrokerInfo(spread=spread, tcc=FxcmTCC()),
    }


def __instantiate_simulated_broker(broker, spread: Union[dict, float],
                                   tcc: TransactionCostsCalculator = None) -> BrokerInfo:
    if isinstance(broker, str):
        # for general brokers we require implicit spreads here
        if spread is None:
            raise ValueError("Spread policy must be specified ! You need pass either fixed spread or dictionary")

        predefined_brokers = __create_brokerage_instances(spread, tcc)

        brk_ctor = predefined_brokers.get(broker)
        if brk_ctor is None:
            raise ValueError(
                f"Unknown broker type '{broker}'\nList of supported brokers: [{','.join(predefined_brokers.keys())}]")

        # instantiate broker
        broker = brk_ctor()

    return broker


def ls_brokers():
    """
    List of simulated brokers supported by default
    """
    return [f'{k}({str(v().tcc)})' for k, v in __create_brokerage_instances(0, None).items()]
