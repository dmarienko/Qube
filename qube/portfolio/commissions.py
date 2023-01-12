import pandas as pd

from qube.datasource import DataSource
from qube.portfolio.Instrument import Instrument
from qube.portfolio.broker_constants import *
from qube.quantitative.tools import infer_series_frequency
from qube.utils import QubeLogger

__logger = QubeLogger.getLogger(__name__)
EXCHANGE_SEPARATOR = ':'

_ent0 = lambda c1: {'USDT': c1}
_entusd0 = lambda c1: {'USD': c1}
_ent1 = lambda c1, c2: {'USDT': c1, 'BNB': c2}
_ent2 = lambda c1, c2, c3, c4: {'USDT': c1, 'USDT_BNB': c2, 'BUSD': c3, 'BUSD_BNB': c4}

_BINANCE_SPOT_FEES = {
    #              maker taker
    'vip0': _ent1([0.1, 0.1], [0.075, 0.075]),
    'vip1': _ent1([0.09, 0.1], [0.0675, 0.0750]),
    'vip2': _ent1([0.08, 0.1], [0.06, 0.075]),
    'vip3': _ent1([0.07, 0.1], [0.0525, 0.075]),
    'vip4': _ent1([0.02, 0.04], [0.015, 0.03]),
    'vip5': _ent1([0.02, 0.04], [0.015, 0.03]),
    'vip6': _ent1([0.02, 0.04], [0.015, 0.03]),
    'vip7': _ent1([0.02, 0.04], [0.015, 0.03]),
    'vip8': _ent1([0.02, 0.04], [0.015, 0.03]),
    'vip9': _ent1([0.02, 0.04], [0.015, 0.03]),
}

_BINANCE_UM_FEES = {
    #              maker taker
    'vip0': _ent2([0.02, 0.04], [0.0180, 0.0360], [-0.0100, 0.023], [-0.0100, 0.0207]),
    'vip1': _ent2([0.016, 0.04], [0.0144, 0.0360], [-0.0100, 0.023], [-0.0100, 0.0207]),
    'vip2': _ent2([0.014, 0.035], [0.0126, 0.0315], [-0.0100, 0.023], [-0.0100, 0.0207]),
    'vip3': _ent2([0.012, 0.032], [0.0108, 0.0288], [-0.0100, 0.023], [-0.0100, 0.0207]),
    'vip4': _ent2([0.01, 0.03], [0.0090, 0.027], [-0.0100, 0.023], [-0.0100, 0.0207]),
    'vip5': _ent2([0.008, 0.027], [0.0072, 0.0243], [-0.0100, 0.023], [-0.0100, 0.0207]),
    'vip6': _ent2([0.006, 0.025], [0.0054, 0.0225], [-0.0100, 0.023], [-0.0100, 0.0207]),
    'vip7': _ent2([0.004, 0.022], [0.0036, 0.0198], [-0.0100, 0.023], [-0.0100, 0.0207]),
    'vip8': _ent2([0.002, 0.02], [0.0018, 0.018], [-0.0100, 0.023], [-0.0100, 0.0207]),
    'vip9': _ent2([0.0, 0.017], [0.0000, 0.0153], [-0.0100, 0.023], [-0.0100, 0.0207]),
}

_BINANCE_FEES = {
    'spot': _BINANCE_SPOT_FEES,
    'um': _BINANCE_UM_FEES,  # TODO
    'cm': None,  # TODO
}

_WOOX_SPOT_FEES = {
    #              maker taker
    't0': _ent0([0.04, 0.04]),
    't1': _ent0([0.03, 0.03]),
    't2': _ent0([0.02, 0.02]),
    't3': _ent0([0.01, 0.01]),
    't4': _ent0([0.0, 0.0]),
    't5': _ent0([0.0, 0.0]),
    't6': _ent0([0.0, 0.0]),
}

_WOOX_FUTURES_FEES = {
    #              maker taker
    't0': _ent0([0.04, 0.04]),
    't1': _ent0([0.03, 0.03]),
    't2': _ent0([0.02, 0.02]),
    't3': _ent0([0.01, 0.01]),
    't4': _ent0([0.0, 0.0]),
    't5': _ent0([0.0, 0.0]),
    't6': _ent0([0.0, 0.0]),
}

_WOOX_FEES = {
    'spot': _WOOX_SPOT_FEES,
    'futures': _WOOX_FUTURES_FEES
}

_FTX_SPOT_FUTURES_FEES = {
    #              maker taker
    't1': _entusd0([0.020, 0.070]),
    't2': _entusd0([0.015, 0.060]),
    't3': _entusd0([0.010, 0.055]),
    't4': _entusd0([0.005, 0.050]),
    't5': _entusd0([0.000, 0.045]),
    't6': _entusd0([0.000, 0.040]),
}

_KRAKEN_SPOT_FEES = {
    #                maker  taker
    '0':    _entusd0([0.16, 0.26]),
    '50K':	_entusd0([0.14,	0.24]),
    '100K':	_entusd0([0.12,	0.22]),
    '250K':	_entusd0([0.10,	0.20]),
    '500K':	_entusd0([0.08,	0.18]),
    '1M':	_entusd0([0.06,	0.16]),
    '2.5M':	_entusd0([0.04,	0.14]),
    '5M':	_entusd0([0.02,	0.12]),
    '10M':	_entusd0([0.00,	0.10])
}

_KRAKEN_FUTURES_FEES = {
    #                maker  taker
    '0':    _entusd0([0.0200, 0.0500]),
    '100K':	_entusd0([0.0150, 0.0400]),
    '1M':	_entusd0([0.0125, 0.0300]),
    '5M':	_entusd0([0.0100, 0.0250]),
    '10M':	_entusd0([0.0075, 0.0200]),
    '20M':	_entusd0([0.0050, 0.0150]),
    '50M':	_entusd0([0.0025, 0.0125]),
    '100M':	_entusd0([0.0000, 0.0100]),
}

_KRAKEN_FEES = {
    'spot': _KRAKEN_SPOT_FEES,
    'futures': _KRAKEN_FUTURES_FEES,
}


class TransactionCostsCalculator:
    def __init__(self, taker, maker):
        self.taker = taker
        self.maker = maker

    def get_execution_cost(self, instrument: Instrument, exec_price, amount, crossed_market=False,
                           conversion_price=1.0):
        if not instrument.is_futures:
            amount = amount * exec_price

        if crossed_market:
            return conversion_price * abs(amount) * self.taker
        else:
            return conversion_price * abs(amount) * self.maker

    def __repr__(self):
        return f'<Maker / Taker: {self.maker * 100:.4f} / {self.taker * 100:.4f}>'


class BinanceRatesCommon(TransactionCostsCalculator):
    """
    Some generic helper for rates
    """

    def __init__(self, asset_type, level, fees_currency='USDT'):
        fees_data = _BINANCE_FEES.get(asset_type)

        if not fees_data:
            raise ValueError(f"Can't find fee rates data for {asset_type} on Binance")

        fees = fees_data.get(level)
        if not fees:
            raise ValueError(f"Can't find fee rates data for {asset_type} | {level} on Binance")

        if isinstance(fees, dict):
            rates = fees.get(fees_currency)
            if rates is None:
                raise ValueError(f"Can't find {asset_type} rates data for {fees_currency} on Binance")
        else:
            rates = fees

        if len(rates) < 2:
            raise ValueError(f"Incorrect rates data for {asset_type} / {fees_currency} on Binance")

        super().__init__(maker=rates[0] / 100, taker=rates[1] / 100, )


class WooXRatesCommon(TransactionCostsCalculator):
    """
    Some generic helper for rates
    """

    def __init__(self, asset_type, level, fees_currency='USDT'):
        fees_data = _WOOX_FEES.get(asset_type)

        if not fees_data:
            raise ValueError(f"Can't find fee rates data for {asset_type} on WooX")

        fees = fees_data.get(level)
        if not fees:
            raise ValueError(f"Can't find fee rates data for {asset_type} | {level} on WooX")

        if isinstance(fees, dict):
            rates = fees.get(fees_currency)
            if rates is None:
                raise ValueError(f"Can't find {asset_type} rates data for {fees_currency} on WooX")
        else:
            rates = fees

        if len(rates) < 2:
            raise ValueError(f"Incorrect rates data for {asset_type} / {fees_currency} on WooX")

        super().__init__(maker=rates[0] / 100, taker=rates[1] / 100, )


class KrakenRatesCommon(TransactionCostsCalculator):
    """
    Kraken rates
    """

    def __init__(self, asset_type, level, fees_currency='USD'):
        fees_data = _KRAKEN_FEES.get(asset_type)

        if not fees_data:
            raise ValueError(f"Can't find fee rates data for {asset_type} on Kraken")

        fees = fees_data.get(level)
        if not fees:
            raise ValueError(f"Can't find fee rates data for {asset_type} | {level} on Kraken")

        if isinstance(fees, dict):
            rates = fees.get(fees_currency)
            if rates is None:
                raise ValueError(f"Can't find {asset_type} rates data for {fees_currency} on Kraken")
        else:
            rates = fees

        if len(rates) < 2:
            raise ValueError(f"Incorrect rates data for {asset_type} / {fees_currency} on Kraken")

        super().__init__(maker=rates[0] / 100, taker=rates[1] / 100)


class FTXRatesCommon(TransactionCostsCalculator):
    """
    Some generic helper for rates
    """

    def __init__(self, asset_type, level, fees_currency='USD'):
        fees_data = _FTX_SPOT_FUTURES_FEES
        fees = fees_data.get(level)
        if not fees:
            raise ValueError(f"Can't find fee rates data for {asset_type} | {level} on FTX")

        if isinstance(fees, dict):
            rates = fees.get(fees_currency)
            if rates is None:
                raise ValueError(f"Can't find {asset_type} rates data for {fees_currency} on FTX")
        else:
            rates = fees

        if len(rates) < 2:
            raise ValueError(f"Incorrect rates data for {asset_type} / {fees_currency} on FTX")

        super().__init__(maker=rates[0] / 100, taker=rates[1] / 100, )


class StockTCC(TransactionCostsCalculator):
    def __init__(self, bps):
        super().__init__(bps, bps)


class ZeroTCC(TransactionCostsCalculator):
    def __init__(self):
        super().__init__(0, 0)


class BitmexTCC(TransactionCostsCalculator):

    def __init__(self, taker=0.05 / 100, maker=-0.01 / 100):
        super().__init__(taker, maker)


class ForexTCC(TransactionCostsCalculator):

    def __init__(self, taker=35 / 1e6, maker=35 / 1e6):
        super().__init__(taker, maker)

    def get_execution_cost(self, instrument, exec_price, amount, crossed_market=False, conversion_price=1.0):
        return conversion_price * abs(amount) * self.taker


class FxcmTCC(TransactionCostsCalculator):
    CTABLE = {
        'EURUSD': 4, 'GBPUSD': 4, 'USDJPY': 4, 'USDCHF': 4,
    }

    def __init__(self):
        super().__init__(0, 0)

    def get_execution_cost(self, instrument, exec_price, amount, crossed_market=False, conversion_price=1.0):
        return conversion_price * abs(amount) * FxcmTCC.CTABLE.get(instrument.symbol, 6) / 1e5


def get_calculator(comm_calc_name: str):
    if comm_calc_name.lower() in {DUKAS_BROKER_NAME, 'dukascopy'}:
        return DukasCommissionsCalculator()
    elif comm_calc_name.lower() in {HITBTC_BROKER_NAME}:
        return HitbtcCommissionsCalculator()
    elif comm_calc_name.lower() in {BITFINEX_BROKER_NAME}:
        return BitfinexCommissionsCalculator()
    elif comm_calc_name.lower() in {POLONIEX_BROKER_NAME}:
        return PoloniexCommissionsCalculator()
    elif comm_calc_name.lower() in {OKEX_BROKER_NAME}:
        return OkexCommissionsCalculator()
    elif comm_calc_name.lower() in {BITMEX_BROKER_NAME}:
        return BitmexCommissionsCalculator()
    elif comm_calc_name.lower() in {BITMEX_NEW_BROKER_NAME}:
        return BitmexNewCommissionsCalculator()
    elif comm_calc_name.lower() in {BITMEX_LIMITS_BROKER_NAME}:
        return BitmexLimitsCommissionsCalculator()
    elif comm_calc_name.lower() in {BINANCE_SPOT_BROKER_NAME}:
        return BinanceSpotCommissionsCalculator()
    elif comm_calc_name.lower() in {BINANCE_USDT_BROKER_NAME}:
        return BinanceUSDTCommissionsCalculator()
    elif comm_calc_name.lower() in {BINANCE_COINM_BROKER_NAME}:
        return BinanceCOINMCommissionsCalculator()
    elif comm_calc_name.lower() in {"crypto_arb", 'crypto_multi_exchange'}:
        return MultiExchangeCryptoCurrencyCommissionsCalculator()

    __logger.warn('Undefined CommissionCalculator %s' % comm_calc_name)
    return {}


def get_total_commissions(commissions_dict):
    """
    Caclulate total commissions value from commissions dict
    :param commissions_dict: commissions dict returned from commissions.calculate
    :return: total commissions
    """
    return sum(sum(commissions_dict.values())) if commissions_dict else 0.0


class ICommissionsCalculator:
    def calculate(self, pfl_log, exec_log: pd.DataFrame = None):
        raise NotImplementedError('Must be implemented in child class %s', self.__class__.__name__)

    @staticmethod
    def _calculate_traded_volume(pfl_log, exec_log: pd.DataFrame = None, is_futures=False) -> dict:
        """
        Finds summary traded volume per instrument.

        :param pfl_log: position manager log (portfolio log)
        :return dict(type->instrument->volume)
                Example: {'MARKET': {EURUSD':100500, 'MSFT':555}} (100,500$ of EURUSD, 555 shares of MSFT in MARKET type)
        """
        if exec_log is not None:
            sum_series_market = {}
            sum_series_limit = {}
            freq = infer_series_frequency(pfl_log.index)
            for inst in exec_log.instrument.unique():
                execinst_market = exec_log[((exec_log.instrument == inst) & (exec_log.type == 'MARKET'))]
                execinst_limit = exec_log[((exec_log.instrument == inst) & (exec_log.type == 'LIMIT'))]
                if is_futures:
                    # quantity futures is already in usd
                    execprice_market = 1
                    execprice_limit = 1
                else:
                    execprice_market = execinst_market.fill_inc_update_price
                    execprice_limit = execinst_limit.fill_inc_update_price
                inst_market_ser = (execinst_market.fill_inc_update_quantity.abs() * execprice_market).resample(
                    freq).sum().replace(False, 0)
                inst_limit_ser = (execinst_limit.fill_inc_update_quantity.abs() * execprice_limit).resample(
                    freq).sum().replace(False, 0)
                # TODO: add column usd_exec_price in exec_log for work on non USD pair
                sum_series_market.update({inst: pd.Series(data=inst_market_ser, index=pfl_log.index).fillna(0)})
                sum_series_limit.update({inst: pd.Series(data=inst_limit_ser, index=pfl_log.index).fillna(0)})
            return {'MARKET': sum_series_market, 'LIMIT': sum_series_limit}
        else:
            pl = pfl_log.filter(regex=r'.*_Pos')
            pl_diff_pos = pl.diff()
            pl_diff_pos = pl_diff_pos.rename(lambda x: x[:-4], axis=1)
            pl = pfl_log.filter(regex=r'.*_Value')
            pl_diff_value = pl.diff()
            pl_diff_value = pl_diff_value.rename(lambda x: x[:-6], axis=1)

            if is_futures:
                for column in pl_diff_value.columns:
                    pl_diff_value[column] = pl_diff_value[column] / pfl_log[column + "_Price"]

            none_zero_pos = pl_diff_pos[pl_diff_pos != 0.0]
            sum_series = {}
            for column in pl_diff_pos.columns:
                idx = none_zero_pos.loc[:, column].dropna().index
                sum_series.update(
                    {column: pd.Series(data=abs(pl_diff_value.loc[idx, column]), index=pfl_log.index).fillna(0)})
            return {'MARKET': sum_series}


class DukasCommissionsCalculator(ICommissionsCalculator):
    def calculate(self, pfl_log, exec_log: pd.DataFrame = None):
        traded_volume = DukasCommissionsCalculator._calculate_traded_volume(pfl_log, exec_log)
        total_commissions = pd.Series(0.0, index=pfl_log.index)
        dukas_outlook_df = DataSource('dukas_outlook').load_data()
        for exec_type in traded_volume:
            for instrument, instrument_volume in traded_volume[exec_type].items():
                if self.is_fx_instrument_static(instrument, dukas_outlook_df):
                    # fx dukas default 35$ per 1MM$ traded (max fx commission rate)
                    total_commissions += 35.0 * instrument_volume / 10e5
                else:
                    # CFDs etc. need add later
                    pass

        return {DUKAS_BROKER_NAME: total_commissions}

    def is_fx_instrument_static(self, instrument, dukas_outlook_df: pd.DataFrame) -> bool:
        if instrument in dukas_outlook_df.index:
            return dukas_outlook_df.loc[instrument]['type'] == 'fx'
        else:
            return False


class CryptoCurrencyCommissionsCalculator(ICommissionsCalculator):
    def __init__(self, fee: float, broker_name, rebate: float = 0):
        self.fee = fee
        self.rebate = rebate
        self.broker_name = broker_name

    def calculate(self, pfl_log, exec_log: pd.DataFrame = None):
        total_commission = pd.Series(0.0, index=pfl_log.index)
        traded_volume = CryptoCurrencyCommissionsCalculator._calculate_traded_volume(pfl_log, exec_log,
                                                                                     self.__is_futures())
        for exec_type in traded_volume:
            for instrument, instrument_volume in traded_volume[exec_type].items():
                fee = self.fee if exec_type == 'MARKET' else self.rebate
                total_commission += fee * instrument_volume
        return {self.broker_name: total_commission}

    def __is_futures(self):
        return self.broker_name in {
            # TODO: redo this !!!
            BITMEX_BROKER_NAME,
            BITMEX_NEW_BROKER_NAME,
            BITMEX_LIMITS_BROKER_NAME,
            BINANCE_USDT_BROKER_NAME,
            BINANCE_COINM_BROKER_NAME
        }


class MultiExchangeCryptoCurrencyCommissionsCalculator(ICommissionsCalculator):

    def calculate(self, pfl_log, exec_log: pd.DataFrame = None):
        result = {}
        brokers = list(set([b.split(EXCHANGE_SEPARATOR)[0] for b in pfl_log.filter(regex=r'.*_Pos').columns]))
        for broker_name in brokers:
            broker_porf_log = pfl_log.filter(regex=broker_name + EXCHANGE_SEPARATOR + ".*")
            broker_exec_log = None if exec_log is None else exec_log[
                exec_log.instrument.str.contains(broker_name + EXCHANGE_SEPARATOR)]
            calc = get_calculator(broker_name)

            result.update(calc.calculate(broker_porf_log, broker_exec_log))
        return result


class HitbtcCommissionsCalculator(CryptoCurrencyCommissionsCalculator):
    def __init__(self):
        # 0.1% for taker orders (max fee)
        super().__init__(0.001, HITBTC_BROKER_NAME)


class BitfinexCommissionsCalculator(CryptoCurrencyCommissionsCalculator):
    def __init__(self):
        # 0.2% for taker orders (max fee)
        super().__init__(0.002, BITFINEX_BROKER_NAME)


class PoloniexCommissionsCalculator(CryptoCurrencyCommissionsCalculator):
    def __init__(self):
        super().__init__(0.002, POLONIEX_BROKER_NAME)


class OkexCommissionsCalculator(CryptoCurrencyCommissionsCalculator):
    def __init__(self):
        super().__init__(0.002, OKEX_BROKER_NAME)


class BitmexCommissionsCalculator(CryptoCurrencyCommissionsCalculator):
    def __init__(self):
        super().__init__(0.00075, BITMEX_BROKER_NAME, -0.00025)


class BitmexNewCommissionsCalculator(CryptoCurrencyCommissionsCalculator):
    def __init__(self):
        super().__init__(0.0005, BITMEX_NEW_BROKER_NAME, -0.0001)


class BitmexLimitsCommissionsCalculator(CryptoCurrencyCommissionsCalculator):
    """
    For estimating if we'd operate only limit orders (no martket at all)
    """

    def __init__(self):
        super().__init__(-0.0001, BITMEX_LIMITS_BROKER_NAME, -0.0001)


class BinanceSpotCommissionsCalculator(CryptoCurrencyCommissionsCalculator):
    def __init__(self):
        super().__init__(0.001, BINANCE_SPOT_BROKER_NAME, 0.001)


class BinanceUSDTCommissionsCalculator(CryptoCurrencyCommissionsCalculator):
    def __init__(self):
        super().__init__(0.0004, BINANCE_USDT_BROKER_NAME, 0.0002)


class BinanceCOINMCommissionsCalculator(CryptoCurrencyCommissionsCalculator):
    def __init__(self):
        super().__init__(0.0005, BINANCE_COINM_BROKER_NAME, 0.0001)
