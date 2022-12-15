from datetime import timedelta
from typing import Union

from qube.portfolio.commissions import StockTCC, TransactionCostsCalculator, ForexTCC, ZeroTCC
from qube.portfolio.Instrument import Instrument
from qube.portfolio.Position import Position, ForexPosition, CryptoPosition, CryptoFuturesPosition


class BrokerInfo:
    """
    Additional info about broker:
      - broker specific transaction cost info / calculus
      - margin levels/call
      - instruments info
      - service info
    There will be for instance DukasBrokerInfo, IBInfo, LimeInfo, etc
    """

    def session_times(self):
        raise NotImplementedError("Do not use BrokerInfo directly !")

    def create_position(self, instrument) -> Position:
        raise NotImplementedError("Do not use BrokerInfo directly !")


class GenericStockBrokerInfo(BrokerInfo):
    """
    Generic stocks broker with fixed spread for all instruments
    """

    def __init__(self, spread: Union[dict, float], tcc: TransactionCostsCalculator = StockTCC(0.05 / 100)):
        self.spread = spread
        self.tcc = tcc
        self._is_variable_spread = isinstance(spread, dict)

    def session_times(self):
        return timedelta(hours=9, minutes=29, seconds=59), timedelta(hours=15, minutes=59, seconds=59)

    def create_position(self, symbol):
        if self._is_variable_spread:

            if symbol in self.spread:
                return Position(Instrument(symbol, False, 0.01, self.spread.get(symbol)), self.tcc)
            else:
                raise ValueError("GenericStockBrokerInfo has no spread information for '%s' !" % symbol)

        # fixed spread
        return Position(Instrument(symbol, False, 0.01, self.spread), self.tcc)


class GenericForexBrokerInfo(BrokerInfo):
    """
    Generic forex broker
    """

    def __init__(self, spread: Union[dict, float], tcc: TransactionCostsCalculator = ForexTCC()):
        self.spread = spread
        self.tcc = tcc
        self._is_variable_spread = isinstance(spread, dict)

    def session_times(self):
        return timedelta(hours=0, minutes=0), timedelta(hours=23, minutes=59, seconds=58)

    def create_position(self, instrument):
        pts = 0.001 if 'JPY' in instrument else 0.00001

        if self._is_variable_spread:

            if instrument in self.spread:
                return ForexPosition(Instrument(instrument, False, pts, spread=self.spread.get(instrument)), self.tcc)
            else:
                raise ValueError("GenericForexBrokerInfo has no spread information for '%s' !" % instrument)

        # fixed spread
        return ForexPosition(Instrument(instrument, False, pts, spread=self.spread), self.tcc)


class GenericCryptoBrokerInfo(BrokerInfo):
    """
    Generic crypto broker info
    """

    def __init__(self, spread: Union[dict, float], tcc: TransactionCostsCalculator = ZeroTCC()):
        self.spread = spread
        self.tcc = tcc
        self._is_variable_spread = isinstance(spread, dict)

    def session_times(self):
        return timedelta(hours=0, minutes=0), timedelta(hours=23, minutes=59, seconds=58)

    def create_position(self, instrument):
        if self._is_variable_spread:

            if instrument in self.spread:
                return CryptoPosition(Instrument(instrument, False, 0.01, spread=self.spread.get(instrument)), self.tcc)
            else:
                raise ValueError("GenericCryptoBrokerInfo has no spread information for '%s' !" % instrument)

        return CryptoPosition(Instrument(instrument, False, 0.01, spread=self.spread), self.tcc)


class GenericCryptoFuturesBrokerInfo(BrokerInfo):
    """
    Generic crypto futures broker info
    """

    def __init__(self, spread: Union[dict, float], tcc: TransactionCostsCalculator = ZeroTCC()):
        self.spread = spread
        self.tcc = tcc
        self._is_variable_spread = isinstance(spread, dict)

    def session_times(self):
        return timedelta(hours=0, minutes=0), timedelta(hours=23, minutes=59, seconds=58)

    def create_position(self, instrument):
        if self._is_variable_spread:

            if instrument in self.spread:
                return CryptoFuturesPosition(Instrument(instrument, True, 0.01,
                                                        spread=self.spread.get(instrument)), self.tcc)
            else:
                raise ValueError("GenericCryptoFuturesBrokerInfo has no spread information for '%s' !" % instrument)

        return CryptoFuturesPosition(Instrument(instrument, True, 0.01, spread=self.spread), self.tcc)
