import numbers

import numpy as np

from qube.portfolio.Position import Position


class IPositionSizer:
    """
    Common interface for any positions size calculator
    """

    def get_position_size(self, signal, position: Position,
                          entry_price: float,
                          stop_price: float = None,
                          take_price: float = None):
        """
        Position size calculator
        :param signal: signal to process
        :param position: current Position object for this instrument
        :param entry_price: price for entrering position (might be current market midprice)
        :param stop_price: planned stop price level
        :param take_price: planned take price level
        """
        raise ValueError("Not implemented method")

    @staticmethod
    def wrap_fixed(value):
        """
        Just small helper
        """
        return FixedSizer(value) if isinstance(value, numbers.Number) else value


class FixedSizer(IPositionSizer):
    def __init__(self, fixed_size):
        self.fixed_size = abs(fixed_size)

    def get_position_size(self, signal, position: Position,
                          entry_price: float, stop_price: float = None, take_price: float = None):
        return signal * self.fixed_size


class FixedRiskSizer(IPositionSizer):
    """
    Calculate position size based on maximal risk per entry
    """

    def __init__(self, capital: float, max_cap_in_risk: float, max_allowed_position=np.inf):
        """
        Create fixed risk sizer calculator instance.
        :param capital: capital allocated
        :param max_cap_in_risk: maximal risked capital (in percentage)
        :param max_allowed_position: limitation for max position size
        """
        self.capital = capital
        self.max_cap_in_risk = max_cap_in_risk
        self.max_allowed_position = max_allowed_position

    def get_position_size(self, signal, position: Position,
                          entry_price: float, stop_price: float = None, take_price: float = None):
        if signal != 0:
            if stop_price > 0:
                direction = np.sign(signal)
                cap = self.capital + max(position.pnl, 0)
                pos_size = direction * min(
                    round(
                        (cap * self.max_cap_in_risk / 100) / abs(stop_price / entry_price - 1)),
                    self.max_allowed_position
                )

                if not position.instrument.is_futures:
                    pos_size = pos_size / entry_price
                else:
                    # using adjustntry price and aligned contract, calc USDT pos size
                    pos_size = round(pos_size, int(position.instrument.futures_contract_size))

                return pos_size
            else:
                print(" >>> FixedRiskSizer: stop is not specified - can't calculate position !")

        return 0