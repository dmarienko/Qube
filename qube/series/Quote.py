from datetime import datetime
from typing import Union

TIME = 'time'
INSTRUMENT = 'instrument'
BID = 'bid'
ASK = 'ask'
BID_SIZE = 'bid_size'
ASK_SIZE = 'ask_size'


class Quote:
    """
        Quote representing data class. Probably we will need methods for compare to another quote later.
    """

    def __init__(self, time: datetime, bid: float, ask: float,
                 bid_size: Union[float, int], ask_size: Union[float, int]):
        self.time = time
        self.bid = bid
        self.ask = ask
        self.bid_size = bid_size
        self.ask_size = ask_size

    def midprice(self):
        """
        Midpoint price
        
        :return: midpoint price
        """
        return 0.5 * (self.ask + self.bid)

    def vmpt(self):
        """
        Volume weighted midprice for this quote. It holds midprice if summary size is zero.

        :return: volume weighted midprice
        """
        _e_size = self.ask_size + self.bid_size
        if _e_size == 0.0:
            return self.midprice()

        return (self.bid * self.ask_size + self.ask * self.bid_size) / _e_size

    def __repr__(self):
        return "[%s]    %.5f (%.1f)  |  %.5f (%.1f)" % (self.time.strftime('%H:%M:%S.%f'),
                                                        self.bid, self.bid_size, self.ask, self.ask_size)
