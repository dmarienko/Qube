from qube.series.BarSeries import BarSeries
from qube.series.Indicators import Ema, KAMA
from qube.simulator.tracking.trackers import TriggerOrder, TriggeredOrdersTracker
from qube.utils.ui_utils import red, green, yellow, blue
from qube.utils.utils import mstruct


class MessageLogger:
    def __init__(self, debug=False):
        self.log = self._log if debug else lambda *args, **kwargs: None
        self.info = self._info if debug else lambda *args, **kwargs: None

    def _log(self, time, event, mesg):
        print(f"\n[{yellow(time)}] -({red(event.upper())})-> {green(mesg)}")

    def _info(self, mesg, *args):
        print(f" {red('->')} {green(mesg)} {' '.join([blue(repr(x)) for x in args])}")


class TestTrader(TriggeredOrdersTracker):
    """
    Simple strategy for testing purposes (2 crossing MA)
    """

    def __init__(self,
                 capital: float,
                 max_cap_in_risk: float,
                 risk_reward_ratio: float,
                 timeframe: str,
                 fast_period: int,
                 slow_period: int,
                 max_risk_skip_pct: float = 3,
                 debug=False):
        super().__init__(
            debug=debug, accurate_stop_execution=True, take_by_limit_orders=True, open_by_limit_orders=True
        )
        self.logger = MessageLogger(debug)

        if fast_period >= slow_period:
            raise ValueError("Fast must be less than slow !")

        self.capital = capital
        self.max_cap_in_risk = max_cap_in_risk
        self.risk_reward_ratio = risk_reward_ratio
        self.max_risk_skip_pct = max_risk_skip_pct

        # indicators
        self.ohlc = BarSeries(timeframe, max_series_length=fast_period + slow_period)
        self._fast = Ema(fast_period)
        self._slow = KAMA(slow_period)
        self.ohlc.attach(self._fast)
        self.ohlc.attach(self._slow)

        self._order = None
        self._id_str = f"TestTrader({capital},{max_cap_in_risk},{risk_reward_ratio},{timeframe},{fast_period},{slow_period})"

    def statistics(self):
        return super().statistics()

    def on_take(self, timestamp, price, is_part_take: bool, closed_size, user_data=None):
        self._order = None

    def on_stop(self, timestamp, price, user_data=None):
        self._order = None

    def on_trigger_fired(self, timestamp, order: TriggerOrder):
        self.logger.log(timestamp, 'FIRED', f'{order} => {order.user_data}')

    def _can_create_new_order(self, direction, entry, stop):
        # check max risk
        risk_pct = 100 * abs(entry - stop) / entry
        if risk_pct > self.max_risk_skip_pct:
            return False

        if self._order is not None:
            if not self._order.fired:
                if (direction > 0 and entry < self._order.price) or (direction < 0 and entry > self._order.price):
                    self.logger.info(f'CNCL {self._order} => {self._order.user_data}')
                    self.cancel(self._order)
                    self._order = None
                    return True
                else:
                    # no improved price so wait until order will be triggered or canceled 
                    return False
            else:
                # order was triggered so no new orders
                return False
        return True

    def _trading_logic(self, instrument: str, time):
        if self._position.quantity != 0:
            return

        if len(self._fast) > 2 and len(self._slow) > 2:
            b1 = self.ohlc[1]
            f1, f2, s1, s2 = self._fast[1], self._fast[2], self._slow[1], self._slow[2]

            # - long 
            if f2 > s2 and f1 < s1:
                entry = b1.high
                stp = min(s2, b1.low)
                size = self.get_position_size(+1, entry, stp)
                take = self.get_take_target(+1, entry, stp)
                risk_pct = 100 * abs(entry - stp) / entry
                if self._can_create_new_order(+1, entry, stp):
                    idstr = f"Long {size} at {entry} (*{take}, x{stp}) | risk: {risk_pct:.2f}"
                    self.logger.log(time, 'BUY', idstr)
                    self._order = self.stop_order(
                        entry, +size, stp, take, comment=idstr,
                        user_data=mstruct(id=idstr, size=+size, entry=entry, take=take, stop=stp)
                    )
                return

            # - short 
            if f2 < s2 and f1 > s1:
                entry = b1.low
                stp = max(s2, b1.high)
                size = self.get_position_size(-1, entry, stp)
                take = self.get_take_target(-1, entry, stp)
                risk_pct = 100 * abs(entry - stp) / entry
                if self._can_create_new_order(-1, entry, stp):
                    idstr = f"Short {size} at {entry} (*{take}, x{stp}) | risk: {risk_pct:.2f}"
                    self.logger.log(time, 'SHORT', idstr)
                    self._order = self.stop_order(
                        entry, size, stp, take, comment=idstr,
                        user_data=mstruct(id=idstr, size=size, entry=entry, take=take, stop=stp)
                    )
                return

    def get_take_target(self, direction, entry, stop):
        # Simple risk reward ratio take target algo
        return round(entry + direction * abs(entry - stop) * self.risk_reward_ratio, 2)

    def get_position_size(self, direction, entry, stop):
        cap = self.capital + max(self._position.pnl, 0)
        return direction * round((cap * self.max_cap_in_risk / 100) / abs(stop / entry - 1))

    def update_market_data(self, instrument: str, time, bid, ask, bsize, asize, is_service_quote, **kwargs):
        if self.ohlc.update_by_data(time, bid, ask, bsize, asize):
            self._trading_logic(instrument, time)

        # - cancel order if price went under stop
        if self._order is not None and not self._order.fired:
            if (self._order.quantity > 0 and bid <= self._order.stop) or (
                    self._order.quantity < 0 and ask >= self._order.stop):
                self.logger.log(time, 'CNCL', f'{self._order} => {self._order.user_data}')
                self.cancel(self._order)
                self._order = None

        super().update_market_data(instrument, time, bid, ask, bsize, asize, is_service_quote, **kwargs)

    def __repr__(self):
        return self._id_str
