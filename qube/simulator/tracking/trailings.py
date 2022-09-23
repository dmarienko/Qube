from typing import Dict

import numpy as np
import pandas as pd

from qube.series.Indicators import ATR, MovingMinMax
from qube.simulator.tracking.trackers import TakeStopTracker


class Pyramiding(TakeStopTracker):
    """
    Pyramiding tracker.

        1. Open position = size on signal, stop at entry - stop_mx * ATR

        2. If price hits entry + next_mx * ATR and # entry < max_positions
            it adds to position: size * pyramiding_factor
            stop = avg_position_price - stop_mx * ATR
            next_add_level = avg_position_price + next_mx * ATR

        3. Skip any other signals if position is open

    Tracker's parameters
    --------------------
        size: basic position size (in contracts). On first entry tracker will buy or sell this amount of contracts.
        stop_mx: how many ATRs we set stop at (stop = entry_price -/+ stop_mx * ATR) [default 3]
        next_mx: how many ATRs next level is (next = entry_price +/- next_mx * ATR) [default 3]
                 At next level tracker may performs following actions:
                     - increase position if level's number >= pyramiding_start_step and it doesn't exceed maximal number of pyramiding positions (see max_position)
                     - pull up/down stop level to breakeven
                     - close position in profit if
        pyramiding_factor: position decaying factor. On every next step we will add to position: prev_size * pyramiding_factor.
                          For example if pyramiding_factor = 0.5 and size = 10
                          - at first initial step tracker open 10 contracts
                          - at next step it add 10 * 0.5 = 5 contracts
                          - at third step it adds 5 * 0.5 = 2 contracts
                          - at 4'th step: 2 * 0.5 = 1 contract
                          - at 5'th step 1 * 0.5 = 0 - so it final step and it will close position in profit if flat_on_max_step is set
                          so 0.5 reproduces classical approach: first 100%, 50%, 25%, 12% ...
                          If pyramiding_factor=1 tracker adds same fixed amount (==size) at every step.
        pyramiding_start_step: level number when it is allowed to increase position.
                         Classical way is to just move stop to breakeven at step 2 (no position increasing) and start pyramiding at step 3
                         (so pyramiding_start_step = 3 for this case)
        max_positions: maximal allowed number of pyramiding steps
        flat_on_max_step: if this flag is set tracker will close position when max number of position increasig steps reached
        atr_period: period of ATR indicator [default 22]
        atr_timeframe: timeframe for ATR indicator [default daily bars]
        round_size: minimal [default 1]

    Example
    -------
        Pyramiding(size=10, stop_mx=3, next_mx=3, pyramiding_factor=0.5,
                   max_positions=5, flat_on_max_step=True, pyramiding_start_step=3, round_size=1)

        - Signal generator produces signal to oen long position, current price is $100.00, ATR=5.00

        - Tracker will open 10 contracts long at $100.00 and set up stop at 100 - 3 * 5 = \$85, next level (#2) is 100 + 3 * 5 = $115

        - If price drops below $85 tracker will close position

        - If price goes above $115 tracker will do following actions:
            - just pull up stop at breakeven level at entry price = $100.00 because pyramiding should start from level N 3 but it's N 2
            - calculate ATR at this moment, let's say it's 7
            - starts waiting for next level (N3) == 115 + 3 * 7 = $136.00

        - When (if) price touches level N3 ($136) tracker will:
            - add to first 10 contracts another 10 * 0.5 = 5 contracts and position now is 15
            - calculate position price, it will be (10 100 + 5 136)/15 = $112
            - calculate ATR, let's say it's 3
            - set stop level at 112 - 3 * 3 = $103
            - calculate next level N4 == 136 + 3 * 3 = $145 (here it uses entry price 136 not average position price !!!)

        - When price reaches level N4 at let's (price is 145.00):
            - add to existing 15 contracts another 5 * 0.5 = 2.5 -> 2 contracts (we round it on round_size=1) and position now is 17
            - position size is (10 100 + 5 136 + 2 * 145) / 17 = 115.88
            - calculate ATR, let's say it's 4
            - move stop to 115.88 - 3 * 4 = 103.88
            - next level (#5) is 145 + 3*4 = $157

        - When price touches level #5 at $157 tracker will close position at take because max_positions is set to 5 and flat_on_max_step=True

    """

    def __init__(self, size, stop_mx=3, next_mx=3, pyramiding_factor=0.5, max_positions=3,
                 flat_on_max_step=False, pyramiding_start_step=3,
                 atr_period=22, atr_timeframe='1d', atr_smoother='sma',
                 round_size=1, debug=False, take_by_limit_orders=False):
        super().__init__(debug, take_by_limit_orders=take_by_limit_orders)
        self.size = size
        self.stop_mx = stop_mx
        self.next_mx = next_mx
        self.pyramiding_factor = pyramiding_factor
        self.pyramiding_start_step = max(abs(pyramiding_start_step), 2)
        self.max_positions = max_positions
        self.flat_on_max_step = flat_on_max_step
        self.atr_period = atr_period
        self.atr_timeframe = atr_timeframe
        self.atr_smoother = atr_smoother
        self.log10_round_size = int(np.log10(max(round_size, 1)))

    def initialize(self):
        self.n_entry = 0
        self.next_level = np.nan

        # indicators stuff
        self.ohlc = self.get_ohlc_series(self.atr_timeframe)
        self.atr = ATR(self.atr_period, self.atr_smoother)
        self.ohlc.attach(self.atr)

    def get_position_size_for_step(self, n):
        n = n - self.pyramiding_start_step + 2
        return np.round(self.size * (self.pyramiding_factor) ** n, self.log10_round_size)

    def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
        tr = self.atr[1]
        qty = self._position.quantity

        if qty != 0 and tr is not None and np.isfinite(tr):
            # --- long position processing
            if qty > 0:
                px = ask
                D = +1

            # --- long position processing
            if qty < 0:
                px = bid
                D = -1

            # price hits target's level
            if (px - self.next_level) * D >= 0:
                # we've aleady reached this level so next will be recomputed
                mesg = f"[{quote_time}] {self._instrument} {px:.3f} touched {self.next_level:.3f} "
                self.next_level = np.nan

                # if we can increase
                if self.n_entry + 1 <= self.max_positions:
                    inc_size = self.get_position_size_for_step(self.n_entry + 1)
                    if inc_size > 0:
                        self.n_entry += 1
                        self.next_level = px + D * self.next_mx * tr

                        if self.n_entry >= self.pyramiding_start_step:
                            mesg += f'step ({self.n_entry}) -> {D * inc_size:+.0f} at ${px:.3f} next: {self.next_level:.3f}'

                            # increase position
                            self.trade(quote_time, qty + D * inc_size, mesg)

                            # average position price
                            avg_price = self._position.cost_usd / self._position.quantity

                            # set new stop
                            n_stop = avg_price - D * self.stop_mx * tr
                        else:
                            # set stop to breakeven only (not increase position !)
                            mesg += f'step ({self.n_entry}) at ${px:.3f} move stop to breakeven next: {self.next_level:.3f}'
                            # average position price
                            avg_price = self._position.cost_usd / self._position.quantity
                            n_stop = avg_price

                        mesg += f", stop: {n_stop:.3f}, avg_price: {avg_price:.3f}"
                        self.stop_at(quote_time, n_stop)
                    else:
                        if self.flat_on_max_step:
                            mesg += "closing position because max possible step is reached and flat_on_max_step"
                            self.trade(quote_time, 0, "Take profit")
                        else:
                            mesg += "position increasing size is zero: skip this step"
                else:
                    # increase position
                    if self.flat_on_max_step:
                        mesg += "closing position because max step is and flat_on_max_step"
                        self.trade(quote_time, 0, "Take profit")
                    else:
                        mesg += f'skip increasing atep: max number of entries ({self.max_positions}) '

                self.debug(mesg)

        super().on_quote(quote_time, bid, ask, bid_size, ask_size, **kwargs)

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        tr = self.atr[1]

        # we skip all signals if position is not flat or indicators not ready
        if self._position.quantity != 0 or tr is None or not np.isfinite(tr):
            return None

        pos = None
        if signal_qty > 0:
            # open initial long
            pos = self.size
            self.n_entry = 1
            self.stop_at(signal_time, ask - self.stop_mx * tr)
            self.next_level = ask + self.next_mx * tr
            self.debug(
                f'[{quote_time}] {self._instrument} step ({self.n_entry}) -> {pos} at ${ask:.3f} stop: {ask - self.stop_mx * tr:.3f}, next: {self.next_level:.3f}'
            )

        elif signal_qty < 0:
            # open initial long
            pos = -self.size
            self.stop_at(signal_time, bid + self.stop_mx * tr)
            self.next_level = bid - self.next_mx * tr
            self.debug(
                f'[{quote_time}] {self._instrument} step ({self.n_entry}) -> {pos} at ${bid:.3f} stop: {bid + self.stop_mx * tr:.3f}, next: {self.next_level:.3f}'
            )
            self.n_entry = 1

        return pos


class RADChandelier(TakeStopTracker):
    """
    RAD chandelier position tracker (trailing stop based on ATR no take target)

    https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/chandelier-exit/
    """

    def __init__(self, size, timeframe, period, stop_risk_mx, atr_smoother='sma', debug=False, take_by_limit_orders=False):
        super().__init__(debug, take_by_limit_orders=take_by_limit_orders)
        self.timeframe = timeframe
        self.period = period
        self.position_size = size
        self.stop_risk_mx = abs(stop_risk_mx)
        self.atr_smoother = atr_smoother

    def initialize(self):
        self.atr = ATR(self.period, self.atr_smoother)
        self.mm = MovingMinMax(self.period)
        self.ohlc = self.get_ohlc_series(self.timeframe)
        self.ohlc.attach(self.atr)
        self.ohlc.attach(self.mm)

        # current stop level
        self.level = None
        self.side = 0  # +1: up trend, -1: down trend
        self._dbg_values = {}

    def statistics(self):
        return super().statistics()

    def get_stops(self):
        return self._stops(1)

    def _stops(self, n):
        av, m = self.atr[n], self.mm[n]
        if av is None or m is None:
            return None, None
        ll, hh = m
        if not np.isfinite(av) or not np.isfinite(ll) or not np.isfinite(hh):
            return None, None
        l_stop = hh - self.stop_risk_mx * av
        s_stop = ll + self.stop_risk_mx * av
        return s_stop, l_stop

    def update_stop_level(self) -> bool:
        if not self.ohlc.is_new_bar:
            return False

        # new bar just started
        s2, l2 = self._stops(2)
        s1, l1 = self._stops(1)
        if s2 is None:
            return False

        c1 = self.ohlc[1].close
        c2 = self.ohlc[2].close

        if c2 > l2 and c1 < l1:
            self.side = -1
            self.level = s1

        if c2 < s2 and c1 > s1:
            self.side = +1
            self.level = l1

        if self.side > 0:
            self.level = max(self.level, l1)

        if self.side < 0:
            self.level = min(self.level, s1)

    def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
        # refresh current stop level
        self.update_stop_level()

        if self.side == 0 or self.level is None:
            return None

        qty = self._position.quantity

        # debug
        # self._dbg_values[self.ohlc[0].time] = {'Side': self.side, 'Level': self.level}

        if qty != 0:
            if qty > 0 and self.level > self.stop:
                self.stop_at(quote_time, self.level)
                self.debug(f'[{quote_time}] {self._instrument} pull up stop to {self.level}')

            if qty < 0 and self.level < self.stop:
                self.stop_at(quote_time, self.level)
                self.debug(f'[{quote_time}] {self._instrument} pull down stop to {self.level}')

        super().on_quote(quote_time, bid, ask, bid_size, ask_size, **kwargs)

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        qty = self._position.quantity

        if qty != 0:
            return None

        if self.side == 0 or self.level is None:
            self.debug(
                f'[{quote_time}] {self._instrument} skip entry indicators are not ready: {self.level} / {self.side}')
            return None

        if signal_qty > 0:
            if self.side > 0 and ask > self.level:
                self.stop_at(signal_time, self.level)
                self.debug(f'[{quote_time}] {self._instrument} entry long at ${ask} stop to {self.level}')
            else:
                self.debug(f'[{quote_time}] {self._instrument} skip long : stop {self.level} is above entry {ask}')
                signal_qty = np.nan

        elif signal_qty < 0:
            if self.side < 0 and bid < self.level:
                self.stop_at(signal_time, self.level)
                self.debug(f'[{quote_time}] {self._instrument} entry short at ${bid} stop to {self.level}')
            else:
                self.debug(f'[{quote_time}] {self._instrument} skip short : stop {self.level} is below entry {bid}')
                signal_qty = np.nan

        # call super method
        return signal_qty * self.position_size