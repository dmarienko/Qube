from typing import Dict

import numpy as np

from qube.simulator.tracking.trackers import TakeStopTracker


class TurtleTracker(TakeStopTracker):
    """
    Simple implemenation of turtles money management system
    """

    def __init__(self,
                 account_size, dollar_per_point, max_units=4, risk_capital_pct=0.01, reinvest_pnl_pct=0,
                 contract_size=100, max_allowed_contracts=200,
                 atr_timeframe='1d', pull_stops_on_incr=False, after_lose_only=False,
                 debug=False, take_by_limit_orders=False):
        """
        Turtles strategy position tracking
        ----------------------------------

        >>> from sklearn.base import TransformerMixin
        >>> from sklearn.pipeline import make_pipeline
        >>> b_e1h = MarketDataComposer(make_pipeline(RollingRange('1h', 10), RangeBreakoutDetector()),
        >>>                                          SingleInstrumentPicker(), None).fit(data, None).predict(data)
        >>> b_x1h = MarketDataComposer(make_pipeline(RollingRange('1h', 6), RangeBreakoutDetector()),
        >>>                                          SingleInstrumentPicker(), None).fit(data, None).predict(data)
        >>> s1h = shift_signals(srows(1 * b_e1h, 2 * b_x1h), '4M59Sec')
        >>> p1h = z_backtest(s1h, data, 'crypto_futures', spread=0.5, execution_logger=True,
        >>>                  trackers=TurtleTracker(3000, None, max_units=4, risk_capital_pct=0.05,
        >>>                                         atr_timeframe='1h',
        >>>                                         max_allowed_contracts=1000, pull_stops_on_incr=True, debug=False))

        It processes signals as following:
          - signals in [-1, +1] designated for open positions
          - signals in [-2, +2] designated for positions closing

        :param accoun_size: starting amount in USD
        :param dollar_per_point: price of 1 point (for example 12.5 for ES mini) if none crypto sizing would be used
        :param max_untis: maximal number of position inreasings
        :param risk_capital_pct: percent of capital in risk (0.01 is 1%)
        :param reinvest_pnl_pct: percent of reinvestment pnl to trading (default is 0)
        :param contract_size: contract size in USD
        :param max_allowed_contracts: maximal allowed contracts to trade
        :param atr_timeframe: timeframe of ATR calculations
        :param pull_stops_on_incr: if true it pull up stop on position's increasing
        :param after_lose_only: if true it's System1 otherwise System2
        :param debug: if true it prints debug messages
        """
        super().__init__(debug, take_by_limit_orders=take_by_limit_orders)
        self.account_size = account_size
        self.dollar_per_point = dollar_per_point
        self.atr_timeframe = atr_timeframe
        self.max_units = max_units
        self.trading_after_lose_only = after_lose_only
        self.pull_stops = pull_stops_on_incr
        self.risk_capital_pct = risk_capital_pct
        self.max_allowed_contracts = max_allowed_contracts
        self.reinvest_pnl_pct = reinvest_pnl_pct
        self.contract_size = contract_size

        if dollar_per_point is None:
            self.calculate_trade_size = self._calculate_trade_size_crypto

    def initialize(self):
        self.days = self.get_ohlc_series(self.atr_timeframe)
        self.N = None
        self.__init_days_counted = 1
        self.__TR_init_sum = 0
        self._n_entries = 0
        self._last_entry_price = np.nan

    def _get_size_at_risk(self):
        return (self.account_size + max(self.reinvest_pnl_pct * self._position.pnl, 0)) * self.risk_capital_pct

    def _calculate_trade_size_crypto(self, direction, vlt, price):
        price2 = price + direction * vlt
        return np.clip(round(self._get_size_at_risk() / ((price2 / price - 1) * self.contract_size)),
                       -self.max_allowed_contracts, self.max_allowed_contracts)

    def _calculate_trade_size_on_dollar_cost(self, direction, vlt, price):
        return min(self.max_units, round(self._get_size_at_risk() / (vlt * self.dollar_per_point))) * direction

    def calculate_trade_size(self, direction, vlt, price):
        return self._calculate_trade_size_on_dollar_cost(direction, vlt, price)

    def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
        daily = self.days
        today, yest = daily[1], daily[2]

        if yest is None or today is None:
            return

        if daily.is_new_bar:
            TR = max(today.high - today.low, today.high - yest.close, yest.close - today.low)
            if self.N is None:
                if self.__init_days_counted <= 21:
                    self.__TR_init_sum += TR
                    self.__init_days_counted += 1
                else:
                    self.N = self.__TR_init_sum / 19
            else:
                self.N = (19 * self.N + TR) / 20

        # increasing position size if possible
        pos = self._position.quantity
        if pos != 0 and self._n_entries < self.max_units:
            n_2 = self.N / 2
            if (pos > 0 and ask > self._last_entry_price + n_2) or (pos < 0 and bid < self._last_entry_price - n_2):
                self._last_entry_price = ask if pos > 0 else bid
                t_size = self.calculate_trade_size(np.sign(pos), self.N, self._last_entry_price)
                self._n_entries += 1

                # new position
                new_pos = pos + t_size

                # increase inventory
                self.trade(quote_time, new_pos, f'increased position to {new_pos} at {self._last_entry_price}')

                # average position price
                avg_price = self._position.cost_usd / self._position.quantity

                # pull stops
                if self.pull_stops:
                    self.stop_at(quote_time, avg_price - self.N * 2 * np.sign(pos))

                self.debug(
                    f"\t[{quote_time}] -> [#{self._n_entries}] {self._instrument} <{avg_price:.2f}> increasing to "
                    f"{new_pos} @ {self._last_entry_price} x {self.stop:.2f}")

        # call stop/take tracker to process sl/tp if need
        super().on_quote(quote_time, bid, ask, bid_size, ask_size, **kwargs)

    def on_signal(self, signal_time, signal, quote_time, bid, ask, bid_size, ask_size):
        if self.N is None:
            return None

        s_type, s_direction, t_size = abs(signal), np.sign(signal), None
        position = self._position.quantity

        # when we want to enter position
        if position == 0 and s_type == 1:
            if not self.trading_after_lose_only or self.last_triggered_event != 'stop':
                self._last_entry_price = ask if s_direction > 0 else bid
                t_size = self.calculate_trade_size(s_direction, self.N, self._last_entry_price)
                self.stop_at(signal_time, self._last_entry_price - self.N * 2 * s_direction)
                self._n_entries = 1
                self.debug(
                    f'\t[{signal_time}] -> [#{self._n_entries}] {self._instrument} {t_size} @ '
                    f'{self._last_entry_price:.2f} x {self.stop:.2f}')
            # clear previous state
            self.last_triggered_event = None

        # when we got to exit signal
        if (position > 0 and signal == -2) or (position < 0 and signal == +2):
            self.last_triggered_event = 'take'
            self._n_entries = 0
            t_size = 0
            self.debug(f'[{signal_time}] -> Close in profit {self._instrument} @ {bid if position > 0 else ask}')
            self.times_to_take.append(signal_time - self._service.last_trade_time)
            self.n_takes += 1

        return t_size

    def statistics(self) -> Dict:
        r = dict()
        return r
