from datetime import datetime
from typing import Callable, List

from numpy import sign, isnan, nan

from qube.portfolio.Instrument import Instrument
from qube.portfolio.commissions import TransactionCostsCalculator, ZeroTCC


class Position:

    def __init__(self, instrument: Instrument, tcc: TransactionCostsCalculator = None):
        # position's instrument
        self.instrument = instrument

        # position's symbol
        self.symbol = instrument.symbol

        # when price updated or position changed
        self.last_update_time = None

        # position size
        self.quantity = 0

        # total cumulative position PnL in USD
        self.pnl: float = 0.0

        # realized cumulative PnL in USD
        self.r_pnl: float = 0

        # update price (actually instrument's price) in quoted currency
        self.price: float = None

        # position market value in USD
        self.market_value_usd: float = 0

        # position cost in USD
        self.cost_usd: float = 0

        # current rate used for conversion to USD
        self.usd_conversion_rate: float = 1

        # position long turnover (in shares)
        self.long_volume: float = 0

        # position short turnover (in shares)
        self.short_volume: float = 0

        # if it needs auxilary quotes to converting PnL and cost to USD
        self.aux_instrument: str = None

        # true if this instrument is quoted in USD (EURUSD, US stocks etc)
        self.is_straight = True

        # position's cost in quoted currency
        self._cost_quoted: float = 0

        # position's market value in quoted currency
        self._market_value: float = 0

        # fixed spread when tested on OHLC data
        # not used when it's being tested on quotes data i
        self.half_spread = abs(instrument.spread) / 2.0

        # transaction costs calculator
        self.tcc = tcc if tcc is not None else ZeroTCC()

        # cumulative commissions
        self.commissions = 0

        self.__exec_cb_list: List[Callable] = []

    def attach_execution_callback(self, callback: Callable):
        """
        Attach callback function to this position object. Example:

        ---------------

        def cb_exec(pos: Position, timestamp: datetime, position_change: int, execution_price):
            print('[%s] executed %d units at %f' % (pos.instrument, position_change, execution_price))

        p = Position(....).attach_execution_callback(cb_exec)
        
        ---------------


        :param callback: callback function
        :return: this position object
        """
        if callback and callback not in self.__exec_cb_list:
            self.__exec_cb_list.append(callback)
        return self

    def detach_execution_callback(self, callback: Callable):
        """
        Remove callback function from position object.

        :param callback: callback function
        :return: this position object
        """
        if callback and callback in self.__exec_cb_list:
            self.__exec_cb_list.remove(callback)
        return self

    def update_position_bid_ask(self, timestamp, n_pos, bid_price, ask_price, usd_conv_rate=1.0, exec_price=None,
                                comment='', crossed_market=True, **kwargs) -> float:
        # realized PnL of this fill
        deal_pnl = 0
        quantity = self.quantity

        if quantity != n_pos:
            pos_change = n_pos - quantity
            direction = sign(pos_change)
            prev_direction = sign(quantity)

            # execution price with:
            # we always buy at ask and sell at bid
            exec_price = exec_price if exec_price else ask_price if direction > 0 else bid_price

            # how many shares are closed/open
            qty_closing = min(abs(self.quantity), abs(pos_change)) * direction if prev_direction != direction else 0
            qty_opening = pos_change if prev_direction == direction else pos_change - qty_closing

            # if we have closed some shares
            new_cost = self._cost_quoted + qty_opening * exec_price
            if self.quantity != 0:
                new_cost += qty_closing * self._cost_quoted / quantity
                deal_pnl = qty_closing * (self._cost_quoted / quantity - exec_price)

            # turnover info
            if direction < 0:
                self.short_volume += abs(pos_change)
            else:
                self.long_volume += abs(pos_change)

            self._cost_quoted = new_cost
            self.quantity = n_pos

            # convert current position's cost to USD
            self.cost_usd = self._cost_quoted / usd_conv_rate

            # convert PnL to USD
            self.r_pnl += deal_pnl / usd_conv_rate

            # calculate transaction costs
            comms = self.tcc.get_execution_cost(self.instrument, exec_price, pos_change, crossed_market, usd_conv_rate)
            self.commissions += comms

            # call executions callback here. Probably we need to add try/catch here ?
            self._call_execution_callback(exec_price, pos_change, timestamp, comms, comment)

        # update self.pnl and self.market_value_usd
        if isinstance(self, (ForexPosition, CryptoPosition, CryptoFuturesPosition)):
            # yes for overridden classes ForexPosition / CryptoPosition need call super's (Position) method
            super(self.__class__, self).update_pnl_bid_ask(timestamp, bid_price, ask_price, usd_conv_rate, **kwargs)
        else:
            self.update_pnl_bid_ask(timestamp, bid_price, ask_price, usd_conv_rate, **kwargs)

        return deal_pnl

    def _call_execution_callback(self, exec_price, pos_change, timestamp, comms, comment=''):
        if self.__exec_cb_list:
            [__f_cb(self, timestamp, pos_change, exec_price, comms, comment) for __f_cb in self.__exec_cb_list]

    def update_pnl_bid_ask(self, timestamp, bid, ask, usd_conv_rate=1.0, **kwargs) -> float:
        # closing long at bid and short at ask
        closing_price = bid if sign(self.quantity) > 0 else ask
        self._market_value = 0
        if not isnan(closing_price):
            self._market_value = self.quantity * closing_price
            self.pnl = (self._market_value - self._cost_quoted) / usd_conv_rate + self.r_pnl
        # price can be NaN - it indicates that price was not known at this moment
        self.price = (bid + ask) / 2.0
        self.last_update_time = timestamp
        # calculate mkt value in USD
        self.market_value_usd = self._market_value / usd_conv_rate
        self.usd_conversion_rate = usd_conv_rate
        return self.pnl

    def update_position(self, timestamp, n_pos, exec_price, **kwargs):
        return self.update_position_bid_ask(timestamp, n_pos, exec_price - self.half_spread,
                                            exec_price + self.half_spread, **kwargs)

    def update_pnl(self, timestamp, price, **kwargs):
        return self.update_pnl_bid_ask(timestamp, price - self.half_spread, price + self.half_spread, **kwargs)

    def __str__(self):
        if isinstance(self.last_update_time, datetime):
            _t_str = self.last_update_time.strftime('%Y-%m-%d %H:%M:%S.%f')
        else:
            _t_str = str(self.last_update_time)
        _p_str = "???" if self.price is None else ('%.5f' % self.price)
        _c_str = "$%.2f" % self.cost_usd
        return '[%s]  %s   %.0f   %s   %.2f  $%.2f / %s' % (self.symbol, _t_str, self.quantity, _p_str, self.pnl,
                                                            self._market_value, _c_str)

    @staticmethod
    def is_aux_straight(instrument: str) -> bool:
        """
        True if aux instrument is quoted in USD (EURUSD, ..) and false for USDJPY etc
        :param aux! instrument:
        :return: bool
        """
        return instrument[:3] != 'USD'

    def __getstate__(self):
        """
        Remove callbacks before pickling
        """
        state = self.__dict__.copy()  # copy, because we don't want to delete callbacks from the original Position instance
        state['_Position__exec_cb_list'] = []
        return state


class ForexPosition(Position):
    _CURRENCIES = ['USD', 'EUR', 'GBP', 'CHF', 'JPY',
                   'AUD', 'NZD', 'CAD', 'SGD', 'HKD',
                   'PLN', 'DKK', 'NOK', 'SEK', 'CNH',
                   'MXN', 'ZAR', 'TRY', 'RUB', 'UAH']

    def __init__(self, instrument: Instrument, tcc: TransactionCostsCalculator = None):
        super().__init__(instrument, tcc)

        instr = instrument.symbol.upper()
        self.base = None
        self.quote = 'USD'
        for c in ForexPosition._CURRENCIES:
            if instr.startswith(c):
                self.base = c
                break

        # check if quote for instrument is different from USD
        last_part = instr[-3:]
        if last_part in ForexPosition._CURRENCIES:
            self.quote = last_part

        # quoted in USD - we have 'straight' instrument : EURUSD ...
        self.is_straight = self.quote == 'USD'

        # seems to be forex pair
        if self.base is not None:
            # basic is USD - we have 'reversed' instrument like USDJPY etc
            self.is_reversed = self.base == 'USD'

            # cross pair if USD not found: EURCHF
            self.is_cross = (not self.is_reversed) & (not self.is_straight)
            if self.is_cross:
                self.aux_instrument = 'USD%s' % self.quote
        else:
            # Not currency
            self.base = instrument.symbol
            self.is_reversed = False
            self.is_cross = False
            if not self.is_straight:
                self.aux_instrument = 'USD%s' % self.quote

        # special case for some currencies (for instance AUDEUR and there is no USDEUR rates)
        if self.aux_instrument is not None:
            if self.quote in ['EUR', 'GBP', 'NZD', 'AUD']:
                self.aux_instrument = '%sUSD' % self.quote
            elif not self.is_straight:
                self.aux_instrument = 'USD%s' % self.quote

    def __usd_conversion_rate(self, c_bid, c_ask, a_bid, a_ask, a_is_straight) -> float:
        """
        Calculates current USD conversion rate for quoted currency (returns 1 if quoted in USD)
        """
        usd_div = 1
        if self.aux_instrument is not None:
            # here we use divider for transform PnL to USD: eurchf -> eurchf / usdchf -> eurusd
            usd_div = a_ask if sign(self.quantity) > 0 else a_bid
            # if instrument is quoted in USD (straight) we need to take reciprocal
            #  eurgbp -> eurgbp / (1/gbpusd) -> eurusd
            if a_is_straight:
                usd_div = 1.0 / usd_div

        # for USDXXX we need convert PnL to USD
        if self.is_reversed:
            usd_div = (c_ask if sign(self.quantity) > 0 else c_bid)

        return usd_div

    def update_position_bid_ask(self, timestamp, n_pos, c_bid, c_ask, a_is_straight=False, a_bid=nan, a_ask=nan,
                                **kwargs) -> float:
        """
        For FOREX cross-pairs we want to do all PnL-related calculations expressed in USD.
        For that we introduce auxilary instrument. For example aux instrument for EURGBP is GBPUSD.

        :param timestamp: time of update 
        :param n_pos: new position
        :param c_bid: quoted bid of instrument
        :param c_ask: quoted ask of instrument
        :param a_is_straight: true if aux instrument is quoted in USD (EURUSD, ..) and false for USDJPY etc
        :param a_bid: quoted bid of auxilary instrument
        :param a_ask: quoted ask of auxilary instrument
        :return: PnL for position in USD
        """
        return super().update_position_bid_ask(timestamp, n_pos, c_bid, c_ask, usd_conv_rate=self.__usd_conversion_rate(
            c_bid, c_ask, a_bid, a_ask, a_is_straight
        ), **kwargs)

    def update_pnl_bid_ask(self, timestamp, c_bid, c_ask, a_is_straight=False, a_bid=nan, a_ask=nan, **kwargs) -> float:
        return super().update_pnl_bid_ask(timestamp, c_bid, c_ask, usd_conv_rate=self.__usd_conversion_rate(
            c_bid, c_ask, a_bid, a_ask, a_is_straight
        ), **kwargs)


class CryptoPosition(Position):
    _CRYPTO_CURRENCIES = ['BTC', 'ETH', 'ETC', 'XBT', 'XRP', 'EOS', 'LTC', 'BCH', 'BSV', 'DOT']
    _CURRENCIES = ['EUR']  # todo: how about USDT etc ????

    def __init__(self, instrument, tcc: TransactionCostsCalculator = None):
        super().__init__(instrument, tcc)

        instr = instrument.symbol.upper()
        self.base = None
        self.quote = 'USD'
        for c in CryptoPosition._CRYPTO_CURRENCIES:
            if instr.startswith(c):
                self.base = c
                break

        # check if quote for instrument is different from USD
        last_part = instr[-3:]
        if last_part in CryptoPosition._CRYPTO_CURRENCIES + CryptoPosition._CURRENCIES:
            self.quote = last_part

        self.is_straight = self.quote == 'USD'

        # seems to be cryptocurrency pair
        if self.base is not None:
            # cross pair if USD not found: ETHBTC
            self.is_cross = (not self.is_straight)
            if self.is_cross:
                self.aux_instrument = '%sUSD' % self.quote if self.quote not in CryptoPosition._CURRENCIES else None
        else:
            # Not currency
            self.base = instr
            self.is_cross = False
            if not self.is_straight:
                self.aux_instrument = '%sUSD' % self.quote

    def __usd_conversion_rate(self, a_bid, a_ask) -> float:
        """
        Calculates current USD conversion rate for quoted currency (returns 1 if quoted in USD)
        """
        usd_div = 1
        if self.aux_instrument is not None:
            # here we use divider for transform PnL to USD : ETHBTC -> ETHBTC / (1/BTCUSD) -> ETHUSD
            usd_div = a_ask if sign(self.quantity) > 0 else a_bid
            usd_div = 1.0 / usd_div
        return usd_div

    def update_position_bid_ask(self, timestamp, n_pos, c_bid, c_ask, a_bid=nan, a_ask=nan, **kwargs) -> float:
        """
        For cryptocurrency cross-pairs we want to do all PnL-related calculations expressed in USD.
        For that we introduce auxiliary instrument. For example aux instrument for BTCETH is ETHUSD.

        :param timestamp: time of update
        :param n_pos: new position
        :param c_bid: quoted bid of instrument
        :param c_ask: quoted ask of instrument
        :param a_bid: quoted bid of auxiliary instrument
        :param a_ask: quoted ask of auxiliary instrument
        :return: PnL for position in USD
        """
        return super().update_position_bid_ask(timestamp, n_pos, c_bid, c_ask,
                                               usd_conv_rate=self.__usd_conversion_rate(a_bid, a_ask), **kwargs)

    def update_pnl_bid_ask(self, timestamp, c_bid, c_ask, a_bid=nan, a_ask=nan, **kwargs) -> float:
        return super().update_pnl_bid_ask(timestamp, c_bid, c_ask,
                                          usd_conv_rate=self.__usd_conversion_rate(a_bid, a_ask), **kwargs)


class CryptoFuturesPosition(CryptoPosition):
    def __init__(self, instrument, tcc: TransactionCostsCalculator = None):
        super().__init__(instrument, tcc)

    # TODO later once we need futures not rated in USD
    def __usd_conversion_rate(self, a_bid, a_ask) -> float:
        return 1.0

    def update_position_bid_ask(self, timestamp, n_pos, c_bid, c_ask, a_bid=nan, a_ask=nan, exec_price=None,
                                comment='', crossed_market=True, **kwargs) -> float:
        # realized PnL of this fill
        deal_pnl = 0
        quantity = self.quantity

        if quantity != n_pos:
            pos_change = n_pos - quantity
            direction = sign(pos_change)
            prev_direction = sign(quantity)

            # execution price with:
            # we always buy at ask and sell at bid if exact price is not specifed
            # exec_price = c_ask if direction > 0 else c_bid
            exec_price = exec_price if exec_price else c_ask if direction > 0 else c_bid

            # how many shares are closed/open
            qty_closing = min(abs(self.quantity), abs(pos_change)) * direction if prev_direction != direction else 0
            qty_opening = pos_change if prev_direction == direction else pos_change - qty_closing

            # if we have closed some shares
            new_cost = self._cost_quoted + qty_opening * exec_price

            usd_conv_rate = 1.0
            if self.quantity != 0:
                new_cost += float(qty_closing) * self._cost_quoted / quantity
                deal_pnl = (1 / exec_price - (1 / (self._cost_quoted / self.quantity))) * qty_closing * exec_price

            # turnover info
            if direction < 0:
                self.short_volume += abs(pos_change)
            else:
                self.long_volume += abs(pos_change)

            self._cost_quoted = new_cost
            self.quantity = n_pos

            # convert current position's cost to USD
            self.cost_usd = self._cost_quoted / usd_conv_rate

            # convert PnL to USD
            self.r_pnl += deal_pnl

            # calculate transaction costs
            comms = self.tcc.get_execution_cost(self.instrument, exec_price, pos_change, crossed_market, usd_conv_rate)
            self.commissions += comms

            # call executions callback here. Probably we need to add try/catch here ?
            self._call_execution_callback(exec_price, pos_change, timestamp, comms, comment)

        self.__update_pnl_bid_ask(timestamp, c_bid, c_ask, self.__usd_conversion_rate(a_bid, a_ask), **kwargs)

        return deal_pnl

    def __update_pnl_bid_ask(self, timestamp, bid, ask, usd_conv_rate=1.0, **kwargs) -> float:
        # closing long at bid and short at ask
        closing_price = bid if sign(self.quantity) > 0 else ask
        self._market_value = 0
        if not isnan(closing_price):
            self._market_value = self.quantity * closing_price
            pnl_futures_correction_multiplier = self.quantity / self._cost_quoted if self._cost_quoted != 0.0 else 0.0
            self.pnl = pnl_futures_correction_multiplier * (
                    self._market_value - self._cost_quoted) / usd_conv_rate + self.r_pnl
        # price can be NaN - it indicates that price was not known at this moment
        self.price = (bid + ask) / 2.0
        self.last_update_time = timestamp
        # calculate mkt value in USD
        self.market_value_usd = self._market_value / usd_conv_rate
        self.usd_conversion_rate = usd_conv_rate
        return self.pnl

    def update_pnl_bid_ask(self, timestamp, c_bid, c_ask, a_bid=nan, a_ask=nan, **kwargs) -> float:
        return self.__update_pnl_bid_ask(timestamp, c_bid, c_ask,
                                         usd_conv_rate=self.__usd_conversion_rate(a_bid, a_ask), **kwargs)
