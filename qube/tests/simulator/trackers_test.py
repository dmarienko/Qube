import unittest

import numpy as np
import pandas as pd
from qube.simulator.multisim import simulation

from qube.utils.utils import mstruct

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)

from qube.simulator.backtester import backtest
from qube.simulator.tracking.trackers import (RADTrailingStopTracker, TakeStopTracker, DispatchTracker, PipelineTracker,
                                              TimeExpirationTracker, TriggeredOrdersTracker,
                                              TriggerOrder, MultiTakeStopTracker, SignalBarTracker)
from qube.simulator.core import Tracker
from qube.tests.utils_for_tests import _read_timeseries_data


def _signals(sdata):
    s = pd.DataFrame.from_dict(sdata, orient='index')
    s.index = pd.DatetimeIndex(s.index)
    return s


class _Test_NoRiskManagementTracker(Tracker):
    def __init__(self, size):
        self.size = size

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        return signal_qty * self.size


class _Test_StopTakeTracker(TakeStopTracker):

    def __init__(self, size, stop_points, take_points, tick_size):
        super().__init__(True)
        self.size = size
        self.tick_size = tick_size
        self.stop_points = stop_points
        self.take_points = take_points

    def on_take(self, timestamp, price, user_data=None):
        print(f" ----> TAKE: {user_data}")

    def on_stop(self, timestamp, price, user_data=None):
        print(f" ----> STOP: {user_data}")

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        mp = (bid + ask) / 2
        if signal_qty > 0:
            if self.stop_points is not None:
                self.debug(f' >> STOP LONG at {mp - self.stop_points * self.tick_size}')
                self.stop_at(signal_time, mp - self.stop_points * self.tick_size, "Stop user data long")

            if self.take_points is not None:
                self.take_at(signal_time, mp + self.take_points * self.tick_size, "Take user data long")

        elif signal_qty < 0:
            if self.stop_points is not None:
                self.debug(f' >> STOP SHORT at {mp - self.stop_points * self.tick_size}')
                self.stop_at(signal_time, mp + self.stop_points * self.tick_size, "Stop user data short")

            if self.take_points is not None:
                self.take_at(signal_time, mp - self.take_points * self.tick_size, "Take user data short")

        return signal_qty * self.size


class Trackers_test(unittest.TestCase):

    def test_dispatcher(self):
        data = _read_timeseries_data('EURUSD', compressed=False, as_dict=True)

        s = _signals({
            '2020-08-17 04:10:00': {'EURUSD': 'regime:trend'},
            '2020-08-17 04:19:59': {'EURUSD': +1},
            '2020-08-17 14:19:59': {'EURUSD': -1},
            '2020-08-17 14:55:59': {'EURUSD': +1},  # this should be flat !
            '2020-08-17 15:00:00': {'EURUSD': 'regime:mr'},
            '2020-08-17 18:19:59': {'EURUSD': 1},
            '2020-08-17 20:19:59': {'EURUSD': 'empty'},
            '2020-08-17 20:24:59': {'EURUSD': 1},  # this should be passed !
            '2020-08-17 23:19:59': {'EURUSD': 0},
        })

        p = backtest(s, data, 'forex', spread=0, execution_logger=True,
                     trackers=DispatchTracker(
                         {
                             'regime:trend': _Test_StopTakeTracker(10000, 50, None, 1e-5),
                             'regime:mr': PipelineTracker(
                                 TimeExpirationTracker('1h', True),
                                 _Test_NoRiskManagementTracker(777)
                             ),
                             'empty': None
                         }, None, flat_position_on_activate=True, debug=True)
                     )

        print(p.executions)
        print(p.trackers_stat)
        execs_log = list(filter(lambda x: x != '', p.executions.comment.values))
        print(execs_log)

        self.assertListEqual(
            ['stop long at 1.18445',
             'stop short at 1.1879499999999998',
             '<regime:mr> activated and flat position',
             'TimeExpirationTracker:: position 777 is expired'],
            execs_log)

    def test_signal_bar_tracker(self):
        class _Test_SignalBarTracker(SignalBarTracker):
            pass

        data = _read_timeseries_data('RM1', compressed=False, as_dict=True)
        s = _signals({
            '2020-08-17 00:05:01': {'RM1': -1},
            '2020-08-17 00:22:00': {'RM1': 0},
        })

        tracker = _Test_SignalBarTracker('5m', 1e-5, impr='improve')
        p = backtest(s, data, 'forex', spread=0, execution_logger=True, trackers=tracker)

        print(p.executions)
        np.testing.assert_array_almost_equal(
            p.executions.exec_price.values,
            [90.0, 55.0],  # todo: here need to check if it's correct behaviour
            err_msg='Executions are not correct !'
        )

    def test_triggered_orders(self):

        class StopOrdersTestTracker(TriggeredOrdersTracker):
            def __init__(self, tick_size):
                super().__init__(True)
                self.tick_size = tick_size
                self.to = None
                self._fired = 0

            def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
                if signal_qty > 0:
                    entry = ask + 50 * self.tick_size
                    self.to = self.stop_order(
                        entry, 1000,
                        entry - 25 * self.tick_size,
                        {entry + 25 * self.tick_size: 1.0},  # test full form of take config {price: 1.0}
                        comment='Test long order', user_data=mstruct(entry_number=1, test=1)
                    )

                if signal_qty < 0:
                    entry = bid - 50 * self.tick_size
                    self.to = self.stop_order(
                        entry, -1000, entry + 100 * self.tick_size, entry - 250 * self.tick_size,
                        comment='Test short order', user_data=mstruct(entry_number=-1, test=-1)
                    )
                return None

            def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
                super().on_quote(quote_time, bid, ask, bid_size, ask_size, **kwargs)

                if self.to is not None:
                    if self.to.fired:
                        print(quote_time, self.to)
                        self.to = None

            def on_trigger_fired(self, timestamp, order: TriggerOrder):
                print(f"\n\t---(FIRED)--> {timestamp} | {order} => {order.user_data} ")
                self._fired += 1

            def on_take(self, timestamp, price, is_partial, closed_amount, user_data=None):
                print(
                    f"\n\t---(TAKE)--> {timestamp} {price} x {closed_amount} | {user_data} [{'PART' if is_partial else 'FULL'}]")
                print(f"\t---| average take price: {self.average_take_price}")

            def on_stop(self, timestamp, price, user_data=None):
                print(f"\n\t---(STOP)--> {timestamp} {price} | {user_data} ")

            def statistics(self):
                return {'fired': self._fired, **super().statistics()}

        data = _read_timeseries_data('EURUSD', compressed=False, as_dict=True)

        s = _signals({
            '2020-08-17 04:19:59': {'EURUSD': +1},
            '2020-08-17 07:19:59': {'EURUSD': -1},
            '2020-08-17 23:19:59': {'EURUSD': 0},
        })

        track = StopOrdersTestTracker(1e-5)
        p = backtest(s, data, 'forex', spread=0, execution_logger=True, trackers=track)

        print(p.executions)
        print(p.trackers_stat)

        self.assertTrue(p.trackers_stat['EURUSD']['fired'] > 0)
        self.assertTrue(p.trackers_stat['EURUSD']['takes'] == 1)
        self.assertTrue(p.trackers_stat['EURUSD']['stops'] == 1)
        np.testing.assert_array_almost_equal(
            p.executions.exec_price.values,
            [1.185980, 1.186230, 1.184655, 1.185655],
            err_msg='Executions are not correct !'
        )

    def test_take_stop_orders(self):
        data = _read_timeseries_data('RM1', compressed=False, as_dict=True)
        s = _signals({
            '2020-08-17 00:00:01': {'RM1': +1},
            '2020-08-17 00:22:00': {'RM1': 0},
        })

        p = backtest(s, data, 'forex', spread=0, execution_logger=True,
                     trackers=_Test_StopTakeTracker(10000, None, 16, 1))

        print(p.executions)
        print(p.trackers_stat)

    def test_multiple_takes_tracker(self):

        class _Test_MultiTakeTracker(MultiTakeStopTracker):
            def __init__(self, size, stop_points, take_config, tick_size):
                super().__init__(True)
                self.size = size
                self.tick_size = tick_size
                self.stop_points = stop_points
                self.take_config = take_config

            def on_take(self, timestamp, price, is_part_take: bool, closed_amount, user_data=None):
                print(
                    f"\t-[{timestamp}]---> TAKE: {closed_amount} @ {price} {'PART' if is_part_take else 'FULL'} -> {user_data}")
                print(f"\t---| average take price: {self.average_take_price}")

            def on_stop(self, timestamp, price, user_data=None):
                print(f"\t-[{timestamp}]---> STOP: {user_data}")

            def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
                mp = (bid + ask) / 2

                if signal_qty > 0:
                    if self.stop_points is not None:
                        self.debug(f'\n::: LONG at {mp} stop {mp - self.stop_points * self.tick_size}')
                        self.stop_at(signal_time, mp - self.stop_points * self.tick_size, "Stopped for long")

                    if self.take_config is not None:
                        for i, (pts, fr, udata) in enumerate(self.take_config, 1):
                            self.partial_take_at(signal_time, mp + i * pts * self.tick_size, fr, udata)

                elif signal_qty < 0:
                    if self.stop_points is not None:
                        self.debug(f'\n::: SHORT at {mp} stop {mp + self.stop_points * self.tick_size}')
                        self.stop_at(signal_time, mp + self.stop_points * self.tick_size, "Stopped for short")

                    if self.take_config is not None:
                        for i, (pts, fr, udata) in enumerate(self.take_config, 1):
                            self.partial_take_at(signal_time, mp - i * pts * self.tick_size, fr, udata)

                return signal_qty * self.size

        data = _read_timeseries_data('EURUSD', compressed=False, as_dict=True)
        s = _signals({
            '2020-08-17 02:20:01': {'EURUSD': +1},  # 1 take + stop
            '2020-08-17 11:20:01': {'EURUSD': +1},  # all takes
            '2020-08-17 14:35:01': {'EURUSD': -1},  # all takes
            '2020-08-17 18:00:00': {'EURUSD': 0},
        })

        p = backtest(s, data, 'forex', spread=0, execution_logger=True,
                     trackers=_Test_MultiTakeTracker(
                         10000, 100, [
                             (50, 1 / 3, 'close 1/3'),  # close 1/3 in 100 pips
                             (50, 1 / 2, 'close 2/3'),  # close 1/2 in another 100 pips
                             (50, 1, 'close 3/3')  # close rest in another 100 pips
                         ], 0.00001))

        print(p.executions)
        print("- - - - - - - - - -")
        print(p.trackers_stat)
        print("- - - - - - - - - -")
        self.assertEqual(p.trackers_stat['EURUSD']['takes'], 7)
        self.assertEqual(p.trackers_stat['EURUSD']['stops'], 1)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # - - - - - - - - - test it as usual take/stop - - - - - - - - - - -
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        data = _read_timeseries_data('RM1', compressed=False, as_dict=True)
        s = _signals({
            '2020-08-17 00:00:01': {'RM1': +1},
            '2020-08-17 00:22:00': {'RM1': 0},
        })

        p1 = backtest(s, data, 'forex', spread=0, execution_logger=True,
                      trackers=_Test_MultiTakeTracker(
                          10000, None, [
                              (16, 1, 'CLOSE ALL'),  # close 1/3 in 100 pips
                          ], 1)
                      )
        print("- As Single take/stop - - - - - - - - -")
        print(p1.executions)
        print("- - - - - - - - - -")
        print(p1.trackers_stat)
        self.assertEqual(p1.trackers_stat['RM1']['takes'], 1)
        self.assertEqual(p1.trackers_stat['RM1']['stops'], 0)

    def test_triggered_order_with_multitake_targets(self):

        class StopOrdersTestMultiTracker(TriggeredOrdersTracker):
            def __init__(self, tick_size):
                super().__init__(True)
                self.tick_size = tick_size
                self.to = None
                self._fired = 0

            def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
                if signal_qty > 0:
                    entry = ask + 50 * self.tick_size
                    self.to = self.stop_order(
                        entry, 1000,
                        entry - 25 * self.tick_size,
                        {
                            entry + 1 * 25 * self.tick_size: 0.5,  # 1/2 at +25
                            entry + 2 * 25 * self.tick_size: 1.0,  # 1/2 at +50
                        },
                        comment='Test long order', user_data=mstruct(entry_number=1, test=1)
                    )

                if signal_qty < 0:
                    entry = bid - 50 * self.tick_size
                    self.to = self.stop_order(
                        entry, -1000, entry + 100 * self.tick_size, entry - 250 * self.tick_size,
                        comment='Test short order', user_data=mstruct(entry_number=-1, test=-1)
                    )
                return None

            def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
                super().on_quote(quote_time, bid, ask, bid_size, ask_size, **kwargs)

                if self.to is not None:
                    if self.to.fired:
                        print(quote_time, self.to)
                        self.to = None

            def on_trigger_fired(self, timestamp, order: TriggerOrder):
                print(f"\n\t---(FIRED)--> {timestamp} | {order} => {order.user_data} ")
                self._fired += 1

            def on_take(self, timestamp, price, is_partial, closed_amount, user_data=None):
                print(
                    f"\n\t---(TAKE)--> {timestamp} {closed_amount} @ {price} | {user_data} [{'PART' if is_partial else 'FULL'}]")
                print(f"\t---| average take price: {self.average_take_price}")

            def on_stop(self, timestamp, price, user_data=None):
                print(f"\n\t---(STOP)--> {timestamp} {price} | {user_data} ")

            def statistics(self):
                return {'fired': self._fired, **super().statistics()}

        data = _read_timeseries_data('EURUSD', compressed=False, as_dict=True)

        s = _signals({
            '2020-08-17 04:19:59': {'EURUSD': +1},
            '2020-08-17 07:19:59': {'EURUSD': -1},
            '2020-08-17 23:19:59': {'EURUSD': 0},
        })

        track = StopOrdersTestMultiTracker(1e-5)
        p = backtest(s, data, 'forex', spread=0, execution_logger=True, trackers=track)

        print(p.executions)
        print(p.trackers_stat)

        self.assertTrue(p.trackers_stat['EURUSD']['fired'] > 0)
        self.assertTrue(p.trackers_stat['EURUSD']['takes'] == 2)
        self.assertTrue(p.trackers_stat['EURUSD']['stops'] == 1)

        np.testing.assert_array_almost_equal(
            p.executions.exec_price.values,
            [1.185980, 1.186230, 1.186480, 1.184655, 1.185655],
            err_msg='Executions are not correct !'
        )

    def test_limit_orders(self):
        """
        Limit orders tests
        """

        class LimitOrdersTestTracker(TriggeredOrdersTracker):
            def __init__(self, tick_size):
                super().__init__(True)
                self.tick_size = tick_size
                self.to = None
                self._fired = 0

            def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
                # on buy signal
                if signal_qty > 0:
                    self.cancel_all()
                    entry = bid - 80 * self.tick_size
                    # print(f" -(B)-> BID: {bid} // entry: {entry}")
                    self.to = self.limit_order(
                        entry, 1000,
                        entry - 25 * self.tick_size,
                        {entry + 25 * self.tick_size: 1.0},
                        comment='Test long limit order',
                        user_data=mstruct(entry_number=1, test=1)
                    )

                if signal_qty < 0:
                    self.cancel_all()
                    entry = ask + 50 * self.tick_size
                    print(f"\n\t -(S)-> ASK: {ask} // entry: {entry}\n")
                    self.to = self.limit_order(
                        entry, -1000,
                        entry + 250 * self.tick_size,
                        entry - 50 * self.tick_size,
                        comment='Test short limit order',
                        user_data=mstruct(entry_number=-1, test=-1)
                    )
                return None

            def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
                super().on_quote(quote_time, bid, ask, bid_size, ask_size, **kwargs)

                if self.to is not None:
                    if self.to.fired:
                        print(quote_time, self.to)
                        self.to = None

            def on_trigger_fired(self, timestamp, order: TriggerOrder):
                print(f"\n\t---(EXECUTED)--> {timestamp} | {order} => {order.user_data} ")
                self._fired += 1

            def on_take(self, timestamp, price, is_partial, closed_amount, user_data=None):
                print(
                    f"\n\t---(TAKE)--> {timestamp} {price} x {closed_amount} | {user_data} [{'PART' if is_partial else 'FULL'}]")
                print(f"\t---| average take price: {self.average_take_price}")

            def on_stop(self, timestamp, price, user_data=None):
                print(f"\n\t---(STOP)--> {timestamp} {price} | {user_data} ")

            def statistics(self):
                return {'fired': self._fired, **super().statistics()}

        data = _read_timeseries_data('EURUSD', compressed=False, as_dict=True)

        s = _signals({
            '2020-08-17 08:25:01': {'EURUSD': +1},
            '2020-08-17 10:25:01': {'EURUSD': +1},

            '2020-08-17 11:50:59': {'EURUSD': -1},
            '2020-08-17 23:19:59': {'EURUSD': 0},
        })

        track = LimitOrdersTestTracker(1e-5)
        p = backtest(s, data, 'forex', spread=0, execution_logger=True, trackers=track)

        print(p.executions)
        print(p.trackers_stat)

        self.assertTrue(p.trackers_stat['EURUSD']['fired'] == 3)
        self.assertTrue(p.trackers_stat['EURUSD']['takes'] == 2)
        self.assertTrue(p.trackers_stat['EURUSD']['stops'] == 1)
        np.testing.assert_array_almost_equal(
            p.executions.exec_price.values,
            [1.183030, 1.183280, 1.184605, 1.184355, 1.185640, 1.18514],
            err_msg='Executions are not correct !'
        )

    def test_trailing_stop(self):
        data = _read_timeseries_data('EURUSD', compressed=False, as_dict=True)
        s0 = _signals({
            '2020-08-17 00:00': {'EURUSD': 0},
            '2020-08-17 09:10': {'EURUSD': 1},
            '2020-08-17 09:30': {'EURUSD': 2},  # <- second trade
            '2020-08-17 23:59': {'EURUSD': 0},
        })

        s1 = _signals({
            '2020-08-17 00:00': {'EURUSD': 0},
            '2020-08-17 06:26': {'EURUSD': -1},
            '2020-08-17 23:59': {'EURUSD': 0},
        })

        tracker1 = RADTrailingStopTracker(1000, '5Min', 5, 2, process_new_signals=True, accurate_stops=True, debug=False)
        tracker2 = RADTrailingStopTracker(1000, '5Min', 5, 2, process_new_signals=False, accurate_stops=True, debug=False)
        tracker3 = RADTrailingStopTracker(1000, '5Min', 5, 2, filter_signals_by_side=False, accurate_stops=True, debug=True)
        r = simulation({
            'Test RAD 1': [s0, tracker1], 
            'Test RAD 2': [s0, tracker2], 
            'Test RAD 3': [s1, tracker3], 
        }, data, 'forex', start='2020-08-17 00:00')

        print(r.results[0].executions)
        np.testing.assert_array_almost_equal(
            r.results[0].executions.exec_price.values, [1.184465, 1.184790, 1.184844],  
            err_msg='Executions are not correct !')

        print(r.results[1].executions)
        np.testing.assert_array_almost_equal(
            r.results[1].executions.exec_price.values, [1.184465, 1.184844],  
            err_msg='Executions are not correct !')

        print(r.results[2].executions)

from pytest import main
if __name__ == '__main__':
    main()