import unittest
import mongomock
from mongomock.gridfs import enable_gridfs_integration
from collections import deque
from datetime import timedelta
from io import StringIO

from dateutil.parser import parse

from qube.portfolio.Instrument import Instrument
from qube.portfolio.Position import Position, ForexPosition, CryptoPosition, CryptoFuturesPosition
from qube.utils import QubeLogger
from qube.datasource.controllers.MongoController import MongoController


enable_gridfs_integration()

def _upd_prt(self, timestamp, price):
    self.update_pnl(parse(timestamp), price)
    print(str(self))


def _upd_prt2(self, timestamp, price):
    self.update_pnl(timestamp, price)
    print(str(self))


def _upd_pos(self, timestamp, pos, price):
    self.update_position(parse(timestamp), pos, price)


class Simulator_Position_test(unittest.TestCase):

    def test_position_class(self):
        Position.upd = _upd_prt
        Position.pos = _upd_pos
        p = Position(Instrument('TEST', False, 0.01, 0.0))

        p.pos('2017-01-01 9:30:00', 5, 100.0)
        p.upd('2017-01-01 9:30:00', 130)
        self.assertEqual(p.pnl, 150)

        p.pos('2017-01-01 9:35:00', 10, 200.0)
        p.upd('2017-01-01 9:35:00', 250)
        self.assertEqual(p.pnl, 1000)

        p.pos('2017-01-01 9:40:00', 0, 300.0)
        p.upd('2017-01-01 9:40:00', 300)
        self.assertEqual(p.pnl, 1500)

        p.upd('2017-01-01 9:45:00', 130)
        self.assertEqual(p.pnl, 1500)

        p.pos('2017-01-01 9:50:00', 10, 50)
        p.pos('2017-01-01 9:50:10', 5, 40)
        p.upd('2017-01-01 9:55:00', 55)
        self.assertEqual(p.pnl, 1475)

        p.pos('2017-01-01 9:56:00', 0, 55)
        p.upd('2017-01-01 9:56:00', 55)
        self.assertEqual(p.pnl, 1475)

        # part position liquidation
        p = Position(Instrument('TEST2', False, 0.01, 0.0))
        p.pos('2017-01-01 10:00:00', 100, 100)
        self.assertEqual(p.pnl, 0)

        p.pos('2017-01-01 10:00:10', 90, 110)
        p.pos('2017-01-01 10:00:15', 80, 110)
        p.pos('2017-01-01 10:00:20', 70, 110)

        p.upd('2017-01-01 10:00:20', 110)
        self.assertEqual(p.pnl, 1000)

        p.upd('2017-01-01 10:00:20', 110)
        self.assertEqual(p.pnl, 1000)

    def test_pnl_calcs(self):
        Position.upd = _upd_prt2
        po = Position(Instrument('TEST3', False, 0.01, 0.0))
        pnls = deque()
        # our old iQt matlab test
        positions = [2, 1, -10, 15, 5, 0]
        prices = [50.0, 51.0, 49.0, 51.0, 53.0, 52.0]
        d0 = parse('2000-01-01 10:00:00')
        for (s, p) in zip(positions, prices):
            po.update_position(d0, s, p)
            pnls.append(po.pnl)
            d0 += timedelta(minutes=5)
        self.assertEqual(deque([0.0, 2.0, 0.0, -20.0, 10.0, 5.0]), pnls)

    def test_pnl_calcs_spread(self):
        Position.upd = _upd_prt2
        po = Position(Instrument('SPREADED', False, 0.01, 2))
        pnls = deque()

        # our old iQt matlab test
        positions = [2, 1, -10, 15, 5, 0]
        prices = [50.0, 51.0, 49.0, 51.0, 53.0, 52.0]

        d0 = parse('2000-01-01 10:00:00')
        for (s, p) in zip(positions, prices):
            po.update_position(d0, s, p)
            pnls.append(po.pnl)
            d0 += timedelta(minutes=5)

        self.assertEqual(deque([-4.0, -2.0, -24.0, -74.0, -44.0, -49.0]), pnls)

    def test_forex_position(self):
        p1 = ForexPosition(Instrument('EURUSD', False, 0.00001, 0))
        self.assertEqual(p1.base, 'EUR')
        self.assertEqual(p1.quote, 'USD')
        self.assertTrue(p1.is_straight)
        self.assertTrue(not p1.is_cross)
        self.assertTrue(not p1.is_reversed)
        self.assertTrue(p1.aux_instrument is None)

        p2 = ForexPosition(Instrument('USDCHF', False, 0.00001, 0))
        self.assertEqual(p2.base, 'USD')
        self.assertEqual(p2.quote, 'CHF')
        self.assertTrue(not p2.is_straight)
        self.assertTrue(not p2.is_cross)
        self.assertTrue(p2.is_reversed)
        self.assertTrue(p2.aux_instrument is None)

        p3 = ForexPosition(Instrument('EURCHF', False, 0.00001, 0))
        self.assertEqual(p3.base, 'EUR')
        self.assertEqual(p3.quote, 'CHF')
        self.assertTrue(not p3.is_straight)
        self.assertTrue(p3.is_cross)
        self.assertTrue(not p3.is_reversed)
        self.assertTrue(p3.aux_instrument == 'USDCHF')

        p4 = ForexPosition(Instrument('DAIDEEUR', False, 0.00001, 0))
        self.assertEqual(p4.base, 'DAIDEEUR')
        self.assertEqual(p4.quote, 'EUR')
        self.assertTrue(not p4.is_straight)
        self.assertTrue(not p4.is_cross)
        self.assertTrue(not p4.is_reversed)
        self.assertTrue(p4.aux_instrument == 'EURUSD')

        p5 = ForexPosition(Instrument('XAUUSD', False, 0.00001, 0))
        self.assertEqual(p5.base, 'XAUUSD')
        self.assertEqual(p5.quote, 'USD')
        self.assertTrue(p5.is_straight)
        self.assertTrue(not p5.is_cross)
        self.assertTrue(not p5.is_reversed)
        self.assertTrue(p5.aux_instrument is None)

    def test_forex_reverted_pairs(self):
        pos_size = 10_000
        p1 = ForexPosition(Instrument('USDJPY', False, 0.001, 0))
        p1.update_position_bid_ask(0, pos_size, 109.000, 109.001)
        pnl = p1.update_pnl_bid_ask(1, 109.300, 109.301)
        self.assertAlmostEqual(pnl, pos_size * (109.300 - 109.001) / 109.300, delta=1e-3)
        p1.update_position_bid_ask(2, 0, 108.000, 108.001)
        pnl = p1.r_pnl
        self.assertAlmostEqual(pnl, -pos_size * (109.001 - 108.000) / 108.000, delta=1e-3)

    def test_crosses1(self):
        p = ForexPosition(Instrument('EURGBP', False, 0.0001, 0))
        p.update_position_bid_ask(0, 1000, 1.1000, 1.1010, a_is_straight=True, a_bid=1.2, a_ask=1.21)
        pnl = p.update_pnl_bid_ask(1, 1.1030, 1.1040, a_is_straight=True, a_bid=1.21, a_ask=1.22)
        self.assertAlmostEqual(pnl, 2.44, delta=0.001)
        p.update_position_bid_ask(2, 0, 1.1100, 1.1210, a_is_straight=True, a_bid=1.2, a_ask=1.21)
        pnl = p.r_pnl
        self.assertAlmostEqual(pnl, 10.890, delta=0.001)

    def test_crosses2(self):
        p1 = ForexPosition(Instrument('EURJPY', False, 0.1, 0))
        p1.update_position_bid_ask(0, 1000, 120.7, 120.7, a_is_straight=False, a_bid=110.0, a_ask=110.0)
        pnl = p1.update_pnl_bid_ask(1, 121.0, 121.0, a_is_straight=False, a_bid=110.0, a_ask=110.0)
        self.assertAlmostEqual(pnl, 2.727, delta=0.001)

        p1.update_position_bid_ask(2, 0, 121.7, 121.7, a_is_straight=False, a_bid=110.1, a_ask=110.2)
        pnl = p1.r_pnl
        self.assertAlmostEqual(pnl, 9.0744, delta=0.001)

    @mongomock.patch(servers=(('localhost', 27017),), on_new='create')
    def test_save_and_position_change_logging(self):
        logger = QubeLogger.getLogger('qube.test.test_pos_change_logging')
        stringio = StringIO()
        QubeLogger.addStreamHandler(logger, stringio)

        def log_execution(pos, dt, pos_change, exec_price, comm, comment=''):
            logger.info(
                '[%s] %+d %s @ %.5f' % (dt.strftime('%d-%b-%Y %H:%M:%S.%f'), pos_change, pos.symbol, exec_price)
            )

        p = Position(Instrument('TEST', False, 0.01, spread=0)).attach_execution_callback(log_execution)
        position_table = "_test_pos_"
        mongo = MongoController()
        mongo.save_data(position_table, p)

        p.update_position(parse('2010-01-01'), 120, 10.24)
        log_str = stringio.getvalue()
        self.assertTrue(all([v in log_str for v in ['TEST', '120', '10.24']]))
        loaded_p = mongo.load_data(position_table)['data']
        self.assertEqual(len(loaded_p._Position__exec_cb_list), 0)  # way to get private variable __exec_cb_list

    def test_crypto_position(self):
        p = CryptoPosition(Instrument('ETHBTC', False, 0.01, 0.01))
        self.assertEqual(p.base, 'ETH')
        self.assertEqual(p.quote, 'BTC')
        self.assertTrue(not p.is_straight)
        self.assertTrue(p.is_cross)
        self.assertTrue(p.aux_instrument == 'BTCUSD')

        p = CryptoPosition(Instrument('BTCUSD', False, 0.01, 0.01))
        self.assertEqual(p.base, 'BTC')
        self.assertEqual(p.quote, 'USD')
        self.assertTrue(p.is_straight)
        self.assertTrue(not p.is_cross)
        self.assertTrue(p.aux_instrument is None)

        p = CryptoPosition(Instrument('BTCEUR', False, 0.01, 0.01))
        self.assertEqual(p.base, 'BTC')
        self.assertEqual(p.quote, 'EUR')
        self.assertTrue(not p.is_straight)
        self.assertTrue(p.is_cross)
        self.assertTrue(p.aux_instrument is None)

        pos_size = 0.002
        p = CryptoPosition(Instrument('ETHBTC', False, 0.01, 0.01))
        p.update_position_bid_ask(0, pos_size, 0.0427, 0.04271, 7.1, 7.102)
        pnl = p.update_pnl_bid_ask(1, 0.04319, 0.04320, 7.1, 7.102)
        self.assertAlmostEqual(pnl, pos_size * (0.04319 - 0.04271) * 7.102, delta=1e-6)

        p = CryptoPosition(Instrument('BTCUSD', False, 0.01, 0.01))
        p.update_position_bid_ask(0, pos_size, 7.1, 7.102)
        pnl = p.update_pnl_bid_ask(1, 7.26, 7.2601)
        self.assertAlmostEqual(pnl, pos_size * (7.26 - 7.102) * 1, delta=1e-6)

    def test_crypto_futures_position_buy(self):
        p = CryptoFuturesPosition(Instrument('XBTUSD', True, 0.5, 0.5))
        self.assertEqual(p.base, 'XBT')
        self.assertEqual(p.quote, 'USD')
        self.assertTrue(p.is_straight)
        self.assertTrue(not p.is_cross)
        self.assertTrue(not p.aux_instrument)
        pos_size = 1000
        p.update_position_bid_ask(0, pos_size, 995, 1000)
        pnl = p.update_pnl_bid_ask(0, 1250, 1255)
        self.assertAlmostEqual(pnl, 250)
        pnl = p.update_pnl_bid_ask(0, 1500, 1555)
        self.assertAlmostEqual(pnl, 500)

        r_pnl = p.update_position_bid_ask(0, 500, 1250, 1255)
        self.assertAlmostEqual(r_pnl, 125.0)
        self.assertAlmostEqual(p.pnl, 250)

        pnl = p.update_pnl_bid_ask(0, 1500, 1555)
        self.assertAlmostEqual(pnl, 375)

        pnl = p.update_pnl_bid_ask(0, 1000, 1005)
        self.assertAlmostEqual(p.r_pnl, 125)
        self.assertAlmostEqual(pnl, 125)

        pnl = p.update_position_bid_ask(0, -100, 500, 505)
        self.assertAlmostEqual(p.r_pnl, -125)
        self.assertAlmostEqual(pnl, -250)


from pytest import main
if __name__ == '__main__':
    main()