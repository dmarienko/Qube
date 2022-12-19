import unittest

from qube.portfolio.performance import *
from qube.datasource import DataSource
from qube.datasource.InMemoryDataSource import InMemoryDataSource
from qube.portfolio.PortfolioLogger import PortfolioLogger
from qube.simulator import SignalTester
from qube.simulator.Brokerage import GenericStockBrokerInfo
from qube.simulator.core import ExecutionLogger
from qube.simulator.utils import split_signals
from qube.tests.simulator.utilities import cross_ma, cumulative_pnl_validation_eod
from qube.utils.DateUtils import DateUtils


class TestStrategy(unittest.TestCase):
    DS_CFG_FILE = 'qube/tests/data/simulator/sim_test_datasource_cfg.json'

    def setUp(self):
        ds_daily = DataSource('simulator::ohlc_data', self.DS_CFG_FILE)

        self.prices = ds_daily.load_data(['XOM', 'AAPL', 'MSFT'], start='2015-01-01', end='2017-01-01')

        # generate positions
        self.positions = {
            instr: cross_ma(self.prices, instr, p_fast=p_F, p_slow=p_S) for (instr, p_F, p_S) in [
                ('AAPL', 5, 50),
                ('MSFT', 5, 50),
            ]
        }

        self.sim = SignalTester(GenericStockBrokerInfo(spread=0), ds_daily)

    def testOHLC(self):
        data_dict = {
            'Date': [DateUtils.get_datetime('2000-01-01 10:00:00'), DateUtils.get_datetime('2000-01-01 10:05:00')],
            'open': [100, 130],
            'high': [150, 150],
            'low': [90, 90],
            'close': [120, 90],
            'volume': [50, 70]
        }

        ds_data = pd.DataFrame(data_dict)
        ds_data.set_index('Date', inplace=True)
        mem_ds = InMemoryDataSource({'EURUSD': ds_data})
        tester = SignalTester(GenericStockBrokerInfo(spread=0), mem_ds)
        signals_dict = {
            'Date': [DateUtils.get_datetime('2000-01-01 10:00:01'),
                     DateUtils.get_datetime('2000-01-01 10:04:59')],
            'EURUSD': [1, 0]
        }
        signals = pd.DataFrame(signals_dict)
        signals.set_index('Date', inplace=True)
        elogger = ExecutionLogger()
        tester.run_signals(signals, execution_logger=elogger)
        exec_log = elogger.get_execution_log()
        self.assertEqual(DateUtils.get_as_string(exec_log.iloc[0].name), '2000-01-01 10:00:01')
        self.assertEqual(exec_log.iloc[0]['exec_price'], 100)
        self.assertEqual(DateUtils.get_as_string(exec_log.iloc[1].name), '2000-01-01 10:04:59')
        self.assertEqual(exec_log.iloc[1]['exec_price'], 120)

    def inception_test(self):
        W = self.sim.run_signals(pd.concat(self.positions.values(), axis=1),
                                 portfolio_logger=PortfolioLogger())

        r = calculate_total_pnl(W.portfolio)

        # how to do validation of PnL
        right_pnl_1 = cumulative_pnl_validation_eod(self.positions['AAPL'], self.prices['AAPL'])
        right_pnl_2 = cumulative_pnl_validation_eod(self.positions['MSFT'], self.prices['MSFT'])
        right_pnl = right_pnl_1 + right_pnl_2

        self.assertAlmostEqual(r['Total_PnL'].sum(), right_pnl[-1], delta=0.01)

    def test_sequence_chunks(self):
        signals = pd.concat(self.positions.values(), axis=1)
        W = self.sim.run_signals(signals, portfolio_logger=PortfolioLogger())
        sim_start = signals.index[0].to_pydatetime()
        sim_end = signals.index[-1].to_pydatetime()
        chunk_amount = 3
        split_dates = DateUtils.splitOnIntervals(sim_start, sim_end, chunk_amount, return_split_dates_only=True)

        split_signals_list = split_signals(signals, split_dates)

        merged_portfolio = None
        for chunk_idx, split_chunk in enumerate(split_signals_list):
            chunk_pl = self.sim.run_signals(split_chunk)
            portf_log = chunk_pl.portfolio
            if merged_portfolio is None:
                merged_portfolio = portf_log
            else:
                to_append = portf_log[1:]
                # merged_portfolio = merged_portfolio.append(to_append)
                merged_portfolio = pd.concat((merged_portfolio[:to_append.index[0]], to_append[1:]), axis=0)

        r = calculate_total_pnl(merged_portfolio, False)

        # how to do validation of PnL
        right_pnl_1 = cumulative_pnl_validation_eod(self.positions['AAPL'], self.prices['AAPL'])
        right_pnl_2 = cumulative_pnl_validation_eod(self.positions['MSFT'], self.prices['MSFT'])
        right_pnl = right_pnl_1 + right_pnl_2

        self.assertAlmostEqual(r['Total_PnL'].sum(), right_pnl[-1], delta=0.01)


from pytest import main
if __name__ == '__main__':
    main()