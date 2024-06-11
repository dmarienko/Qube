import unittest

import numpy as np
import pandas as pd
import pandas.testing as pdt

from qube.datasource import DataSource
from qube.portfolio.PortfolioLogger import PortfolioLogger
from qube.simulator import SignalTester
from qube.simulator import utils
from qube.simulator.Brokerage import GenericStockBrokerInfo
from qube.simulator.utils import merge_portfolio_log_chunks, ls_brokers
from qube.simulator.utils import rolling_forward_test_split
from qube.tests.simulator.utilities import cross_ma
from qube.utils.DateUtils import DateUtils


class SignalsUtilTest(unittest.TestCase):
    DS_CFG_FILE = 'qube/tests/data/simulator/sim_test_datasource_cfg.json'

    def setUp(self):
        self.sigs = self._gen_pos({
            'XXX': [
                '2000-01-01 9:30', 100,
                '2000-01-01 10:30', None,
                '2000-01-01 16:00', 0,
                '2000-01-02 9:30', -100,
                '2000-01-03 16:00', 0,
            ],
            'YYY': [
                '2000-01-01 9:30', 100,
                '2000-01-01 16:00', 0,
                '2000-01-02 9:30', 200,
                '2000-01-05 16:00', 0,
            ]})

    def test_split_1(self):
        print('test_split_1:')
        print(self.sigs)
        res = utils.split_signals(self.sigs, DateUtils.get_datetime_ls(['2000-01-02 09:30']))
        self.assertEqual(len(res), 2)
        print(res[0])
        pdt.assert_frame_equal(res[0], self._get_by_dict({'2000-01-01 09:30': [100, 100],
                                                          '2000-01-01 10:30': [100, 100],
                                                          '2000-01-01 16:00': [0, 0],
                                                          '2000-01-02 09:30': [-100, 200]
                                                          }))

        print(res[1])
        pdt.assert_frame_equal(res[1], self._get_by_dict({'2000-01-02 9:30': [-100, 200],
                                                          '2000-01-03 16:00': [0, 200],
                                                          '2000-01-05 16:00': [0, 0]
                                                          }))

    def test_split_2(self):
        print('\ntest_split_2')
        print(self.sigs)
        res = utils.split_signals(self.sigs, DateUtils.get_datetime_ls(['2000-01-01 10:31:00', '2000-01-03 16:00:00']))
        self.assertEqual(len(res), 3)

        print(res[0])
        pdt.assert_frame_equal(res[0], self._get_by_dict({'2000-01-01 09:30': [100, 100],
                                                          '2000-01-01 10:30': [100, 100],
                                                          }))

        print(res[1])
        pdt.assert_frame_equal(res[1], self._get_by_dict({'2000-01-01 10:30': [100, 100],
                                                          '2000-01-01 16:00': [0, 0],
                                                          '2000-01-02 09:30': [-100, 200],
                                                          '2000-01-03 16:00': [0, 200],
                                                          }))
        print(res[2])
        pdt.assert_frame_equal(res[2], self._get_by_dict({'2000-01-03 16:00': [0, 200],
                                                          '2000-01-05 16:00': [0, 0],
                                                          }))

    def test_merge_chunks1(self):
        columns = ['MSFT_PnL', 'AAPL_PnL']
        chunk1_pl = self._get_by_dict({'2000-01-02 9:30': [-100, 200],
                                       '2000-01-03 16:00': [0, 200],
                                       '2000-01-03 16:02': [0, 200], }, columns=columns)

        chunk2_pl = self._get_by_dict({
            '2000-01-03 16:00': [0, 0],
            '2000-01-05 16:00': [0, 0]}, columns=columns)

        r = utils.merge_portfolio_log_chunks([chunk1_pl, chunk2_pl], split_cumulative=False)
        print(r)
        pdt.assert_frame_equal(r, self._get_by_dict({
            '2000-01-02 9:30': [-100, 200],
            '2000-01-03 16:00': [0, 200],
            '2000-01-05 16:00': [0, 0]
        }, columns=columns))

    def test_merge_chunks2(self):
        columns = ['MSFT_PnL', 'AAPL_PnL']
        chunk1_pl = self._get_by_dict({'2000-01-02 9:30': [-100, 200],
                                       '2000-01-03 16:01': [0, 200],
                                       '2000-01-03 16:02': [0, 200], }, columns=columns)

        chunk2_pl = self._get_by_dict({
            '2000-01-03 16:00': [0, 0],
            '2000-01-05 16:00': [0, 0]}, columns=columns)

        r = utils.merge_portfolio_log_chunks([chunk1_pl, chunk2_pl], split_cumulative=False)
        print(r)
        pdt.assert_frame_equal(r, self._get_by_dict({
            '2000-01-02 9:30': [-100, 200],
            '2000-01-03 16:00': [-100, 200],
            '2000-01-05 16:00': [0, 0]
        }, columns=columns))

    def test_merge_chunks3(self):
        columns = ['MSFT_PnL', 'AAPL_PnL']
        chunk1_pl = self._get_by_dict({'2000-01-02 9:30': [-100, 200],
                                       '2000-01-03 15:59': [0, 200]}, columns=columns)

        chunk2_pl = self._get_by_dict({
            '2000-01-03 16:00': [0, 0],
            '2000-01-05 16:00': [0, 0]}, columns=columns)

        r = utils.merge_portfolio_log_chunks([chunk1_pl, chunk2_pl], split_cumulative=False)
        print(r)
        pdt.assert_frame_equal(r, self._get_by_dict({
            '2000-01-02 9:30': [-100, 200],
            '2000-01-03 15:59': [0, 200],
            '2000-01-03 16:00': [0, 0],
            '2000-01-05 16:00': [0, 0]
        }, columns=columns))

    def test_merge_chunks4(self):
        columns = ['MSFT_PnL', 'AAPL_PnL', 'XYZ']
        chunk1_pl = self._get_by_dict({'2000-01-02 9:30': [-100, 200, 10],
                                       '2000-01-03 16:00': [0, 200, 20]}, columns=columns)

        chunk2_pl = self._get_by_dict({
            '2000-01-03 16:00': [0, 0, 30],
            '2000-01-05 16:00': [0, 0, 40]}, columns=columns)

        r = utils.merge_portfolio_log_chunks([chunk1_pl, chunk2_pl], split_cumulative=False)
        print(r)
        pdt.assert_frame_equal(r, self._get_by_dict({
            '2000-01-02 9:30': [-100, 200, 10],
            '2000-01-03 16:00': [0, 200, 30],
            '2000-01-05 16:00': [0, 0, 40]
        }, columns=columns))

    def test_hours_rolling_forward_test_split(self):
        # generate df
        df = pd.Series([i for i in range(200)], pd.date_range('2015-01-10 10:00:00', periods=200, freq='5min'))
        result_train = []
        result_test = []
        r = rolling_forward_test_split(df, 12, 10, 'H')
        for train, test in r:
            result_test.append(test)
            result_train.append(train)

        self.assertEqual(str(result_train[0][0]), '2015-01-10T10:00:00.000000000')
        # end after 12 hours
        self.assertEqual(str(result_train[0][-1]), '2015-01-10T21:55:00.000000000')
        self.assertEqual(str(result_test[0][0]), '2015-01-10T22:00:00.000000000')
        # end after 10 hours
        self.assertEqual(str(result_train[0][-1]), '2015-01-10T21:55:00.000000000')
        # last 'test' day same as last day in df
        self.assertEqual(np.datetime64(df.index[-1]), result_test[-1][-1])

    def test_months_rolling_forward_test_split(self):
        # generate df
        df = pd.Series([i for i in range(200)], pd.date_range('2015-01-02 00:00:00', periods=200, freq='1D'))
        result_train = []
        result_test = []
        r = rolling_forward_test_split(df, 2, 1, 'M')
        for train, test in r:
            result_test.append(test)
            result_train.append(train)

        self.assertEqual(str(result_train[0][0]), '2015-01-02T00:00:00.000000000')

        # 'training' end after 2 months
        self.assertEqual(str(result_train[0][-1]), '2015-02-28T00:00:00.000000000')

        # 'test' starts after end 'training'
        self.assertEqual(str(result_test[0][0]), '2015-03-01T00:00:00.000000000')

        # 'test' end after 1 month
        self.assertEqual(str(result_test[0][-1]), '2015-03-31T00:00:00.000000000')

        # next 'training' starts after last end 'training' start + 'test' period
        self.assertEqual(str(result_train[1][0]), '2015-02-01T00:00:00.000000000')

        # and end after 2 months
        self.assertEqual(str(result_train[1][-1]), '2015-03-31T00:00:00.000000000')

        # last day in test periods, same as last day in df
        self.assertEqual(np.datetime64(df.index[-1]), result_test[-1][-1])

    def test_skip_days_rolling_forward_test_split(self):
        # generate df
        df = pd.Series([i for i in range(200)], pd.date_range('2015-01-02 00:00:00', periods=200, freq='3D'))
        result_train = []
        result_test = []
        r = rolling_forward_test_split(df, 1, 1, 'W')
        for train, test in r:
            result_test.append(test)
            result_train.append(train)

        self.assertEqual(len(result_train), 85)

        # last day in test periods, same as last day in df
        self.assertEqual(np.datetime64(df.index[-1]), result_test[-1][-1])

    def test_merge_chunks_flow(self):
        # run with 1 process
        ds_daily = DataSource('simulator::ohlc_data', self.DS_CFG_FILE)

        prices = ds_daily.load_data(['XOM', 'AAPL', 'MSFT'], start='2015-01-01', end='2017-01-01')
        positions = {instr: cross_ma(prices, instr, p_fast=p_F, p_slow=p_S)
                     for (instr, p_F, p_S) in [
                         ('AAPL', 5, 50),
                         ('MSFT', 5, 50),
                     ]
                     }

        signals = pd.concat(positions.values(), axis=1)
        sim = SignalTester(GenericStockBrokerInfo(spread=0),
                           DataSource(ds_daily.data_source_name, ds_daily.config_path))
        pl1 = sim.run_signals(signals, portfolio_logger=PortfolioLogger(), verbose=True)
        # >>> we don't need to split because it's already splitted in portoflio
        # pl1_log = split_cumulative_pnl(pl1.portfolio)
        pl1_log = pl1.portfolio

        sim_start = signals.index[0].to_pydatetime()
        sim_end = signals.index[-1].to_pydatetime()

        # prepare intervals to split testing on chunks
        split_dates = DateUtils.splitOnIntervals(sim_start, sim_end, 3, return_split_dates_only=True)

        split_signals_list = utils.split_signals(signals, split_dates)

        pl_chunks = []
        merged_portfolio = None
        for chunk_idx, split_chunk in enumerate(split_signals_list):
            chunk_pl = sim.run_signals(split_chunk, portfolio_logger=PortfolioLogger())
            portf_log = chunk_pl.portfolio
            pl_chunks.append(portf_log)
            if merged_portfolio is None:
                # >>> we don't need to split because it's already splitted in portoflio
                # merged_portfolio = split_cumulative_pnl(portf_log)
                merged_portfolio = portf_log
            else:
                # >>> we don't need to split because it's already splitted in portoflio
                # to_append = split_cumulative_pnl(portf_log[1:])
                to_append = portf_log[1:]
                merged_portfolio = pd.concat((merged_portfolio[:to_append.index[0]], to_append[1:]), axis=0)

        merged_chunks = merge_portfolio_log_chunks(pl_chunks)

        # finally test
        np.testing.assert_array_almost_equal(pl1_log, merged_portfolio, decimal=5)

    def _get_by_dict(self, data_dict, columns=['XXX', 'YYY']):
        res = pd.DataFrame.from_dict(data_dict, orient='index', dtype='float')
        res.index = pd.to_datetime(res.index)
        res.columns = columns
        res.sort_index(inplace=True)
        return res

    # Handy positions generator from dictionary
    def _gen_pos(self, p_dict: dict):
        sgen = lambda name, sigs: pd.Series(data=sigs[1:len(sigs):2], index=pd.to_datetime(sigs[0:-1:2]), name=name)
        return pd.concat([sgen(s, p) for (s, p) in p_dict.items()], axis=1)

    def test_list_of_brokers(self):
        for x in ls_brokers():
            print(x)


from pytest import main
if __name__ == '__main__':
    main()