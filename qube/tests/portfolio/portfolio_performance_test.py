import unittest
from os.path import join

from qube.configs.Properties import get_root_dir
from qube.portfolio.PortfolioLogger import PortfolioLogger
from qube.portfolio.performance import *
from qube.simulator import SignalTester
from qube.simulator.Brokerage import GenericStockBrokerInfo
from qube.tests.simulator.signal_tester_test import MockDataSource, gen_pos


class PortfolioStatsTests(unittest.TestCase):
    DS_CFG_PATH = 'qube/tests/ds_test_cfg.json'

    d_rets = pd.Series(np.array([np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
                       index=pd.date_range('2000-1-30', periods=9, freq='D'))

    def test_cagr(self):
        self.assertAlmostEqual(cagr(self.d_rets), 1.913593, delta=0.001)

        year_returns = pd.Series(np.array([3., 3., 3.]) / 100, index=pd.date_range('2000-1-30', periods=3, freq='A'))
        self.assertAlmostEqual(cagr(year_returns, YEARLY), 0.03, delta=0.001)

    def test_aggregate(self):
        data = pd.Series(np.array([0., 1., 0., 1., 0., 1., 0., 1., 0.]) / 100,
                         index=pd.date_range('2000-1-30', periods=9, freq='D'))
        np.testing.assert_almost_equal(aggregate_returns(data, convert_to='Y'), [0.04060401])
        np.testing.assert_almost_equal(aggregate_returns(data, convert_to='M'), [0.01, 0.030301])
        np.testing.assert_almost_equal(aggregate_returns(data, convert_to='W'), [0.0, 0.04060401, 0.0])
        np.testing.assert_almost_equal(aggregate_returns(data, convert_to='D'), data.values)

    def test_sharpe(self):
        benchmark = pd.Series(np.array([0., 1., 0., 1., 0., 1., 0., 1., 0.]) / 100,
                              index=pd.date_range('2000-1-30', periods=9, freq='D'))

        self.assertAlmostEqual(sharpe_ratio(self.d_rets, 0.0), 1.7238613961706866)
        self.assertAlmostEqual(sharpe_ratio(self.d_rets, benchmark), 0.34111411441060574)

    def test_sortino(self):
        self.assertAlmostEqual(2.605531251673693, sortino_ratio(self.d_rets, 0.0))

        incr_returns = pd.Series(np.array([np.nan, 1., 10., 1., 2., 3., 2., 1., 1.]) / 100,
                                 index=pd.date_range('2000-1-30', periods=9, freq='D'))
        self.assertAlmostEqual(np.inf, sortino_ratio(incr_returns, 0.0))

        zero_returns = pd.Series([0, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.isnan(sortino_ratio(zero_returns, 0.0)))

    def test_stat_sheet(self):
        portfolio = pd.read_csv(join(get_root_dir(), 'tests/data/portfolios/portfolio1.csv'), index_col='Date',
                                parse_dates=True)

        sheet = portfolio_stats(portfolio, 100000, commissions='dukas')

        print(100 * sheet['cagr'])
        print(sheet['sharpe'])
        print(sheet['sortino'])
        print(sheet['calmar'])
        print(100 * sheet['annual_volatility'])
        print('MaxDD: $%.2f, %0.2f%%' % (sheet['mdd_usd'], 100 * sheet['drawdown_pct']))
        print(sheet['mdd_start'])
        print(sheet['mdd_peak'])
        print(sheet['mdd_recover'])
        print(sheet['dd_stat'])
        print('Stability: %f' % sheet['stability'])
        print('Tail ratio: %f' % sheet['tail_ratio'])

        # Value-At-Risk
        print('VaR: $%0.2f' % sheet['var'])

        if 'alpha' in sheet:
            print('Alpha: %f, Beta: %f' % (sheet['alpha'], sheet['beta']))

        if sheet['broker_commissions']:
            print('Broker Commissions: $%0.2f' % (sheet['broker_commissions']))

        print('-------------------------------------------------------------------------------------')
        print(sorted(list(sheet.keys())))
        print('-------------------------------------------------------------------------------------')

    def test_collect_entries_data(self):
        ds = MockDataSource('2000-01-01', 10000, amplitudes=(10, 30, 5), freq='5 min')

        pos = gen_pos({
            'XXX': [
                '2000-01-01 9:30', 100,
                '2000-01-01 9:35', 100,
                '2000-01-01 9:45', 0,
                '2000-01-01 10:00', -100,
                '2000-01-01 10:15', 0,
                '2000-01-01 11:00', 100,
                '2000-01-01 11:10', 200,
                '2000-01-01 11:30', -200,
                '2000-01-01 11:45', -100,
                '2000-01-01 11:55', 100,
                '2000-01-01 12:00', 0
            ]
        })
        plogger = PortfolioLogger()
        sim = SignalTester(GenericStockBrokerInfo(spread=0), ds)
        sim.run_signals(pos, portfolio_logger=plogger)
        pfl_log = split_cumulative_pnl(plogger.get_portfolio_log())
        expected_res = [('2000-01-01 09:30', '2000-01-01 09:45', 0.0, -800.0, -300.0),
                        ('2000-01-01 10:00', '2000-01-01 10:15', 500.0, -100.0, -100.0),
                        ('2000-01-01 11:00', '2000-01-01 11:30', 1000.0, -400.0, 1000.0),
                        ('2000-01-01 11:30', '2000-01-01 11:55', 1600.0, 400.0, 600.0),
                        ('2000-01-01 11:55', '2000-01-01 12:00', -500.0, -600.0, -600.0)]
        s = collect_entries_data(pfl_log)['XXX']
        self.assertEqual(len(expected_res), len(s))
        [self.assertEqual(expected_res[i],
                          (idx.strftime('%Y-%m-%d %H:%M'), s.at[idx, 'Closed'].strftime('%Y-%m-%d %H:%M'),
                           s.at[idx, 'MaxPL'], s.at[idx, 'MinPL'], s.at[idx, 'SignalPL'])) for i, idx in
         enumerate(s.index[:])]


from pytest import main
if __name__ == '__main__':
    main()