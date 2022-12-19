from unittest import TestCase

import numpy as np
import pandas as pd

from qube.quantitative.tools import column_vector
from qube.portfolio.allocating import tang_portfolio, effective_portfolio, gmv_portfolio, \
    runnig_portfolio_allocator, olmar_portfolio


class PortfolioAllocatingTest(TestCase):

    def setUp(self):
        self.rets_1 = np.array([[1, 5, 3], [3, 2, 5], [2, 1, 8],
                                [9, 4, 1], [24, 0, 0], [14, 0, 4],
                                [1, 1, 4], [2, 1, 2], [9, 5, 5],
                                [5, 1, 55], [10, 1, 0]])
        self.rets_2 = np.array([[24, 9, 2], [8, 12, 1], [14, 15, 0],
                                [11, 8, 1], [2, 8, 11], [11, 9, 1],
                                [6, 15, 21], [18, 2, 4], [10, 5, 3],
                                [25, 21, 25], [20, 2, 11]])
        self.columns = ['INSTR_1', 'INSTR_2', 'INSTR_3']

    def test_tang_portfolio(self):
        df = pd.DataFrame(self.rets_1, columns=self.columns)
        res_1 = tang_portfolio(self.rets_1, 0.01)
        res_2 = tang_portfolio(self.rets_2, 0.5)
        res_df = tang_portfolio(df, 0.01)
        np.testing.assert_almost_equal(res_1, np.array([0.2020, 0.7430, 0.0550]), decimal=4)
        np.testing.assert_almost_equal(res_2, np.array([0.4819, 0.5740, -0.0559]), decimal=4)
        np.testing.assert_equal(res_df.values.flatten(), res_1)
        self.assertEqual(res_df.columns.tolist(), self.columns)

    def test_gmv_portfolio(self):
        df = pd.DataFrame(self.rets_1, columns=self.columns)
        res_1 = gmv_portfolio(self.rets_1)
        res_2 = gmv_portfolio(self.rets_2)
        res_df = gmv_portfolio(df)
        np.testing.assert_almost_equal(res_1, np.array([[0.1283, 0.8376, 0.0341]]), decimal=4)
        np.testing.assert_almost_equal(res_2, np.array([[0.3611, 0.5811, 0.0578]]), decimal=4)
        np.testing.assert_equal(res_df.values, res_1)
        self.assertEqual(res_df.columns.tolist(), self.columns)

    def test_effective_portfolio(self):
        df = pd.DataFrame(self.rets_1, columns=self.columns)
        res_1 = effective_portfolio(self.rets_1, 0)
        res_2 = effective_portfolio(self.rets_2, 2)
        res_df = effective_portfolio(df, 0)
        np.testing.assert_almost_equal(res_1, np.array([[-0.2680, 1.3466, -0.0786]]), decimal=4)
        np.testing.assert_almost_equal(res_2, np.array([[-1.0913, 0.6654, 1.4259]]), decimal=4)
        np.testing.assert_equal(res_df.values, res_1)
        self.assertEqual(res_df.columns.tolist(), self.columns)

    def test_olmar_portfolio(self):
        array_prices = np.array([[55.23, 55.45], [58.48, 59.01], [72.49, 72.75], [34.49, 34.61], [76.71, 77.11]])
        array_volumes = np.array(
            [[999090, 20283], [1187729, 26500], [1003032, 18687], [44733854, 522926], [3786887, 64493]])
        prices = {}
        volumes = {}
        for i, price_ser in enumerate(array_prices):
            prices['INSTR_%i' % i] = pd.DataFrame(data={'close': price_ser},
                                                  index=pd.date_range('2018-01-01 01:00:00', periods=2, freq='1Min'))
            volumes['INSTR_%i' % i] = pd.DataFrame(data=array_volumes[i])
        b_t = [0.19955795, 0.20172189, 0.20055742, 0.19949981, 0.19866294]
        res = olmar_portfolio(prices, b_t, 1, volumes)
        np.testing.assert_almost_equal(res, np.array([0.26448988, 0., 0.35834384, 0.37716628, 0.]))

    def test_running_allocator(self):
        globals()["_tmp"] = False

        def dumb_allocator(rets):
            rets = column_vector(rets)
            globals()["_tmp"] = not globals()["_tmp"]
            if globals()["_tmp"]:
                return np.array([0])
            else:
                return np.array([100])

        df = pd.DataFrame(np.random.rand(25, 1))
        df.index = pd.date_range('2018-01-01 01:00:00', periods=25, freq='15Min')
        res_1 = runnig_portfolio_allocator(dumb_allocator, df, 2, 1, 'H')
        # print(res_1)
        first_hour_vals = res_1.loc['2018-01-01 03:00:00':'2018-01-01 03:59:59'].mean()[0]
        second_hour_vals = res_1.loc['2018-01-01 04:00:00':'2018-01-01 04:59:59'].mean()[0]
        third_hour_vals = res_1.loc['2018-01-01 05:00:00':'2018-01-01 05:59:59'].mean()[0]
        self.assertEqual(first_hour_vals, 0)
        self.assertEqual(second_hour_vals, 100)
        self.assertEqual(third_hour_vals, 0)


from pytest import main
if __name__ == '__main__':
    main()