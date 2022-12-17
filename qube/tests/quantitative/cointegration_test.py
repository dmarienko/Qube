import unittest
from os.path import join

import numpy as np
import pandas as pd

from qube.quantitative.stats.cointegration import johansen
from qube.configs.Properties import get_root_dir


class CointTest(unittest.TestCase):

    def test_johansen(self):
        data = pd.read_csv(join(get_root_dir(), 'tests/quantitative/coint_test.csv'), delimiter=',', header=0)
        result = johansen(data.values, 0, 9, trace=True)

        test_1 = np.array([
            [307.689, 153.634, 159.529, 171.090],
            [205.384, 120.367, 125.618, 135.982],
            [129.133, 91.109, 95.754, 104.964],
            [83.310, 65.820, 69.819, 77.820],
            [52.520, 44.493, 47.855, 54.681],
            [30.200, 27.067, 29.796, 35.463],
            [13.842, 13.429, 15.494, 19.935],
            [0.412, 2.705, 3.841, 6.635],
        ])

        test_2 = np.array([
            [102.305, 49.285, 52.362, 58.663],
            [76.251, 43.295, 46.230, 52.307],
            [45.823, 37.279, 40.076, 45.866],
            [30.791, 31.238, 33.878, 39.369],
            [22.319, 25.124, 27.586, 32.717],
            [16.359, 18.893, 21.131, 25.865],
            [13.430, 12.297, 14.264, 18.520],
            [0.412, 2.705, 3.841, 6.635],
        ])

        eig_test = np.array([0.466148653574356, 0.373619286471005, 0.245063548771434,
                             0.172130365006779, 0.127967097275594, 0.0954883143596804,
                             0.0790888071962733, 0.00252281858123386])

        r1 = [np.array([result.lr1[i], result.cvt[i, 0], result.cvt[i, 1], result.cvt[i, 2]])
              for i in range(test_1.shape[0])]

        r2 = [np.array([result.lr2[i], result.cvm[i, 0], result.cvm[i, 1], result.cvm[i, 2]])
              for i in range(test_1.shape[0])]

        # test trace statistics
        np.testing.assert_almost_equal(r1, test_1, 3, err_msg='Trace statistics are not equal')

        # test eigen statistics
        np.testing.assert_almost_equal(r2, test_2, 3, err_msg='Eigen statistics are not equal')

        # test eigen values
        np.testing.assert_almost_equal(result.eig, eig_test, 4, err_msg='Eigen values are not equal')


from pytest import main
if __name__ == '__main__':
    main()