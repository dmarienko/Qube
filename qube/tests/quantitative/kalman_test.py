import unittest

import numpy as np

from qube.quantitative.ta.kalman import kalman_regression_estimator


class Kalman_LR_Estimator(unittest.TestCase):

    def test_lr_kalman(self):
        b, e, q = kalman_regression_estimator(np.array([1, 2, 3, 4, 5]),
                                              np.array([1, 2, 3, 4, 5]),
                                              1, 1, True)
        np.testing.assert_almost_equal(b,
                                       np.array([[1, 1, 1, 1, 1],
                                                 [0., 0., 0., 0., 0.]]),
                                       decimal=3)

        np.testing.assert_almost_equal(q,
                                       np.array([1., 10., 24.6, 40.31504065, 57.2563352]),
                                       decimal=3)

        b, e, q = kalman_regression_estimator(np.array([1, 2, 3, 4, 5]),
                                              np.array([1, 2, 3, 4, 5]),
                                              1, 1, False)

        np.testing.assert_almost_equal(b,
                                       np.array([[1., 1., 1., 1., 1.]]),
                                       decimal=3)

        b, e, q = kalman_regression_estimator(np.array([[2, 2], [3, 3], [4, 4], [5, 5]]),
                                              np.array([2, 3, 4, 5]),
                                              1, 1, False)

        np.testing.assert_almost_equal(b,
                                       np.array([[1., 0.51351351, 0.50013808, 0.50000078],
                                                 [1., 0.51351351, 0.50013808, 0.50000078]]),
                                       decimal=3)


from pytest import main
if __name__ == '__main__':
    main()