import numpy as np
import pandas as pd
from numba import float64
from numba.experimental import jitclass

from qube.quantitative.tools import nans, isscalar, add_constant, column_vector
from qube.utils.utils import njit_optional


def kalman_regression_estimator(x, y, vb, vm, intercept=True):
    """
    Kalman filter for estimating coefficients for linear regression: y ~ x
    
    y_hat = beta_0 + beta_1 * x
    
    it returns:
    betas estimations
    error = y - y_hat
    Q - error's variation    
    
    :param x: dependent variable (also can be matrix NxM where each column presents separate regressor)
              number of rows (N) must be equal to number of elements in y
    :param y: target variable
    :param vb: state variance 
    :param vm: measurement variance
    :param intercept: estimates intercept term if true 
    :return: betas, error, error variation
    """
    x = x.values if isinstance(x, pd.Series) else x
    y = y.values if isinstance(y, pd.Series) else y

    if intercept:
        x_ext_const = add_constant(x, 1.0, prepend=False)
    else:
        x_ext_const = column_vector(x)

    n_regressor_series = x_ext_const.shape[1]
    y_len = len(y)

    if isscalar(vm):
        vm = np.repeat(vm, y_len)
    elif len(vm) != y_len:
        raise ValueError('if Vm is vector it must have same size as y')

    R = np.zeros((n_regressor_series, n_regressor_series))
    P = np.zeros((n_regressor_series, n_regressor_series))
    betas = nans((n_regressor_series, x.shape[0]))
    y_hat = nans(y_len)
    error = nans(y_len)
    Q = nans(y_len)

    # first estimate for beta is p1/p2
    if intercept:
        betas[:, 0] = np.hstack((y[0] / x[0], 0))
    else:
        betas[:, 0] = y[0] / x[0]

    for t in range(y_len):
        if t > 0:
            betas[:, t] = betas[:, t - 1]
            R = P + vb

        y_hat[t] = x_ext_const[t, :].dot(betas[:, t])
        error[t] = y[t] - y_hat[t]

        Q[t] = x_ext_const[t, :].dot(R.dot(x_ext_const[t, :].T)) + vm[t]
        K = R.dot(x_ext_const[t, :]).T / Q[t]
        betas[:, t] = betas[:, t] + K * error[t]
        P = R - K * x_ext_const[t, :] * R

    return betas, error, Q


spec = [
    ('process_variance', float64),
    ('estimated_measurement_variance', float64),
    ('posteri_estimate', float64),
    ('posteri_error_estimate', float64),
    ('pp', float64)
]


@jitclass(spec)
class KalmanFilterSmoother:
    def __init__(self, process_variance, estimated_measurement_variance):
        self.process_variance = process_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = np.nan
        self.posteri_error_estimate = 1

    def update(self, measurement):
        if np.isnan(self.posteri_estimate):
            self.posteri_estimate = measurement

        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        K = priori_error_estimate / (priori_error_estimate + self.estimated_measurement_variance)
        self.posteri_estimate = priori_estimate + K * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - K) * priori_error_estimate
        self.pp = priori_error_estimate

    def estimation(self):
        return self.posteri_estimate

    def err_estimation(self):
        return self.posteri_error_estimate


def kf_smoother(x, pvar, mvar):
    """
    Smoothing Kalman Filter

    :param x: input series
    :param pvar: process variation
    :param mvar: measurement variation
    :return: smoothed series
    """
    kf = KalmanFilterSmoother(pvar, mvar)
    x = column_vector(x).flatten()
    return calc_kf_smoother(x, kf)


@njit_optional
def calc_kf_smoother(x, kf):
    nn = len(x)
    x_smoothed = np.zeros(nn)
    covar = np.zeros(nn)
    for i in range(nn):
        kf.update(x[i])
        x_smoothed[i] = kf.estimation()
        covar[i] = kf.err_estimation()
    return x_smoothed, covar
