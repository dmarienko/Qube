import numpy as np
import pandas as pd

from qube.quantitative.tools import column_vector, nans
from qube.utils.utils import mstruct, njit_optional, jit_optional


@njit_optional
def __fast_ols(x, y):
    n = len(x)
    p, _, _, _ = np.linalg.lstsq(x, y, rcond=-1)
    r2 = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) ** 2 / (
                (n * np.sum(x ** 2) - np.sum(x) ** 2) * (n * np.sum(y ** 2) - np.sum(y) ** 2))
    return p[0][0], p[1][0], r2


def fast_ols(x, y):
    b, c, r2 = __fast_ols(column_vector(x), column_vector(y))
    return mstruct(const=c, beta=b, r2=r2)


@jit_optional
def fast_alpha(x, order=1, factor=10, min_threshold=1e-10):
    """
    Returns alpha based on following calculations:
    
    alpha = exp(-F*(1 - R2))
    
    where R2 - r squared metric from regression of x data against straight line y = x
    """
    x = x[~np.isnan(x)]
    x_max, x_min = np.max(x), np.min(x)

    if x_max - x_min > min_threshold:
        yy = 2 * (x - x_min) / (x_max - x_min) - 1
        xx = np.vander(np.linspace(-1, 1, len(yy)), order + 1)
        slope, intercept, r2 = __fast_ols(xx, yy.reshape(-1, 1))
    else:
        slope, intercept, r2 = np.inf, 0, 0

    return np.exp(-factor * (1 - r2)), r2, slope, intercept


@jit_optional
def __rolling_slope(x, period, alpha_factor):
    ri = nans((len(x), 2))

    for i in range(period, x.shape[0]):
        a, r2, s, _ = fast_alpha(x[i - period:i], factor=alpha_factor)
        ri[i, :] = [r2, s]

    return ri


def rolling_slope(x, period, alpha_factor=10):
    """
    Calculates slope/R2 on rolling basis for series from x 
    returns DataFrame with 2 columns: R2, Slope
    """
    return pd.DataFrame(__rolling_slope(column_vector(x), period, alpha_factor), index=x.index, columns=['R2', 'Slope'])
