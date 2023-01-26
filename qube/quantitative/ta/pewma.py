import numpy as np
import pandas as pd

from qube.utils.utils import mstruct, njit_optional

"""
    Implementation of probabilistic exponential weighted ma (https://sci-hub.shop/10.1109/SSP.2012.6319708)
"""


@njit_optional(cache=True)
def norm_pdf(x):
    return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)


@njit_optional(cache=True)
def lognorm_pdf(x, s):
    return np.exp(-np.log(x) ** 2 / (2 * s ** 2)) / (x * s * np.sqrt(2 * np.pi))


@njit_optional(cache=True)
def _pwma(_x, a, beta, T):
    _mean, _std, _var = np.zeros(_x.shape), np.zeros(_x.shape), np.zeros(_x.shape)
    _mean[0] = _x[0]

    for i in range(1, len(_x)):
        i_1 = i - 1
        diff = _x[i] - _mean[i_1]
        p = norm_pdf(diff / _std[i_1]) if _std[i_1] != 0 else 0  # Prob of observing diff
        a_t = a * (1 - beta * p) if i_1 > T else 1 - 1 / i  # weight to give to this point
        incr = (1 - a_t) * diff

        # Update Mean, Var, Std
        v = a_t * (_var[i_1] + diff * incr)
        _mean[i] = _mean[i_1] + incr
        _var[i] = v
        _std[i] = np.sqrt(v)
    return _mean, _var, _std


def pwma(x, alpha, beta, T):
    m, v, s = _pwma(x.values, alpha, beta, T)
    return pd.DataFrame({'m': m, 'v': v, 's': s}, index=x.index)


@njit_optional(cache=True)
def _pwma_outliers_detector(x, a, beta, T, z_th, dist):
    x0 = 0 if np.isnan(x[0]) else x[0]
    s1, s2, s1_n, std_n = x0, x0 ** 2, x0, 0

    s1a, stda, za, probs = np.zeros(x.shape), np.zeros(x.shape), np.zeros(x.shape), np.zeros(x.shape)
    uba, lba = np.zeros(x.shape), np.zeros(x.shape)
    outliers = []

    for i in range(0, len(x)):
        s1 = s1_n
        std = std_n
        xi = x[i]

        z_t = ((xi - s1) / std) if (std != 0 and not np.isnan(xi)) else 0
        ub, lb = (z_t + z_th) * std + s1, (z_t - z_th) * std + s1

        # find probability
        p = norm_pdf(z_t)
        a_t = a * (1 - beta * p) if i + 1 >= T else 1 - 1 / (i + 1)

        # Update Mean, Var, Std
        if not np.isnan(xi):
            s1 = a_t * s1 + (1 - a_t) * xi
            s2 = a_t * s2 + (1 - a_t) * xi ** 2
            s1_n = s1
            std_n = np.sqrt(abs(s2 - np.square(s1)))

        # detects outlier
        if abs(z_t) >= z_th:
            outliers.append(i)

        s1a[i] = s1_n
        stda[i] = std_n
        probs[i] = p
        za[i] = z_t

        # upper and lower boundaries
        ub, lb = s1_n + z_th * std_n, s1_n - z_th * std_n
        uba[i] = ub
        lba[i] = lb
        # print('[%d] %.3f  -> s1_n: %.3f  s1: %.3f Z: %.3f s: %.3f s_n: %.3f' % (i, x[i], s1_n, s1, z_t, std, std_n))
    return s1a, stda, probs, za, uba, lba, outliers


def pwma_outliers_detector(x, alpha, beta, T=30, threshold=0.05, dist='norm'):
    import scipy.stats
    z_thr = scipy.stats.norm.ppf(1 - threshold / 2)
    m, s, p, z, u, l, oi = _pwma_outliers_detector(x.values, alpha, beta, T, z_thr, dist)
    res = pd.DataFrame({'m': m, 's': s, 'z': z, 'u': u, 'l': l, 'p': p}, index=x.index)

    return mstruct(
        m=res.m, s=res.s, z=res.z, u=res.u, l=res.l, p=res.p,
        outliers=x.iloc[oi] if len(oi) else None,
        z_bounds=(z_thr, -z_thr)
    )

# z = (x - m) / s
# upper = (z + z_th) * s + m 
#

# --- test ---
# V = pd.Series([1.603, 1.238, 1.492, 1.742, 1.58, 1.335, 1.353, 1.356, 1.992, 1.705, 2.008, 1.228, 1.484, 1.071, 1.537, 1.843, 3.883, 3.177, 2.461, 2.243, 2.312, 2.597, 3.513, 3.638, 3.737, 3.753, 3.795, 3.987, 4.157, 3.641, 2.884, 2.576, 2.36, 2.557, 4.088, 4.296, 4.0, 4.365, 4.0, 4.168, 4.647, 5.087, 5.172, 5.326, 5.47, 6.125, 5.009, 5.329])
# m, o = pwma_outliers_detector(V, 0.95, 0.5, 30, 0.05)
# pd.concat((V, m[['m','s']]), axis=1).tail()
# print(o)
# 8     1.992
# 10    2.008
# 16    3.883
# 17    3.177
# 22    3.513
# 23    3.638
# 24    3.737
# 34    4.088
# 35    4.296
# dtype: float64
