import math

import numpy as np
from scipy.linalg import LinAlgError
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import adfuller

from qube.charting.plot_helpers import sbp
from qube.quantitative.tools import isscalar


def j_divergence(p, q):
    """
    Calculates J-divergence distance between data vectors p and q
    
    :param p: vector 1
    :param q: vector 2 
    :return: distance 
    """
    meanP = np.mean(p)
    meanQ = np.mean(q)

    sigmaP = np.std(p)
    sigmaQ = np.std(q)

    return 0.5 * (((meanP - meanQ) ** 2) * (1 / sigmaP + 1 / sigmaQ) + sigmaQ / sigmaP + sigmaP / sigmaQ - 2)


def __divisors(n, n0):
    # Find all divisors of the natural number N greater or equal to N0
    return [j for j in range(n0, math.floor(n / 2) + 1) if (n / j) == math.floor(n / j)]


def __rs_calc(z, n):
    # calculate (R/S)_n for given n
    m = math.floor(len(z) / n)
    Y = np.reshape(z.copy(), (m, n)).T
    E = np.mean(Y, axis=0)
    S = np.std(Y, axis=0, ddof=1)
    for i in range(0, m):
        Y[:, i] = Y[:, i] - E[i]
    Y = np.cumsum(Y, axis=0)

    # find the ranges of cummulative series
    mm = np.max(Y) - np.min(Y)

    # rescale the ranges by the standard deviations
    return np.mean(mm / S)


def hurst_rs(x, d=50, display_results=True):
    """
    Calculate the Hurst exponent using R/S analysis.
    It calculates the Hurst exponent of time series X using
    the R/S analysis of Hurst [2], corrected for small sample bias [1,3,4].
    If a vector of increasing natural numbers is given as the second input
    parameter, i.e. hurst_rs(X,D), then it defines the box sizes that the
    sample is divided into (the values in D have to be divisors of the
    length of series X). If D is a scalar (default value D = 50) it is
    treated as the smallest box size that the sample can be divided into.
    In this case the optimal sample size opt_n and the vector of divisors
    for this size are automatically computed.
    opt_n is defined as the length that possesses the most divisors among
    series shorter than X by no more than 1%. The input series X is
    truncated at the opt_n-th value.
    H,HE,HT,PV95 = hurst_rs(X) returns the uncorrected empirical and theoretical Hurst exponents and the empirical
    95% confidence intervals PV95 (see [4]).

    References:
    [1] A.A.Anis, E.H.Lloyd (1976) The expected value of the adjusted rescaled Hurst range of independent normal summands, Biometrica 63,283-298.
    [2] H.E.Hurst (1951) Long-term storage capacity of reservoirs, Transactions of the American Society of Civil Engineers 116, 770-808.
    [3] E.E.Peters (1994) Fractal Market Analysis, Wiley.
    [4] R.Weron (2002) Estimating long range dependence: finite sample properties and confidence intervals, Physica A 312, 285-299.

    :param x:
    :param d:
    :param display_results:
    :return:
    """
    if isscalar(d):
        # for scalar d set dmin=d and find the 'optimal' vector d
        dmin = d
        # find such a natural number OptN that possesses the largest number of
        # divisors among all natural numbers in the interval [0.99*N,N]
        N = len(x)
        n0 = math.floor(0.99 * N)
        dv = np.zeros(N - n0 + 1)
        for i in range(n0, N + 1):
            dv[i - n0] = len(__divisors(i + 1, dmin))

        opt_n = n0 + np.argmax(dv)
        # use the first OptN values of x for further analysis
        x = x[:opt_n + 1]
        # find the divisors of x
        d = __divisors(opt_n + 1, dmin)
    else:
        opt_n = len(x)

    N = len(d)
    rs_e = np.zeros(N)
    ers = np.zeros(N)

    # calculate empirical R/S
    for i in range(N):
        rs_e[i] = __rs_calc(x, d[i])

    for i in range(N):
        n = d[i]
        K = np.array(range(1, n))
        ratio = (n - 0.5) / n * np.sum(np.sqrt((np.ones((1, n - 1)) * n - K) / K))
        if n > 340:
            ers[i] = ratio / np.sqrt(0.5 * np.pi * n)
        else:
            ers[i] = (gamma(0.5 * (n - 1)) * ratio) / (gamma(0.5 * n) * np.sqrt(np.pi))

    # calculate the Anis-Lloyd/Peters corrected Hurst exponent
    # compute the Hurst exponent as the slope on a loglog scale
    ers_al = np.sqrt(0.5 * np.pi * np.array(d, dtype=float))
    pal = np.polyfit(np.log10(d), np.log10(rs_e - ers + ers_al), 1)
    hal = pal[0]

    # Calculate the empirical and theoretical Hurst exponents
    pe = np.polyfit(np.log10(d), np.log10(rs_e), 1)
    he = pe[0]
    p = np.polyfit(np.log10(d), np.log10(ers), 1)
    ht = p[0]

    # Compute empirical confidence intervals (see [4])
    L = np.log2(opt_n)
    # R/S-AL (min(divisor)>50) two-sided empirical confidence intervals
    pval95 = [0.5 - np.exp(-7.33 * np.log(np.log(L)) + 4.21), np.exp(-7.20 * np.log(np.log(L)) + 4.04) + 0.5];

    conf_int = np.array([
        [0.5 - np.exp(-7.35 * np.log(np.log(L)) + 4.06), np.exp(-7.07 * np.log(np.log(L)) + 3.75) + 0.5, .90],
        [*pval95, .95],
        [0.5 - np.exp(-7.19 * np.log(np.log(L)) + 4.34), np.exp(-7.51 * np.log(np.log(L)) + 4.58) + 0.5, .99]])

    if display_results:
        print('---------------------------------------------------------------')
        print('R/S-AL using %d divisors (%d, ..., %d) for a sample of %d values' % (len(d), d[0], d[-1], opt_n))
        print('Corrected theoretical Hurst exponent    %0.4f' % 0.5)
        print('Corrected empirical Hurst exponent      %0.4f' % hal)
        print('Theoretical Hurst exponent              %0.4f' % ht)
        print('Empirical Hurst exponent                %0.4f' % he)
        print('---------------------------------------------------------------')

        # Display empirical confidence intervals
        print('R/S-AL (min(divisor) > 50) two-sided empirical confidence intervals')
        print('--- conf_lo   conf_hi   level ---------------------------------')
        print(conf_int)
        print('---------------------------------------------------------------')

    return hal, he, ht, pval95


def hurst(series, max_lag=20):
    """
    Simplest Hurst exponent helps test whether the time series is:
    (1) A Random Walk (H ~ 0.5)
    (2) Trending (H > 0.5)
    (3) Mean reverting (H < 0.5)
    """
    tau, lagvec = [], []

    # Step through the different lags
    for lag in range(2, max_lag):
        # Produce price different with lag
        pp = np.subtract(series[lag:], series[:-lag])

        # Write the different lags into a vector
        lagvec.append(lag)

        # Calculate the variance of the difference
        tau.append(np.sqrt(np.std(pp)))

    # Linear fit to a double-log graph to get power
    m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)

    # Calculate hurst
    return m[0] * 2


def percentile_rank(x: np.ndarray, v, pctls=np.arange(1, 101)):
    """
    Find percentile rank of value v
    :param x: values array
    :param v: vakue to be ranked
    :param pctls: percentiles
    :return: rank

    >>> percentile_rank(np.random.randn(1000), 1.69)
    >>> 95
    >>> percentile_rank(np.random.randn(1000), 1.69, [10,50,100])
    >>> 2
    """
    return np.argmax(np.sign(np.append(np.percentile(x, pctls), np.inf) - v))


def adfuller_report(x, **kwargs):
    """
    ADF test with report
    """
    result = adfuller(x, **kwargs)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %.9f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def kde(array, cut_down=True, bw_method='scott'):
    """
    Kernel dense estimation
    """
    if cut_down:
        bins, counts = np.unique(array, return_counts=True)
        f_mean = counts.mean()
        f_above_mean = bins[counts > f_mean]
        bounds = [f_above_mean.min(), f_above_mean.max()]
        array = array[np.bitwise_and(bounds[0] < array, array < bounds[1])]
    return gaussian_kde(array, bw_method=bw_method)


def mode_estimation(array, cut_down=True, bw_method='scott'):
    """
    Returns mode from estimated empirical distribution of values
    """
    kernel = kde(array, cut_down=cut_down, bw_method=bw_method)
    height = kernel.pdf(array)
    x0 = array[np.argmax(height)]
    span = array.max() - array.min()
    dx = span / 4
    bounds = np.array([[x0 - dx, x0 + dx]])
    linear_constraint = [{'type': 'ineq', 'fun': lambda x: x - 0.5}]
    results = minimize(lambda x: -kernel(x)[0], x0=x0, bounds=bounds, constraints=linear_constraint)
    return results.x[0]


def safe_mode_estimation(array):
    if len(array) == 0:
        return np.nan

    try:
        m0 = mode_estimation(array, False)
    except (LinAlgError, ValueError):
        m0 = np.mean(array)
    return m0


def cmp_to_norm(xs, xranges=None):
    """
    Compare distribution from xs against normal using estimated mean and std
    """
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns

    _m, _s = np.mean(xs), np.std(xs)
    fit = stats.norm.pdf(sorted(xs), _m, _s)  # this is a fitting indeed

    sbp(12, 1)
    plt.plot(sorted(xs), fit, 'r--', lw=2, label='N(%.2f, %.2f)' % (_m, _s))
    plt.legend(loc='upper right')

    sns.kdeplot(xs, color='g', label='Data', shade=True)
    if xranges is not None and len(xranges) > 1:
        plt.xlim(xranges)
    plt.legend(loc='upper right')

    sbp(12, 2)
    stats.probplot(xs, dist="norm", sparams=(_m, _s), plot=plt)
