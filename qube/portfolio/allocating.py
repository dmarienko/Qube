from typing import Union, Callable
from collections import OrderedDict

import numpy as np
import pandas as pd

from qube.quantitative.tools import column_vector
from qube.simulator.utils import rolling_forward_test_split


def tang_portfolio(rets: Union[pd.DataFrame, pd.Series, np.ndarray], risk_free=0.01):
    """
    Calculate tangential portfolio weights

    :param rets: assets returns in columns
    :param risk_free: risk-free rate
    :return: weight coefficient for every asset
    """
    df_columns = rets.columns.tolist() if isinstance(rets, pd.DataFrame) else None
    rets = column_vector(rets)
    n_cols = rets.shape[1]

    cov_mx = np.cov(rets.T)
    means = np.array([np.mean(rets, axis=0)])

    i_cov_mx = np.linalg.inv(cov_mx)
    top = i_cov_mx @ (means - risk_free * np.ones((1, n_cols))).T
    bottom = np.ones((1, n_cols)) @ i_cov_mx @ (means.T - risk_free * np.ones((n_cols, 1)))
    tang = (top / bottom.T).T
    
    if df_columns:
        return pd.DataFrame(tang, columns=df_columns)

    return tang.flatten()


def gmv_portfolio(rets: Union[pd.DataFrame, pd.Series, np.ndarray]):
    """
    Global minimum variance portfolio weights

    :param rets: assets returns in columns
    :return: weight coefficient for every asset
    """
    df_columns = rets.columns.tolist() if isinstance(rets, pd.DataFrame) else None
    rets = column_vector(rets)
    n_cols = rets.shape[1]

    cov_mx = np.cov(rets.T)
    am = np.ones((n_cols + 1, n_cols + 1))
    am[0:n_cols, 0:n_cols] = 2 * cov_mx
    am[n_cols, n_cols] = 0
    b = np.zeros((1, n_cols + 1)).T
    b[n_cols, 0] = 1

    zm = np.linalg.inv(am) @ b
    gmv = zm[:n_cols].T
    
    if df_columns:
        return pd.DataFrame(gmv, columns=df_columns)
    
    return gmv


def effective_portfolio(rets: Union[pd.DataFrame, pd.Series, np.ndarray], expected_mean=0):
    """
    Effective portfolio weights

    :param rets: assets returns in columns
    :param expected_mean: expected mean return of portfolio
    :return: weight coefficient for every asset
    """
    df_columns = rets.columns.tolist() if isinstance(rets, pd.DataFrame) else None
    rets = column_vector(rets)
    n_cols = rets.shape[1]
    cov_mx = np.cov(rets.T)
    am = np.ones((n_cols + 2, n_cols + 2))

    am[0:n_cols, 0:n_cols] = 2 * cov_mx
    means = np.mean(rets, axis=0)

    am[:-2, n_cols] = means
    am[n_cols, :-2] = means

    am[-2:, -2:] = 0
    b = np.zeros((1, n_cols + 2)).T
    b[n_cols, 0] = expected_mean
    b[n_cols + 1, 0] = 1

    zm = np.linalg.inv(am) @ b
    eff = zm[:n_cols].T

    if df_columns:
        return pd.DataFrame(eff, columns=df_columns)

    return eff


def simplex_projection(v, b=1):
    """Projection vectors to the simplex domain

    Implemented according to the paper: Efficient projections onto the
    l1-ball for learning in high dimensions, John Duchi, et al. ICML 2008.
    Implementation Time: 2011 June 17 by Bin@libin AT pmail.ntu.edu.sg
    Optimization Problem: min_{w}\| w - v \|_{2}^{2}
    s.t. sum_{i=1}^{m}=z, w_{i}\geq 0

    Input: A vector v \in R^{m}, and a scalar z > 0 (default=1)
    Output: Projection vector w

    :Example:
    >>> proj = simplex_projection([.4 ,.3, -.4, .5])
    >>> print proj
    array([ 0.33333333, 0.23333333, 0. , 0.43333333])
    >>> print proj.sum()
    1.0

    Original matlab implementation: John Duchi (jduchi@cs.berkeley.edu)
    Python-port: Copyright 2012 by Thomas Wiecki (thomas.wiecki@gmail.com).
    """

    v = np.asarray(v)
    p = len(v)

    # Sort v into u in descending order
    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)

    rho = np.where(u > (sv - b) / np.arange(1, p+1))[0][-1]
    theta = np.max([0, (sv[rho] - b) / (rho+1)])
    w = (v - theta)
    w[w < 0] = 0
    return w


def olmar_portfolio(prices: Union[dict, OrderedDict], b_t: Union[list, np.ndarray, pd.DataFrame],
                    eps=1, volumes: Union[dict, OrderedDict]=None):
    """
    :param prices: dict or OrderedDict with ohlc DataFrames
    :param b_t: list, np.ndarray, pd.DataFrame for each every price series
    :param eps: olmar epsilon
    :param volumes: volumes for price series
    :return: olmar values each price series
    """
    x_tilde = np.zeros(len(prices))
    for i, instr in enumerate(prices):
        instr_price = prices[instr].close
        if volumes and instr in volumes:
            vwa_price = np.dot(instr_price, volumes[instr]) / np.sum(volumes[instr])
        else:
            vwa_price = np.mean(instr_price)
        x_tilde[i] = vwa_price/instr_price.iloc[-1]

    ###########################
    # Inside of OLMAR (algo 2)
    x_bar = x_tilde.mean()

    # Calculate terms for lambda (lam)
    dot_prod = np.dot(b_t, x_tilde)
    num = eps - dot_prod
    denom = (np.linalg.norm((x_tilde-x_bar)))**2

    # test for divide-by-zero case
    if denom == 0.0:
        lam = 0 # no portolio update
    else:
        lam = max(0, num/denom)
    b = b_t + lam*(x_tilde-x_bar)
    b_norm = simplex_projection(b)
    return b_norm


def runnig_portfolio_allocator(allocator: Callable, rets: pd.DataFrame,
                               hist_period: int, work_period: int, units='D', **kwargs):
    """
    Portflio allocator on running window basis.

    >>> rets

                         EUR        CHF        NOK              DKK             JPY
    time
    2016-01-03 22:10:00	-0.000350	-0.000015	-1.899417e-04	-4.572062e-05	-3.113275e-07
    2016-01-03 22:15:00	-0.000010	0.000160	1.152844e-05	-4.665920e-06	-7.263403e-07
    ...

    >>> w = runnig_portfolio_allocator(effective_portfolio, rets, 4, 1, 'W')
    >>> pnl = (w * rets).sum(axis=1).cumsum()

    :param allocator: allocating function
    :param rets: returns as dataframe
    :param hist_period: historical period used for calculation
    :param work_period: period when allocation is applied
    :param units: period units ('H', 'D', 'W', 'M', 'Q', 'Y') default 'D' - daily
    :param kwargs: allocator function arguments (if need)
    :return: running weights as dataframe object
    """
    weights = []
    for hist_idx, work_idx in rolling_forward_test_split(rets, hist_period, work_period, units=units):
        hist_r = rets.loc[hist_idx]
        w = allocator(hist_r, **kwargs)
        weights.append(pd.DataFrame(np.repeat(column_vector(w), len(work_idx), axis=0), index=work_idx, columns=rets.columns))
        
    return pd.concat(weights, axis=0)