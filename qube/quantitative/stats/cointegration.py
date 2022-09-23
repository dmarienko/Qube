import numpy as np
import statsmodels.tsa.tsatools as tsat
from numpy import diff as tdiff
from numpy import zeros, ones, flipud, log
from numpy.linalg import inv, eig, cholesky as chol

from qube.quantitative.ta.indicators import detrend

"""
   Johansen cointegration ported from matlab (econometrix library)
   Also there is another python implementation https://github.com/iisayoo/johansen
"""

__ejcp0 = np.array([
    [2.9762, 4.1296, 6.9406],
    [9.4748, 11.2246, 15.0923],
    [15.7175, 17.7961, 22.2519],
    [21.8370, 24.1592, 29.0609],
    [27.9160, 30.4428, 35.7359],
    [33.9271, 36.6301, 42.2333],
    [39.9085, 42.7679, 48.6606],
    [45.8930, 48.8795, 55.0335],
    [51.8528, 54.9629, 61.3449],
    [57.7954, 61.0404, 67.6415],
    [63.7248, 67.0756, 73.8856],
    [69.6513, 73.0946, 80.0937]])

__ejcp1 = np.array([
    [2.7055, 3.8415, 6.6349],
    [12.2971, 14.2639, 18.5200],
    [18.8928, 21.1314, 25.8650],
    [25.1236, 27.5858, 32.7172],
    [31.2379, 33.8777, 39.3693],
    [37.2786, 40.0763, 45.8662],
    [43.2947, 46.2299, 52.3069],
    [49.2855, 52.3622, 58.6634],
    [55.2412, 58.4332, 64.9960],
    [61.2041, 64.5040, 71.2525],
    [67.1307, 70.5392, 77.4877],
    [73.0563, 76.5734, 83.7105]])

__ejcp2 = np.array([
    [2.7055, 3.8415, 6.6349],
    [15.0006, 17.1481, 21.7465],
    [21.8731, 24.2522, 29.2631],
    [28.2398, 30.8151, 36.1930],
    [34.4202, 37.1646, 42.8612],
    [40.5244, 43.4183, 49.4095],
    [46.5583, 49.5875, 55.8171],
    [52.5858, 55.7302, 62.1741],
    [58.5316, 61.8051, 68.5030],
    [64.5292, 67.9040, 74.7434],
    [70.4630, 73.9355, 81.0678],
    [76.4081, 79.9878, 87.2395]])


def c_sja(n, p):
    """
    Find critical values for Johansen trace statistic
    ------------------------------------------------------------
    USAGE:  jc = c_sjt(n,p)
    where:    n = dimension of the VAR system
                  NOTE: routine doesn't work for n > 12
              p = order of time polynomial in the null-hypothesis
                    p = -1, no deterministic part
                    p =  0, for constant term
                    p =  1, for constant plus time-trend
                    p >  1  returns no critical values
    ------------------------------------------------------------
    RETURNS: a (3x1) vector of percentiles for the trace
             statistic for [90% 95% 99%]
    ------------------------------------------------------------
    NOTES: for n > 12, the function returns a (3x1) vector of zeros.
           The values returned by the function were generated using
           a method described in MacKinnon (1996), using his FORTRAN
           program johdist.f
    ------------------------------------------------------------
    SEE ALSO: johansen()
    ------------------------------------------------------------
    % References: MacKinnon, Haug, Michelis (1996) 'Numerical distribution
    functions of likelihood ratio tests for cointegration',
    Queen's University Institute for Economic Research Discussion paper.
    -------------------------------------------------------

    written by:
    James P. LeSage, Dept of Economics
    University of Toledo
    2801 W. Bancroft St,
    Toledo, OH 43606
    jlesage@spatial-econometrics.com

    these are the values from Johansen's 1995 book
    for comparison to the MacKinnon values
    jcp0 = [ 2.98   4.14   7.02
            10.35  12.21  16.16
            21.58  24.08  29.19
            36.58  39.71  46.00
            55.54  59.24  66.71
            78.30  86.36  91.12
           104.93 109.93 119.58
           135.16 140.74 151.70
           169.30 175.47 187.82
           207.21 214.07 226.95
           248.77 256.23 270.47
           293.83 301.95 318.14];
    """
    if (p > 1) or (p < -1):
        jc = np.zeros(3)
    elif (n > 12) or (n < 1):
        jc = np.zeros(3)
    elif p == -1:
        jc = __ejcp0[n - 1, :]
    elif p == 0:
        jc = __ejcp1[n - 1, :]
    elif p == 1:
        jc = __ejcp2[n - 1, :]
    else:
        raise ValueError('invalid p')
    return jc


__tjcp0 = np.array([
    [2.9762, 4.1296, 6.9406],
    [10.4741, 12.3212, 16.3640],
    [21.7781, 24.2761, 29.5147],
    [37.0339, 40.1749, 46.5716],
    [56.2839, 60.0627, 67.6367],
    [79.5329, 83.9383, 92.7136],
    [106.7351, 111.7797, 121.7375],
    [137.9954, 143.6691, 154.7977],
    [173.2292, 179.5199, 191.8122],
    [212.4721, 219.4051, 232.8291],
    [255.6732, 263.2603, 277.9962],
    [302.9054, 311.1288, 326.9716]])

__tjcp1 = np.array([
    [2.7055, 3.8415, 6.6349],
    [13.4294, 15.4943, 19.9349],
    [27.0669, 29.7961, 35.4628],
    [44.4929, 47.8545, 54.6815],
    [65.8202, 69.8189, 77.8202],
    [91.1090, 95.7542, 104.9637],
    [120.3673, 125.6185, 135.9825],
    [153.6341, 159.5290, 171.0905],
    [190.8714, 197.3772, 210.0366],
    [232.1030, 239.2468, 253.2526],
    [277.3740, 285.1402, 300.2821],
    [326.5354, 334.9795, 351.2150]])

__tjcp2 = np.array([
    [2.7055, 3.8415, 6.6349],
    [16.1619, 18.3985, 23.1485],
    [32.0645, 35.0116, 41.0815],
    [51.6492, 55.2459, 62.5202],
    [75.1027, 79.3422, 87.7748],
    [102.4674, 107.3429, 116.9829],
    [133.7852, 139.2780, 150.0778],
    [169.0618, 175.1584, 187.1891],
    [208.3582, 215.1268, 228.2226],
    [251.6293, 259.0267, 273.3838],
    [298.8836, 306.8988, 322.4264],
    [350.1125, 358.7190, 375.3203]])


def c_sjt(n, p):
    if (p > 1) or (p < -1):
        jc = np.zeros(3)
    elif (n > 12) or (n < 1):
        jc = np.zeros(3)
    elif p == -1:
        jc = __tjcp0[n - 1, :]
    elif p == 0:
        jc = __tjcp1[n - 1, :]
    elif p == 1:
        jc = __tjcp2[n - 1, :]
    else:
        raise ValueError('invalid p')
    return jc


def johansen(x, p, k, trace=False):
    """
    Performs Johansen cointegration tests
    
    USAGE: result = johansen(x, p, k)
    where:      x = input matrix of time-series in levels, (nobs x m)
                p = order of time polynomial in the null-hypothesis
                    p = -1, no deterministic part
                    p =  0, for constant term
                    p =  1, for constant plus time-trend
                    p >  1, for higher order polynomial
                k = number of lagged difference terms used when
                    computing the estimator
    
    RETURNS: a results structure:
             result.eig  = eigenvalues  (m x 1)
             result.evec = eigenvectors (m x m), where first
                           r columns are normalized coint vectors
             result.lr1  = likelihood ratio trace statistic for r=0 to m-1
                           (m x 1) vector
             result.lr2  = maximum eigenvalue statistic for r=0 to m-1
                           (m x 1) vector
             result.cvt  = critical values for trace statistic
                           (m x 3) vector [90% 95% 99%]
             result.cvm  = critical values for max eigen value statistic
                           (m x 3) vector [90% 95% 99%]
             result.ind  = index of co-integrating variables ordered by
                           size of the eigenvalues from large to small
    -------------------------------------------------------
    NOTE: c_sja(), c_sjt() provide critical values generated using
          a method of MacKinnon (1994, 1996).
          critical values are available for n<=12 and -1 <= p <= 1,
          zeros are returned for other cases.
    -------------------------------------------------------
    SEE ALSO: prt_coint, a function that prints results
    -------------------------------------------------------
    References: Johansen (1988), 'Statistical Analysis of Co-integration
    vectors', Journal of Economic Dynamics and Control, 12, pp. 231-254.
    MacKinnon, Haug, Michelis (1996) 'Numerical distribution
    functions of likelihood ratio tests for cointegration',
    Queen's University Institute for Economic Research Discussion paper.
    (see also: MacKinnon's JBES 1994 article
    -------------------------------------------------------

    written by:
    James P. LeSage, Dept of Economics
    University of Toledo
    2801 W. Bancroft St,
    Toledo, OH 43606
    jlesage@spatial-econometrics.com
    """

    def __rows(x):
        return x.shape[0]

    def __trimr(x, front, end):
        return x[front:-end] if end > 0 else x[front:]

    def __resid(y, x):
        return y - np.dot(x, np.dot(np.linalg.pinv(x), y))

    nobs, m = x.shape

    # why this? f is detrend transformed series, p is detrend data
    if p > -1:
        f = 0
    else:
        f = p

    x = detrend(x, p)
    dx = tdiff(x, 1, axis=0)
    z = tsat.lagmat(dx, k)  # [k-1:]
    z = __trimr(z, k, 0)
    z = detrend(z, f)
    dx = __trimr(dx, k, 0)

    dx = detrend(dx, f)
    r0t = __resid(dx, z)  # diff on lagged diffs
    lx = x[:-k]
    lx = __trimr(lx, 1, 0)
    dx = detrend(lx, f)
    rkt = __resid(dx, z)  # level on lagged diffs
    skk = np.dot(rkt.T, rkt) / __rows(rkt)
    sk0 = np.dot(rkt.T, r0t) / __rows(rkt)
    s00 = np.dot(r0t.T, r0t) / __rows(r0t)
    sig = np.dot(sk0, np.dot(inv(s00), sk0.T))
    tmp = inv(skk)
    au, du = eig(np.dot(tmp, sig))  # au is eval, du is evec

    # Normalize the eigen vectors such that (du'skk*du) = I
    temp = inv(chol(np.dot(du.T, np.dot(skk, du))))
    dt = np.dot(du, temp)

    # JP: the next part can be done much  easier
    #      NOTE: At this point, the eigenvectors are aligned by column. To
    #            physically move the column elements using the MATLAB sort,
    #            take the transpose to put the eigenvectors across the row
    # dt = transpose(dt)

    # sort eigenvalues and vectors
    auind = np.argsort(au)
    aind = flipud(auind)
    a = au[aind]
    d = dt[:, aind]

    # NOTE: The eigenvectors have been sorted by row based on auind and moved to array "d".
    #       Put the eigenvectors back in column format after the sort by taking the
    #       transpose of "d". Since the eigenvectors have been physically moved, there is
    #       no need for aind at all. To preserve existing programming, aind is reset back to
    #       1, 2, 3, ....
    # d  =  transpose(d)
    # test = np.dot(transpose(d), np.dot(skk, d))

    # EXPLANATION:  The MATLAB sort function sorts from low to high. The flip realigns
    # auind to go from the largest to the smallest eigenvalue (now aind). The original procedure
    # physically moved the rows of dt (to d) based on the alignment in aind and then used
    # aind as a column index to address the eigenvectors from high to low. This is a double
    # sort. If you wanted to extract the eigenvector corresponding to the largest eigenvalue by,
    # using aind as a reference, you would get the correct eigenvector, but with sorted
    # coefficients and, therefore, any follow-on calculation would seem to be in error.
    # If alternative programming methods are used to evaluate the eigenvalues, e.g. Frame method
    # followed by a root extraction on the characteristic equation, then the roots can be
    # quickly sorted. One by one, the corresponding eigenvectors can be generated. The resultant
    # array can be operated on using the Cholesky transformation, which enables a unit
    # diagonalization of skk. But nowhere along the way are the coefficients within the
    # eigenvector array ever changed. The final value of the "beta" array using either method
    # should be the same.

    # Compute the trace and max eigenvalue statistics
    lr1 = zeros(m)
    lr2 = zeros(m)
    cvm = zeros((m, 3))
    cvt = zeros((m, 3))
    iota = ones(m)
    t, junk = rkt.shape
    for i in range(0, m):
        tmp = __trimr(log(iota - a), i, 0)
        lr1[i] = -t * np.sum(tmp, 0)  # columnsum ?
        lr2[i] = -t * log(1 - a[i])
        cvm[i, :] = c_sja(m - i, p)
        cvt[i, :] = c_sjt(m - i, p)
        aind[i] = i

    class ResultHolder(object):
        pass

    result = ResultHolder()

    # set up results structure
    result.rkt = rkt
    result.r0t = r0t
    result.eig = a
    result.evec = d  # transposed compared to matlab ?
    result.lr1 = lr1
    result.lr2 = lr2
    result.cvt = cvt
    result.cvm = cvm
    result.ind = aind
    result.meth = 'johansen'

    if trace:
        # print('--( Trace Statistics )------------------------------------------------')
        # print('variable statistic Crit-90% Crit-95%  Crit-99%')
        # for i in range(len(result.lr1)):
        #     print('r =', i, '\t', result.lr1[i], result.cvt[i, 0], result.cvt[i, 1], result.cvt[i, 2])
        # print('--(Eigen Statistics )-------------------------------------------------')
        # print('variable statistic Crit-90% Crit-95%  Crit-99%')
        # for i in range(len(result.lr2)):
        #     print('r =', i, '\t', result.lr2[i], result.cvm[i, 0], result.cvm[i, 1], result.cvm[i, 2])
        # print('--------------------------------------------------')
        # print('eigenvectors:\n', result.evec)
        # print('--------------------------------------------------')
        # print('eigenvalues:\n', result.eig)
        # print('--------------------------------------------------')
        import pandas as pd

        n_stats = len(result.lr1)
        __columns = ['Statistics', 'Crit-90%', 'Crit-95%', 'Crit-99%']
        s00 = pd.DataFrame(index=range(n_stats), columns=__columns)
        for i in range(n_stats):
            s00.iloc[i] = [result.lr1[i], result.cvt[i, 0], result.cvt[i, 1], result.cvt[i, 2]]

        n_stats = len(result.lr2)
        s01 = pd.DataFrame(index=range(n_stats), columns=__columns)
        for i in range(n_stats):
            s01.iloc[i] = [result.lr2[i], result.cvm[i, 0], result.cvm[i, 1], result.cvm[i, 2]]
        s0 = pd.concat((s00, s01), keys=['Trace Statistics', 'Eigen Statistics'])
        s0.index = s0.index.rename(['Statistics', 'Variable'])
        print('\n', s0)

        print('\n\t--- Eigenvectors ---\n')
        print(pd.DataFrame(data=result.evec, columns=['v%d' % x for x in range(len(result.evec))]))

        print('\n\t--- Eigenvalues ---\n')
        print(pd.Series({i: x for i, x in enumerate(result.eig)}, name='Eigenvalues'))

    return result
