import unittest

import numpy as np
import pandas as pd

from qube.portfolio.drawdown import absmaxdd, dd_freq_stats


class AbsMaxDDtest(unittest.TestCase):

    # test np array
    def testNp(self):
        res = absmaxdd(np.array([1, 2, 3, 1]))
        self.assertEqual(res[0], 2)
        self.assertEqual(res[1], 2)
        self.assertEqual(res[3], 3)

    def testIndexes(self):
        x = [1, 2, 3, 1]
        mdd, i0, i1, i2, dd = absmaxdd(x)
        self.assertEqual(x[i2], 1)

    # test pd series
    def testPd(self):
        rng = pd.date_range('1/1/2011', periods=6, freq='H')
        ts = pd.Series([1, 2, 3, 1, 8, 11], index=rng)
        res = absmaxdd(ts)
        self.assertEqual(res[0], 2)
        self.assertEqual(res[1], 2)
        self.assertEqual(res[3], 4)
        self.assertEqual(str(res[4].index[1]), '2011-01-01 01:00:00')

    # test increasing
    def testIncreasing(self):
        res = absmaxdd((np.array([1, 2, 3, 4])))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], 0)

    # test Lowering
    def testLowering(self):
        res = absmaxdd((np.array([10, 9, 5, 4])))
        self.assertEqual(res[0], 6)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[3], 3)

    # test incorrect data
    def testIncorrect(self):
        try:
            res = absmaxdd('string')
        except TypeError as ex:
            cur_exception = ex
        self.assertEqual(type(cur_exception), TypeError)
        cur_exception = None

        try:
            res = absmaxdd(10)
        except TypeError as ex:
            cur_exception = ex
        self.assertEqual(type(cur_exception), TypeError)

    # test pd series
    def testDD_stat(self):
        x = [1, 2, 3, 1, 8, 11, 6, 10, 12, 11, 10, 100]
        _, _, _, _, ds = absmaxdd(pd.Series(x, index=pd.date_range('1/1/2011', periods=len(x), freq='H')))
        dd_stat = dd_freq_stats(ds)
        print(dd_stat)

        # None data
        self.assertIsNone(dd_freq_stats([0]))


from pytest import main
if __name__ == '__main__':
    main()