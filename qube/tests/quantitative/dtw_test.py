import unittest

import numpy as np
import pandas as pd

from qube.quantitative.ta.dtw import dtw_distance, dtw_window_distance, dtw_keogh_lower_bound


class DTWTest(unittest.TestCase):

    def test_dtw_dist(self):
        x = np.linspace(0, 50, 100)
        ts1 = pd.Series(3.1 * np.sin(x / 1.5) + 3.5).values
        ts2 = pd.Series(2.2 * np.sin(x / 3.5 + 2.4) + 3.2).values
        ts3 = pd.Series(0.04 * x + 3.0).values

        self.assertAlmostEqual(dtw_distance(ts1, ts2), 17.9297184686, delta=0.00001)
        self.assertAlmostEqual(dtw_distance(ts1, ts3), 21.5494948244, delta=0.00001)
        self.assertAlmostEqual(dtw_window_distance(ts1, ts2, 10), 18.5965518384, delta=0.00001)
        self.assertAlmostEqual(dtw_window_distance(ts1, ts3, 10), 22.4724828468, delta=0.00001)
        self.assertAlmostEqual(dtw_keogh_lower_bound(ts1, ts2, 20), 6.25389235159, delta=0.00001)
        self.assertAlmostEqual(dtw_keogh_lower_bound(ts1, ts3, 20), 19.9595478694, delta=0.00001)


from pytest import main
if __name__ == '__main__':
    main()