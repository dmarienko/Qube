import importlib
import unittest
from datetime import datetime, timedelta
from time import sleep

import ntplib

from qube.utils import ntp


class NtpTest(unittest.TestCase):

    def setUp(self):
        importlib.reload(ntp)  # reloading ntp

    def test_ntp(self):
        self.assertIsNone(ntp._offset)
        sleep(0.1)  # still should be None as for TEST env we start it manually
        self.assertIsNone(ntp._offset)
        ntp._controlling_thread.start()
        sleep(0.2)
        self.assertIsNotNone(ntp._offset)
        print("offset %f" % ntp._offset)
        # let's test the logic is fine
        self.assertAlmostEqual(
            ntp.get_now().timestamp(),
            (datetime.now() + timedelta(seconds=ntp.get_offset())).timestamp(),
            places=3,
        )

        # let's check get_now really match with real NTP time
        ntp_client = ntplib.NTPClient()
        response = ntp_client.request(ntp.NTP_SERVERS_LIST[0])
        self.assertAlmostEqual(ntp.get_now().timestamp(), response.tx_time, places=0)


from pytest import main

if __name__ == "__main__":
    main()
