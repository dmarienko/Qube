import threading
import traceback
from datetime import datetime, timedelta
from time import sleep

import ntplib

from qube.utils import QubeLogger
from qube.configs import Properties

NTP_SERVERS_LIST = ["time.google.com", "pool.ntp.org", "time.windows.com"]

__CORRECT_EVERY_SECONDS = 60 * 5
__SLEEP_CORRECT_THREAD = 10

_offset = None  # never use it explicitly but for tests! Always use get_offset()
__logger = QubeLogger.getLogger(__name__)


def __correct_offset():
    global _offset
    ntp_client = ntplib.NTPClient()
    for ntp_url in NTP_SERVERS_LIST:
        try:
            response = ntp_client.request(ntp_url)
            _offset = response.offset
            return
        except:
            __logger.warn('%s NTP server request exception %s:' % (ntp_url, traceback.format_exc()))
    __logger.error('Unable to get ntp offset from neither of NTP servers list %s', NTP_SERVERS_LIST)


def __correct_offset_runnable():
    __logger.info('NTP offset controller thread is started')
    last_corrected_dt = None
    while True:
        if last_corrected_dt is None or datetime.now() - last_corrected_dt > timedelta(
                seconds=__CORRECT_EVERY_SECONDS):  # correct every 5 mins
            __correct_offset()
            last_corrected_dt = datetime.now()
        sleep(__SLEEP_CORRECT_THREAD)  # sleep 10 seconds


_controlling_thread = threading.Thread(target=__correct_offset_runnable, daemon=True)
if Properties.get_env() != Properties.TEST_ENV:
    _controlling_thread.start()


def get_now(tz: datetime.tzinfo = None):
    return datetime.now(tz) + timedelta(seconds=get_offset())


def get_offset():
    global _offset
    if _offset is None:
        __correct_offset()
        if _offset is None:  # if something really went wrong
            __logger.warn('Unable to get ntp offset value. Very unexpected!')
            _offset = 0.0
    return _offset
