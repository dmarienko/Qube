from calendar import timegm
from datetime import datetime, timedelta, time, date

import pytz


class DateUtils:
    DEFAULT_TIME_ZONE = pytz.timezone('UTC')
    DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    DEFAULT_DATETIME_FORMAT_MSEC = "%Y-%m-%d %H:%M:%S.%fms"
    DEFAULT_DATETIME_FORMAT_MCSEC = "%Y-%m-%d %H:%M:%S.%fmcs"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d"
    KDB_DATE_FORMAT = "%Y.%m.%d"
    KDB_DATETIME_FORMAT = "%Y.%m.%dD%H:%M:%S"
    KDB_DATETIME_FORMAT_MCSEC = "%Y.%m.%dD%H:%M:%S.%fmcs"

    _ALL_DATE_PATTERNS = None

    MIN_DATE = datetime(1900, 1, 1)

    def __init__(self):
        pass

    @staticmethod
    def int_to_datetime(time: int):
        """
        :param time: long java time format
        :return: datetime
        """
        return datetime.utcfromtimestamp(time / 1e3) if time else time

    @staticmethod
    def datetime_to_int(dt: datetime):
        """
        :param dt: datetime
        :return: time in ms as long (java format)
        """
        return int(timegm(dt.timetuple()) * 1e3 + dt.microsecond / 1e3)

    @staticmethod
    def get_as_string(dt, pattern=DEFAULT_DATETIME_FORMAT):
        if isinstance(dt, (datetime, date)):
            strftime = dt.strftime(pattern)
            if pattern.endswith(".%fms"):
                strftime = strftime[:-5]
            if pattern.endswith(".%fmcs"):
                strftime = strftime[:-3]
            return strftime
        elif isinstance(dt, str):
            tmp = DateUtils.__get_datetime_with_pattern(dt, is_process_parsing=False)[0]
            if tmp:  # here we process if user passed: 'now', '-1d' etc.
                assert isinstance(tmp, datetime)
                return DateUtils.get_as_string(tmp, pattern)
            else:
                return dt
        else:
            return dt

    @staticmethod
    def get_datetime(date_time):
        if isinstance(date_time, str):
            return DateUtils.__get_datetime_with_pattern(date_time)[0]
        else:
            return date_time

    @staticmethod
    def get_datetime_ls(date_time_ls):
        return [DateUtils.get_datetime(idate) for idate in date_time_ls]

    @staticmethod
    def __get_datetime_with_pattern(date_time: str, is_process_parsing=True):
        lowered = date_time.lower()
        if lowered == 'now' or lowered == 'today':
            return DateUtils.get_now(), DateUtils.DEFAULT_DATETIME_FORMAT

        if date_time.startswith('-'):
            # time_unit_to_subtract = DateUtils.TIME_UNIT_DICT[date_as_string[-1]]
            # if time_unit_to_subtract is None:
            #     raise ValueError("can't identify time_unit_to_subtract {}", date_as_string[-1])
            delta = None
            amount_to_subtract = int(date_time[1:-1])
            if lowered[-1] == 'd':
                delta = timedelta(days=amount_to_subtract)
            elif lowered[-1] == 'h':
                delta = timedelta(hours=amount_to_subtract)
            elif lowered[-1] == 'w':
                delta = timedelta(weeks=amount_to_subtract)

            if delta is None:
                raise ValueError("can't identify time_unit_to_subtract '{0}'".format(date_time[-1]))

            return DateUtils.get_now() - delta, DateUtils.DEFAULT_DATETIME_FORMAT

        if is_process_parsing:
            result = None
            for pat in DateUtils._all_dateformats():
                try:
                    result = datetime.strptime(date_time, pat), pat
                    break
                except ValueError:
                    pass

            if result is None:
                raise ValueError("Unable transform value '{0}' into datetime".format(date_time))

            return result
        else:
            return None, DateUtils.DEFAULT_DATETIME_FORMAT

    @staticmethod
    def _all_dateformats():
        if DateUtils._ALL_DATE_PATTERNS is None:
            DateUtils._ALL_DATE_PATTERNS = (list)(DateUtils._dateformats_generator())
        return DateUtils._ALL_DATE_PATTERNS

    @staticmethod
    def _dateformats_generator():
        "Yield all combinations of valid date formats."
        years = ("%Y",)
        months = ("%m", "%b")
        days = ("%d",)
        times = ("", "%H:%M:%S", "%H:%M:%S.%f", "%H:%M")
        separators = {" ": ("-", "/", "."), "D": "."}

        for year in years:
            for month in months:
                for day in days:
                    for args in ((year, month, day), (month, day, year)):
                        for dt_sep in separators.keys():
                            for d_sep in separators[dt_sep]:
                                date = d_sep.join(args)
                                for time in times:
                                    yield dt_sep.join((date, time)).strip()

    @staticmethod
    def set_time(date, hour=0, minute=0, second=0, microsecond=0):
        """
        Set new time to datetime object
        :param date:
        :param hour:
        :param minute:
        :param second:
        :param microsecond:
        :return:
        """
        return datetime.combine(date.date() if isinstance(date, datetime) else date,
                                time(hour=hour, minute=minute, second=second, microsecond=microsecond))

    @staticmethod
    def format_kdb_date(dt):
        return DateUtils.__format_kdb(dt, DateUtils.KDB_DATE_FORMAT)

    @staticmethod
    def format_kdb_datetime(dt):
        return DateUtils.__format_kdb(dt, DateUtils.KDB_DATETIME_FORMAT)

    @staticmethod
    def format_kdb_datetime_msec(dt):
        return DateUtils.__format_kdb(dt, DateUtils.KDB_DATETIME_FORMAT + ".%fms")

    @staticmethod
    def format_kdb_datetime_usec(dt):
        return DateUtils.__format_kdb(dt, DateUtils.KDB_DATETIME_FORMAT + ".%f")

    @staticmethod
    def __format_kdb(dt, pattern):
        if isinstance(dt, str):
            dt = DateUtils.get_datetime(dt)
        return DateUtils.get_as_string(dt, pattern)

    @staticmethod
    def daterange(start_date, end_date):
        """
        Generate range of dates for given interval. It includes also start and end dates.
        :param start_date: starting date
        :param end_date: terminal date
        :return: generated range of dates (onlt dates without time)
        """
        for n in range(int((end_date - start_date).days) + 1):
            yield (start_date + timedelta(n)).date()

    @staticmethod
    def get_now():
        return datetime.now()

    @staticmethod
    def round_time(time, period_msec):
        """
        Rounds time to nearest period's end.
        __round_time('2017-01-01 13:31:23', timedelta(minutes=5).total_seconds()*10**3) -> '2017-01-01 13:30:00'
        :param time: datetime object
        :param period_msec: period in msec
        :return:
        """
        if period_msec == 0:
            return time

        t_msec = time.microsecond / 1000 + (time.second + time.minute * 60 + time.hour * 3600 + time.day * 86400) * 1000
        return time - timedelta(milliseconds=(t_msec % period_msec))

    @staticmethod
    def round_time_by(dt=None, value: int = -1, units='hours'):
        if dt is None: dt = datetime.now()

        if units == 'hours':
            delta = timedelta(minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond)
        elif units == 'minutes':
            delta = timedelta(seconds=dt.second, microseconds=dt.microsecond)
        elif units == 'seconds':
            delta = timedelta(microseconds=dt.microsecond)
        else:
            raise ValueError('units can be only hours, minutes or seconds')
        dt -= delta

        if value < 0:
            args = {units: getattr(dt, units[:-1]) % abs(value)}
            dt -= timedelta(**args)
        elif value > 0:
            args = {units: getattr(dt, units[:-1]) % abs(value) + value}
            dt += timedelta(**args)
        return dt

    @staticmethod
    def splitOnIntervals(start_date, end_date, amount, return_split_dates_only=False):
        diff_secs = end_date.timestamp() - start_date.timestamp()
        result = []
        next = start_date
        for i in range(amount - 1):
            prev = next
            next += timedelta(seconds=diff_secs / amount)
            result.append((prev, next))
        result.append((next, end_date))
        return result if not return_split_dates_only else [pair[1] for pair in result][:-1]


def hour_in_range(h, start, end):
    """
    Check if hour h in range  [start, end]
    """
    if start > end:
        return ((h >= start) & (h <= 23)) | ((h >= 0) & (h <= end))
    return (start <= h) & (h <= end)
