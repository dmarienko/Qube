import unittest
from datetime import datetime

from qube.utils.DateUtils import DateUtils


class TestDateUtils(unittest.TestCase):
    def test_basic(self):
        print('now : ' + DateUtils.get_as_string(DateUtils.get_datetime('now')))
        print('NOW : ' + DateUtils.get_as_string(DateUtils.get_datetime('NOW')))
        print('Today : ' + DateUtils.get_as_string(DateUtils.get_datetime('Today')))

        print('-1d : ' + DateUtils.get_as_string(DateUtils.get_datetime('-1d')))
        print('-2D : ' + DateUtils.get_as_string(DateUtils.get_datetime('-2D')))
        print('-1w : ' + DateUtils.get_as_string(DateUtils.get_datetime('-1w')))
        print('-2W : ' + DateUtils.get_as_string(DateUtils.get_datetime('-2W')))

        self.assertRaises(ValueError, lambda: DateUtils.get_datetime('-2Z'))

        self.assertEqual(DateUtils.get_as_string(DateUtils.get_datetime('2016-10-25'), DateUtils.DEFAULT_DATE_FORMAT),
                         '2016-10-25')
        self.assertEqual(DateUtils.get_as_string(DateUtils.get_datetime('2016-10-25 10:20')), '2016-10-25 10:20:00')
        self.assertEqual(DateUtils.get_as_string(DateUtils.get_datetime('2016-10-25 10:20:25')), '2016-10-25 10:20:25')
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.get_datetime('2016.10.25'), DateUtils.DEFAULT_DATETIME_FORMAT),
            '2016-10-25 00:00:00')
        # TODO WTF not works!
        # self.assertEqual(DateUtils.get_as_string(DateUtils.get_datetime('2016.10.05'), '%-1d-%b-%y'), '5-Oct-16')
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.get_datetime('2016-01-01 01:30:52.123'), "%Y-%m-%d %H:%M:%S.%f"),
            '2016-01-01 01:30:52.123000')
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.get_datetime('2016.01.01 01:30:52.123'),
                                    DateUtils.DEFAULT_DATETIME_FORMAT_MSEC),
            '2016-01-01 01:30:52.123')

    def test_basic_2(self):
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.get_datetime('2016-May-25 10:20'), DateUtils.DEFAULT_DATE_FORMAT),
            '2016-05-25')
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.get_datetime('2016/May/25 10:20'), DateUtils.DEFAULT_DATE_FORMAT),
            '2016-05-25')
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.get_datetime('2016.May.25 10:20:00'),
                                    DateUtils.DEFAULT_DATETIME_FORMAT_MSEC),
            '2016-05-25 10:20:00.000')

    def test_basic_3(self):
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.get_datetime('2016-May-25 10:20').date(), DateUtils.KDB_DATE_FORMAT),
            '2016.05.25')

    def test_accepts_diff_types(self):
        dt = datetime(2012, 11, 12, 10, 30, 35)
        self.assertEqual(DateUtils.get_datetime(dt), dt)

        self.assertEqual(DateUtils.get_as_string('2016-01-01', DateUtils.DEFAULT_DATE_FORMAT), '2016-01-01')

    def test_kdb(self):
        self.assertEqual(DateUtils.get_as_string(DateUtils.get_datetime('2016.01.01D01:30:00')), '2016-01-01 01:30:00')
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.get_datetime('2016.01.01D01:30:52.123'), "%Y-%m-%d %H:%M:%S.%f"),
            '2016-01-01 01:30:52.123000')

    def test_kdb_format(self):
        dt = DateUtils.get_datetime('2016-01-01 01:30:52.123456')
        self.assertEqual(DateUtils.format_kdb_datetime(dt), '2016.01.01D01:30:52')
        self.assertEqual(DateUtils.format_kdb_datetime_msec(dt), '2016.01.01D01:30:52.123')
        self.assertEqual(DateUtils.format_kdb_datetime_usec(dt), '2016.01.01D01:30:52.123456')
        self.assertEqual(DateUtils.format_kdb_date(dt), '2016.01.01')

        self.assertEqual(DateUtils.format_kdb_datetime('2015-10-23'), '2015.10.23D00:00:00')
        self.assertEqual(DateUtils.format_kdb_date('2015-10-23 10:23'), '2015.10.23')
        self.assertEqual(DateUtils.format_kdb_datetime_msec('2015-10-23 10:23'), '2015.10.23D10:23:00.000')
        self.assertEqual(DateUtils.format_kdb_datetime_usec('2015-10-23 10:23'), '2015.10.23D10:23:00.000000')

    def testRoundDownTime(self):
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.round_time_by(datetime(2015, 10, 25, 10, 00, 23), units='minutes')),
            "2015-10-25 10:00:00")
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.round_time_by(datetime(2015, 10, 25, 10, 00, 31), units='minutes')),
            "2015-10-25 10:00:00")
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.round_time_by(datetime(2015, 10, 25, 10, 00, 31), units='minutes')),
            "2015-10-25 10:00:00")
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.round_time_by(datetime(2015, 10, 25, 10, 00, 31), 1, 'minutes')),
            "2015-10-25 10:01:00")
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.round_time_by(datetime(2015, 10, 25, 10, 00, 31), 5, 'minutes')),
            "2015-10-25 10:05:00")
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.round_time_by(datetime(2015, 10, 25, 10, 00, 31), -5, 'minutes')),
            "2015-10-25 10:00:00")
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.round_time_by(datetime(2015, 10, 25, 10, 00, 31), -24, 'hours')),
            "2015-10-25 00:00:00")
        self.assertEqual(
            DateUtils.get_as_string(DateUtils.round_time_by(datetime(2015, 10, 25, 10, 00, 31), -10, 'seconds')),
            "2015-10-25 10:00:30")

    def testGetList(self):
        self.assertEqual(DateUtils.get_datetime_ls(['2000-01-01 10:30', '2009-11-23']),
                         [datetime(2000, 1, 1, 10, 30), datetime(2009, 11, 23, 0, 0)])

    def testSplitOnIntervals(self):
        self.assertEqual(
            DateUtils.splitOnIntervals(DateUtils.get_datetime('2010-01-01'), DateUtils.get_datetime('2012-01-01'), 1),
            [(DateUtils.get_datetime('2010-01-01'), DateUtils.get_datetime('2012-01-01'))])

        self.assertEqual(
            DateUtils.splitOnIntervals(DateUtils.get_datetime('2010-01-01'), DateUtils.get_datetime('2012-01-01'), 2),
            [(DateUtils.get_datetime('2010-01-01'), DateUtils.get_datetime('2011-01-01')),
             (DateUtils.get_datetime('2011-01-01'), DateUtils.get_datetime('2012-01-01'))])

        self.assertEqual(
            DateUtils.splitOnIntervals(DateUtils.get_datetime('2009-01-01'), DateUtils.get_datetime('2012-01-01'), 3),
            [(DateUtils.get_datetime('2009-01-01'), DateUtils.get_datetime('2010-01-01')),
             (DateUtils.get_datetime('2010-01-01'), DateUtils.get_datetime('2011-01-01')),
             (DateUtils.get_datetime('2011-01-01'), DateUtils.get_datetime('2012-01-01'))])

        self.assertEqual(
            DateUtils.splitOnIntervals(DateUtils.get_datetime('2009-01-01'), DateUtils.get_datetime('2012-01-01'), 3,
                                       return_split_dates_only=True),
            DateUtils.get_datetime_ls(['2010-01-01', '2011-01-01']))

    def int_test_all_dateformats(self):
        self.assertIsNone(DateUtils._ALL_DATE_PATTERNS)
        dateformats = DateUtils._all_dateformats()
        print(dateformats)
        self.assertEqual(len((list)(dateformats)), 64)
        self.assertIsNotNone(DateUtils._ALL_DATE_PATTERNS)
        dateformats = DateUtils._all_dateformats()
        self.assertEqual(len((list)(dateformats)), 64)


from pytest import main
if __name__ == '__main__':
    main()