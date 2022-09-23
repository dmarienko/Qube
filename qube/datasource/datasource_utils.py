from datetime import datetime, timezone
from typing import Union

import pandas as pd

from qube.datasource.DataSource import DataSource
from qube.series.Series import Series
from qube.utils.DateUtils import DateUtils


def get_ds_data_nback(datasource: Union[DataSource, str], nback, instruments,
                      freq: Union[int, str] = None, date_from: Union[str, datetime] = None,
                      accept_holidays=True, populate_local_time=True):
    """
    Helpful method for specific strategy TestingModel (see ITestingModel) implementation to
    preload specified 'nback' bars series

    :param datasource: datasource to load data from. Example 'kdb::dukas'
    :param nback: amount of bars / ticks to load
    :param instruments: list of instruments to load data for
    :param freq: size of the bars to load. supported formats: int (IRA) or str format (pandas frequency format).
                 An example of values to populate '5Min' or 300.
                 Default is None - means tick data.
    :param date_from: date to load previous data from. default is None - means load latest (from now) nback bars.
    :accept_holidays: adding weekends and holidays to each week
    :return: Dict[instrument:pd.DataFrame]
    """
    if isinstance(datasource, str):
        d_source = DataSource(datasource)
        to_close_ds = True
    else:
        d_source = datasource
        to_close_ds = False

    if isinstance(instruments, str):
        instruments = [instruments]
    start = _estimate_start_by_nback(nback, freq=freq, date_from=date_from, accept_holidays=accept_holidays)
    result = d_source.load_data(instruments, start, date_from, timeframe=freq, populate_local_time=populate_local_time)
    for instr in result.keys():
        result[instr] = result[instr].iloc[-nback:]
    if to_close_ds:
        d_source.close()
    return result


def _estimate_start_by_nback(nback: int, nback_unit: str = None, freq: Union[int, str] = None,
                             date_from: Union[str, datetime] = None, accept_holidays=True):
    """
    Method is used to estimate start date to begin load data from
    when we want to load specified 'nback' bars of 'freq' timeframe back from 'date_from' date.
    There is no any reason in explicit use of that method!

    :param nback: amount of bars / ticks to load or pd.TimeDelta unit if param nback_unit exists.
    :param nback_unit: pd.TimeDelta unit for nback.
    :param freq: size of the bars to load.
    :param date_from: date to load previous data from. default is None - means load latest (from now) nback bars.
    :accept_holidays: adding weekends and holidays to each week
    :return: estimation of start date (return type: dt.datetime)
    """
    __ADDITIONAL_DAYS_PER_WEEK = 4
    __NOW_DATA_TZ = timezone.utc

    if nback_unit:
        nback_delta = pd.Timedelta(nback, unit=nback_unit)
    else:
        if freq:
            freq = freq if isinstance(freq, str) else Series.get_pd_freq_by_seconds_freq_val(freq)
            delta_freq = pd.Timedelta(freq)
        else:
            delta_freq = pd.Timedelta(seconds=1)

        nback_delta: pd.Timedelta = delta_freq * nback
    if accept_holidays:
        nback_weeks = int(nback_delta.days / 7)
        nback_delta += pd.Timedelta(days=__ADDITIONAL_DAYS_PER_WEEK) * (nback_weeks + 1)

    date_from = DateUtils.get_datetime(date_from) if date_from else datetime.now(__NOW_DATA_TZ)
    return date_from - nback_delta
