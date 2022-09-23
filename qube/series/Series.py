from typing import Union, List

import pandas as pd

from qube.series.Indicators import Indicator


class Series:
    """
    Basic abstract class for all series like objects
    """

    def __init__(self, timeframe: Union[int, str]):
        if isinstance(timeframe, str):
            self._resample_rule = timeframe
        elif isinstance(timeframe, int) and timeframe > 0:
            self._resample_rule = self.get_pd_freq_by_seconds_freq_val(timeframe)
        else:
            raise ValueError('freq must be either positive int seconds value or pandas offset alias format!')

        self._timeframe = pd.Timedelta(self._resample_rule)
        self._timeframe_value, self._timeframe_units = self.__get_timeframe_value(self._timeframe)

        # here we set ths flag to true when new bar is formed
        self.is_new_bar = False

        # indicators
        self.indicators: List[Indicator] = []

    def __get_timeframe_value(self, timeframe):
        seconds = int(timeframe.total_seconds())
        minutes = seconds / 60
        hours = minutes / 60
        if seconds < 60:
            timeframe_value = seconds
            timeframe_units = 'seconds'
        elif minutes < 60:
            timeframe_value = minutes
            timeframe_units = 'minutes'
        else:
            timeframe_value = hours
            timeframe_units = 'hours'
        return timeframe_value, timeframe_units

    @staticmethod
    def get_pd_freq_by_seconds_freq_val(seconds: int):
        minutes = seconds / 60
        hours = minutes / 60
        days = hours / 24
        years = days / 365
        if seconds < 60:
            resample_rule = "%ds" % seconds
        elif minutes < 60:
            resample_rule = "%dMin" % minutes
        elif hours < 24:
            resample_rule = "%dH" % hours
        elif days < 365:
            resample_rule = "%dD" % days
        else:
            resample_rule = "%dA" % years
        return resample_rule

    def __getitem__(self, idx: Union[int, slice]):
        raise NotImplementedError('Must be implemented in %s' % self.__class__.__name__)

    def __len__(self):
        """
        Returns length of this series
        """
        raise NotImplementedError('Must be implemented in %s' % self.__class__.__name__)

    def to_frame(self) -> pd.DataFrame:
        raise NotImplementedError('Must be implemented in %s' % self.__class__.__name__)

    def attach(self, indicator: Indicator):
        """
        Attach new indicator to this series

        :param indicator:
        :return: self
        """
        raise NotImplementedError('Must be implemented in %s' % self.__class__.__name__)
