from typing import Union

import numpy as np
import pandas as pd

from qube.series.BarSeries import BarSeries, Bar


class DoubleSeries(BarSeries):
    """
    Series of double values
    """

    def __init__(self, timeframe: Union[int, str],
                 series: pd.Series = None,
                 max_series_length=np.inf,
                 use_first_value=True):
        super().__init__(timeframe, series, max_series_length=max_series_length)
        self.__is_open = use_first_value

    def update_by_value(self, time, value: float) -> bool:
        if not self.series[self._TM_ID]:
            self._BarSeries__add_new_bar(time, value, 0)

            # Here we disable first notification because first bar may be incomplete
            self.is_new_bar = False
        elif time - self.series[self._TM_ID][-1] >= self._timeframe:

            # first we update indicators by currect last bar
            self.__update_all_indicators(self._BarSeries__bar_at(0), True)

            # then add new bar
            self._BarSeries__add_new_bar(time, value, 0)
        else:
            self._BarSeries__update_last_bar(time, value, 0)

        # update indicators by new data
        self.__update_all_indicators(self._BarSeries__bar_at(0), False)

        return self.is_new_bar

    def __update_all_indicators(self, bar: Bar, is_new_bar: bool):
        _last_val = bar.open if self.__is_open else bar.close
        for i in self.indicators:
            if i.need_bar_input():
                i.update(bar, is_new_bar)
            else:
                i.update(_last_val, is_new_bar)
