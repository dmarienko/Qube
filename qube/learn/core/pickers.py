import re
from typing import Union, Dict

import pandas as pd

from qube.quantitative.tools import ohlc_resample
from qube.learn.core.data_utils import _get_top_names, detect_data_type, make_dataframe_from_dict


class Resampler:

    def __init__(self):
        self.start_ = None
        self.stop_ = None

    def for_range(self, start, stop=None):
        """
        If we need to select range of data to operate on
        :param start: starting date
        :param stop: end date
        :return:
        """
        self.start_ = start
        self.stop_ = stop
        return self

    def _select(self, x):
        nf = self.start_ is not None or self.stop_ is not None

        if nf:
            s = x.index[0] if self.start_ is None else self.start_
            e = x.index[-1] if (self.stop_ is None or self.stop_ == 'now') else self.stop_
            return x[s:e]
        return x

    def _resample(self, data, timeframe, tz):
        """
        Resample data to timeframe
        :param data:
        :return:
        """
        r = data
        if timeframe is not None:
            if isinstance(data, pd.DataFrame):
                cols = data.columns
                if isinstance(cols, pd.MultiIndex):
                    symbols = _get_top_names(cols)
                    return self._select(
                        pd.concat([ohlc_resample(data[c], timeframe, resample_tz=tz, non_ohlc_columns_aggregator='sum') for c in symbols], axis=1,
                                  keys=symbols))

            # all the rest cases
            r = ohlc_resample(data, timeframe, resample_tz=tz, non_ohlc_columns_aggregator='sum')

        return self._select(r)


class AbstractDataPicker(Resampler):
    """
    Generic abstract data picker
    """

    def __init__(self, rules=None, timeframe=None, tz='UTC'):
        super(AbstractDataPicker, self).__init__()
        rules = [] if rules is None else rules
        self.rules = rules if isinstance(rules, (list, tuple, set)) else [rules]
        self.timeframe = timeframe
        self.tz = tz

    def _is_selected(self, s, rules):
        if not rules:
            return True
        for r in rules:
            if re.match(r, s):
                return True
        return False

    def iterdata(self, data, selected_symbols, data_type, symbols, entries_types):
        raise NotImplementedError('Method must be implemented !')

    def iterate(self, data):
        info = detect_data_type(data)

        selected = info.symbols
        if self.rules:
            seen = set()
            seen_add = seen.add
            # selected = [s for s in info.symbols if self._is_selected(s, self.rules)]

            # check rules and select passed symbols
            selected_seq = [s for r in self.rules for s in info.symbols if self._is_selected(s, [r])]

            # make list of selection in order of rules been passed
            selected = [x for x in selected_seq if not (x in seen or seen_add(x))]

        return self.iterdata(data, selected, info.type, info.symbols, info.subtypes)

    def take(self, data, nth: Union[str, int] = 0):
        """
        Helper method to take n-th iteration
        :param data: data to be iterated
        :param nth: if int it returns n-th iteration of data
        :return: data or none if not matched
        """
        v = None

        # if we look for some specific pattern in symbols
        if isinstance(nth, str):
            v = next((data for (s, data) in self.iterate(data) if re.match(nth, s)), None)

        # if we just asked for n-th record
        if isinstance(nth, int):
            v = next((data for (i, (s, data)) in enumerate(self.iterate(data)) if i == nth), None)

        return v

    def as_datasource(self, data) -> Dict:
        """
        Return prepared data ready to be used in simulator
        
        :param data: input data
        :return: {symbol : preprocessed_data}
        """
        _ds = {}
        for s, data in self.iterate(data):
            if isinstance(s, (list, tuple)):
                _ds = {**_ds, **{k: data[k] for k in s}}
            else:
                _ds[s] = data
        return _ds


class SingleInstrumentPicker(AbstractDataPicker):
    """
    Iterate symbol by symbol
    """

    def __init__(self, rules=None, timeframe=None, tz='UTC'):
        super().__init__(rules=rules, timeframe=timeframe, tz=tz)

    def iterdata(self, data, selected_symbols, data_type, symbols, entries_types):
        if data_type == 'dict' or data_type == 'multi':
            # iterate every dict entry or column from multi index dataframe
            for s in selected_symbols:
                yield s, self._resample(data[s], self.timeframe, self.tz)
        elif data_type == 'ohlc' or data_type == 'series' or data_type == 'ticks':
            # just single series
            yield symbols[0], self._resample(data, self.timeframe, self.tz)
        else:
            raise ValueError(f"Unknown data type '{data_type}'")


class PortfolioPicker(AbstractDataPicker):
    """
    Iterate whole portfolio
    """

    def __init__(self, rules=None, timeframe=None, tz='UTC'):
        super().__init__(rules=rules, timeframe=timeframe, tz=tz)

    def iterdata(self, data, selected_symbols, data_type, symbols, entries_types):
        if data_type == 'dict':
            if entries_types:
                if len(entries_types) > 1:
                    raise ValueError(
                        "Dictionary contains data with different types so not sure how to merge them into portfolio !")
            else:
                raise ValueError("Couldn't detect types of dictionary items so not sure how to deal with it !")

            subtype = list(entries_types)[0]
            yield selected_symbols, make_dataframe_from_dict(
                {s: self._resample(data[s], self.timeframe, self.tz) for s in selected_symbols}, subtype)

        elif data_type == 'multi':
            yield selected_symbols, self._resample(data[selected_symbols], self.timeframe, self.tz)

        elif data_type == 'ohlc' or data_type == 'series' or data_type == 'ticks':
            yield symbols, self._resample(data, self.timeframe, self.tz)

        else:
            raise ValueError(f"Unknown data type '{data_type}'")
