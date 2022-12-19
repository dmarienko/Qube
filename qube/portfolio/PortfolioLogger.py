import collections
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Union, List

import pandas as pd

from qube.portfolio.Position import Position
from qube.utils.DateUtils import DateUtils
from qube.datasource.controllers.MongoController import MongoController


class PortfolioLogger:
    OUTPUT_DATE_PATTERN = '%Y-%m-%d %H:%M:%S.%f'
    SEPARATOR = ','
    MAX_TRACKING_BUFFER_FOR_AUTOSAVE = 10

    def __init__(self, log_frequency_sec=300, autosave_log_id=None, **kwargs):
        """
            :param log_frequency_sec: 300 - means physically adding a log record every 5 min. 0 - on every update (tick logger)
        """
        max_tracking_buff = kwargs.get('max_tracking_buff', None)
        self.log_frequency = timedelta(seconds=log_frequency_sec)
        self.log_frequency_sec = log_frequency_sec
        self._log_freq_msec = log_frequency_sec * 10 ** 3

        self.watching_positions: List[Position] = []
        self.column_names: List[str] = []
        self.index_buff = collections.deque(maxlen=max_tracking_buff)
        self.rows_buff = collections.deque(maxlen=max_tracking_buff)
        self.autosave_log_id = autosave_log_id
        self.__save_log_executor = ThreadPoolExecutor(1)  # 1 is important! single thread pool!
        self.__mongo_controller: Union[MongoController, None] = None
        self._last_pm_log_time = DateUtils.MIN_DATE

    def add_positions_for_watching(self, positions: Union[Position, List]):
        if isinstance(positions, Position):
            positions = [positions]

        if self.watching_positions:  # is not first add_positions_for_watching call?
            current_pos_names = [p.instrument.symbol for p in self.watching_positions]
            for p in positions:
                if p.instrument not in current_pos_names:
                    self.watching_positions.append(p)
                    self.column_names.extend(
                        map(lambda x: p.instrument.symbol + x, ['_Pos', '_PnL', '_Price', '_Value', '_Commissions']))

        else:
            self.watching_positions = positions
            [self.column_names.extend(
                map(lambda x: p.instrument.symbol + x, ['_Pos', '_PnL', '_Price', '_Value', '_Commissions'])) for p in
                positions]

        # adding init positions to beffers only if there are all last_update_time specified
        if sum([pos.last_update_time is None for pos in positions]) == 0:
            self.notify_update()

    def notify_update(self):
        prices_time = max(
            [pos.last_update_time if pos.last_update_time else DateUtils.MIN_DATE for pos in self.watching_positions])
        if prices_time >= self._last_pm_log_time + self.log_frequency:
            pm_log_time = DateUtils.round_time(prices_time, self._log_freq_msec)
            self._append_positions_to_buffers(pm_log_time)
            self._last_pm_log_time = pm_log_time

    def get_portfolio_log(self) -> pd.DataFrame:
        """
        Returns logged portfolio as pandas DataFrame

        :return: portfolio log as pandas data frame
        """
        if self.watching_positions:
            prices_time = max([pos.last_update_time if pos.last_update_time else DateUtils.MIN_DATE for pos in
                               self.watching_positions])
            final_row = [];
            [final_row.extend([pos.quantity, pos.pnl, pos.price, pos.market_value_usd, pos.commissions]) for pos in
             self.watching_positions]
            if prices_time >= self._last_pm_log_time + self.log_frequency:
                last_log_time = DateUtils.round_time(prices_time, self._log_freq_msec)
                return pd.DataFrame(data=list(self.rows_buff) + [final_row],
                                    index=list(self.index_buff) + [last_log_time], columns=self.column_names,
                                    dtype=float)
            else:
                self.rows_buff[-1] = final_row
                return pd.DataFrame(data=list(self.rows_buff), index=list(self.index_buff), columns=self.column_names,
                                    dtype=float)
        else:
            return pd.DataFrame()

    def clear(self):
        self.index_buff.clear()
        self.rows_buff.clear()

    def __len__(self):
        return len(self.index_buff)

    def wait_for_all_rows_saved(self):
        self.__save_log_executor.shutdown(True)

    def _autosave_final_row(self):
        prices_time = max([
            pos.last_update_time if pos.last_update_time else DateUtils.MIN_DATE for pos in self.watching_positions
        ])
        if prices_time > DateUtils.MIN_DATE:  # doing if there were any updates
            final_row = []
            [final_row.extend([pos.quantity, pos.pnl, pos.price, pos.market_value_usd, pos.commissions]) for pos in self.watching_positions]
            final_log_time = DateUtils.round_time(prices_time, self._log_freq_msec)
            self.__autosave_row(final_log_time, final_row, True)

    def _append_positions_to_buffers(self, pm_log_time):
        row = []
        [row.extend([pos.quantity, pos.pnl, pos.price, pos.market_value_usd, pos.commissions]) for pos in self.watching_positions]
        self.rows_buff.append(row)
        self.index_buff.append(pm_log_time)

        is_first_record = self._last_pm_log_time <= DateUtils.MIN_DATE
        self.__autosave_row(pm_log_time, row, check_if_update=is_first_record)

    def __autosave_row(self, pm_log_time, row, check_if_update):
        if self.autosave_log_id:
            self.__save_log_executor.submit(self.__save_log_update, *(pm_log_time, row, check_if_update))

    def __save_log_update(self, log_time, row, check_if_update):
        if not self.__mongo_controller:
            self.__mongo_controller = MongoController()
        if check_if_update:
            self.__del_last_saved_row(self.autosave_log_id, log_time)
        self.__mongo_controller.append_data(self.autosave_log_id,
                                            pd.Series(data=row, index=self.column_names, name=log_time))

    def __del_last_saved_row(self, autosave_log_id, log_time):
        self.__mongo_controller = MongoController()
        aggr_query = [{'$group': {'_id': 0, 'last_date': {'$max': "$__index"}}}]
        pl_last = self.__mongo_controller.load_aggregate(autosave_log_id, aggr_query)
        if pl_last:
            last_log_time = pl_last[0]['last_date']
            if last_log_time >= log_time:
                self.__mongo_controller.delete_records(autosave_log_id, {'__index': {'$gte': log_time}})

    def _format_record(self, dt, row):
        dt_as_string = dt.strftime(self.OUTPUT_DATE_PATTERN) if dt is not None else 'None'
        return dt_as_string + self.SEPARATOR + self.SEPARATOR.join([str(e) for e in row])
