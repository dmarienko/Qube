from functools import reduce
import re
import hashlib
import types
from datetime import timedelta
from enum import Enum
from itertools import product
from typing import Any, List, Union, Dict

import numpy as np
import pandas as pd

from qube import runtime_env
from qube.datasource import DataSource
from qube.portfolio.commissions import (
    _KRAKEN_FUTURES_FEES,
    _KRAKEN_SPOT_FEES,
    TransactionCostsCalculator,
    ZeroTCC,
    BinanceRatesCommon,
    WooXRatesCommon,
    StockTCC,
    ForexTCC,
    BitmexTCC,
    FxcmTCC,
    FTXRatesCommon,
    KrakenRatesCommon,
)
from qube.portfolio.performance import split_cumulative_pnl
from qube.quantitative.tools import infer_series_frequency
from qube.simulator.Brokerage import (
    BrokerInfo,
    GenericStockBrokerInfo,
    GenericForexBrokerInfo,
    GenericCryptoBrokerInfo,
    GenericCryptoFuturesBrokerInfo,
)
from qube.utils.DateUtils import DateUtils


def split_signals(signals: pd.Series, split_intervals_dates) -> List[pd.Series]:
    """
    Split signals by intervals
    :param signals:
    :param split_intervals_dates:
    :return:
    """
    signals_intervals_dates = []
    for split_date in split_intervals_dates:
        search_idx = signals.index.searchsorted(split_date)
        nearest_signal_date = (
            search_idx if split_date in signals.index else search_idx - 1
        )
        signals_intervals_dates.append(signals.index[nearest_signal_date])

    signals.ffill(inplace=True)

    split_start_date = signals.index[0]
    result = []
    for split_date in signals_intervals_dates:
        result.append(signals.loc[split_start_date:split_date])
        split_start_date = split_date

    result.append(signals.loc[split_start_date : signals.index[-1]])
    return result


def rolling_forward_test_split(
    x, training_period: int, test_period: int, units: str = None
):
    """
    Split data into training and testing **rolling** periods.

    Example:

    >>> for train_idx, test_idx in rolling_forward_test_split(np.array(range(15)), 5, 3):
    >>>     print('Train:', train_idx, ' Test:', test_idx)

    > Train: [1 2 3 4 5]  Test: [6 7 8]
      Train: [4 5 6 7 8]  Test: [9 10 11]
      Train: [7 8 9 10 11]  Test: [12 13 14]

    Also it allows splitting using calendar periods (see units for that).
    Example of 2w / 1w splitting:

    >>> Y = pd.Series(np.arange(30), index=pd.date_range('2000-01-01', periods=30))
    >>> for train_idx, test_idx in rolling_forward_test_split(Y, 2, 1, units='W'):
    >>>     print('Train:', Y.loc[train_idx], '\\n Test:', Y.loc[test_idx])

    :param x: data
    :param training_period: number observations for training period
    :param test_period: number observations for testing period
    :param units: period units if training_period and test_period is the period date: {'H', 'D', 'W', 'M', 'Q', 'Y'}
    :return:
    """
    # unit formats from pd.TimeDelta and formats for pd.resample
    units_format = {"H": "H", "D": "D", "W": "W", "M": "MS", "Q": "QS", "Y": "AS"}

    if units:
        if units.upper() not in units_format:
            raise ValueError(
                'Wrong value for "units" parameter. Only %s values are valid'
                % ",".join(units_format.keys())
            )
        else:
            if not isinstance(x, (pd.Series, pd.DataFrame)) or not isinstance(
                x.index, pd.DatetimeIndex
            ):
                raise ValueError(
                    'Data must be passed as pd.DataFrame or pd.Series when "units" specified'
                )

            if isinstance(x, pd.Series):
                x = x.to_frame()

            resampled = x.resample(units_format[units.upper()]).mean().index
            resampled = resampled - pd.DateOffset(seconds=1)

            for i in range(0, len(resampled), test_period):
                if (
                    len(resampled) - 1 < i + training_period
                    or resampled[i + training_period] > x.index[-1]
                ):
                    # no data for next training period
                    break
                training_df = x[resampled[i] : resampled[i + training_period]]
                whole_period = i + training_period + test_period
                if (
                    len(resampled) - 1 < whole_period
                    or resampled[whole_period] > x.index[-1]
                ):
                    # if there is not all data for test period or it's just last month,
                    # we don't need restrict the end date
                    test_df = x[resampled[i + training_period] :]
                else:
                    test_df = x[
                        resampled[i + training_period] : resampled[whole_period]
                    ]

                if training_df.empty or test_df.empty:
                    continue
                yield (np.array(training_df.index), np.array(test_df.index))
    else:
        n_obs = x.shape[0]
        i_shift = (n_obs - training_period - test_period) % test_period
        for i in range(i_shift + training_period, n_obs, test_period):
            yield (
                np.array(range(i - training_period, i)),
                np.array(range(i, i + test_period)),
            )


def merge_portfolio_log_chunks(portfolio_log_chunks, split_cumulative=True):
    merged_chunks = None
    for split_chunk in portfolio_log_chunks:
        to_append = (
            split_cumulative_pnl(split_chunk) if split_cumulative else split_chunk
        )
        if merged_chunks is not None:
            new_chunk_border_date = to_append.index[0]
            prev_chunk_last_date = merged_chunks.index[-1]
            if prev_chunk_last_date >= new_chunk_border_date:
                if prev_chunk_last_date > new_chunk_border_date:
                    merged_chunks = merged_chunks[:new_chunk_border_date]

                pnl_cols = [
                    col
                    for col in merged_chunks.columns
                    if col.endswith("_PnL") or col.endswith("_Commissions")
                ]
                prev_chunk_last_PnLs = merged_chunks.iloc[-1].filter(
                    regex=r".*_PnL|.*_Commissions"
                )
                to_append.loc[new_chunk_border_date, pnl_cols] = prev_chunk_last_PnLs

                if merged_chunks.index[-1] == new_chunk_border_date:
                    merged_chunks = merged_chunks.iloc[:-1]

                merged_chunks = pd.concat(
                    [merged_chunks, to_append]
                )  # , ignore_index=True)

            else:  # if chunks not crossing and borders are not equals
                merged_chunks = pd.concat(
                    [merged_chunks, to_append]
                )  # , ignore_index=True)
        else:
            merged_chunks = to_append
    return merged_chunks


def shift_signals(
    sigs: Union[pd.Series, pd.DataFrame],
    forward: str = None,
    days=0,
    hours=0,
    minutes=0,
    seconds=0,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Shift signal time into future.

    :return: shifted signals frame
    """
    n_sigs = sigs.copy()
    f0 = pd.Timedelta(forward if forward is not None else 0)
    n_sigs.index = (
        n_sigs.index
        + f0
        + pd.DateOffset(days=days, hours=hours, minutes=minutes, seconds=seconds)
    )
    return n_sigs


class dstype(Enum):
    """
    Enumeration for datasource type (protected visibility)
    """

    UNKNOWN = -1  # not recognized datasource
    BIDASK = 0  # for bid/ask source
    OHLC = 1  # for OHLC data
    PRICE = 2  # for midprice/price data


class dsinfo:
    def __init__(self, load_block_amnt, type, freq):
        # amount of days per one data reading from datasource
        self.load_block_amnt = load_block_amnt
        # datasource type (see dstype class)
        self.type = type
        # data timeframe frequency (timedelta object)
        self.freq = freq


def recognize_datasource_structure(
    data_src: DataSource, series_names, start, end, logger=None
) -> dsinfo:
    """
    Trying to infer structure of provided datasource.

    :param data_src: data source for recognize
    :param series_names: instruments names to test
    :param start: start time
    :param end: end time
    :param logger: logge
    :return: structure with inferred info
    """

    if logger is not None:
        logger.info("Trying to infer minimal datasource data loading size ...")

    for name in series_names:
        d0 = start.date()

        # - try to speed up everything and ask datasource if there is any data ranges
        try:
            _s, _ = data_src.get_range(name)
            if _s is not None:
                d0 = pd.Timestamp(_s).date()
        except:
            pass

        test_data = None
        while (test_data is None or test_data.empty) and d0 <= end.date():
            # trying to load any data from simulation start to {d+1}:10:00
            try:
                test_data = data_src.load_data(
                    name,
                    d0,
                    DateUtils.set_time(d0 + timedelta(days=1), hour=10, minute=0),
                ).get(name.upper())
            except Exception as ex:
                if logger:
                    logger.warning(
                        "Error loading '%s' instrument: %s" % (name, str(ex))
                    )

            d0 += timedelta(days=1)

        ds_type = dstype.UNKNOWN

        # if we found any data let's look at it's structure
        if (test_data is not None) and not test_data.empty:
            # check if it's OHLC data
            if test_data.columns.isin(["Close", "Open", "close", "open"]).any():
                ds_type = dstype.OHLC

            # check if it's tick data
            if test_data.columns.isin(["Price", "Mid", "MidPrice", "price"]).any():
                ds_type = dstype.PRICE

            if (
                test_data.columns.isin(["Ask", "ask", "askprice"]).any()
                and test_data.columns.isin(["Bid", "bid", "bidprice"]).any()
            ):
                ds_type = dstype.BIDASK

            # default is daily or weekly - so we can load without any limits
            ld_blk_amnt = (end - start) + timedelta(weeks=2)
            freq = timedelta(days=1)

            # if we have more than 1 records try to infer frequency
            if test_data.shape[0] > 1:

                # try to infer series frequency (timedelta object)
                freq = infer_series_frequency(test_data)

                # if datasource provides ticks/seconds frequent data - we load one day of data per time
                if freq < timedelta(minutes=1):
                    ld_blk_amnt = timedelta(days=1)
                elif freq < timedelta(days=1):
                    # all timeframes >=1 min but < 1 day we can load by weekly blocks
                    ld_blk_amnt = timedelta(weeks=1)

            # cracked datasource info
            if logger is not None:
                logger.info(
                    "Inferred %s structure. Data frequency is %s. Loading block size is %d days"
                    % (
                        ds_type,
                        "TICKS" if freq < timedelta(seconds=1) else str(freq),
                        ld_blk_amnt.days,
                    )
                )

            return dsinfo(ld_blk_amnt, ds_type, freq)

    raise ValueError(
        f"Can't infer frequency for specified datasource '{data_src.get_name() if data_src is not None else data_src}'"
    )


def __is_list_in(cols: List[str], df: pd.DataFrame):
    return all(list(map(lambda x: x in df.columns, cols)))


_DEFAULT_DAILY_SESSION_START = pd.Timedelta("9:29:59")
_DEFAULT_DAILY_SESSION_END = pd.Timedelta("15:59:59")


def convert_ohlc_to_ticks(
    ohlc: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    spread: Union[float, Dict[str, float]] = 0.0,
    freq=None,
    default_size=1e12,
    session_start=None,
    session_end=None,
    reverse_order_for_bullish_bars=False,
) -> pd.DataFrame:
    """
    Present OHLC data (from pandas DataFrame or dict of DataFrame) as quotes
    """

    # spread getter helper
    def _get_spread(sprds, name):
        return (
            sprds.get(name, 0.0)
            if isinstance(sprds, dict)
            else sprds if isinstance(sprds, float) else 0.0
        )

    if isinstance(ohlc, dict):
        return {
            k: convert_ohlc_to_ticks(
                v,
                _get_spread(spread, k),
                freq=freq,
                default_size=default_size,
                session_start=session_start,
                session_end=session_end,
                reverse_order_for_bullish_bars=reverse_order_for_bullish_bars,
            )
            for k, v in ohlc.items()
        }

    if not isinstance(ohlc, pd.DataFrame):
        raise ValueError("Only pandas DataFrame is supported")

    # if just OHLC without bid/ask separation
    is_4 = __is_list_in(["open", "high", "low", "close"], ohlc)
    is_2 = __is_list_in(["open", "close"], ohlc) if not is_4 else False

    # if OHLC with bid/ask series
    is_4_ba = __is_list_in(
        [
            "open_bid",
            "open_ask",
            "high_bid",
            "high_ask",
            "low_bid",
            "low_ask",
            "close_bid",
            "close_ask",
        ],
        ohlc,
    )
    is_2_ba = (
        __is_list_in(["open_bid", "open_ask", "close_bid", "close_ask"], ohlc)
        if not is_4_ba
        else False
    )

    if not is_2 and not is_4:
        if not is_2_ba and not is_4_ba:
            raise ValueError(
                "Input dataframe doesn't look like OHLC data ! "
                "At least open, close columns must be presented !"
            )

    # times
    freq = pd.Timedelta(infer_series_frequency(ohlc)) if freq is None else freq

    # special case for daily data to cover stocks
    if freq == pd.Timedelta("1D"):
        init_shift = (
            session_start if session_start is not None else _DEFAULT_DAILY_SESSION_START
        )
        in_middle1 = pd.Timedelta("11:00:00")  # some time in the middle for low price
        in_middle2 = pd.Timedelta("12:00:00")  # some time in the middle for high price
        before_close = (
            session_end if session_end is not None else _DEFAULT_DAILY_SESSION_END
        )
    else:
        init_shift = pd.Timedelta(0)
        in_middle1 = freq / 2 - freq / 10  # some time in the middle for low price
        in_middle2 = freq / 2 + freq / 10  # some time in the middle for high price
        before_close = freq - freq / 5  # time for close price near the end of interval

    if is_2 or is_4:
        z = ohlc.open.shift(1, freq=init_shift)
        p1 = pd.DataFrame(
            {
                "bid": z - spread,
                "ask": z + spread,
                "bidvol": default_size,
                "askvol": default_size,
                "is_real": True,
            }
        )

        # if high/low are presented
        p2, p3 = None, None
        if is_4:
            if reverse_order_for_bullish_bars:
                # - new logic: O,H,L,C for bearish and O,L,H,C for fullish bars
                is_bull = ohlc.close >= ohlc.open
                z2 = ohlc.low.where(is_bull, ohlc.high).shift(1, freq=in_middle1)
                z3 = ohlc.high.where(is_bull, ohlc.low).shift(1, freq=in_middle2)
            else:
                # - old logic: O,L,H,C for all bars
                z2 = ohlc.low.shift(1, freq=in_middle1)
                z3 = ohlc.high.shift(1, freq=in_middle2)

            p2 = pd.DataFrame(
                {
                    "bid": z2 - spread,
                    "ask": z2 + spread,
                    "bidvol": np.nan,
                    "askvol": np.nan,
                    "is_real": 0,
                }
            )
            p3 = pd.DataFrame(
                {
                    "bid": z3 - spread,
                    "ask": z3 + spread,
                    "bidvol": np.nan,
                    "askvol": np.nan,
                    "is_real": 0,
                }
            )

        z = ohlc.close.shift(1, freq=before_close)
        p4 = pd.DataFrame(
            {
                "bid": z - spread,
                "ask": z + spread,
                "bidvol": default_size,
                "askvol": default_size,
                "is_real": 1,
            }
        )
    else:
        # if bid/ask series are presented
        zb, za = ohlc.open_bid.shift(1, freq=init_shift), ohlc.open_ask.shift(
            1, freq=init_shift
        )
        p1 = pd.DataFrame(
            {
                "bid": zb,
                "ask": za,
                "bidvol": default_size,
                "askvol": default_size,
                "is_real": 1,
            }
        )

        # if high/low are presented
        p2, p3 = None, None
        if is_4_ba:
            if reverse_order_for_bullish_bars:
                is_bull = ohlc.close_bid >= ohlc.open_ask
                # - new logic: O,H,L,C for bearish and O,L,H,C for fullish bars
                zb2, za2 = ohlc.low_bid.where(is_bull, ohlc.high_bid).shift(
                    1, freq=in_middle1
                ), ohlc.low_ask.where(is_bull, ohlc.high_ask).shift(1, freq=in_middle1)
                zb3, za3 = ohlc.high_bid.where(is_bull, ohlc.low_bid).shift(
                    1, freq=in_middle2
                ), ohlc.high_ask.where(is_bull, ohlc.low_ask).shift(1, freq=in_middle2)
            else:
                # - old logic: O,L,H,C for all bars
                zb2, za2 = ohlc.low_bid.shift(1, freq=in_middle1), ohlc.low_ask.shift(
                    1, freq=in_middle1
                )
                zb3, za3 = ohlc.high_bid.shift(1, freq=in_middle2), ohlc.high_ask.shift(
                    1, freq=in_middle2
                )
            p2 = pd.DataFrame(
                {
                    "bid": zb2,
                    "ask": za2,
                    "bidvol": np.nan,
                    "askvol": np.nan,
                    "is_real": 0,
                }
            )
            p3 = pd.DataFrame(
                {
                    "bid": zb3,
                    "ask": za3,
                    "bidvol": np.nan,
                    "askvol": np.nan,
                    "is_real": 0,
                }
            )

        zb = ohlc.close_bid.shift(1, freq=before_close), ohlc.close_ask.shift(
            1, freq=before_close
        )
        p4 = pd.DataFrame(
            {
                "bid": zb,
                "ask": za,
                "bidvol": default_size,
                "askvol": default_size,
                "is_real": 1,
            }
        )

    # final dataframe
    return pd.concat((p1, p2, p3, p4), axis=0).sort_index()


def load_tick_price_block(
    data_src: DataSource,
    info: Union[None, dsinfo],
    instruments: List[str],
    t_start,
    spread_info: Union[float, Dict[str, float]] = 0.0,
    exec_by_new_update=False,
    default_volume=1e12,
    logger=None,
    broker_info: BrokerInfo = None,
) -> pd.DataFrame:
    """
    Loading tick prices block from datasource.

    It returns pandas dataframe like that:

                                                SPY               |            XOM
                                    bid     ask    bidvol  askvol |  bid     ask    bidvol  askvol
    2000-01-01 09:30:01.123456      50.0    50.01    100    200   |  35.0    35.02    400    100
    2000-01-01 09:30:01.456888      50.0    50.01    100    200   |  35.0    35.02    400    100
    2000-01-01 09:30:02.123456      50.0    50.01    100    200   |  35.0    35.02    400    100

    :param data_src: data source
    :param info: datasource info
    :param instruments: list of instruments
    :param spread_info: dictionary of instrument's bid/ask spreads or fixed float value (default 0.0)
    :param t_start: block's start time
    :param exec_by_new_update: if true execution on next quote
    :param default_volume: default bid/ask volumes for OHLC to ticks convertor
    :param logger: logger
    :param broker_info: information about broker's sessions time etc (to prepare ticks from OHLC)
    :return: prices as combined dataframe
    """
    t_start = pd.Timestamp(t_start) if isinstance(t_start, str) else t_start

    # if there is no any info about data structure we could try to recognixe it here
    if info is None:
        info = recognize_datasource_structure(
            data_src, instruments, t_start, t_start + pd.Timedelta("30d"), logger
        )

    # load prices from datasource
    price_dict = data_src.load_data(
        instruments, start=t_start, end=t_start + info.load_block_amnt
    )

    if info.type == dstype.OHLC:
        session_start, session_end = None, None
        if broker_info is not None:
            bst = broker_info.session_times()
            if len(bst) > 1:
                session_start, session_end = pd.Timedelta(bst[0]), pd.Timedelta(bst[1])
        price_dict = convert_ohlc_to_ticks(
            price_dict,
            spread_info,
            freq=info.freq,
            default_size=default_volume,
            session_start=session_start,
            session_end=session_end,
            reverse_order_for_bullish_bars=True,
        )
    else:
        pass

    # generate keys for combined data - it takes care about repeated symbols
    keys = [
        "%s_%d" % (v, instruments[:i].count(v) + 1) if instruments.count(v) > 1 else v
        for i, v in enumerate(instruments)
    ]

    # concat them to single dataframe
    prices_df = pd.concat(
        [
            price_dict[instr].filter(
                items=["bid", "ask", "bidvol", "askvol", "is_real"]
            )
            for instr in instruments
        ],
        axis=1,
        keys=keys,
    )
    if exec_by_new_update:
        prices_df = prices_df.bfill().ffill()
    else:
        prices_df = prices_df.ffill().bfill()

    # 2022-Dec-29: if we have more than 1 instrument in this block and they are not intesected in time
    #              so it might happened that first instrument has all NaNs in 'is_real' field
    #              as result all simulation would be skipped for this price block
    #              we need to get combined (from all instruments) 'is_real' column
    #              and propagate it to all instruments to avoid exclusion from simulation
    if "is_real" in prices_df.columns.get_level_values(level=1):
        idx = pd.IndexSlice
        # we need to use 'keys' here instead of instruments because we may have multiple
        # entries for single symbol in case when we use aux instruments
        combined = reduce(
            lambda x, y: x.combine_first(y),
            [prices_df[s]["is_real"].copy().fillna(-1).astype(int) for s in keys],
        ).replace(-1, np.nan)
        for s in keys:
            prices_df.loc[idx[:, (s, "is_real")]] = combined

    if not prices_df.empty:
        if logger is not None:
            logger.info(
                "Loaded %d tick price records [%s ~ %s]"
                % (len(prices_df), str(prices_df.index[0]), str(prices_df.index[-1]))
            )
    else:
        if logger is not None:
            logger.info(
                "Loaded empty prices data for request [%s ~ %s]"
                % (str(t_start), str(t_start + info.load_block_amnt))
            )

    return prices_df


def generate_simulation_identificator(clz, brok, date) -> str:
    """
    Create simulation ID from class, broker and simulation date
    """
    return hashlib.sha256(("%s/%s/%s" % (clz, brok, date)).encode("utf-8")).hexdigest()[
        :3
    ].upper() + pd.Timestamp(date).strftime("%y%m%d%H%M")


def _wrap_single_list(param_grid: Union[List, Dict]):
    """
    Wraps all non list values as single
    :param param_grid:
    :return:
    """
    as_list = lambda x: x if isinstance(x, (tuple, list, dict, np.ndarray)) else [x]
    if isinstance(param_grid, list):
        return [_wrap_single_list(ps) for ps in param_grid]
    return {k: as_list(v) for k, v in param_grid.items()}


def permutate_params(
    parameters: dict,
    conditions: Union[types.FunctionType, list, tuple] = None,
    wrap_as_list=False,
) -> List[Dict]:
    """
    Generate list of all permutations for given parameters and theirs possible values

    Example:

    >>> def foo(par1, par2):
    >>>     print(par1)
    >>>     print(par2)
    >>>
    >>> # permutate all values and call function for every permutation
    >>> [foo(**z) for z in permutate_params({
    >>>                                       'par1' : [1,2,3],
    >>>                                       'par2' : [True, False]
    >>>                                     }, conditions=lambda par1, par2: par1<=2 and par2==True)]

    1
    True
    2
    True

    :param conditions: list of filtering functions
    :param parameters: dictionary
    :param wrap_as_list: if True (default) it wraps all non list values as single lists (required for sklearn)
    :return: list of permutations
    """
    if conditions is None:
        conditions = []
    elif isinstance(conditions, types.FunctionType):
        conditions = [conditions]
    elif isinstance(conditions, (tuple, list)):
        if not all([isinstance(e, types.FunctionType) for e in conditions]):
            raise ValueError("every condition must be a function")
    else:
        raise ValueError("conditions must be of type of function, list or tuple")

    args = []
    vals = []
    for k, v in parameters.items():
        args.append(k)
        vals.append([v] if not isinstance(v, (list, tuple)) else v)
    d = [dict(zip(args, p)) for p in product(*vals)]
    result = []
    for params_set in d:
        conditions_met = True
        for cond_func in conditions:
            func_param_args = cond_func.__code__.co_varnames
            func_param_values = [params_set[arg] for arg in func_param_args]
            if not cond_func(*func_param_values):
                conditions_met = False
                break
        if conditions_met:
            result.append(params_set)

    # if we need to follow sklearn rules we should wrap every noniterable as list
    return _wrap_single_list(result) if wrap_as_list else result


def __create_brokerage_instances(
    spread: Union[dict, float], tcc: TransactionCostsCalculator = None
) -> dict:
    """
    Some predefined list of broerages
    """

    def _binance_generator(broker_class, mtype, currencies):
        def _f(_cl, _t, _i, _c):
            return lambda: _cl(spread=spread, tcc=BinanceRatesCommon(_t, _i, _c))

        return {
            f"binance_{mtype}_vip{i}_{c.lower()}": _f(
                broker_class, mtype, f"vip{i}", c.upper()
            )
            for i in range(10)
            for c in currencies
        }

    def _woox_generator(broker_class, mtype, currencies):
        def _f(_cl, _t, _i, _c):
            return lambda: _cl(spread=spread, tcc=WooXRatesCommon(_t, _i, _c))

        return {
            f"woox_{mtype}_t{i}_{c.lower()}": _f(
                broker_class, mtype, f"t{i}", c.upper()
            )
            for i in range(7)
            for c in currencies
        }

    def _ftx_generator(broker_class, currencies):
        def _f(_cl, _i, _c):
            return lambda: _cl(spread=spread, tcc=FTXRatesCommon(None, _i, _c))

        return {
            f"ftx_t{i + 1}_{c.lower()}": _f(broker_class, f"t{i + 1}", c.upper())
            for i in range(6)
            for c in currencies
        }

    def _kraken_generator(broker_class, mtype, currencies):
        def _f(_cl, _t, _i, _c):
            return lambda: _cl(spread=spread, tcc=KrakenRatesCommon(_t, _i, _c))

        if mtype.lower() == "spot":
            return {
                f"kraken_spot_{i}_{c.lower()}": _f(broker_class, mtype, i, c.upper())
                for i in _KRAKEN_SPOT_FEES.keys()
                for c in currencies
            }

        if mtype.lower() == "futures":
            return {
                f"kraken_futures_{i}_{c.lower()}": _f(broker_class, mtype, i, c.upper())
                for i in _KRAKEN_FUTURES_FEES.keys()
                for c in currencies
            }

        raise ValueError(f"Unknown instruments type {mtype} for Kraken exchange !")

    return {
        "stock": lambda: GenericStockBrokerInfo(
            spread=spread, tcc=StockTCC(0.05 / 100) if tcc is None else tcc
        ),
        "forex": lambda: GenericForexBrokerInfo(
            spread=spread, tcc=ForexTCC(35 / 1e6, 35 / 1e6) if tcc is None else tcc
        ),
        "crypto": lambda: GenericCryptoBrokerInfo(
            spread=spread, tcc=ZeroTCC() if tcc is None else tcc
        ),
        "crypto_futures": lambda: GenericCryptoFuturesBrokerInfo(
            spread=spread, tcc=ZeroTCC() if tcc is None else tcc
        ),
        # --- some predefined ---
        "bitmex": lambda: GenericCryptoFuturesBrokerInfo(
            spread=spread, tcc=BitmexTCC()
        ),
        "bitmex_vip": lambda: GenericCryptoFuturesBrokerInfo(
            spread=spread, tcc=BitmexTCC(0.03 / 100, -0.01 / 100)
        ),
        # - Binance spot
        **_binance_generator(GenericCryptoBrokerInfo, "spot", ["USDT", "BNB"]),
        # - Binance um
        **_binance_generator(
            GenericCryptoFuturesBrokerInfo,
            "um",
            ["USDT", "USDT_BNB", "BUSD", "BUSD_BNB"],
        ),
        # - WooX spot
        **_woox_generator(GenericCryptoBrokerInfo, "spot", ["USDT"]),
        # - WooX spot
        **_woox_generator(GenericCryptoFuturesBrokerInfo, "futures", ["USDT"]),
        # - FTX
        **_ftx_generator(GenericCryptoFuturesBrokerInfo, ["USD"]),
        # - Kraken spot
        **_kraken_generator(GenericCryptoBrokerInfo, "spot", ["USD"]),
        # - Kraken futures
        **_kraken_generator(GenericCryptoFuturesBrokerInfo, "futures", ["USD"]),
        "dukas": lambda: GenericForexBrokerInfo(
            spread=spread, tcc=ForexTCC(35 / 1e6, 35 / 1e6)
        ),
        "dukas_premium": lambda: GenericForexBrokerInfo(
            spread=spread, tcc=ForexTCC(17.5 / 1e6, 17.5 / 1e6)
        ),
        "fxcm": lambda: GenericForexBrokerInfo(spread=spread, tcc=FxcmTCC()),
    }


def __instantiate_simulated_broker(
    broker, spread: Union[dict, float], tcc: TransactionCostsCalculator = None
) -> BrokerInfo:
    if isinstance(broker, str):
        # for general brokers we require implicit spreads here
        if spread is None:
            raise ValueError(
                "Spread policy must be specified ! You need pass either fixed spread or dictionary"
            )

        predefined_brokers = __create_brokerage_instances(spread, tcc)

        brk_ctor = predefined_brokers.get(broker)
        if brk_ctor is None:
            raise ValueError(
                f"Unknown broker type '{broker}'\nList of supported brokers: [{','.join(predefined_brokers.keys())}]"
            )

        # instantiate broker
        broker = brk_ctor()

    return broker


def get_trading_cost_calculator(name: str) -> TransactionCostsCalculator:
    predefined_brokers = __create_brokerage_instances(0, None)
    brk_ctor = predefined_brokers.get(name)
    if brk_ctor:
        return brk_ctor().tcc
    raise ValueError(f"Can't find {name} TCC !")


def ls_brokers() -> List[str]:
    """
    List of simulated brokers supported by default
    """
    return [
        f"{k}({str(v().tcc)})" for k, v in __create_brokerage_instances(0, None).items()
    ]


def _progress_bar(description="[Backtest]"):
    """
    Default progress bar (based on tqdm)
    """
    if runtime_env() == "notebook":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    class __MyProgress:
        def __init__(self, descr):
            self.p = tqdm(
                total=100, unit_divisor=1, unit_scale=1, unit=" quotes", desc=descr
            )

        def __call__(self, i, label=None):
            d = i - self.p.n
            if d > 0:
                self.p.update(d)

    return __MyProgress(description)


def variate(clz, *args, conditions=None, **kwargs) -> Dict[str, Any]:
    """
    Make variations of parameters for simulation tests
    Example:

    >>>    class MomentumStrategy_Ex1_test:
    >>>       def __init__(self, p1, lookback_period=10, filter_type='sma', skip_entries_flag=False):
    >>>            self.p1, self.lookback_period, self.filter_type, self.skip_entries_flag = p1, lookback_period, filter_type, skip_entries_flag
    >>>
    >>>        def __repr__(self):
    >>>            return self.__class__.__name__ + f"({self.p1},{self.lookback_period},{self.filter_type},{self.skip_entries_flag})"
    >>>
    >>>    variate(MomentumStrategy_Ex1_test, 10, lookback_period=[1,2,3], filter_type=['ema', 'sma'], skip_entries_flag=[True, False])

    Output:
    >>>    {
    >>>        'MSE1t_(lp=1,ft=ema,sef=True)':  MomentumStrategy_Ex1_test(10,1,ema,True),
    >>>        'MSE1t_(lp=1,ft=ema,sef=False)': MomentumStrategy_Ex1_test(10,1,ema,False),
    >>>        'MSE1t_(lp=1,ft=sma,sef=True)':  MomentumStrategy_Ex1_test(10,1,sma,True),
    >>>        'MSE1t_(lp=1,ft=sma,sef=False)': MomentumStrategy_Ex1_test(10,1,sma,False),
    >>>        'MSE1t_(lp=2,ft=ema,sef=True)':  MomentumStrategy_Ex1_test(10,2,ema,True),
    >>>        'MSE1t_(lp=2,ft=ema,sef=False)': MomentumStrategy_Ex1_test(10,2,ema,False),
    >>>        'MSE1t_(lp=2,ft=sma,sef=True)':  MomentumStrategy_Ex1_test(10,2,sma,True),
    >>>        'MSE1t_(lp=2,ft=sma,sef=False)': MomentumStrategy_Ex1_test(10,2,sma,False),
    >>>        'MSE1t_(lp=3,ft=ema,sef=True)':  MomentumStrategy_Ex1_test(10,3,ema,True),
    >>>        'MSE1t_(lp=3,ft=ema,sef=False)': MomentumStrategy_Ex1_test(10,3,ema,False),
    >>>        'MSE1t_(lp=3,ft=sma,sef=True)':  MomentumStrategy_Ex1_test(10,3,sma,True),
    >>>        'MSE1t_(lp=3,ft=sma,sef=False)': MomentumStrategy_Ex1_test(10,3,sma,False)
    >>>    }

    and using in simuation:
    >>>    r = simulation(
    >>>             variate(MomentumStrategy_Ex1_test, 10, lookback_period=[1,2,3], filter_type=['ema', 'sma'], skip_entries_flag=[True, False]),
    >>>             data,
    >>>             'binance_um_vip0_usdt', start='2021-11-01',  stop='2023-06-01')
    """

    def _cmprss(xs: str):
        return "".join([x[0] for x in re.split("((?<!-)(?=[A-Z]))|_|(\d)", xs) if x])

    sfx = _cmprss(clz.__name__)
    to_excl = [s for s, v in kwargs.items() if not isinstance(v, (list, set, tuple))]
    dic2str = lambda ds: [
        _cmprss(k) + "=" + str(v) for k, v in ds.items() if k not in to_excl
    ]

    return {
        f"{sfx}_({ ','.join(dic2str(z)) })": clz(*args, **z)
        for z in permutate_params(kwargs, conditions=conditions)
    }
