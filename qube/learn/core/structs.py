from datetime import timedelta
from dataclasses import dataclass
from typing import Union, List, Dict

QLEARN_VERSION = '0.0.5'
_FIELD_EXACT_TIME = 'exact_time'
_FIELD_MARKET_INFO = 'market_info_'


@dataclass
class MarketInfo:
    symbols: Union[List[str], None]
    column: str
    timezone: str = 'UTC'
    session_start = timedelta(hours=0, minutes=0)
    session_end = timedelta(hours=23, minutes=59, seconds=58)
    tick_sizes: dict = None
    tick_prices: dict = None
    debug: bool = False
