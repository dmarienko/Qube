from dataclasses import dataclass


@dataclass
class Instrument:
    # symbol in upper case
    symbol: str

    # true if it's futures
    is_futures: bool

    # tick size
    tick_size: float

    # avergae spread size
    spread: float

    # futures contract sizje
    futures_contract_size: float = 1

    # true if it needs other instrument to convert to base currency
    base_currency_instrument: 'Instrument' = None
