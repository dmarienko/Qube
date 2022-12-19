import unittest
from collections import defaultdict
from dataclasses import dataclass

from tqdm import tqdm

from qube.learn.core.data_utils import merge_ticks_from_dict
from qube.series.BarSeries import BarSeries
from qube.series.Indicators import (Indicator, Sma, Ema, ATR, Returns,
                                    RollingStd)
from qube.series.Quote import Quote
from qube.tests.data.test_mdf import generate_feed


def create_indicators(timeframe, period1, period2, vol_period=15):
    @dataclass
    class Container:
        ohlc: BarSeries  # basic OHLC timeseries
        ma1: Indicator  # ema 1
        ma2: Indicator  # ema 2
        vol: Indicator  # atr volatility
        vol_std: Indicator  # standard deviation of returns as volatility proxy

    # 1. create series
    ohlc = BarSeries(timeframe)

    # 2.a create fast ma
    ohlc.attach(i1 := Ema(period1))

    # 2.b create slow ma
    ohlc.attach(i2 := Ema(period2))

    # 2.c create volatility
    ohlc.attach(i3 := ATR(vol_period))

    # 2.d another way to measure volatility: std on returns
    rets = Returns(100)  # just multiply returns by 100 to get percentages
    m_model = Sma(vol_period)
    i4 = RollingStd(vol_period, m_model)
    rets.attach(m_model).attach(i4)

    # 3. attach all indicators to basic series
    ohlc.attach(rets)

    return Container(ohlc=ohlc, ma1=i1, ma2=i2, vol=i3, vol_std=i4)


class IndicatorDemoTest(unittest.TestCase):

    def test_demo_indicator(self):
        """
        Example of using indicators framework
        """
        instruments = ['BTCUSDT', 'ETHUSDT']

        # - prepare 'market data' udates { symbol: [quotes] }
        m_qts = merge_ticks_from_dict(
            {s: generate_feed("2022-01-01", 10.0, 10_000) for s in instruments},
            instruments=instruments
        )
        market_quotes_updates = defaultdict(list)
        for s in instruments:
            for t, a, b, av, bv in zip(m_qts[s].index, m_qts[s].ask, m_qts[s].bid, m_qts[s].askvol, m_qts[s].bidvol):
                market_quotes_updates[t].append((s, Quote(t, b, a, bv, av)))

        # create structures
        xdata = {k: create_indicators('1Min', 2, 7) for k in instruments}
        # print(xdata)

        # - now we emulate updates by quotes data for every instrument
        for t, qs in market_quotes_updates.items():
            for s, q in qs:
                xd = xdata[s]
                # - update by quote: returns true if new bar is just formed
                if xd.ohlc.update_by_quote(q):

                    # - some 'logic': check if fast MA on closes crossed slow MA
                    if len(xd.ma1) > 1 and len(xd.ma2) > 1:
                        if xd.ma1[2] < xd.ma2[2] and xd.ma1[1] > xd.ma2[1]:
                            print(f"""{s} \t| {q.time}: cross UP and ATR: {xd.vol[1]:.3f} | Std ret: {xd.vol_std[1]:.3f}%""")

        print(">> Volatility 1 (ATR)", xdata[instruments[0]].vol[:])
        print(">> Volatility 2 (Std on returns)", xdata[instruments[0]].vol_std[:])
        self.assertTrue(len(xdata[instruments[0]].ohlc) > 0)


from pytest import main
if __name__ == '__main__':
    main()