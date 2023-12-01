from typing import Union
import pandas as pd
import numpy as np
from qube.quantitative.tools import scols 

from qube.simulator.core import SimulationResult


def effr(xs):
    sad = abs(xs.diff()).sum()
    dif = xs[-1] - xs[0]
    
    # if not np.isfinite(dif) or not np.isfinite(sad):
    # print('---', dif/sad, dif, sad)
        
    return (dif / sad) if sad != 0.0 else 0


def signals_statistics(backtest: Union[SimulationResult, pd.DataFrame], symbol: str, 
                       prices: Union[dict, pd.DataFrame], short_stats=False, fixed_exit_time_after=None) -> pd.DataFrame:
    """
    Calculates statistics of executed signals using Qube execution log records
    """
    if isinstance(prices, dict):
        prices = prices[symbol]
        
    if isinstance(backtest, SimulationResult):
        ex = backtest.executions[backtest.executions.instrument==symbol].copy()
    elif isinstance(backtest, pd.DataFrame) and all(backtest.columns.isin(['instrument', 'quantity', 'exec_price', 'commissions', 'comment'])):
        ex = backtest[backtest.instrument==symbol].copy()
        
    ex = ex.assign(timestamp=ex.index.values)
    qcs = ex.quantity.cumsum()
    pex = ex.assign(qcs=qcs, qcs1=qcs.shift(1), quantity1=ex.quantity.shift(1))
    fixed_exit_time = pd.Timedelta(fixed_exit_time_after) if fixed_exit_time_after is not None else 0
    if fixed_exit_time != 0: 
        print('WARN: Overriding actual postion time by new time interval !')

    sgns = {}
    o = None
    for t, (q, q1, cq, cq1, pe, c) in zip(pex.index, pex[['quantity', 'quantity1', 'qcs', 'qcs1', 'exec_price', 'commissions']].values):
        dr = np.sign(cq)

        if cq != 0 and (cq1 == 0 or np.isnan(cq1)):
            # print('OP', t)
            o = [t, cq, pe, c, dr]
            continue

        if cq == 0 and cq1 != 0:
            # print(o[0],  'LIQ', t)
            sgns[o[0]] = {
                'direction': o[4],
                'entry_time':o[0], 'quantity': o[1], 'entry_price': o[2], 
                'exit_time': t, 'exit_price': pe,
                'commissions': o[3] + c,
            }
            continue

        if np.sign(cq) != np.sign(cq1) and np.isfinite(cq1) and cq1 != 0:
            # print(o[0],  'REV', t)
            sgns[o[0]] = {
                'direction': o[4],
                'entry_time':o[0], 'quantity': o[1], 'entry_price': o[2], 
                'exit_time': t, 'exit_price': pe,
                'commissions': o[3] + c,
            }
            o = [t, cq, pe, c, dr]

    sgns = pd.DataFrame.from_dict(sgns, orient='index')
    ampl_x = sgns.direction * (sgns.exit_price - sgns.entry_price)
    sgns = sgns.assign(
        hold_time = sgns.exit_time - sgns.entry_time,
        signed_change = sgns.exit_price - sgns.entry_price,
        pnl = (sgns.exit_price - sgns.entry_price) * (sgns.quantity / sgns.entry_price ),
        ampl_x = ampl_x,
        ampl_x_pct = ampl_x / sgns.entry_price,
    )

    if not short_stats:
        stats, idx  = {}, 0
        for te, (pe, px, d, tx) in zip(sgns.index, sgns[['entry_price', 'exit_price', 'direction', 'exit_time']].values):
            # if we want to override actual exit by new time based interval
            tx = (te + fixed_exit_time) if fixed_exit_time != 0 else tx 
            
            p_close = prices.close[te:tx]
            if p_close.empty:
                # tx = prices.index[prices.index.get_loc(tx, method='bfill')]
                tx = prices.index[prices.index.get_indexer([tx], method='bfill')[0]]
                p_close = prices.close[te:tx]
                
            p_highs = prices.high[te:tx]
            p_lows = prices.low[te:tx]

            if 0:
                ampl_max = max(p_close) if d > 0 else min(p_close)
                ampl_min = min(p_close) if d > 0 else max(p_close)
                ampl_time_max = (p_close.idxmax() - te) if d > 0 else (p_close.idxmin() - te)
                ampl_time_min = (p_close.idxmin() - te) if d > 0 else (p_close.idxmax() - te)
            else:
                ampl_max = (max(p_highs) - pe) if d > 0 else (pe - min(p_lows))
                ampl_min = (pe - min(p_lows)) if d > 0 else (max(p_highs) - pe)
                ampl_time_max = (p_highs.idxmax() - te) if d > 0 else (p_lows.idxmin() - te)
                ampl_time_min = (p_lows.idxmin() - te) if d > 0 else (p_highs.idxmax() - te)
            ampl_max_pct = ampl_max / pe
            ampl_min_pct = ampl_min / pe

            stats[te] = dict(
                er_x = d * effr(p_close),

                ampl_max = ampl_max,
                ampl_max_pct = ampl_max_pct,
                ampl_time_max = ampl_time_max,
                er_max = d * effr(p_close[te:te+ampl_time_max]),

                ampl_min = ampl_min,
                ampl_min_pct = ampl_min_pct,
                ampl_time_min = ampl_time_min,
                er_min = d * effr(p_close[te:te+ampl_time_min]),

                index = idx
            )
            idx += 1

        sgns = scols(sgns, pd.DataFrame.from_dict(stats, orient='index'))
        
    return sgns
