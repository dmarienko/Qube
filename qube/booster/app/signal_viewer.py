import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - some plotly customizations
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import plotly.io as pio

from qube.charting.lookinglass import LookingGlass
from qube.charting.plot_helpers import install_plotly_helpers
from qube.quantitative.tools import infer_series_frequency, ohlc_resample
from qube.utils.utils import dict2struct

_as_ts = lambda x: pd.Timestamp(x)

pio.templates.default = "plotly_dark"
install_plotly_helpers()


def aslist(o):
    return o if isinstance(o, (list, tuple)) else [o]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def show_signals(data, signals, start, end, t_frame='15Min', indicators={}, title='', data_margins='1d'):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    F1 = pd.Timedelta(t_frame)

    def _dict2struct(sx):
        if 'take_at' in sx and isinstance(sx['take_at'], dict):
            sx['take_at'] = [{'take': float(k), 'fraction': v} for k, v in sx['take_at'].items()]
        return dict2struct(sx)

    # get signals and select
    selected = [_dict2struct(s) for s in signals if start <= _as_ts(s["time"]) <= end]
    if not selected:
        return {}

    r_start = start
    r_end = _as_ts(aslist(selected[-1].risk_hit_time)[-1])

    r_start_v = r_start - pd.Timedelta(data_margins)
    r_end_v = r_end + pd.Timedelta(data_margins)
    data = data[r_start_v: r_end_v]

    if F1 > pd.Timedelta(infer_series_frequency(data[:100])):
        data = ohlc_resample(data, F1)

    g = LookingGlass(data, indicators, backend='plotly')
    lv = g.look(r_start_v, r_end_v, title=title)

    for s in selected:
        s.time = _as_ts(s.time)
        s.risk_hit_time = [_as_ts(x) for x in aslist(s.risk_hit_time)] if s.risk_hit_time else None
        entry_type = '?'

        # temporary fix
        if hasattr(s, 'entry'):
            s.entry.time = _as_ts(s.entry.time)
            s.entry.level.detected_time = _as_ts(s.entry.level.detected_time)
            s.entry.level.time = _as_ts(s.entry.level.time)
            s.entry.level.broken_time = _as_ts(s.entry.level.broken_time)
            entry_type = s.entry.entry_type

            # - show level
            l = s.entry.level

            _b_thresh = l.price + l.side * l.break_zone
            _U_thresh, _B_thresh = l.price + l.touch_zone, l.price - l.touch_zone

            lv = lv.rlinex(
                l.time, l.broken_time, l.price, c='red' if l.side < 0 else 'yellow'
            ).rlinex(
                l.time, l.broken_time, _b_thresh, c='#903030' if l.side < 0 else '#909030', lw=1, ls='dot'
            )

            lv = lv.rlinex(
                l.time, l.broken_time, _U_thresh, c='#903030' if l.side < 0 else '#909030', lw=0.5, ls='dash'
            ) if l.side < 0 else lv

            lv = lv.rlinex(
                l.time, l.broken_time, _B_thresh, c='#903030' if l.side < 0 else '#909030', lw=0.5, ls='dash'
            ) if l.side > 0 else lv

            # - entry
        lt0, lt1 = s.time - F1, s.time
        lv = lv.rlinex(lt0, lt1 + 2 * F1, s.price, c='white', ls='dot', lw=1.5)

        _rel = lt1 + 5 * F1
        if s.risk_hit_time:
            descr = f"{s.time.strftime('%H:%M')}: {'BUY' if s.direction > 0 else 'SLD'} {s.amount} @ {s.price:.2f} ({entry_type})"
            lv = lv.add_annotation(x=s.time, y=s.price, ax=0, ay=-20 if s.direction < 0 else 20,
                                   text=descr, arrowwidth=0.1, showarrow=True)

            x0, y0 = s.time, s.price
            for hit_time, hit_price, hit_take in zip(s.risk_hit_time, aslist(s.risk_hit_price),
                                                     aslist(s.risk_hit_take)):
                risk_txt = f"{hit_time.strftime('%H:%M')} {'TAKE' if hit_take else 'STOP'} at @ {hit_price:.2f}" if hit_time else ''
                c_text = '#10f010' if hit_take else '#ff0000'

                # arrow to stop | take
                lv = lv.arrow(x0, y0, hit_time, hit_price, text='', lw=2, c=c_text)

                _ay = -20 if (s.direction > 0 and hit_take) or (s.direction < 0 and not hit_take) else 20
                lv = lv.add_annotation(x=hit_time, y=hit_price, ax=0, ay=_ay,
                                       text=risk_txt, font={'size': 9, 'color': c_text}, arrowwidth=0.1, showarrow=True)

                # next arrow start
                x0, y0 = hit_time, hit_price

                _rel = hit_time

        if s.stop_at is not None:
            lv = lv.rlinex(lt0 - F1, _rel, s.stop_at, c='red', ls='dot', lw=2)

        if s.take_at is not None:
            lv = lv.rlinex(lt0 - F1, _rel, s.take_at, c='green', ls='dot', lw=2)

        # Info potantially contains important pattern points, stored as json string
        if hasattr(s, 'info'):
            if len(s.info) > 0:
                sig_info = np.array(json.loads(s.info))
                df = pd.DataFrame.from_records(sig_info)

                df['time'] = pd.to_datetime(df['time'], utc=True)
                df = df.set_index("time")
                df = df.dropna()

                lv = lv.add_traces([
                    go.Scatter(x=df.index, y=df.price, mode='markers', marker_symbol='cross-thin-open',
                               marker={'color': 'white', 'size': 15})
                ])

    return lv.hover(h=800)
