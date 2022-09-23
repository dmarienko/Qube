import copy

import pandas

_pd2hc_kind = {
    "bar": "column",
    "barh": "bar",
    "area": "area",
    "line": "line",
    "pie": "pie"
}


def pd2hc_kind(kind):
    if kind not in _pd2hc_kind:
        raise ValueError("%(kind)s plots are not yet supported" % locals())
    return _pd2hc_kind[kind]


_pd2hc_linestyle = {
    "-": "Solid",
    "--": "Dash",
    "-.": "DashDot",
    ":": "Dot"
}

_default_series_opt = {
    'connectNulls': True,
}

_default_y_axis = {
    'opposite': False,
    'resize': {
        'enabled': True
    },
    'offset': 0}

_default_tooltip = {
    'valueDecimals': 2,
    'split': True
}

_default_boost = {
    'seriesThreshold': 20
}

_default_chart = {
    'zoomType': 'x'
}

_default_multichart_height = 150
_default_resize_height = 190


def pd2hc_linestyle(linestyle):
    if linestyle not in _pd2hc_linestyle:
        raise ValueError("%(linestyle)s linestyles are not yet supported" % locals())
    return _pd2hc_linestyle[linestyle]


def json_encode(obj):
    return pandas.io.json.dumps(obj)


def serialize(df, output_type="javascript", chart_type="stock", *args, **kwargs):
    def serialize_chart(df, output, *args, **kwargs):
        output["chart"] = copy.deepcopy(_default_chart)
        if 'render_to' in kwargs:
            output['chart']['renderTo'] = kwargs['render_to']
        if "figsize" in kwargs:
            output["chart"]["width"] = kwargs["figsize"][0]
            output["chart"]["height"] = kwargs["figsize"][1]
        if "kind" in kwargs:
            output["chart"]["type"] = pd2hc_kind(kwargs["kind"])
        if kwargs.get('polar'):
            output['chart']['polar'] = True

    def serialize_colors(df, output, *args, **kwargs):
        if kwargs.get("colors"):
            output['colors'] = kwargs["colors"]

    def serialize_credits(df, output, *args, **kwargs):
        pass

    def serialize_data(df, output, *args, **kwargs):
        pass

    def serialize_drilldown(df, output, *args, **kwargs):
        pass

    def serialize_exporting(df, output, *args, **kwargs):
        pass

    def serialize_labels(df, output, *args, **kwargs):
        pass

    def serialize_legend(df, output, *args, **kwargs):
        output["legend"] = {
            "enabled": kwargs.get("legend", True)
        }

    def serialize_loading(df, output, *args, **kwargs):
        pass

    def serialize_navigation(df, output, *args, **kwargs):
        pass

    def serialize_noData(df, output, *args, **kwargs):
        pass

    def serialize_pane(df, output, *args, **kwargs):
        pass

    def serialize_plotOptions(df, output, *args, **kwargs):
        pass

    def serialize_boost(df, output, *args, **kwargs):
        output['boost'] = _default_boost
        if 'boost' in kwargs:
            output['boost'].update(kwargs['boost'])

    def serialize_series(df, output, *args, **kwargs):
        def is_secondary(c, **kwargs):
            return c in kwargs.get("secondary_y", [])

        if kwargs.get('sort_columns'):
            df = df.sort_index()
        series = df.to_dict('series')
        output["series"] = []
        for name, data in series.items():
            if df[name].dtype.kind in "biufc":
                sec = is_secondary(name, **kwargs)
                d = {
                    "name": name if not sec or not kwargs.get("mark_right", True) else name + " (right)",
                    "yAxis": int(sec),
                    "data": list(zip(df.index, data.values.tolist()))
                }
                d.update(_default_series_opt)
                if kwargs.get('polar'):
                    d['data'] = [v for k, v in d['data']]
                if kwargs.get("kind") == "area" and kwargs.get("stacked", True):
                    d["stacking"] = 'normal'
                if kwargs.get("style"):
                    d["dashStyle"] = pd2hc_linestyle(kwargs["style"].get(name, "-"))
                if kwargs.get('series_opt'):
                    d.update(kwargs['series_opt'])
                output["series"].append(d)

    def serialize_subtitle(df, output, *args, **kwargs):
        pass

    def serialize_title(df, output, *args, **kwargs):
        if "title" in kwargs:
            output["title"] = {"text": kwargs["title"]}

    def serialize_tooltip(df, output, *args, **kwargs):
        output['tooltip'] = _default_tooltip
        if 'tooltip' in kwargs:
            output['tooltip'].update(kwargs['tooltip'])

    def serialize_xAxis(df, output, *args, **kwargs):
        output["xAxis"] = {}
        if df.index.name:
            output["xAxis"]["title"] = {"text": df.index.name}
        if df.index.dtype.kind in "M":
            output["xAxis"]["type"] = "datetime"
        if df.index.dtype.kind == 'O':
            output['xAxis']['categories'] = sorted(list(df.index)) if kwargs.get('sort_columns') else list(df.index)
        if kwargs.get("grid"):
            output["xAxis"]["gridLineWidth"] = 1
            output["xAxis"]["gridLineDashStyle"] = "Dot"
        if kwargs.get("loglog") or kwargs.get("logx"):
            output["xAxis"]["type"] = 'logarithmic'
        if "xlim" in kwargs:
            output["xAxis"]["min"] = kwargs["xlim"][0]
            output["xAxis"]["max"] = kwargs["xlim"][1]
        if "rot" in kwargs:
            output["xAxis"]["labels"] = {"rotation": kwargs["rot"]}
        if "fontsize" in kwargs:
            output["xAxis"].setdefault("labels", {})["style"] = {"fontSize": kwargs["fontsize"]}
        if "xticks" in kwargs:
            output["xAxis"]["tickPositions"] = kwargs["xticks"]

    def serialize_yAxis(df, output, *args, **kwargs):
        yAxis = _default_y_axis
        if kwargs.get("grid"):
            yAxis["gridLineWidth"] = 1
            yAxis["gridLineDashStyle"] = "Dot"
        if kwargs.get("loglog") or kwargs.get("logy"):
            yAxis["type"] = 'logarithmic'
        if "ylim" in kwargs:
            yAxis["min"] = kwargs["ylim"][0]
            yAxis["max"] = kwargs["ylim"][1]
        if "rot" in kwargs:
            yAxis["labels"] = {"rotation": kwargs["rot"]}
        if "fontsize" in kwargs:
            yAxis.setdefault("labels", {})["style"] = {"fontSize": kwargs["fontsize"]}
        if "yticks" in kwargs:
            yAxis["tickPositions"] = kwargs["yticks"]
        output["yAxis"] = [yAxis]
        if kwargs.get("secondary_y"):
            yAxis2 = copy.deepcopy(yAxis)
            yAxis2["opposite"] = True
            output["yAxis"].append(yAxis2)

    def serialize_zoom(df, output, *args, **kwargs):
        if "zoom" in kwargs:
            if kwargs["zoom"] not in ("x", "y", "xy"):
                raise ValueError("zoom must be in ('x', 'y', 'xy')")
            output["chart"]["zoomType"] = kwargs["zoom"]

    output = {}
    df_copy = copy.deepcopy(df)
    if "x" in kwargs:
        df_copy.index = df_copy.pop(kwargs["x"])
    if kwargs.get("use_index", True) is False:
        df_copy = df_copy.reset_index()
    if "y" in kwargs:
        df_copy = pandas.DataFrame(df_copy, columns=kwargs["y"])
    serialize_chart(df_copy, output, *args, **kwargs)
    serialize_colors(df_copy, output, *args, **kwargs)
    serialize_credits(df_copy, output, *args, **kwargs)
    serialize_data(df_copy, output, *args, **kwargs)
    serialize_drilldown(df_copy, output, *args, **kwargs)
    serialize_exporting(df_copy, output, *args, **kwargs)
    serialize_labels(df_copy, output, *args, **kwargs)
    serialize_legend(df_copy, output, *args, **kwargs)
    serialize_loading(df_copy, output, *args, **kwargs)
    serialize_navigation(df_copy, output, *args, **kwargs)
    serialize_noData(df_copy, output, *args, **kwargs)
    serialize_pane(df_copy, output, *args, **kwargs)
    serialize_plotOptions(df_copy, output, *args, **kwargs)
    serialize_series(df_copy, output, *args, **kwargs)
    serialize_subtitle(df_copy, output, *args, **kwargs)
    serialize_title(df_copy, output, *args, **kwargs)
    serialize_tooltip(df_copy, output, *args, **kwargs)
    serialize_xAxis(df_copy, output, *args, **kwargs)
    serialize_yAxis(df_copy, output, *args, **kwargs)
    serialize_zoom(df_copy, output, *args, **kwargs)
    serialize_boost(df_copy, output, *args, **kwargs)
    if output_type == "dict":
        return output
    if output_type == "json":
        return json_encode(output)
    if chart_type == "stock":
        return "new Highcharts.StockChart(%s);" % json_encode(output)
    return "new Highcharts.Chart(%s);" % json_encode(output)


def union_charts(dict_charts, heights=None):
    if not heights:
        heights = [_default_multichart_height] * len(dict_charts)
    main_chart = copy.deepcopy(dict_charts[0])
    main_chart['series'] = []
    main_chart['yAxis'] = []
    main_chart['title'] = ''
    main_chart['chart']['height'] = 0
    main_chart['chart']['type'] = ''
    main_chart['colors'] = []
    for height in heights:
        main_chart['chart']['height'] += height
    main_chart['chart']['height'] += _default_resize_height
    for idx, chart in enumerate(dict_charts):
        for idx_series in range(len(chart['series'])):
            chart['series'][idx_series]['yAxis'] = idx
            if chart['chart'].get('type'):
                chart['series'][idx_series]['type'] = chart['chart']['type']
        main_chart['series'].extend(chart['series'])
        chart['yAxis'][0] = copy.deepcopy(_default_y_axis)
        chart['yAxis'][0]['height'] = '%spx' % heights[idx]
        chart['yAxis'][0]['top'] = '%spx' % (sum(heights[:idx]))
        chart['yAxis'][0]['title'] = chart['title']
        main_chart['colors'].extend(chart['colors'])
        main_chart['yAxis'].extend(chart['yAxis'])
    return "new Highcharts.StockChart(%s);" % json_encode(main_chart)
