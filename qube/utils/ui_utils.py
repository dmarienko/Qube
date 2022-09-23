from qube.utils.utils import runtime_env, mstruct


def __wrap_with_color(code):
    def inner(text, bold=False):
        c = code
        if bold:
            c = "1;%s" % c
        return "\033[%sm%s\033[0m" % (c, text)

    return inner


red, green, yellow, blue, magenta, cyan, white = (
    __wrap_with_color('31'),
    __wrap_with_color('32'),
    __wrap_with_color('33'),
    __wrap_with_color('34'),
    __wrap_with_color('35'),
    __wrap_with_color('36'),
    __wrap_with_color('37'),
)


def ui_progress_bar(name: str, width='99%'):
    if runtime_env() == 'notebook':
        import ipywidgets as widgets
        V, H, L = widgets.VBox, widgets.HBox, widgets.Label
        layout = {'width': width}
        progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout=layout)
        info = widgets.HTML('---', layout={**layout, **{'border': '1px dashed #205020'}})
        h = V([L(name, layout={**layout, **{'border': '1px solid #205020'}}), info, progress],
              layout=widgets.Layout(grid_raw_gap='1px'))
        return mstruct(panel=h, progress=progress, info=info, name=name)

    # empty progress (for running in shell)
    return mstruct(panel=None, progress=mstruct(value=None, style=mstruct(bar_color=None)), info=mstruct(value=None),
                   name=name)


def ui_confirmation_dialog(confirmation_callback, cancel_callback, output,
                           text="Are you sure want to run optimization?", confirm_text="Run", cancel_text="Cancel"):
    if runtime_env() == 'notebook':
        import ipywidgets as widgets
        confirm_button = widgets.Button(description=confirm_text, button_style="success")
        confirm_button.on_click(confirmation_callback)
        cancel_button = widgets.Button(description=cancel_text, button_style="danger")
        cancel_button.on_click(cancel_callback)
        return widgets.VBox([widgets.Label(text), widgets.HBox([confirm_button, cancel_button]), output])
