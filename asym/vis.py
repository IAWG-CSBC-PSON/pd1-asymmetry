from functools import lru_cache, partial
import pandas as pd
import numpy as np

import bokeh
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.server.server import Server
from bokeh.palettes import viridis, inferno, Colorblind
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    AutocompleteInput,
    Button,
    Circle,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    PreText,
    RadioButtonGroup,
    RangeSlider,
    Select,
    TextInput,
)
from bokeh.plotting import figure
from bokeh.transform import linear_cmap, factor_cmap

CELL_IMAGE_METRICS = (["mean", "sd", "min", "max"], [np.mean, np.std, np.min, np.max])

DOWNLOAD_JS = """
function table_to_csv(source) {
    const columns = Object.keys(source.data)
    const nrows = source.get_length()
    const lines = [columns.join(',')]

    for (let i = 0; i < nrows; i++) {
        let row = [];
        for (let j = 0; j < columns.length; j++) {
            const column = columns[j]
            row.push(source.data[column][i].toString())
        }
        lines.push(row.join(','))
    }
    return lines.join('\\n').concat('\\n')
}


const filename = 'data_result.csv'
const filetext = table_to_csv(source)
const blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' })

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename)
} else {
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.target = '_blank'
    link.style.visibility = 'hidden'
    link.dispatchEvent(new MouseEvent('click'))
}
"""


def round_signif(x, n=2):
    return float(
        np.format_float_positional(
            x, precision=n, unique=False, fractional=False, trim="k"
        )
    )


def prepare_server(
    doc,
    input_data,
    cell_stack,
    cell_markers=None,
    default_umap_marker=None,
    default_cell_marker=None,
):
    @lru_cache()
    def get_data(m=None):
        d = input_data
        if m is not None:
            m_v = d[m]
            if np.issubdtype(m_v.dtype, np.number):
                d["marker_val_num"] = m_v
            else:
                d["marker_val_cat"] = m_v
        if "marker_val_num" not in d:
            d["marker_val_num"] = np.arange(d.shape[0])
        if "marker_val_cat" not in d:
            d["marker_val_cat"] = np.full(d.shape[0], "a")
        return d

    @lru_cache()
    def get_cat_colors(n):
        return Colorblind[max(3, n)][:n]

    @lru_cache()
    def marker_cols(lower=False):
        markers = list(sorted(input_data.columns, key=lambda x: x.lower()))
        if lower:
            return {x.lower(): x for x in markers}
        return markers

    @lru_cache()
    def image_markers(lower=False, mapping=False):
        if mapping:
            return {
                y: j
                for j, y in sorted(
                    (
                        (i, x)
                        for i, x in enumerate(image_markers(lower=lower, mapping=False))
                    ),
                    key=lambda x: x[1].lower(),
                )
            }
        if lower:
            return [x.lower() for x in image_markers(lower=False, mapping=False)]
        return (
            cell_markers
            if cell_markers is not None
            else [f"Marker {i + 1}" for i in range(cell_stack.shape[1])]
        )

    input_data = input_data.copy()

    # Marker selection for UMAP plots
    ###########################################################################

    if default_umap_marker is None:
        default_umap_marker = marker_cols()[0]
    marker_select = Select(
        value=default_umap_marker, options=marker_cols(), title="Color UMAP by"
    )
    marker_input = AutocompleteInput(
        completions=marker_cols() + list(marker_cols(lower=True).keys()),
        min_characters=1,
        placeholder="Search for marker",
    )
    marker_slider = RangeSlider(
        start=0, end=1, value=(0, 1), step=0.1, orientation="vertical", direction="rtl"
    )

    # Data sources
    ###########################################################################

    source = ColumnDataSource(data=get_data(None))
    image_source = ColumnDataSource(data=dict(image=[], dw=[], dh=[]))

    # UMAP scatter plot for numeric data
    ###########################################################################

    umap_figure = figure(
        plot_width=800,
        plot_height=500,
        tools="pan,wheel_zoom,lasso_select,box_select,tap,reset",
        active_scroll="wheel_zoom",
        active_drag="box_select",
        active_tap="tap",
        toolbar_location="left",
    )
    marker_mapper = linear_cmap(
        field_name="marker_val_num",
        palette=inferno(10)[:-1],
        low=0,
        high=500,
        high_color=None,
    )
    umap_scatter_renderer = umap_figure.circle(
        "d1",
        "d2",
        size=8,
        source=source,
        fill_alpha=0.5,
        line_alpha=0.9,
        fill_color=marker_mapper,
        line_color=marker_mapper,
        selection_fill_alpha=0.8,
        selection_line_alpha=1,
        selection_line_color="black",
        nonselection_alpha=0.2,
        hover_line_color="black",
    )
    umap_color_bar = ColorBar(
        color_mapper=marker_mapper["transform"], width=12, location=(0, 0)
    )
    umap_figure.add_layout(umap_color_bar, "right")
    umap_figure.add_tools(
        HoverTool(tooltips=None, renderers=[umap_scatter_renderer], mode="mouse")
    )

    # UMAP scatter plot for categorical data
    ###########################################################################

    umap_figure_cat = figure(
        plot_width=800,
        plot_height=500,
        tools="pan,wheel_zoom,lasso_select,box_select,tap,reset",
        active_scroll="wheel_zoom",
        active_drag="box_select",
        active_tap="tap",
        x_range=umap_figure.x_range,
        y_range=umap_figure.y_range,
        toolbar_location="left",
    )
    marker_mapper_cat = factor_cmap(
        field_name="marker_val_cat", palette=["#000000"], factors=["a"]
    )
    umap_figure_cat.circle(
        "d1",
        "d2",
        size=8,
        source=source,
        legend_field="marker_val_cat",
        alpha=0.7,
        fill_color=marker_mapper_cat,
        line_color=None,
        selection_alpha=0.9,
        selection_line_color="black",
        nonselection_alpha=0.5,
    )
    umap_figure_cat.legend.location = "top_right"
    umap_figure_cat.legend.orientation = "vertical"

    # Cell picture plot
    ###########################################################################

    default_cell_marker = (
        0
        if default_cell_marker is None
        else image_markers(mapping=True)[default_cell_marker]
    )
    cell_markers_select = Select(
        value=str(default_cell_marker),
        options=list((str(i), x) for x, i in image_markers(mapping=True).items()),
        title="Marker cell image",
    )
    cell_marker_input = AutocompleteInput(
        completions=list(image_markers()) + list(image_markers(lower=True)),
        min_characters=1,
        placeholder="Search for marker",
    )
    cell_slider = RangeSlider(
        start=0, end=1, value=(0, 1), orientation="vertical", direction="rtl"
    )
    metric_select = RadioButtonGroup(active=0, labels=CELL_IMAGE_METRICS[0])
    stats = PreText(text="", width=100)

    cell_mapper = bokeh.models.mappers.LinearColorMapper(
        viridis(20), low=0, high=1000, high_color=None
    )
    cell_color_bar = ColorBar(color_mapper=cell_mapper, width=12, location=(0, 0))
    cell_figure = figure(
        plot_width=450,
        plot_height=350,
        tools="pan,wheel_zoom,reset",
        toolbar_location="left",
    )
    cell_image = cell_figure.image(
        image="image",
        color_mapper=cell_mapper,
        x=0,
        y=0,
        dw="dw",
        dh="dh",
        source=image_source,
    )
    cell_figure.add_layout(cell_color_bar, "right")

    # Edit data of selected cells
    ###########################################################################

    edit_selection_col = TextInput(title="Column")
    edit_selection_val = TextInput(title="Value")
    edit_selection_submit = Button(
        label="Submit change", button_type="primary", align=("start", "end")
    )
    edit_selecton_log = PreText(text="")

    download_button = Button(
        label="Download cell data", button_type="success", align=("start", "end")
    )
    download_button.js_on_click(CustomJS(args=dict(source=source), code=DOWNLOAD_JS))

    # Callbacks for buttons and widgets
    ###########################################################################

    def marker_change(attrname, old, new):
        update()

    def cell_slider_change(attrname, old, new):
        cell_mapper.low = new[0]
        cell_mapper.high = new[1]

    def marker_slider_change(attrname, old, new):
        marker_mapper["transform"].low = new[0]
        marker_mapper["transform"].high = new[1]

    def update(update_range=True):
        m = marker_select.value
        d = get_data(m)
        source.data = d
        numeric_marker = np.issubdtype(d[m].dtype, np.number)
        if not numeric_marker:
            levels = list(sorted(set(d["marker_val_cat"])))
            marker_mapper_cat["transform"].palette = get_cat_colors(len(levels))
            marker_mapper_cat["transform"].factors = levels
        elif update_range and numeric_marker:
            marker_max = round_signif(d["marker_val_num"].max())
            marker_min = round_signif(d["marker_val_num"].min())
            marker_slider.start = marker_min
            marker_slider.end = marker_max
            marker_slider.step = round_signif((marker_max - marker_min) / 50)
            marker_slider.value = tuple(
                map(round_signif, np.percentile(d["marker_val_num"], [5, 95]))
            )
        umap_figure.visible = numeric_marker
        umap_figure_cat.visible = not numeric_marker
        marker_slider.visible = numeric_marker

    def selection_change(attrname, old, new):
        m = marker_select.value
        data = get_data(m)
        selected = source.selected.indices
        if not selected:
            return
        data = data.iloc[selected, :]
        mean_image = CELL_IMAGE_METRICS[1][metric_select.active](
            cell_stack[selected, int(cell_markers_select.value), :, :], axis=0
        )
        image_source.data = {
            "image": [mean_image],
            "dw": [cell_stack.shape[1]],
            "dh": [cell_stack.shape[2]],
        }
        image_extr = round_signif(mean_image.min()), round_signif(mean_image.max())
        cell_slider.start = image_extr[0]
        cell_slider.end = image_extr[1]
        cell_slider.step = round_signif((image_extr[1] - image_extr[0]) / 50)
        cell_slider.value = image_extr
        stats.text = "n cells: " + str(len(selected))

    def mark_selection():
        get_data.cache_clear()
        col = edit_selection_col.value
        if col is None or col == "":
            return
        if col not in input_data:
            input_data[col] = np.full(input_data.shape[0], "NA")
        col_type = input_data[col].dtype
        idx = source.selected.indices
        try:
            val = np.full(len(idx), edit_selection_val.value).astype(col_type)
            input_data.loc[input_data.index[idx], col] = val
        except Exception as e:
            edit_selecton_log.text = (
                f'Failed to edit cells. Exception: "{e}"\n' + edit_selecton_log.text
            )
        else:
            edit_selecton_log.text = (
                f'Edited {len(source.selected.indices)} cells. {col}="{edit_selection_val.value}"\n'
                + edit_selecton_log.text
            )
        update(update_range=False)
        old_marker_cols = set(marker_cols())
        marker_cols.cache_clear()
        if old_marker_cols != set(marker_cols()):
            marker_select.options = marker_cols()
            marker_input.completions = marker_cols() + list(
                marker_cols(lower=True).keys()
            )

    def autocomplete_change(attrname, old, new):
        if new not in marker_cols():
            try:
                new = marker_cols(lower=True)[new]
            except KeyError:
                return
        marker_select.value = new
        marker_input.value = None

    def autocomplete_cell_change(attrname, old, new):
        try:
            idx = image_markers(mapping=True)[new]
        except KeyError:
            try:
                idx = image_markers(lower=True, mapping=True)[new]
            except KeyError:
                return
        cell_markers_select.value = str(idx)
        cell_marker_input.value = None

    source.selected.on_change("indices", selection_change)
    marker_select.on_change("value", marker_change)
    marker_slider.on_change("value", marker_slider_change)
    cell_slider.on_change("value", cell_slider_change)
    metric_select.on_change("active", selection_change)
    cell_markers_select.on_change("value", selection_change)
    edit_selection_submit.on_click(mark_selection)
    marker_input.on_change("value", autocomplete_change)
    cell_marker_input.on_change("value", autocomplete_cell_change)

    # set up layout
    layout = column(
        row(
            column(
                marker_select,
                marker_input,
                row(umap_figure, marker_slider),
                umap_figure_cat,
            ),
            column(
                cell_markers_select,
                cell_marker_input,
                metric_select,
                row(cell_figure, cell_slider),
                stats,
            ),
        ),
        Div(text="Change data for selected cells"),
        row(
            edit_selection_col,
            edit_selection_val,
            edit_selection_submit,
            download_button,
        ),
        edit_selecton_log,
    )

    # initialize
    update()

    doc.add_root(layout)
    doc.title = "UMAP projection"


def run_server(
    cell_stack,
    input_data,
    port=5000,
    markers=None,
    default_umap_marker=None,
    default_cell_marker=None,
    server_kwargs={},
):
    print(server_kwargs)
    apps = {
        "/": Application(
            FunctionHandler(
                partial(
                    prepare_server,
                    cell_stack=cell_stack,
                    input_data=input_data,
                    cell_markers=markers,
                    default_umap_marker=default_umap_marker,
                    default_cell_marker=default_cell_marker,
                )
            )
        )
    }
    server = Server(apps, port=port, **server_kwargs)
    server.run_until_shutdown()
