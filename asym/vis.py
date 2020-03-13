from functools import lru_cache
import pandas as pd
import numpy as np

import bokeh
from bokeh.palettes import viridis, inferno
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, PreText, Select, ColorBar, RangeSlider, Div
from bokeh.plotting import figure
from bokeh.transform import linear_cmap

CELL_IMAGE_METRICS = {"mean": np.mean, "sd": np.std, "min": np.min, "max": np.max}


def prepare_server(doc, input_data, cell_stack, cell_markers=None):
    @lru_cache()
    def get_data(m):
        d = input_data.copy()
        if m is not None:
            d["marker_val"] = d[m]
        return d

    cell_markers = (
        cell_markers
        if cell_markers is not None
        else [f"Marker {i + 1}" for i in range(cell_stack.shape[1])]
    )
    cell_markers = [(str(i), x) for i, x in enumerate(cell_markers)]

    # set up widgets

    marker_cols = list(sorted(input_data.select_dtypes(include=np.number).columns))

    stats = PreText(text="", width=200)
    marker_select = Select(
        value=marker_cols[0], options=marker_cols, title="Color UMAP by"
    )

    cell_markers_select = Select(
        value=cell_markers[0][0], options=cell_markers, title="Marker cell image"
    )
    marker_slider = RangeSlider(start=0, end=1, value=(0, 1), step=0.1)
    cell_slider = RangeSlider(start=0, end=1, value=(0, 1))
    metric_select = Select(
        value="mean",
        options=list(CELL_IMAGE_METRICS.keys()),
        title="Image aggregation method",
    )

    # set up plots

    source = ColumnDataSource(data=input_data)
    image_source = ColumnDataSource(data=dict(image=[], dw=[], dh=[]))

    marker_mapper = linear_cmap(
        field_name="marker_val",
        palette=inferno(8)[:-1],
        low=0,
        high=500,
        high_color=None,
    )

    umap_figure = figure(
        plot_width=800,
        plot_height=500,
        tools="pan,wheel_zoom,lasso_select,box_select,tap,reset",
    )
    umap_figure.circle(
        "d1",
        "d2",
        size=4,
        source=source,
        line_color=marker_mapper,
        color=marker_mapper,
        selection_color="orange",
        alpha=0.6,
        nonselection_alpha=0.4,
        selection_alpha=0.8,
    )
    umap_color_bar = ColorBar(
        color_mapper=marker_mapper["transform"], width=8, location=(0, 0)
    )
    umap_figure.add_layout(umap_color_bar, "right")

    cell_mapper = bokeh.models.mappers.LinearColorMapper(
        viridis(10), low=0, high=1000, high_color=None
    )
    cell_color_bar = ColorBar(color_mapper=cell_mapper, width=8, location=(0, 0))
    cell_figure = figure(plot_width=450, plot_height=350, tools="pan,wheel_zoom,reset")
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

    # set up callbacks

    def marker_change(attrname, old, new):
        # marker_select.options = nix(new, marker_cols)
        update()

    def cell_slider_change(attrname, old, new):
        cell_mapper.low = new[0]
        cell_mapper.high = new[1]

    def marker_slider_change(attrname, old, new):
        marker_mapper["transform"].low = new[0]
        marker_mapper["transform"].high = new[1]

    def update(selected=None):
        m = marker_select.value
        d = get_data(m)
        source.data = d
        marker_max = d["marker_val"].max()
        marker_min = d["marker_val"].min()
        marker_slider.start = marker_min
        marker_slider.end = marker_max
        marker_slider.value = tuple(np.percentile(d["marker_val"], [5, 95]))
        umap_figure.title.text = "UMAP with marker %s" % (m)

    def selection_change(attrname, old, new):
        m = marker_select.value
        data = get_data(m)
        selected = source.selected.indices
        if not selected:
            return
        data = data.iloc[selected, :]
        mean_image = CELL_IMAGE_METRICS[metric_select.value](
            cell_stack[selected, cell_markers_select.value, :, :], axis=0
        )
        image_source.data = {
            "image": [mean_image],
            "dw": [cell_stack.shape[1]],
            "dh": [cell_stack.shape[2]],
        }
        image_extr = mean_image.min(), mean_image.max()
        cell_slider.start = image_extr[0]
        cell_slider.end = image_extr[1]
        cell_slider.value = image_extr
        stats.text = "n cells: " + str(len(selected))

    source.selected.on_change("indices", selection_change)
    marker_select.on_change("value", marker_change)
    marker_slider.on_change("value", marker_slider_change)
    cell_slider.on_change("value", cell_slider_change)
    metric_select.on_change("value", selection_change)
    cell_markers_select.on_change("value", selection_change)

    # set up layout
    layout = row(
        column(umap_figure, marker_slider),
        column(marker_select, stats, metric_select),
        column(cell_markers_select, cell_figure, cell_slider),
    )

    # initialize
    update()

    doc.add_root(layout)
    doc.title = "UMAP projection"
