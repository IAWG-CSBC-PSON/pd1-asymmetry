from functools import lru_cache
from os.path import dirname, join

import pandas as pd
import numpy as np
import matplotlib as mpl

import bokeh
from bokeh.palettes import viridis, inferno
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, PreText, Select, ColorBar, RangeSlider
from bokeh.plotting import figure
from bokeh.transform import linear_cmap

input_data = pd.read_csv("umap_pd1+_all_channels.csv", index_col=False)
cell_stack = np.load("all_cells_tensor.npy")

@lru_cache()
def get_data(m):
    d = input_data
    if m is not None:
        d["marker_val"] = d[m]
    return d

# set up widgets

marker_cols = list(input_data.select_dtypes(include=np.number).columns)

stats = PreText(text='', width=200)
marker_select = Select(value="CD25", options=marker_cols)

marker_slider = RangeSlider(start=0, end=1, value=(0, 1))
cell_slider = RangeSlider(start=0, end=1, value=(0, 1))

# set up plots

source = ColumnDataSource(data=input_data)
image_source = ColumnDataSource(data=dict(image=[], dw=[], dh=[]))

marker_mapper = linear_cmap(field_name="marker_val", palette=inferno(8)[:-1], low=0, high=500, high_color=None)

umap_figure = figure(plot_width=800, plot_height=500,
              tools='pan,wheel_zoom,lasso_select,box_select,reset')
umap_figure.circle('d1', 'd2', size=3, source=source,
            line_color=marker_mapper, color=marker_mapper,
            selection_color="orange", alpha=0.6, nonselection_alpha=0.4, selection_alpha=0.8
)
umap_color_bar = ColorBar(color_mapper=marker_mapper['transform'], width=8, location=(0,0))
umap_figure.add_layout(umap_color_bar, "right")

cell_mapper = bokeh.models.mappers.LinearColorMapper(viridis(10), low=0, high=1000, high_color=None)
cell_color_bar = ColorBar(color_mapper=cell_mapper, width=8, location=(0,0))
cell_figure = figure(plot_width=450, plot_height=350,
              tools='pan,wheel_zoom,reset')
cell_image = cell_figure.image(
    image="image", color_mapper=cell_mapper,
    x=0, y=0, dw="dw", dh="dh", source=image_source
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
    marker_slider.value = (marker_min, marker_max)
    umap_figure.title.text = 'UMAP with marker %s' % (m)

def selection_change(attrname, old, new):
    m = marker_select.value
    data = get_data(m)
    selected = source.selected.indices
    if selected:
        data = data.iloc[selected, :]
    mean_image = np.mean(cell_stack[selected, :, :], axis=0)
    image_source.data = {
        "image": [mean_image],
        "dw": [cell_stack.shape[1]],
        "dh": [cell_stack.shape[2]],
    }
    image_extr = mean_image.min(), mean_image.max()
    cell_slider.start = image_extr[0]
    cell_slider.end = image_extr[1]
    cell_slider.value = (image_extr[0], image_extr[1])
    stats.text = "n cells: " + str(len(selected))


source.selected.on_change('indices', selection_change)
marker_select.on_change('value', marker_change)
marker_slider.on_change("value", marker_slider_change)
cell_slider.on_change("value", cell_slider_change)

# set up layout
widgets = column(marker_select, stats)
main_row = row(column(umap_figure, marker_slider), widgets, column(cell_figure, cell_slider))
layout = column(main_row)

# initialize
update()

curdoc().add_root(layout)
curdoc().title = "UMAP projection"
