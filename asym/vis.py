''' Create a simple stocks correlation dashboard.
Choose stocks to compare in the drop down widgets, and make selections
on the plots to update the summary and histograms accordingly.
.. note::
    Running this example requires downloading sample data. See
    the included `README`_ for more information.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve stocks
at your command prompt. Then navigate to the URL
    http://localhost:5006/stocks
.. _README: https://github.com/bokeh/bokeh/blob/master/examples/app/stocks/README.md
'''
from functools import lru_cache
from os.path import dirname, join

import pandas as pd
import numpy as np
import matplotlib as mpl

import bokeh
from bokeh.palettes import viridis
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, PreText, Select
from bokeh.plotting import figure

umap_data = pd.read_csv("test.csv", index_col=False)
cell_stack = np.load("reg005_X04_Y04_tensor.npy")
input_data = pd.read_csv("reg005_X04_Y04_tensor_data.csv", index_col=False)
input_data["u1"] = umap_data["u1"][:len(input_data.index)]
input_data["u2"] = umap_data["u2"][:len(input_data.index)]

def nix(val, lst):
    return [x for x in lst if x != val]

@lru_cache()
def get_data(m):
    d = input_data
    # if m is not None:
        # d["marker_color"] = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl.cm.viridis(mpl.colors.Normalize()(d[m]))]
    return d

# set up widgets

marker_cols = list(input_data.select_dtypes(include=np.number).columns)

stats = PreText(text='', width=500)
marker_select = Select(value="CD25", options=marker_cols)

# set up plots

source = ColumnDataSource(data=input_data)
image_source = ColumnDataSource(data=dict(image=[], dw=[], dh=[]))

umap_figure = figure(plot_width=350, plot_height=350,
              tools='pan,wheel_zoom,lasso_select,box_select,reset')
umap_figure.circle('u1', 'u2', size=1, source=source,
            selection_color="orange", alpha=0.6, nonselection_alpha=0.1, selection_alpha=0.4)

cell_figure = figure(plot_width=350, plot_height=350,
              tools='pan,wheel_zoom,reset')
cell_image = cell_figure.image(
    image="image", color_mapper=bokeh.models.mappers.LinearColorMapper(viridis(5), low=0, high=1000),
    x=0, y=0, dw="dw", dh="dh", source=image_source
)

# set up callbacks

def marker_change(attrname, old, new):
    # marker_select.options = nix(new, marker_cols)
    update()

def update(selected=None):
    m = marker_select.value
    source.data = get_data(m)
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
    stats.text = "n cells: " + str(len(selected))


source.selected.on_change('indices', selection_change)
marker_select.on_change('value', marker_change)

# set up layout
widgets = column(marker_select, stats)
main_row = row(umap_figure, widgets, cell_figure)
layout = column(main_row)

# initialize
update()

curdoc().add_root(layout)
curdoc().title = "UMAP projection"
