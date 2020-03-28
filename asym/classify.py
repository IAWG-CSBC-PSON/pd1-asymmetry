from base64 import b64decode
from functools import lru_cache, partial
from io import BytesIO
import json
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
    BooleanFormatter,
    Button,
    Circle,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    DataTable,
    Div,
    FileInput,
    HoverTool,
    PreText,
    RadioButtonGroup,
    RangeSlider,
    Select,
    TableColumn,
    TextAreaInput,
    TextInput,
)
from bokeh.plotting import figure
from bokeh.transform import linear_cmap, factor_cmap

from .common import DOWNLOAD_JS, round_signif

CELL_IMAGE_METRICS = (["mean", "sd", "min", "max"], [np.mean, np.std, np.min, np.max])

CUSTOM_CSS = r"""
<style>
.edit_log textarea.bk {
    overflow-y: scroll;
    font-family: Courier, Monaco, monospace;
}
</style>
"""


class ColumnEditor:
    def __init__(
        self,
        source,
        container,
        log_widget=None,
        editor_delete_callback=None,
        external_edit_callback=None,
    ):
        self.source = source
        self.container = container
        self.log_widget = log_widget
        self.editor_delete_callback = editor_delete_callback
        self.external_edit_callback = external_edit_callback
        self.edit_callback = self._make_edit_callback()
        self.delete_callback = self._make_delete_callback()
        self.widgets = self._make_edit_widgets()
        self.widget_row = self._add_widgets()

    def _add_widgets(self):
        widget_row = row(
            self.widgets["delete_button"],
            self.widgets["input_col"],
            self.widgets["input_val"],
            self.widgets["value_row"],
        )
        self.container.children.append(widget_row)
        return widget_row

    def _make_delete_callback(self):
        def delete_callback():
            idx = next(
                i for i, x in enumerate(self.container.children) if x is self.widget_row
            )
            del self.container.children[idx]
            if self.editor_delete_callback:
                self.editor_delete_callback(self)

        return delete_callback

    def _make_edit_widgets(self):
        delete_button = Button(
            label="-", button_type="warning", align=("start", "end"), width=50
        )
        delete_button.on_click(self.delete_callback)
        return {
            "input_col": TextInput(title="Column", width=150),
            "input_val": TextInput(title="Value", width=100),
            "delete_button": delete_button,
            "value_row": row(align=("start", "end")),
            "value_buttons": {},
        }

    def _log(self, text):
        if self.log_widget:
            new_text = text + self.log_widget.value
            self.log_widget.value = new_text[: min(1000, len(new_text))]

    def _make_edit_callback(self):
        def edit_callback(val=None):
            data = self.source.data
            col = self.widgets["input_col"].value
            if val is None:
                val = self.widgets["input_val"].value
            if col is None or col == "" or not self.source.selected:
                return
            self.widgets["input_col"].disabled = True
            if col not in data:
                data[col] = np.full(
                    len(next(iter(data.values()))), "NA", dtype=np.object
                )
            col_type = data[col].dtype
            idx = self.source.selected.indices
            try:
                col_data = data[col]
                col_data[idx] = np.full(len(idx), val).astype(col_type)
                data[col] = col_data
                marked_data = data["marked"]
                marked_data[idx] = "✓"
                data["marked"] = marked_data
            except Exception as e:
                self._log(f'Failed to edit cells. Exception: "{e}"\n')
            else:
                self._log(f'Edited {len(idx)} cells. {col}="{val}"\n')
                if self.external_edit_callback:
                    self.external_edit_callback()
            self._add_value_buttons(col)

        return edit_callback

    def _add_value_buttons(self, col):
        vals = set(self.source.data[col])
        if len(vals) > 10:
            self._log(f'Too many values in "{col}". Not showing buttons')
            return
        buttons = self.widgets["value_buttons"]
        if all(v in buttons for v in vals):
            return
        self.widgets["value_row"].children = []
        for val in sorted(vals):
            try:
                button = buttons[val]
            except KeyError:
                button = Button(label=val, align=("start", "end"), width=50)
                button.on_click(partial(self.edit_callback, val=val))
                buttons[val] = button
            self.widgets["value_row"].children.append(button)


def parse_contour(contour):
    return list(zip(*(zip(*json.loads(x)) for x in contour)))


def prepare_server(
    doc, input_data, cell_stack, cell_markers=None, default_cell_marker=None
):
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

    # Data sources
    ###########################################################################

    def prepare_data(input_data):
        data = input_data.copy()
        if "contour" in data and not all(x in data for x in ["contour_x", "contour_y"]):
            contour = parse_contour(data["contour"])
            data["contour_x"] = contour[0]
            data["contour_y"] = contour[1]
        if "marked" not in data:
            data["marked"] = np.full(data.shape[0], "")
        source.data = data

    source = ColumnDataSource(data={})
    prepare_data(input_data)
    image_source = ColumnDataSource(data=dict(image=[], dw=[], dh=[], contour_x=[], contour_y=[]))

    # Cell picture plot
    ###########################################################################

    def add_outline():
        data = source.data
        if all(x in data for x in ["contour_x", "contour_y"]):
            cell_outline = cell_figure.patches(
                xs="contour_x",
                ys="contour_y",
                fill_color=None,
                line_color="red",
                name="cell_outline",
                source=image_source,
            )
            cell_outline.level = "overlay"
        else:
            cell_outline = cell_figure.select(name="cell_outline")
            for x in cell_outline:
                cell_figure.renderers.remove(x)

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
    add_outline()
    cell_figure.add_layout(cell_color_bar, "right")

    # Edit data of selected cells
    ###########################################################################

    marker_edit_container = column()
    marker_edit_instances = []

    def add_marker_edit_callback():
        editor = ColumnEditor(
            source,
            marker_edit_container,
            log_widget=edit_selecton_log,
            editor_delete_callback=delete_marker_edit_callback,
            external_edit_callback=edit_selection_callback,
        )
        marker_edit_instances.append(editor)

    def delete_marker_edit_callback(editor):
        idx = next(i for i, x in enumerate(marker_edit_instances) if x is editor)
        del marker_edit_instances[idx]

    file_name_text = Div()

    add_marker_edit_button = Button(
        label="+", button_type="success", align=("start", "end"), width=50
    )
    add_marker_edit_button.on_click(add_marker_edit_callback)

    edit_selection_submit = Button(
        label="Submit change", button_type="primary", align=("start", "end")
    )
    download_button = Button(
        label="Download edited data", button_type="success", align=("start", "end")
    )
    download_button.js_on_click(CustomJS(args=dict(source=source), code=DOWNLOAD_JS))

    edit_selecton_log = TextAreaInput(
        value="", disabled=True, css_classes=["edit_log"], cols=30, rows=10
    )

    upload_file_input = FileInput(accept="text/csv", align=("end", "end"))

    # Cell table
    ###########################################################################

    default_data_table_cols = [TableColumn(field="marked", title="Seen", width=20)]

    data_table = DataTable(source=source, columns=default_data_table_cols, width=800)

    # Callbacks for buttons and widgets
    ###########################################################################

    def cell_slider_change(attrname, old, new):
        cell_mapper.low = new[0]
        cell_mapper.high = new[1]

    def selection_change(attrname, old, new):
        selected = source.selected.indices
        data = source.data
        if not selected:
            return
        mean_image = CELL_IMAGE_METRICS[1][metric_select.active](
            cell_stack[selected, int(cell_markers_select.value), :, :], axis=0
        )
        image_data = {
            "image": [mean_image],
            "dw": [cell_stack.shape[2]],
            "dh": [cell_stack.shape[3]],
        }
        for coord in ["contour_x", "contour_y"]:
            try:
                image_data[coord] = list(data[coord][selected])
                print(image_data[coord])
            except KeyError:
                pass
        image_source.data = image_data
        image_extr = round_signif(mean_image.min()), round_signif(mean_image.max())
        cell_slider.start = image_extr[0]
        cell_slider.end = image_extr[1]
        cell_slider.step = round_signif((image_extr[1] - image_extr[0]) / 50)
        cell_slider.value = image_extr
        stats.text = "n cells: " + str(len(selected))

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

    def data_change(attrname, old, new):
        new_keys = [n for n in new.keys() if n not in set(old.keys())]
        for n in new_keys:
            data_table.columns.append(TableColumn(field=n, title=n))

    def edit_selection_submit_click():
        for x in marker_edit_instances:
            x.edit_callback()

    def edit_selection_callback():
        idx = source.selected.indices
        try:
            if len(idx) == 1 and all(
                source.data[x.widgets["input_col"].value][idx] != "NA"
                for x in marker_edit_instances
            ):
                source.selected.indices = [idx[0] + 1]
        except KeyError:
            pass

    def upload_file_callback(attrname, old, new):
        try:
            data_text = b64decode(new)
            data = pd.read_csv(BytesIO(data_text))
        except Exception:
            file_name_text.text = f"Error loading file {upload_file_input.filename}"
            return
        file_name_text.text = f"Editing file {upload_file_input.filename}"
        data_table.columns = default_data_table_cols
        prepare_data(data)
        add_outline()

    source.selected.on_change("indices", selection_change)
    source.on_change("data", data_change)
    cell_slider.on_change("value", cell_slider_change)
    metric_select.on_change("active", selection_change)
    cell_markers_select.on_change("value", selection_change)
    cell_marker_input.on_change("value", autocomplete_cell_change)
    edit_selection_submit.on_click(edit_selection_submit_click)
    upload_file_input.on_change("value", upload_file_callback)

    style_div = Div(text=CUSTOM_CSS)

    # set up layout
    layout = column(
        row(
            column(data_table),
            column(
                cell_markers_select,
                cell_marker_input,
                metric_select,
                row(cell_figure, cell_slider),
                stats,
            ),
        ),
        file_name_text,
        marker_edit_container,
        add_marker_edit_button,
        row(edit_selection_submit, download_button, upload_file_input),
        edit_selecton_log,
        style_div,
    )

    doc.add_root(layout)
    doc.title = "Cell classifier"


def run_server(
    cell_stack,
    input_data,
    port=5000,
    markers=None,
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
                    default_cell_marker=default_cell_marker,
                )
            )
        )
    }
    server = Server(apps, port=port, **server_kwargs)
    server.run_until_shutdown()
