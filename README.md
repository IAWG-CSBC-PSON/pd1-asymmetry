# PD1 asymmetry challenge

Challenge organizer: Yury Goltsev

Challenge participants:
* Cole Harris
* Clemens Hug
* Rumana Rashid
* Geoffrey Schau

## asym python package

The asym package contains code to run a variational autoencoder network to extract
features from a stack of single cell images.

These features can be further reduced by running UMAP and then visualized
interactively using the builtin Bokeh app.

### Installation

    pip install git+https://github.com/IAWG-CSBC-PSON/pd1-asymmetry.git

### Usage example

Some example cell stacks and cell marker data are in the pd1_project folder.

To visualize and example UMAP embedding run

    asym vis pd1_project/all_cells_tensor.npy pd1_project/umap_pd1+_all_channels.csv

This starts up the interactive Bokeh server. While using the app, keep the server running
in the background. To exit the server press CTRL+C.

While the server is running, browse to localhost:5000 in any webbrowser to access the app.

## PD1 asymmetry project

Data for the PD1 project are available at https://www.synapse.org/#!Synapse:syn22009464/files/.

The pd1_project folder contains multiple Jupyter notebooks for the pre-processing
and analysis of the PD1 asymmetry project. In order to run them, several dependencies need
to be installed:

### Create conda environment

Conda environments contain a set of packages required for a project,
keeping them separate from other projects.

To create a conda environment called `pd1` for the challenge:

    conda env create --name pd1 --file pd1_project/pd1-conda-environment.yaml

### Activate conda environment

    conda activate pd1
