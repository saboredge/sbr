[![Documentation Status](https://readthedocs.org/projects/sbr/badge/?version=latest)](https://sbr.readthedocs.io/en/latest/?badge=latest)
# sbr: 

Library of usefule functions for analyzing public datasets with machine learning using TensorFlow.

## Contents
- [Repo Structure](#repo-structure)
- [Getting started](#getting-started)
- [Development environment](#development-environment)
- [Install data]
  - [GTEX](#gtex)
- [Make docs](#make-docs)
- [Future](#future)


## Repo Structure

This repository is structured as follows:
 - `docs`:  sphinx documentation for this library, hosted on readthedocs.io
 - `.readthedocs.yaml`: configuration for deploying sbr.readthedocs.io
 - `src/sbr`: all the code for the sbr pypi package
 - `src/sbr/preprocessing`: functions to help with preprocessing; analogous to `tf.keras.preprocessing`
 - `src/sbr/datasets`: functions for creating tfds-type datasets; analogous to `tf.data.Datasets`
 - `sbr_env.yml`: a tested conda environment
 - `data`: functions under `src/sbr/datasets/structured` may write supplemental, useful metadata here
 - `src/tests`: regression unit tests; run with `cd src; nosetests`
 - `LICENSE`: terms of use and licensing of all the code in this repository
 - `pyproject.toml`: for pypi deployment

## Getting started

Start with setting up a development environment for running conda and tensorflow and conda. Then install a dataset, and readthedocs.

Note: these notes use `mamba`, which works on top of `conda`, but is a bit faster. You can replace `mamba` with `conda`, as preferred.

## Development environment

Install miniconda (or conda) and create an environment like the following:

```
conda env create -f sbr_env.yml
```

## Install data

Documentation for additional public datasets will be added here.

### GTEx

Make the GTEx dataset:

```
mamba install -c conda-forge tensorflow-datasets
tfds build --register_checksums  --overwrite src/sbr/datasets/structured/gtex
```

## Make docs:

For more help, see this article: https://saboredge.com/Generating-documentation-from-python

```
mamba install python3-sphinx 
mamba install myst-parser 
mamba install sphinx_rtd_theme
mkdir -p docs/source
sphinx-apidoc -f -o docs/source src/sbr
cp docs/index.rst docs/conf.py docs/source
cd docs; make html; cd .. # makes local copy of documentation, or see https://sbr.readthedocs.io
```

## Future

* pypi package
* git attr
* Generative model functions
* Graph model functions
* Gene expression visualization functions (heatmap, volcano, t-SNE, M-A, etc.)
* TCGA dataset
* Single-cell - HubMap, HPA (Human Protein Atlas)
* Brain (Alan Brain Atlas)



