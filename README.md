
# Welcome to phoenix

PHase-space Optimization and EstimatioN In JAX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/anschaible/phoenix/ci.yml?branch=main)](https://github.com/anschaible/phoenix/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/phoenix/badge/)](https://phoenix.readthedocs.io/)
[![codecov](https://codecov.io/gh/anschaible/phoenix/branch/main/graph/badge.svg)](https://codecov.io/gh/anschaible/phoenix)

![PHOENIX](phoenix_logo.png)


## Installation

The Python package `phoenix` can be installed from PyPI:

```
python -m pip install phoenix
```

## Development installation

If you want to contribute to the development of `phoenix`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/anschaible/phoenix.git
cd phoenix
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).

