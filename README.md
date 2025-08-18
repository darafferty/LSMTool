# LSMTool: the LOFAR Sky Model Tool

LSMTool allows the manipulation of LOFAR sky models (in the makesourcedb format).

Author:
* David Rafferty

Based on contributed scripts by:
* Bjoern Adebahr
* Francesco de Gasperin
* Reinout van Weeren

Contents:
* __docs/__: documentation
* __tests/__: contains test sky models and scripts useful for validation
* __bin/__: contains lsmtool executable
* __lsmtool/__: contains the main LSMTool scripts
* __lsmtool/operations/__: contains the modules for operations
* __parsets/__: some example parsets


The following operations are available:
* SELECT: Select sources by source or patch properties
* REMOVE: Remove sources by source or patch properties
* TRANSFER: Transfer a patch scheme from one sky model to another
* GROUP: Group sources into patches
* UNGROUP: Remove patches
* MOVE: Move a source or patch position
* MERGE: Merge two or more patches into one
* CONCATENATE: Concatenate two sky models
* ADD: Add a source
* SETPATCHPOSITIONS: Calculate and set patch positions
* PLOT: Plot the sky model
* COMPARE: Compare source fluxes and positions of two sky models

For details, please see the [full documentation](https://lsmtool.readthedocs.io/en/latest/).


## Dependencies

LSMTool depends on the following packages:

* [Astropy](https://www.astropy.org/) (version 3.2 or later)
* [EveryBeam](https://git.astron.nl/RD/EveryBeam.git) (version 0.6.1 or later)
* [Matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/)
* [Pillow](https://pypi.org/project/pillow/) (any version except 10.x)
* [PyBDSF](https://github.com/lofar-astron/PyBDSF.git)
* [Python-Casacore](https://github.com/casacore/python-casacore.git)
* [PyVO](https://github.com/astropy/pyvo)
* [Scipy](https://scipy.org/) (version 0.11 or later)
* [Shapely](https://github.com/shapely/shapely)

These packages will normally be installed automatically.


## Downloading and Installing

LSMTool uses `PyBDSF` as default source finder. If you wish, you can use [SoFiA-2](https://gitlab.com/SoFiA-Admin/SoFiA-2/) instead, which can be installed as extra.

### Install the latest release from PyPI

```
    pip install lsmtool
```
or with `SoFiA-2` as extra:
```
    pip install lsmtool[sofia]
```

### Install the latest developer version
```
    pip install git+https://git.astron.nl/RD/LSMTool.git
```
or with `SoFiA-2` as extra:
```
    pip install lsmtool[sofia] git+https://git.astron.nl/RD/LSMTool.git
```

### Install a faster mean-shift grouping algorithm
If you have a C++11-compliant compiler, you can build a faster
version of the mean shift grouping algorithm by compiling it
yourself:
```
    git clone https://git.astron.nl/RD/LSMTool.git
    cd LSMTool
    sed -Ei 's/^(BUILD_EXTENSIONS = ).*/\1"ON"/' pyproject.toml
    pip install .
```
Note that the C++ version will give slightly different results compared to the
Python version, but such differences are not expected to be important
in practice.


## Testing

If you've cloned the repository, and installed the software from source, you can test that the installation works as expected:
```
    pip install pytest
    pytest
```
If no errors occur, LSMTool is installed correctly.


## Usage

The LSMTool executable can be used from the command line with a parset that defines the steps
to be done. E.g.:

    $ lsmtool model.sky lsmtool.parset

The parset follows the usual DP3 format. E.g.:

    # Select individual sources with Stokes I fluxes above 1 Jy
    LSMTool.Steps.select.Operation = SELECT
    LSMTool.Steps.select.FilterExpression = I > 1.0 Jy
    LSMTool.Steps.select.OutFile = out_model.sky

LSMTool can also be used in Python scripts by importing the lsmtool module. E.g.:

    >>> import lsmtool
    >>> skymod = lsmtool.load('model.sky')
    >>> skymod.select('I > 1.0 Jy')
    >>> skymod.write('out_model.sky')

For further details, please see the [full documentation](https://lsmtool.readthedocs.io/en/latest/).
