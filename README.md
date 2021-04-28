LSMTool: the LOFAR Sky Model Tool
=================================

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

For details, please see the [full documentation](http://www.astron.nl/citt/lsmtool).

Installation
------------

LSMTool is already installed on the LOFAR CEP3 cluster. Users on CEP3
should run the following commands before using LSMTool:

    source ~rafferty/init_lsmtool

If you want to install LSMTool yourself, follow the instructions below.

### Dependencies

* [Numpy](http://www.numpy.org)
* [Matplotlib](http://www.matplotlib.org)
* [Astropy](http://www.astropy.org)
* [PyVO](http://pyvo.readthedocs.org) (optional, for VO access)

### Downloading and Installing

Get the latest developer version by cloning the git repository:

    git clone https://github.com/darafferty/LSMTool.git

Then install with:

    cd LSMTool
    python setup.py install

If you have a C++11-compliant compiler, you can build a faster
version of the mean shift grouping algorithm with:

    cd LSMTool
    python setup.py install --build_c_extentions

Note that the C++ version will give slightly different results compared to the
Python version, but such differences are not expected to be important
in practice.

### Testing

You can test that the installation worked with:

    python setup.py test

If no errors occur, LSMTool is installed correctly.


Usage
-----

The LSMTool executable can be used from the command line with a parset that defines the steps
to be done. E.g.:

    $ lsmtool model.sky lsmtool.parset

The parset follows the usual NDPPP/BBS format. E.g.:

    # Select individual sources with Stokes I fluxes above 1 Jy
    LSMTool.Steps.select.Operation = SELECT
    LSMTool.Steps.select.FilterExpression = I > 1.0 Jy
    LSMTool.Steps.select.OutFile = out_model.sky

LSMTool can also be used in Python scripts by importing the lsmtool module. E.g.:

    >>> import lsmtool
    >>> skymod = lsmtool.load('model.sky')
    >>> skymod.select('I > 1.0 Jy')
    >>> skymod.write('out_model.sky')

For further details, please see the [full documentation](http://www.astron.nl/citt/lsmtool).
