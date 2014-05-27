LSMTool: the LOFAR Local Sky Model Tool
=======================================

LSMTool allows the manipulation of LOFAR sky models (in the makesourcedb format).

Authors:
* David Rafferty
Based on contributed scripts from:
* Björn Adebahr
* Francesco de Gasperin
* Reinout van Weeren

Contents:
* __doc/__: documentation
* __tests/__: contains test sky models and scripts useful for validation
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
* PLOT: Plot the sky model

Usage
-----
LSMTool can be used from the command line with a parset that defines the steps
to be done. E.g.:

    $ lsmtool.py model.sky lsmtool.parset

The parset follows the usual NDPPP/BBS format. E.g.:

    # Select individual sources with Stokes I fluxes above 1 Jy
    LSMTool.Steps.select.Operation = SELECT
    LSMTool.Steps.select.FilterExpression = I > 1.0 Jy
    LSMTool.Steps.select.OutFile = out_model.sky

LSMTool can also be used in Python scripts by importing the lsmtool module. E.g.:

    >>> import lsmtool
    >>> skymod = lsmtool.skymodel.load('model.sky')
    >>> skymod.select('I > 1.0 Jy')
    >>> skymod.write('out_model.sky')

For further details, please see the documentation.
