# -*- coding: utf-8 -*-
#
# Defines filter functions used by the remove and select operations
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

import logging
log = logging.getLogger('LSMTool.Filter')
try:
    dict.iteritems
except AttributeError:
    # Python 3
    def itervalues(d):
        return iter(d.values())
    def iteritems(d):
        return iter(d.items())
else:
    # Python 2
    def itervalues(d):
        return d.itervalues()
    def iteritems(d):
        return d.iteritems()


def filter(LSM, filterExpression, exclusive=False, aggregate=None,
    applyBeam=False, useRegEx=False, force=True):
    """
    Filters the sky model, keeping all sources that meet the given expression.

    After filtering, the sky model contains only those sources for which the
    given filter expression is true.

    Parameters
    ----------
    filterExpression : str, dict, list, or numpy array
        - If string:
            A string specifying the filter expression in the form:
            '<property> <operator> <value> [<units>]'
            (e.g., 'I <= 10.5 Jy').

        - If dict:
            The filter can also be given as a dictionary in the form:
            {'filterProp':property, 'filterOper':operator,
                'filterVal':value, 'filterUnits':units}

        - If list:
            The filter can also be given as a list of:
            [property, operator, value] or
            [property, operator, value, units]

        - If numpy array:
            The indices to filter on can be specified directly as a numpy array
            of row or patch indices such as:
            array([ 0,  2, 19, 20, 31, 37])

            or as a numpy array of bools with the same length as the sky model.

            If a numpy array is given and the indices correspond to patches, then
            set aggregate=True.

        The property to filter on must be one of the following:
            - a valid column name
            - the filename of a mask image

        Supported operators are:
            - !=
            - <=
            - >=
            - >
            - <
            - = (or '==')
        Units are optional and must be specified as required by astropy.units.
    exclusive : bool, optional
        If False, sources that meet the filter expression are kept. If True,
        sources that do not meet the filter expression are kept.
    aggregate : str or bool, optional
        If set, the values are aggregated over the patch members. The following
        aggregation functions are available:
            - 'sum': sum of patch values
            - 'mean': mean of patch values
            - 'wmean': Stokes I weighted mean of patch values
            - 'min': minimum of patch values
            - 'max': maximum of patch values
            - True: only valid when the filter indices are specified directly as
                a numpy array. If True, filtering is done on patches instead of
                sources.
    applyBeam : bool, optional
        If True, apparent fluxes will be used.
    useRegEx : bool, optional
        If True, string matching will use regular expression matching. If
        False, string matching uses Unix filename matching.
    force : bool, optional
        If True, filters that result in empty sky models are allowed. If
        False, such filters are not applied and the sky model is unaffected.

    Examples
    --------
    Filter on column 'I' (Stokes I flux). This filter will select all sources
    with Stokes I flux greater than 1.5 Jy::

        >>> LSM = lsmtool.load('sky.model')
        >>> filter(LSM, 'I > 1.5 Jy')
        INFO: Filtered out 1102 sources.

    If the sky model has patches and the filter is desired per patch, use
    ``aggregate = True``::

        >>> filter(LSM, 'I > 1.5 Jy', aggregate=True)

    Filter on source names, keeping those that match "src*_1?"::

        >>> filter(LSM, 'Name == src*_1?')

    Use a CASA clean mask image to keep sources that lie in masked regions::

        >>> filter(LSM, 'clean_mask.mask == True')

    Filter on patch size::

        >>> sizes = LSM.getPatchSizes(units='arcsec', weight=True)
        >>> indices = numpy.where(sizes <= maj_cut_arcsec)
        >>> filter(LSM, indices, aggregate=True)

    or more simply::

        >>> sizes = LSM.getPatchSizes(units='arcsec', weight=True)
        >>> filter(LSM, sizes < maj_cut_arcsec, aggregate=True)

    """
    import numpy as np
    import math as m
    import os
    from astropy.table import Table

    if len(LSM) == 0:
        log.error('Sky model is empty.')
        return

    if filterExpression is None:
        raise ValueError('No filter expression specified.')

    if LSM.hasPatches and aggregate is not None:
        nrows = len(LSM.getPatchNames())
    else:
        nrows = len(LSM)

    filt = None
    if type(filterExpression) is list:
        try:
            if len(filterExpression) == 3:
                filterProp, filterOperStr, filterVal = filterExpression
                filterUnits = None
                filterOper, f = convertOperStr(filterOperStr)
            elif len(filterExpression) == 4:
                filterProp, filterOperStr, filterVal, filterUnits = filterExpression
                filterOper, f = convertOperStr(filterOperStr)
        except Exception:
            raise ValueError("Could not parse filter.")

    elif type(filterExpression) is dict:
        if ('filterProp' in filterExpression.keys() and
            'filterOper' in filterExpression.keys() and
            'filterVal' in filterExpression.keys()):
            filterProp = filterExpression['filterProp']
            filterOperStr = filterExpression['filterOper']
            filterOper, f = convertOperStr(filterOperStr)
            filterVal = filterExpression['filterVal']
        else:
            raise ValueError("Could not parse filter.")
        if 'filterUnits' in filterExpression.keys():
            filterUnits = filterExpression['filterUnits']
        else:
            filterUnits = None

    elif type(filterExpression) is str:
        # Parse the filter expression
        filterProp, filterOper, filterVal, filterUnits = parseFilter(filterExpression)

    elif type(filterExpression) is np.ndarray:
        # Array of indices / bools
        if np.result_type(filterExpression) == 'bool':
            if len(filterExpression) == nrows:
                filt = [i for i in range(len(filterExpression)) if
                    filterExpression[i]]
            else:
                raise ValueError("Boolean filter arrays be of same length as "
                    "the sky model.")
        else:
            filt = filterExpression.tolist()

    else:
        raise ValueError("Could not parse filter.")

    if filt is None and filterProp is None:
        raise ValueError('Filter expression not understood')

    if filt is None:
        # Get the column values to filter on
        if LSM._verifyColName(filterProp, quiet=True) in LSM.table.colnames:
            filterProp = LSM._verifyColName(filterProp)
            colVals = LSM.getColValues(filterProp, units=filterUnits,
                aggregate=aggregate, applyBeam=applyBeam)
        else:
            # Assume filterProp is a mask filename and try to load mask
            if os.path.exists(filterProp):
                mask = filterProp
                RARad = LSM.getColValues('Ra', units='radian')
                DecRad = LSM.getColValues('Dec', units='radian')
                colVals = getMaskValues(mask, RARad, DecRad)
            else:
                raise ValueError('Could not parse filter.')

        # Do the filtering
        if colVals is None:
            raise ValueError("Could not parse filter.")
        filt = getFilterIndices(colVals, filterOper, filterVal, useRegEx=useRegEx)

    if exclusive:
        filt = [i for i in range(nrows) if i not in filt]
    if len(filt) == 0:
        if force:
            LSM.table.remove_rows(range(len(LSM.table)))
            LSM.hasPatches = False
            if exclusive:
                log.info('Removed all sources.')
                return
            else:
                log.info('Kept zero sources.')
                return
        else:
            raise RuntimeError('Filter would result in an empty sky model. '
                'Use force=True to override.')
    if len(filt) == len(LSM):
        if exclusive:
            log.info('Removed zero sources.')
        else:
            log.info('Kept all sources.')
        return

    if LSM.hasPatches and aggregate is not None:
        sourcesToKeep = LSM.getPatchNames()[filt]
        allsources = LSM.getColValues('Patch')
        indicesToKeep = [i for i in range(len(LSM)) if allsources[i] in sourcesToKeep]
        nPatchesOrig = len(LSM.table.groups)
        LSM.table = LSM.table[indicesToKeep]
        LSM._updateGroups()
        nPatchesNew = len(LSM.table.groups)
        if nPatchesOrig - nPatchesNew == 1:
            plustr = ''
        else:
            plustr = 'es'
        if exclusive:
            log.info('Removed {0} patch{1}.'.format(nPatchesOrig-nPatchesNew, plustr))
        else:
            if nPatchesNew > 1:
                plustr = 'es'
            log.info('Kept {0} patch{1}.'.format(nPatchesNew, plustr))
    else:
        nRowsOrig = len(LSM)
        LSM.table = LSM.table[filt]
        nRowsNew = len(LSM)
        if LSM.hasPatches:
            LSM._updateGroups()
        if exclusive:
            if nRowsOrig - nRowsNew == 1:
                plustr = ''
            else:
                plustr = 's'
            log.info('Removed {0} source{1}.'.format(nRowsOrig-nRowsNew, plustr))
        else:
            if nRowsNew == 1:
                plustr = ''
            else:
                plustr = 's'
            log.info('Kept {0} source{1}.'.format(nRowsNew, plustr))

    if type(filterExpression) is np.ndarray:
        history = 'with array of indices/bools'
    else:
        _, filterOperStr = convertOperStr(filterOper)
        history = '{0} {1} {2} {3}'.format(filterProp, filterOperStr, filterVal, filterUnits)
    LSM._addHistory('FILTER ({0})'.format(history))
    LSM._info()


def parseFilter(filterExpression):
    """
    Takes a filter expression and returns tuple of (property, operation, val,
    units), all as strings.
    """
    try:
        from ..tableio import allowedColumnNames
        from ..tableio import allowedColumnDefaults
    except:
        from .tableio import allowedColumnNames
        from .tableio import allowedColumnDefaults
    from itertools import groupby

    # Get operator function
    filterOper, filterOperStr = convertOperStr(filterExpression)
    if filterOper is None:
        return (None, None, None, None)

    filterParts = filterExpression.split(filterOperStr)
    if len(filterParts) != 2:
        raise ValueError("Filter expression must be of the form '<property> "
            "<operator> <value> <unit>'\nE.g., 'Flux >= 10 Jy'")

    # Get the column to filter on
    filterProp = filterParts[0].strip().lower()
    if filterProp not in allowedColumnNames:
        log.warn('"{0}" is not a valid column. Trying it as a mask '
            'filename instead...'.format(filterProp))
        filterProp = filterParts[0].strip() # don't use lower-case version if filename
        filterVal = filterParts[1].strip()
        return (filterProp, filterOper, input2bool(filterVal), None)

    # Get the filter value(s)
    filterValAndUnits = filterParts[1].strip()
    if allowedColumnDefaults[filterProp] == 'N/A':
        # Column values are strings. Allow only '==' and '!=' operators
        if filterOperStr not in ['=', '==', '!=']:
            raise ValueError("Filter operator '{0}' not allow with string columns. "
                "Supported operators are '!=' or '=' (or '==')".format(filterOperStr))

        # Check for a list of values
        if '[' in filterValAndUnits and ']' in filterValAndUnits:
            filterVal = filterValAndUnits.split(']')[0].strip('[')
            filterValParts = filterVal.split(',')
            filterVal = []
            for val in filterValParts:
                val = val.strip()
                val = val.strip('"')
                val = val.strip("'")
                filterVal.append(val)
        else:
            filterVal = filterValAndUnits.split(' ')[0].strip()
            filterVal = filterVal.strip('"')
            filterVal = filterVal.strip("'")
    else:
        # The column to filter is type float
        try:
            parts = [''.join(g).strip() for _, g in groupby(filterValAndUnits,
                str.isalpha)]
            if len(parts) > 1:
                if type(parts[1]) is str:
                    if parts[1].lower() == 'e':
                        # Check if number uses exponential notation (e.g., 1e8)
                        parts = [parts[0] + parts[1] + parts[2]] + parts[3:]
            filterVal = float(parts[0])
        except ValueError:
            raise ValueError('Filter value not understood.')

    # Try and get the units (only if filterVal is not a string)
    if type(filterVal) is str:
        filterUnits = None
    else:
        try:
            filterUnits = filterValAndUnits.split(']')
            if len(filterUnits) == 1:
                parts = [''.join(g).strip() for _, g in groupby(filterUnits[0],
                    str.isalpha)]
                if type(parts[1]) is str:
                    if parts[1].lower() == 'e':
                        # Check if number uses exponential notation (e.g., 1e8)
                        parts = [parts[0] + parts[1] + parts[2]] + parts[3:]
                filterUnits = parts[1]
            else:
                filterUnits = filterUnits[1].strip()
        except IndexError:
            filterUnits = None
        if filterUnits == '':
            filterUnits = None

    return (filterProp, filterOper, filterVal, filterUnits)


def input2bool(invar):
    if isinstance(invar, bool):
        return invar
    elif isinstance(invar, str):
        if invar.upper() == 'TRUE' or invar == '1':
            return True
        elif invar.upper() == 'FALSE' or invar == '0':
            return False
        else:
            raise ValueError('Cannot convert string "'+invar+'" to boolean!')
    elif isinstance(invar, int) or isinstance(invar, float):
        return bool(invar)
    else:
        raise TypeError('Unsupported data type:'+str(type(invar)))


def convertOperStr(operStr):
    """
    Returns operator function corresponding to string.
    """
    import operator as op

    filterOperStr = None
    ops = {'!=':op.ne, '<=':op.le, '>=':op.ge, '>':op.gt, '<':op.lt,
        '==':op.eq, '=':op.eq}

    if type(operStr) is str:
        for op in ops:
            if op in operStr:
                if filterOperStr is None:
                    filterOperStr = op
                elif len(op) > len(filterOperStr):
                    # Pick longer match
                    filterOperStr = op
    else:
        for k, v in iteritems(ops):
            if v == operStr:
                filterOperStr = k
    if filterOperStr is None:
        return None, None

    return ops[filterOperStr], filterOperStr


def getFilterIndices(colVals, filterOper, filterVal, useRegEx=False):
    """
    Returns the indices that correspond to the input filter expression.
    """
    import operator as op
    import fnmatch
    import re

    if type(filterVal) is not list:
        filterVal = [filterVal]
    filterInd = []
    for val in filterVal:
        if type(val) is str:
            # String -> use regular expression or Unix filename matching search
            if filterOper is op.eq:
                if useRegEx:
                    filt = [i for i, item in enumerate(colVals) if re.search(val, item) is not None]
                else:
                    filt = [i for i, item in enumerate(colVals) if fnmatch.fnmatch(item, val)]
            elif filterOper is op.ne:
                if useRegEx:
                    filt = [i for i, item in enumerate(colVals) if re.search(val, item) is None]
                else:
                    filt = [i for i, item in enumerate(colVals) if not fnmatch.fnmatch(item, val)]
            else:
                raise ValueError("Filter operator '{0}' not allow with string columns. "
                    "Supported operators are '!=' or '=' (or '==')".format(filterOper))
        else:
            filtBool = filterOper(colVals, val)
            filt = [f for f in range(len(colVals)) if filtBool[f]]
        filterInd += filt

    return filterInd


def getMaskValues(mask, RARad, DecRad):
    """
    Returns an array of mask values for each (RA, Dec) pair in radians.
    """
    import math
    import pyrap.images as pim
    import numpy as np

    maskdata = pim.image(mask)
    maskval = maskdata.getdata()[0][0]

    vals = []
    for raRad, decRad in zip(RARad, DecRad):
        (a, b, _, _) = maskdata.toworld([0, 0, 0, 0])
        (_, _, pixY, pixX) = maskdata.topixel([a, b, decRad, raRad])
        try:
            if maskval[int(pixY), int(pixX)] and pixX >= 0 and pixY >=0:
                vals.append(True)
            else:
                vals.append(False)
        except IndexError:
            vals.append(False)

    return np.array(vals)
