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


def filter(LSM, filterExpression, exclusive=False, aggregate=False, weight=False,
    applyBeam=False, useRegEx=False):
    """
    Filters the sky model, keeping all sources that meet the given expression.

    After filtering, the sky model contains only those sources for which the
    given filter expression is true.

    Parameters
    ----------
    filterExpression : str or dict
        A string specifying the filter expression in the form:
            '<property> <operator> <value> [<units>]'
        (e.g., 'I <= 10.5 Jy'). These elements can also be given as a
        dictionary in the form:
            {'filterProp':property, 'filterOper':operator,
                'filterVal':value, 'filterUnits':units}
        or as a list:
            [property, operator, value] or
            [property, operator, value, units]
        The property to filter on must be a valid column name or the filename
        of a mask image.

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
    aggregate : bool, optional
        If True, values are aggregated by patch before filtering. There,
        filtering will be done by patch.
    weight : bool, optional
        If True, aggregated values will be calculated when appropriate using
        the Stokes I fluxes of sources in each patch as weights.
    applyBeam : bool, optional
        If True, apparent fluxes will be used.
    useRegEx : bool, optional
        If True, string matching will use regular expression matching. If
        False, string matching uses Unix filename matching.

    Examples
    --------
    Filter on column 'I' (Stokes I flux). This filter will select all sources
    with Stokes I flux greater than 1.5 Jy::

        >>> s = SkyModel('sky.model')
        >>> s.filter('I > 1.5 Jy')
        INFO: Filtered out 1102 sources.

    If the sky model has patches and the filter is desired per patch, use
    ``aggregate = True``::

        >>> s.filter('I > 1.5 Jy', aggregate=True)

    Filter on source names, keeping those that match "src*_1?"::

        >>> s.filter('Name == src*_1?')

    Use a CASA clean mask image to keep sources that lie in masked regions::

        >>> s.filter('clean_mask.mask == True')

    """
    import numpy as np
    import math as m
    import os

    if filterExpression is None:
        logging.error('Please specify a filter expression.')
        return 1

    if type(filterExpression) is list:
        if len(filterExpression) == 3:
            filterProp, filterOperStr, filterVal = filterExpression
            filterUnits = None
            filterOper, f = convertOperStr(filterOperStr)
        elif len(filterExpression) == 4:
            filterProp, filterOperStr, filterVal, filterUnits = filterExpression
            filterOper, f = convertOperStr(filterOperStr)
        else:
            logging.error("Please specify filter list as "
                "[property, operator, value, units].")
            return 1

    elif type(filterExpression) is dict:
        if ('filterProp' in filterExpression.keys() and
            'filterOper' in filterExpression.keys() and
            'filterVal' in filterExpression.keys()):
            filterProp = filterExpression['filterProp']
            filterOperStr = filterExpression['filterOper']
            filterOper, f = convertOperStr(filterOperStr)
            filterVal = filterExpression['filterVal']
        else:
            logging.error("Please specify filter dictionary as "
                "{'filterProp':property, 'filterOper':operator, "
                "'filterVal':value, 'filterUnits':units}")
            return 1
        if 'filterUnits' in filterExpression.keys():
            filterUnits = filterExpression['filterUnits']
        else:
            filterUnits = None

    elif type(filterExpression) is str:
        # Parse the filter expression
        filterProp, filterOper, filterVal, filterUnits = parseFilter(filterExpression)
    else:
        return 1

    # Get the column values to filter on
    if LSM._verifyColName(filterProp) in LSM.table.colnames:
        filterProp = LSM._verifyColName(filterProp)
        colVals = LSM.getColValues(filterProp, units=filterUnits,
            aggregate=aggregate, weight=weight, applyBeam=applyBeam)
    else:
        # Assume filterProp is a mask filename and try to load mask
        if os.path.exists(fileName):
            RARad = LSM.getColValues('RA', units='radian')
            DecRad = LSM.getColValues('Dec', units='radian')
            colVals = getMaskValues(mask, RARad, DecRad)
            if colVals is None:
                return 1
        else:
            return 1

    # Do the filtering
    if colVals is None:
        return 1
    filt = getFilterIndices(colVals, filterOper, filterVal, useRegEx=useRegEx)
    if exclusive:
        filt = [i for i in range(len(colVals)) if i not in filt]
    if len(filt) == 0:
        logging.error('Filter would result in an empty sky model.')
        return 1
    if len(filt) == len(colVals):
        logging.info('Filtered out 0 sources.')
        return 0

    if LSM._hasPatches and aggregate:
        sourcesToKeep = LSM.getColValues('Patch', aggregate=True)[filt]
        def filterByName(tab, key_colnames):
            if tab['Patch'][0] in sourcesToKeep:
                return True
            else:
                return False
        nPatchesOrig = len(LSM.table.groups)
        LSM.table = LSM.table.groups.filter(filterByName) # filter
        LSM.table = LSM.table.group_by('Patch') # regroup
        nPatchesNew = len(LSM.table.groups)
        if nPatchesOrig-nPatchesNew == 1:
            plustr = ''
        else:
            plustr = 'es'
        if exclusive:
            logging.info('Removed {0} patch{1}.'.format(nPatchesOrig-nPatchesNew, plustr))
        else:
            logging.info('Kept {0} patch{1}.'.format(nPatchesNew, plustr))
    else:
        nRowsOrig = len(LSM.table)
        LSM.table = LSM.table[filt]
        nRowsNew = len(LSM.table)
        if LSM._hasPatches:
            LSM.table = LSM.table.group_by('Patch') # regroup
        if nRowsOrig-nRowsNew == 1:
            plustr = ''
        else:
            plustr = 's'
        if exclusive:
            logging.info('Removed {0} source{1}.'.format(nRowsOrig-nRowsNew, plustr))
        else:
            logging.info('Kept {0} source{1}.'.format(nRowsNew, plustr))

    return 0


def parseFilter(filterExpression):
    """
    Takes a filter expression and returns tuple of
    (property, operation, val, units), all as strings
    """
    try:
        from tableio import allowedColumnNames
        from tableio import allowedColumnDefaults
    except ImportError:
        from lsmtool.tableio import allowedColumnNames
        from lsmtool.tableio import allowedColumnDefaults

    # Get operator function
    filterOper, filterOperStr = convertOperStr(filterExpression)
    if filterOper is None:
        return (None, None, None, None)

    filterParts = filterExpression.split(filterOperStr)
    if len(filterParts) != 2:
        logging.error("Filter expression must be of the form '<property> "
            "<operator> <value> <unit>'\nE.g., 'Flux >= 10 Jy'")
        return (None, None, None, None)

    # Get the column to filter on
    filterProp = filterParts[0].strip().lower()
    if filterProp not in allowedColumnNames:
        logging.error('Column name "{0}" is not a valid column.'.format(colName))
        return (None, None, None, None)

    # Get the filter value(s)
    filterValAndUnits = filterParts[1].strip()
    if allowedColumnDefaults[filterProp] == 'N/A':
        # Column values are strings. Allow only '==' and '!=' operators
        if filterOperStr not in ['=', '==', '!=']:
            logging.error("Filter operator '{0}' not allow with string columns. "
                "Supported operators are '!=' or '=' (or '==')".format(filterOperStr))
            return (None, None, None, None)

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
            filterVal = float(filterValAndUnits.split(' ')[0].strip())
        except ValueError:
            logging.error('Filter value not understood. Make sure the value is '
                'separated from the units (if any)')
            return (None, None, None, None)

    # Try and get the units
    try:
        filterUnits = filterValAndUnits.split(']')
        if len(filterUnits) == 1:
            filterUnits = filterUnits[0].split(' ')[1].strip()
        else:
            filterUnits = filterUnits[1].strip()
    except IndexError:
        filterUnits = None
    if filterUnits == '':
        filterUnits = None

    if type(filterVal) is str and filterUnits is not None:
        logging.error('Units are not allowed with string columns.')
        return (None, None, None, None)

    return (filterProp, filterOper, filterVal, filterUnits)


def convertOperStr(operStr):
    """
    Returns operator function corresponding to string.
    """
    import operator as op

    filterOperStr = None
    ops = {'!=':op.ne, '<=':op.le, '>=':op.ge, '>':op.gt, '<':op.lt,
        '==':op.eq, '=':op.eq}
    for op in ops:
        if op in operStr:
            if filterOperStr is None:
                filterOperStr = op
            elif len(op) > len(filterOperStr):
                # Pick longer match
                filterOperStr = op
    if filterOperStr is None:
        logging.error("Filter operator '{0}' not understood. Supported "
            "operators are '!=', '<=', '>=', '>', '<', '=' (or '==')".
            format(operStr))
        return None, None

    return ops[filterOperStr], filterOperStr


def getFilterIndices(colVals, filterOper, filterVal, useRegEx=False):
    """
    Returns the indices that correspond to the input filter expression
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
                logging.error("Filter operator '{0}' not allow with string columns. "
                    "Supported operators are '!=' or '=' (or '==')".format(filterOper))
                return None
        else:
            filtBool = filterOper(colVals, val)
            filt = [f for f in range(len(colVals)) if filtBool[f]]
        filterInd += filt

    return filterInd


def getMaskValues(mask, RARad, DecRad):
    """
    Returns an array of mask values for each (RA, Dec) pair in radians
    """
    import math
    import pyrap

    try:
        maskdata = pyrap.images.image(mask)
        maskval = maskdata.getdata()[0][0]
    except:
        loggin.error("Error opening mask file '{0}'".format(mask))
        return None

    vals = []
    for raRad, decRad in zip(RARad, DecRad):
        (a, b, _, _) = maskdata.toworld([0, 0, 0, 0])
        (_, _, pixY, pixX) = maskdata.topixel([a, b, decRad, raRad])
        try:
            # != is a XOR for booleans
            if (not maskval[math.floor(pixY)][math.floor(pixX)]) != False:
                vals.append(True)
            else:
                vals.append(False)
        except:
            vals.append(False)

    return np.array(vals)
