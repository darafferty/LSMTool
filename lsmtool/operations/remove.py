#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This operation implements removal of sources using a filter
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
log = logging.getLogger('LSMTool.REMOVE')
log.debug('Loading REMOVE module.')


def run(step, parset, LSM):

    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )
    filterExpression = parset.getString('.'.join(["LSMTool.Steps", step, "FilterExpression"]), '' )
    aggregate = parset.getString('.'.join(["LSMTool.Steps", step, "Aggregate"]), '' )
    applyBeam = parset.getBool('.'.join(["LSMTool.Steps", step, "ApplyBeam"]), False )

    if filterExpression == '':
        filterExpression = None
    if aggregate == '':
        aggregate = None

    try:
        remove(LSM, filterExpression, aggregate=aggregate, applyBeam=applyBeam)
        result = 0
    except Exception as e:
        log.error(e.message)
        result = 1

    # Write to outFile
    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result


def remove(LSM, filterExpression, aggregate=None, applyBeam=None,
    useRegEx=False, force=True):
    """
    Filters the sky model, removing all sources that meet the given expression.

    After filtering, the sky model contains only those sources for which the
    given filter expression is false.

    Parameters
    ----------
    LSM : SkyModel object
        Input sky model
    filterExpression : str, dict, list, or numpy array
        - If string:
            A string specifying the filter expression in the form:
            '<property> <operator> <value> [<units>]'
            (e.g., 'I <= 10.5 Jy')

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
            set aggregate=True

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
        Units are optional and must be specified as required by astropy.units
    aggregate : str, optional
        If set, the array returned will be of values aggregated
        over the patch members. The following aggregation functions are
        available:
            - 'sum': sum of patch values
            - 'mean': mean of patch values
            - 'wmean': Stokes I weighted mean of patch values
            - 'min': minimum of patch values
            - 'max': maximum of patch values
            - True: only valid when the filter indices are specified directly as
                a numpy array. If True, filtering is done on patches instead of
                sources
    applyBeam : bool, optional
        If True, apparent fluxes will be used
    useRegEx : bool, optional
        If True, string matching will use regular expression matching. If
        False, string matching uses Unix filename matching

    Examples
    --------
    Filter on column 'I' (Stokes I flux). This filter will remove all sources
    with Stokes I flux greater than 1.5 Jy::

        >>> LSM = lsmtool.load('sky.model')
        >>> remove(LSM, 'I > 1.5 Jy')
        INFO: Removed 1102 sources.

    If the sky model has patches and the filter is desired per patch, use
    ``aggregate = function``. For example, to select on the sum of the patch
    fluxes::

        >>> remove(LSM, 'I > 1.5 Jy', aggregate='sum')

    Filter on source names, removing those that match "src*_1?"::

        >>> remove(LSM, 'Name == src*_1?')

    Use a CASA clean mask image to remove sources that lie in masked regions::

        >>> remove(LSM, 'clean_mask.mask == True')

    """
    from . import _filter

    _filter.filter(LSM, filterExpression, aggregate=aggregate,
        applyBeam=applyBeam, useRegEx=useRegEx, exclusive=True, force=force)
