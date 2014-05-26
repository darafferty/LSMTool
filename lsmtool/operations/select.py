#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This operation implements selection of sources using a filter
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

logging.debug('Loading SELECT module.')


def run(step, parset, LSM):

    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )
    filterExpression = parset.getString('.'.join(["LSMTool.Steps", step, "FilterExpression"]), '' )
    aggregate = parset.getBool('.'.join(["LSMTool.Steps", step, "Aggregate"]), False )
    beamMS = parset.getFloat('.'.join(["LSMTool.Steps", step, "BeamMS"]), 1. )

    if filterExpression == '':
        filterExpression = None
    LSM = select(LSM, filterExpression, aggregate=aggregate, weight=weight,
        beamMS=beamMS, useRegEx=useRegEx)

    # Write to outFile
    if outFile == '' or outFile is None:
        outFile = LSM._fileName
    LSM.writeFile(outFile, clobber=True)

    return 0


def select(LSM, filterExpression, aggregate=False, weight=False,
    beamMS=None, useRegEx=False):
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
            [property, operator, value, value]
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
    aggregate : bool, optional
        If True, values are aggregated by patch before filtering. There,
        filtering will be done by patch.
    weight : bool, optional
        If True, aggregated values will be calculated when appropriate using
        the Stokes I fluxes of sources in each patch as weights
    beamMS : string, optional
        Measurement set from which the primary beam will be estimated. Either
        beamMS or beamFWHM and position must be specified for ApparentFlux
        filters.
    useRegEx : bool, optional
        If True, string matching will use regular expression matching. If
        False, string matching uses Unix filename matching.

    Examples
    --------
    Filter on column 'I' (Stokes I flux). This filter will select all sources
    with Stokes I flux greater than 1.5 Jy::

        >>> s = SkyModel('sky.model')
        >>> s.select('I > 1.5 Jy')
        INFO: Filtered out 1102 sources.

    If the sky model has patches and the filter is desired per patch, use
    ``aggregate = True``::

        >>> s.select('I > 1.5 Jy', aggregate=True)

    Filter on source names, keeping those that match "src*_1?"::

        >>> s.select('Name == src*_1?')

    Use a CASA clean mask image to keep sources that lie in masked regions::

        >>> s.filter('clean_mask.mask == True')

    """
    from _filter import filter

    return filter(LSM, filterExpression, aggregate=aggregate, weight=weight,
        beamMS=beamMS, useRegEx=useRegEx, exclusive=False)
