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

logging.debug('Loading REMOVE module.')


def run(step, parset, LSM):

    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )
    filterExpression = parset.getString('.'.join(["LSMTool.Steps", step, "FilterExpression"]), '' )
    aggregate = parset.getBool('.'.join(["LSMTool.Steps", step, "Aggregate"]), False )
    weight = parset.getBool('.'.join(["LSMTool.Steps", step, "Weight"]), False )
    applyBeam = parset.getBool('.'.join(["LSMTool.Steps", step, "ApplyBeam"]), False )

    if filterExpression == '':
        filterExpression = None
    LSM = remove(LSM, filterExpression, aggregate=aggregate, weight=weight,
        applyBeam=applyBeam)

    # Write to outFile
    if outFile != '':
        LSM.write(outFile, clobber=True)

    return 0


def remove(LSM, filterExpression, aggregate=False, weight=False,
    applyBeam=None, useRegEx=False):
    """
    Filters the sky model, removing all sources that meet the given expression.

    After filtering, the sky model contains only those sources for which the
    given filter expression is false.

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
    applyBeam : bool, optional
        If True, apparent fluxes will be used.
    useRegEx : bool, optional
        If True, string matching will use regular expression matching. If
        False, string matching uses Unix filename matching.

    Examples
    --------
    Filter on column 'I' (Stokes I flux). This filter will remove all sources
    with Stokes I flux greater than 1.5 Jy::

        >>> s.remove('I > 1.5 Jy')
        INFO: Filtered out 1102 sources.

    If the sky model has patches and the filter is desired per patch, use
    ``aggregate = True``::

        >>> s.remove('I > 1.5 Jy', aggregate=True)

    Filter on source names, removing those that match "src*_1?"::

        >>> s.remove('Name == src*_1?')

    Use a CASA clean mask image to remove sources that lie in masked regions::

        >>> s.remove('clean_mask.mask == True')

    """
    from _filter import filter

    return filter(LSM, filterExpression, aggregate=aggregate, weight=weight,
        applyBeam=applyBeam, useRegEx=useRegEx, exclusive=True)
