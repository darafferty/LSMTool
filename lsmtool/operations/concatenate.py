#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This operation implements joining of two sky models
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

log = logging.getLogger('LSMTool.CONCATENATE')
log.debug('Loading CONCATENATE module.')


def run(step, parset, LSM):

    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )
    skyModel2 = parset.getString('.'.join(["LSMTool.Steps", step, "Skymodel2"]), '' )
    matchBy = parset.getString('.'.join(["LSMTool.Steps", step, "MatchBy"]), 'name' )
    radius = parset.getString('.'.join(["LSMTool.Steps", step, "Radius"]), '0.1' )
    keep = parset.getString('.'.join(["LSMTool.Steps", step, "Keep"]), 'all' )
    inheritPatches = parset.getBool('.'.join(["LSMTool.Steps", step, "InheritPatches"]), False )

    try:
        concatenate(LSM, skyModel2, matchBy, radius, keep, inheritPatches)
        result = 0
    except Exception as e:
        log.error(e.message)
        result = 1

    # Write to outFile
    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result


def concatenate(LSM1, LSM2, matchBy='name', radius=0.1, keep='all',
    inheritPatches=False):
    """
    Concatenate two sky models

    Note that the mergesourcedb tool performs a similar function (but on
    SourceDB data sets, not on sky model files; see
    http://www.lofar.org/operations/doku.php?id=engineering:software:tools:makesourcedb#format_string
    for details).

    Parameters
    ----------
    LSM1 : SkyModel object
        Parent sky model
    LSM2 : SkyModel object
        Sky model to concatenate with the parent sky model
    matchBy : str, optional
        Determines how duplicate sources are determined:
        - 'name' => duplicates are identified by name
        - 'position' => duplicates are identified by radius. Sources within the
            radius specified by the radius parameter are considered duplicates
    radius : float or str, optional
        Radius in degrees (if float) or 'value unit' (if str; e.g., '30 arcsec')
        for matching when matchBy='position'
    keep : str, optional
        Determines how duplicates are treated:
        - 'all' => all duplicates are kept; those with identical names are re-
            named
        - 'from1' => duplicates kept are those from sky model 1 (LSM1)
        - 'from2' => duplicates kept are those from sky model 2 (LSM2)
    inheritPatches : bool, optional
        If True, duplicates inherit the patch name from the parent sky model. If
        False, duplicates keep their own patch names.

    Examples
    --------
    Concatenate two sky models, identifying duplicates by matching to the source
    names. When duplicates are found, keep the source from the parent sky model
    and discard the duplicate from second sky model (this might be useful when
    merging two sky models that have some overlap)::

        >>> LSM1 = lsmtool.load('sky1.model')
        >>> LSM2 = lsmtool.load('sky2.model')
        >>> concatenate(LSM1, LSM2, matchBy='name', keep='from1')

    Concatenate two sky models, identifying duplicates by matching to the source
    positions within a radius of 10 arcsec. When duplicates are found, keep the
    source from the second sky model and discard the duplicate from the parent
    sky model (this might be useful when replacing parts of a low-resolution
    sky model with a high-resolution one)::

        >>> LSM2 = lsmtool.load('high_res_sky.model')
        >>> concatenate(LSM1, LSM2, matchBy='position', radius=10.0/3600.0,
            keep='from2')

    """
    from astropy.table import vstack, Column
    from ..operations_lib import matchSky
    from ..skymodel import SkyModel
    import numpy as np

    if type(LSM2) is str:
        LSM2 = SkyModel(LSM2)

    if len(LSM1) == 0:
        log.info('Parent sky model is empty. Concatenated sky model is '
            'copy of secondary sky model.')
        LSM1.table = LSM2.table
        LSM1._updateGroups()
        LSM1._info()
        return
    if len(LSM2) == 0:
        log.info('Secondary sky model is empty. Parent sky model left '
            'unaltered.')
        return

    if (LSM1.hasPatches and not LSM2.hasPatches):
         LSM2.group('every')
    if (LSM2.hasPatches and not LSM1.hasPatches):
         LSM2.ungroup()

    # Make sure spectral index entries are of same length
    if 'SpectralIndex' in LSM1.getColNames() and 'SpectralIndex' in LSM2.getColNames():
        len1 = len(LSM1.getColValues('SpectralIndex')[0])
        len2 = len(LSM2.getColValues('SpectralIndex')[0])

        if len1 < len2:
            oldspec = LSM1.getColValues('SpectralIndex')
            newspec = []
            for specind in oldspec:
                speclist = specind.tolist()
                while len(speclist) < len2:
                    speclist.append(0.0)
                newspec.append(speclist)
            specCol = Column(name='SpectralIndex', data=np.array(newspec, dtype=np.float))
            specIndx = LSM1.table.keys().index('SpectralIndex')
            LSM1.table.remove_column('SpectralIndex')
            LSM1.table.add_column(specCol, index=specIndx)

        elif len1 > len2:
            oldspec = LSM2.getColValues('SpectralIndex')
            newspec = []
            for specind in oldspec:
                speclist = specind.tolist()
                while len(speclist) < len1:
                    speclist.append(0.0)
                newspec.append(speclist)
            specCol = Column(name='SpectralIndex', data=np.array(newspec, dtype=np.float))
            specIndx = LSM2.table.keys().index('SpectralIndex')
            LSM2.table.remove_column('SpectralIndex')
            LSM2.table.add_column(specCol, index=specIndx)

    # Fill masked values and merge defaults and RA, Dec formaters
    table1 = LSM1.table.filled()
    table2 = LSM2.table.filled()
    for entry in table1.meta:
        if LSM1._verifyColName(entry, quiet=True) is not None:
            if entry in table2.meta.keys():
                table1.meta[entry] = table2.meta[entry]
    table1['Ra'].format = table2['Ra'].format
    table1['Dec'].format = table2['Dec'].format

    # Now concatenate the tables
    if matchBy.lower() == 'name':
        LSM1.table = vstack([table1, table2], metadata_conflicts='silent')
    elif matchBy.lower() == 'position':
        matches1, matches2 = matchSky(LSM1, LSM2, radius=radius)
        matchCol1 = np.array(range(len(LSM1)))
        matchCol2 = np.array(range(len(LSM2))) + len(LSM1)

        # Set values to be the same for the matches
        matchCol2[matches2] = matchCol1[matches1]

        # Now add columns and stack
        col1 = Column(name='match', data=matchCol1)
        col2 = Column(name='match', data=matchCol2)
        table1.add_column(col1)
        table2.add_column(col2)
        LSM1.table = vstack([table1, table2], metadata_conflicts='silent')

    if keep == 'from1' or keep == 'from2':
        # Remove any duplicates
        if matchBy.lower() == 'name':
            colName = 'Name'
        elif matchBy.lower() == 'position':
            colName = 'match'
        else:
            raise ValueError('Invalid matchBy parameter.')
        vals = LSM1.table[colName]
        for val in vals:
            valsCur = LSM1.table[colName]
            toRemove = []
            indx = np.where(valsCur == val)[0]
            if len(indx) > 1:
                if keep == 'from1':
                    toRemove.append(indx[1:])
                else:
                    toRemove.append(indx[0])
                    if inheritPatches and LSM1.hasPatches:
                        LSM1.table['Patch'][indx[1:]] = LSM1.table['Patch'][indx[0]]
                LSM1.table.remove_rows(toRemove)

    # Rename any duplicates
    check_duplicates = True
    while check_duplicates:
        names = LSM1.getColValues('Name')
        check_duplicates = False
        for name in set(names):
            indx = np.where(names == name)[0]
            if len(indx) > 1:
                check_duplicates = True
                LSM1.table['Name'][indx[0]] = name + '_1'
                LSM1.table['Name'][indx[1]] = name + '_2'

    if matchBy.lower() == 'position':
        LSM1.table.remove_column('match')

    LSM1._updateGroups()
    history = "matchBy = '{0}', ".format(matchBy)
    if matchBy.lower() == 'position':
        history += "radius = {0}".format(radius)
    history += " keep = '{0}'".format(keep)
    LSM1._addHistory("CONCATENATE ({0})".format(history))
    LSM1._info()

