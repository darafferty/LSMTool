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

logging.debug('Loading CONCATENATE module.')


def run(step, parset, LSM):

    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )
    skyModel2 = parset.getString('.'.join(["LSMTool.Steps", step, "Skymodel2"]), '' )
    matchBy = parset.getString('.'.join(["LSMTool.Steps", step, "MatchBy"]), '' )
    radius = parset.getString('.'.join(["LSMTool.Steps", step, "Radius"]), '' )
    keep = parset.getString('.'.join(["LSMTool.Steps", step, "KeepMatches"]), '' )

    result = concatenate(LSM, skyModel2, matchBy, radius, keep)

    # Write to outFile
    if outFile != '':
        LSM.write(outFile, clobber=True)

    return result


def concatenate(LSM1, LSM2, matchBy='name', radius=10.0, keep='all'):
    """
    Concatenate two sky models
    """
    from astropy.table import vstack, Column
    from astropy.coordinates import ICRS
    from astropy import units as u
    import numpy as np
    from .. import skymodel

    if type(LSM2) is str:
        LSM2 = skymodel.SkyModel(LSM2)

    if (LSM1._hasPatches and not LSM2._hasPatches):
         LSM2.group('every')
    if (LSM2._hasPatches and not LSM1._hasPatches):
         LSM1.group('every')
    table1 = LSM1.table.copy()
    table2 = LSM2.table.copy()

    # Due to a bug in astropy, the spectral index column must be removed before
    # joining
    nameCol1 = table1['Name']
    nameCol2 = table2['Name']
    spCol1 = table1['SpectralIndex']
    spCol1Indx = table1.index_column('SpectralIndex')
    spCol2 = table2['SpectralIndex']
    table1.remove_column('SpectralIndex')
    table2.remove_column('SpectralIndex')

    if matchBy.lower() == 'name':
        LSM1.table = vstack([table1, table2])
    if matchBy.lower() == 'patch':
        LSM1.table = vstack([table1, table2])
    elif matchBy.lower() == 'radius':
        # Create catalogs
        catalog1 = ICRS(LSM1.getColValues('RA'), LSM1.getColValues('Dec'),
            unit=(u.degree, u.degree))
        catalog2 = ICRS(LSM2.getColValues('RA'), LSM2.getColValues('Dec'),
            unit=(u.degree, u.degree))
        idx, d2d, d3d = catalog1.match_to_catalog_sky(catalog2)

        matches = np.where(d2d.value <= radius)

        matchCol1 = np.array(range(len(LSM1)))
        matchCol2 = np.array(range(len(LSM2))) + len(LSM1)

        # Set values to be the same for the matches
        matchCol2[idx[matches]] = matchCol1[matches]

        # Now add columns and stack
        col1 = Column(name='match', data=matchCol1)
        col2 = Column(name='match', data=matchCol2)
        table1.add_column(col1)
        table2.add_column(col2)
        LSM1.table = vstack([table1, table2])

    # Add spectral index column back
    spData = np.zeros((len(LSM1), 2), dtype=np.float)
    for i, name in enumerate(LSM1.getColValues('Name')):
        if name in nameCol2:
            indx = np.where(nameCol2.data == name)
            spData[i] = spCol2[indx]
        else:
            indx = np.where(nameCol1.data == name)
            spData[i] = spCol1[indx]
    spCol = Column(name='SpectralIndex', data=spData)
    LSM1.table.add_column(spCol, index=spCol1Indx)

    if keep == 'from1' or keep == 'from2':
        # Remove any duplicates
        if matchBy.lower() == 'name':
            colName = 'Name'
        elif matchBy.lower() == 'radius':
            colName = 'match'
        vals = LSM1.table[colName]
        toRemove = []
        for val in vals:
            indx = np.where(vals == val)[0]
            if len(indx) > 1:
                if keep == 'from1':
                    toRemove.append(indx[1:])
                else:
                    toRemove.append(indx[0])
        LSM1.table.remove_rows(toRemove)

    # Rename any duplicates
    names = LSM1.getColValues('Name')
    for name in set(names):
        indx = np.where(names == name)[0]
        if len(indx) > 1:
            LSM1.table['Name'][indx[0]] = name + '_1'
            LSM1.table['Name'][indx[1]] = name + '_2'

    if matchBy.lower() == 'radius':
        LSM1.table.remove_column('match')

    if LSM1._hasPatches:
        LSM1._updateGroups()
    return 0

