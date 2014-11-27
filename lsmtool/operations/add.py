#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This operation implements adding of sources to the sky model
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

log = logging.getLogger('LSMTool.ADD')
log.debug('Loading ADD module.')


def run(step, parset, LSM):

    from ..tableio import allowedColumnNames

    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )
    colNamesVals = {}
    for colName in allowedColumnNames:
        val = parset.getString('.'.join(["LSMTool.Steps",
            step, allowedColumnNames[colName]]), '' )
        if val != '':
            try:
                val = float(val)
            except ValueError:
                pass
            colNamesVals[allowedColumnNames[colName]] = val

    try:
        add(LSM, colNamesVals)
        result = 0
    except Exception as e:
        log.error(e.message)
        result = 1

    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result


def add(LSM, colNamesVals):
    """
    Add a source to the sky model.

    Parameters
    ----------
    LSM : SkyModel object
        Input sky model
    colNamesVals : dict
        A dictionary that specifies the column values for the source to be added

    Examples:
    ---------
    Add a point source::

        >>> source = {'Name':'src1', 'Type':'POINT', 'Ra':'12:32:10.1',
            'Dec':'23.43.21.21', 'I':2.134}
        >>> add(LSM, source)

    """
    sourceNames = LSM.getColValues('Name').tolist()
    if colNamesVals['Name'] in sourceNames:
        raise ValueError('A source with the same name already exists.')

    LSM.setRowValues(colNamesVals)
    LSM._updateGroups()
    LSM._addHistory("ADD (source '{0}')".format(colNamesVals['Name']))
    LSM._info()
