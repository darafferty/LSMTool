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

logging.debug('Loading ADD module.')


def run(step, parset, LSM):

    from ..tableio import outputColumnNames

    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )
    colNamesVals = {}
    for colName in outputColumnNames:
        val = parset.getString('.'.join(["LSMTool.Steps",
            step, outputColumnNames[colName]]), '' )
        if val != '':
            try:
                val = float(val)
            except ValueError:
                pass
            colNamesVals[outputColumnNames[colName]] = val


    result = add(LSM, colNamesVals)

    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result


def add(LSM, colNamesVals):
    """
    Add a source to the sky model.

    Parameters
    ----------
    LSM : SkyModel object
        Input sky model.
    colNamesVals : dict
        A dictionary that specifies the row values for the source to be added.

    Examples:
    ---------
    Add a point source::

        >>> source = {'Name':'src1', 'Type':'POINT', 'Ra':'12:32:10.1',
            'Dec':'23.43.21.21', 'I':2.134}
        >>> add(LSM, source)

    """
    result = LSM.setRowValues(colNamesVals)
    LSM._info()
    return result
