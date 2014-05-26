#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This operation implements grouping of sources into patches
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

logging.debug('Loading GROUP module.')


def run(step, parset, LSM):

    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )
    name = parset.getString('.'.join(["LSMTool.Steps", step, "Name"]), '' )
    position = parset.getString('.'.join(["LSMTool.Steps", step, "Position"]), '' )
    shift = parset.getString('.'.join(["LSMTool.Steps", step, "Shift"]), '' )

    result = move(LSM, name, position, shift)

    # Write to outFile
    if outFile == '' or outFile is None:
        outFile = LSM._fileName
    LSM.writeFile(outFile, clobber=True)

    return result


def move(LSM, name, position=None, shift=None):

    if position is None and shift is None:
        logging.error("One of positon or shift must be specified.".format(rowName))
        return 1

    sourceNames = LSM.getColValues('Name')

    if rowName in sourceNames:
        indx = LSM._getNameIndx(rowName)
        if position is not None:
            LSM.table['RA-HMS'][indx] = position[0]
            LSM.table['Dec-DMS'][indx] = position[1]
            LSM.table['RA'][indx] = tableio.convertRAdeg(position[0])
            LSM.table['Dec'][indx] = tableio.convertDecdeg(position[1])
        elif shift is not None:
            RA = LSM.table['RA'][indx] + shift[0]
            Dec = LSM.table['Dec'][indx] + shift[1]
            LSM.table['RA'][indx] = RA
            LSM.table['Dec'][indx] = Dec
            LSM.table['RA-HMS'][indx] = tableio.convertRAHHMMSS(RA)
            LSM.table['Dec-DMS'][indx] = convertDecDDMMSS(Dec)

    elif LSM._hasPatches:
        patchNames = self.getColValues('Patch', aggregate=True)
        if rowName in patchNames:
            if position is not None:
                LSM.table.meta[rowName] = position
            elif shift is not None:
                position = LSM.table.meta[rowName]
                LSM.table.meta[rowName] = [position[0] + shift[0], position[1] + shift[1]]
        else:
            logging.error("Row name '{0}' not recognized.".format(rowName))
            return 1
    else:
        logging.error("Row name '{0}' not recognized.".format(rowName))
        return 1
