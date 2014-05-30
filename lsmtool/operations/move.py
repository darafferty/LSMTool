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
    position = parset.getFloatVector('.'.join(["LSMTool.Steps", step, "Position"]), [] )
    shift = parset.getFloatVector('.'.join(["LSMTool.Steps", step, "Shift"]), [] )

    if len(position) < 2:
        position = None
    if len(shift) < 2:
        shift = None
    result = move(LSM, name, position, shift)

    # Write to outFile
    if outFile != '':
        LSM.write(outFile, clobber=True)

    return result


def move(LSM, name, position=None, shift=None):
    """
    Move or shift a source.

    If both a position and a shift are specified, the source is moved to the
    new position and then shifted.

    Parameters
    ----------
    name : str
        Source name.
    position : list, optional
        A list specifying a new position as [RA, Dec] in either makesourcedb
        format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
        [123.2312, 23.3422]).
    shift : list, optional
        A list specifying the shift as [RAShift, DecShift] in
        in degrees (e.g., [0.02312, 0.00342]).

    Examples
    --------
    Move source '1609.6+6556' to a new position::

        >>> s.move('1609.6+6556', position=['16:10:00', '+65.57.00'])

    Shift the source by 10 arcsec in Dec::

        >>> s.move('1609.6+6556', shift=[0.0, 10.0/3600.0])

    """
    import tableio

    if position is None and shift is None:
        logging.error("One of positon or shift must be specified.")
        return 1

    sourceNames = LSM.getColValues('Name')

    if name in sourceNames:
        indx = LSM._getNameIndx(name)
        if position is not None:
            if type(position[0]) is str:
                LSM.table['RA-HMS'][indx] = position[0]
                LSM.table['RA'][indx] = tableio.convertRAdeg(position[0])
            elif type(position[0]) is float:
                LSM.table['RA'][indx] = position[0]
                LSM.table['RA-HMS'][indx] = tableio.convertRAHHMMSS(position[0])
            else:
                loggin.error('Postion not understood.')
            if type(position[1]) is str:
                LSM.table['Dec-DMS'][indx] = position[1]
                LSM.table['Dec'][indx] = tableio.convertDecdeg(position[1])
            elif type(position[1]) is float:
                LSM.table['Dec'][indx] = position[1]
                LSM.table['Dec-DMS'][indx] = tableio.convertDecDDMMSS(position[1])
            else:
                loggin.error('Postion not understood.')
        if shift is not None:
            RA = LSM.table['RA'][indx] + shift[0]
            Dec = LSM.table['Dec'][indx] + shift[1]
            LSM.table['RA'][indx] = RA
            LSM.table['Dec'][indx] = Dec
            LSM.table['RA-HMS'][indx] = tableio.convertRAHHMMSS(RA)
            LSM.table['Dec-DMS'][indx] = tableio.convertDecDDMMSS(Dec)
        return 0
    elif LSM._hasPatches:
        patchNames = LSM.getColValues('Patch', aggregate=True)
        if name in patchNames:
            if position is not None:
                if type(position[0]) is str:
                    position[0] = tableio.convertRAdeg(position[0])
                if type(position[1]) is str:
                    position[1] = tableio.convertDecdeg(position[1])
                LSM.table.meta[name] = position
            if shift is not None:
                position = LSM.table.meta[name]
                LSM.table.meta[name] = [position[0] + shift[0], position[1] + shift[1]]
            return 0
        else:
            logging.error("Row name '{0}' not recognized.".format(name))
            return 1
    else:
        logging.error("Row name '{0}' not recognized.".format(name))
        return 1
