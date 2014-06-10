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
    if outFile != '' and result == 0:
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
    try:
        from .. import tableio
    except:
        import tableio
    from tableio import RA2Angle, Dec2Angle

    if position is None and shift is None:
        logging.error("One of positon or shift must be specified.")
        return 1

    sourceNames = LSM.getColValues('Name')

    if name in sourceNames:
        indx = LSM._getNameIndx(name)
        if position is not None:
            try:
                LSM.table['Ra'][indx] = tableio.RA2Angle(position[0])[0]
                LSM.table['Dec'][indx] = tableio.Dec2Angle(position[1])[0]
            except:
                loggin.error('Postion not understood.')
        if shift is not None:
            RA = LSM.table['Ra'][indx] + tableio.RA2Angle(shift[0])
            Dec = LSM.table['Dec'][indx] + tableio.Dec2Angle(shift[1])
            LSM.table['Ra'][indx] = tableio.RA2Angle(RA)[0]
            LSM.table['Dec'][indx] = tableio.Dec2Angle(Dec)[0]
        return 0
    elif LSM._hasPatches:
        patchNames = LSM.getColValues('Patch', aggregate=True)
        if name in patchNames:
            if position is not None:
                try:
                    position[0] = tableio.RA2Angle(position[0])[0]
                    position[1] = tableio.Dec2Angle(position[1])[0]
                except:
                    loggin.error('Postion not understood.')
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
