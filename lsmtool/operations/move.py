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

log = logging.getLogger('LSMTool.MOVE')
log.debug('Loading MOVE module.')


def run(step, parset, LSM):

    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )
    name = parset.getString('.'.join(["LSMTool.Steps", step, "Name"]), '' )
    position = parset.getStringVector('.'.join(["LSMTool.Steps", step, "Position"]), [] )
    shift = parset.getStringVector('.'.join(["LSMTool.Steps", step, "Shift"]), [] )

    if len(position) < 2:
        position = None
    if len(shift) < 2:
        shift = None

    try:
        move(LSM, name, position, shift)
        result = 0
    except Exception as e:
        log.error(e.message)
        result = 1

    # Write to outFile
    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result


def move(LSM, name, position=None, shift=None, xyshift=None, fitsFile=None):
    """
    Move or shift a source or sources.

    If both a position and a shift are specified, a source is moved to the
    new position and then shifted. Note that only a single source can be
    moved to a new position. However, multiple sources can be shifted.

    If an xyshift is specified, a FITS file must also be specified to define
    the WCS system. If a position, a shift, and an xyshift are all specified,
    a source is moved to the new position, shifted in RA and Dec, and then
    shifted in x and y.

    Parameters
    ----------
    LSM : SkyModel object
        Input sky model
    name : str or list
        Source name or list of names (can include wildcards)
    position : list, optional
        A list specifying a new position as [RA, Dec] in either makesourcedb
        format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
        [123.2312, 23.3422])
    shift : list, optional
        A list specifying the shift as [RAShift, DecShift] in degrees (e.g.,
        [0.02312, 0.00342])
    xyshift : list, optional
        A list specifying the shift as [xShift, yShift] in pixels. A FITS file
        must be specified with the fitsFILE argument
    fitsFile : str, optional
        A FITS file from which to take WCS information to transform the pixel
        coordinates to RA and Dec values. The xyshift argument must be specfied
        for this to be useful

    Examples
    --------
    Move source '1609.6+6556' to a new position::

        >>> LSM = lsmtool.load('sky.model')
        >>> move(LSM, '1609.6+6556', position=['16:10:00', '+65.57.00'])

    Shift the source by 10 arcsec in Dec::

        >>> move(LSM, '1609.6+6556', shift=[0.0, 10.0/3600.0])

    Shift all sources by 10 pixels in x::

        >>> move(LSM, '*', xyshift=[10, 0], fitsFile='image.fits')

    """
    from .. import tableio
    from astropy.io.fits import getheader
    from astropy import wcs

    if len(LSM) == 0:
        log.error('Sky model is empty.')
        return

    if fitsFile is not None:
        if xyshift is None:
            log.warn("A FITS file is specified, but no xyshift is specified.")
        hdr = getheader(fitsFile, 1)
        w = wcs.WCS(hdr)
    elif xyshift is not None:
        raise ValueError("A FITS file must be specified to use xyshift.")
    if position is None and shift is None and xyshift is None:
        raise ValueError("One of positon, shift, or xyshift must be specified.")

    sourceNames = LSM.getColValues('Name')
    table = LSM.table.copy()
    indx = LSM._getNameIndx(name)
    if indx is not None:
        if position is not None:
            if len(indx) > 1:
                raise ValueError('Only one source can be moved to a new position')
            try:
                table['Ra'][indx] = tableio.RA2Angle(position[0])[0]
                table['Dec'][indx] = tableio.Dec2Angle(position[1])[0]
            except Exception as e:
                raise ValueError('Could not parse position: {0}'.format(e.message))
        if shift is not None:
            for ind in indx:
                RA = LSM.table['Ra'][ind] + tableio.RA2Angle(shift[0])
                Dec = LSM.table['Dec'][ind] + tableio.Dec2Angle(shift[1])
                table['Ra'][ind] = tableio.RA2Angle(RA)[0]
                table['Dec'][ind] = tableio.Dec2Angle(Dec)[0]
        if xyshift is not None:
            for ind in indx:
                radec = np.array([LSM.table['Ra'][ind], LSM.table['Dec'][ind]])
                xy = w.wcs_world2pix(radec, 1)
                xy[0] += xyshift[0]
                xy[1] += xyshift[1]
                RADec = w.wcs_pix2world(xy, 1)
                table['Ra'][ind] = tableio.RA2Angle(RADec[0])[0]
                table['Dec'][ind] = tableio.Dec2Angle(RADec[1])[0]
        LSM.table = table
    elif LSM.hasPatches:
        indx = LSM._getNameIndx(name, patch=True)
        patchNames = LSM.getPatchNames()
        if indx is not None:
            if position is not None:
                if len(indx) > 1:
                    raise ValueError('Only one source can be moved to a new position')
                try:
                    position[0] = tableio.RA2Angle(position[0])[0]
                    position[1] = tableio.Dec2Angle(position[1])[0]
                except Exception as e:
                    raise ValueError('Could not parse position: {0}'.format(e.message))
                table.meta[name] = position
            if shift is not None:
                for ind in indx:
                    pname = patchNames[ind]
                    position = LSM.table.meta[name]
                    table.meta[name] = [position[0] + tableio.RA2Angle(shift[0]),
                        position[1] + tableio.Dec2Angle(shift[1])]
            if xyshift is not None:
                for ind in indx:
                    pname = patchNames[ind]
                    radec = np.array(LSM.table.meta[name])
                    xy = w.wcs_world2pix(radec, 1)
                    xy[0] += xyshift[0]
                    xy[1] += xyshift[1]
                    RADec = w.wcs_pix2world(xy, 1)
                    table.meta[name] = [RADec[0], RADec[1]]
            LSM.table = table
        else:
            raise ValueError("Could not find patch '{0}'.".format(name))
    else:
        raise ValueError("Could not find source '{0}'.".format(name))

    history = "'{0}' by {1} deg".format(name, shift)
    if position is not None:
        history += ', moved to {0}'.format(position)
    if shift is not None:
        history += ', shifted by {1} degrees'.format(shift)
    if xyshift is not None:
        history += ', shifted by {1} pixels'.format(shift)
    LSM._addHistory("Move ({0})".format(history))
