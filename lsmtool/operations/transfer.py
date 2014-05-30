#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This operation implements transferring of patches from one sky model to another
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

logging.debug('Loading TRANSFER module.')


def run(step, parset, LSM):

    patchFile = parset.getString('.'.join(["LSMTool.Steps", step, "PatchFile"]), '' )
    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )

    result = transfer(LSM, patchFile)

    # Write to outFile
    if outFile != '':
        LSM.write(outFile, clobber=True)

    return result


def transfer(LSM, patchFile, method='mid'):
    """
    Transfer patches from the input sky model.

    Sources with the same name as those in patchFile will be grouped into
    the patches defined in patchFile. Sources that do not appear in patchFile
    will be placed into separate patches (one per source).

    Parameters
    ----------
    patchFile : str
        Input sky model from which to transfer patches.
    method : str, optional
        Method to use in setting patch positons: 'mid', 'mean', or 'wmean'

    """
    import skymodel

    masterLSM = skymodel.SkyModel(patchFile)
    masterNames = masterLSM.getColValues('Name')
    masterPatchNames = masterLSM.getColValues('Patch')

    # Group LSM by source. This ensures that any sources not in the master
    # sky model are given a patch of their own
    LSM.group('every')
    names = LSM.getColValues('Name')
    patchNames = LSM.getColValues('Patch')

    for i, name in enumerate(names):
        indx = LSM._getNameIndx(name)[0]
        if indx is not None:
            patchNames[i] = masterPatchNames[indx]

    LSM.setColValues('Patch', patchNames)
    LSM._updateGroups(method=method)
    LSM.info()
    return 0
