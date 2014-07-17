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
    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result


def transfer(LSM, patchSkyModel):
    """
    Transfer patches from the input sky model.

    Sources with the same name as those in patchSkyModel will be grouped into
    the patches defined in patchSkyModel. Sources that do not appear in patchSkyModel
    will be placed into separate patches (one per source). Patch positions are
    not transferred.

    Parameters
    ----------
    patchSkyModel : str or SkyModel object
        Input sky model from which to transfer patches.

    Examples
    --------
    Transfer patches from one sky model to another and set their positions::

        >>> LSM = lsmtool.load('sky.model')
        >>> transfer(LSM, 'master_sky.model')
        >>> setPatchPositions(LSM, method='mid')

    """
    try:
        from ..skymodel import SkyModel
    except:
        from .skymodel import SkyModel

    if type(patchSkyModel) is str:
        masterLSM = SkyModel(patchSkyModel)
    else:
        masterLSM = patchSkyModel
    masterNames = masterLSM.getColValues('Name').tolist()
    masterPatchNames = masterLSM.getColValues('Patch').tolist()

    # Group LSM by source. This ensures that any sources not in the master
    # sky model are given a patch of their own
    logging.debug('Grouping master sky model by one patch per source.')
    LSM.group('every')

    logging.debug('Transferring patches.')
    names = LSM.getColValues('Name')
    patchNames = LSM.getColValues('Patch').tolist()

    toIndx = [i for i in range(len(LSM)) if names[i] in masterNames]
    masterIndx = [masterNames.index(name) for name in names[toIndx]]
    for i, indx in enumerate(toIndx):
        patchNames[indx] = masterPatchNames[masterIndx[i]]

    LSM.setColValues('Patch', patchNames)
    LSM._updateGroups()
    LSM._info()
    return 0
