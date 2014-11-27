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

log = logging.getLogger('LSMTool.TRANSFER')
log.debug('Loading TRANSFER module.')


def run(step, parset, LSM):

    patchFile = parset.getString('.'.join(["LSMTool.Steps", step, "PatchFile"]), '' )
    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )

    try:
        transfer(LSM, patchFile)
        result = 0
    except Exception as e:
        log.error(e.message)
        result = 1

    # Write to outFile
    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result


def transfer(LSM, patchSkyModel, matchBy='name', radius=0.1):
    """
    Transfer patches from the input sky model.

    Sources matching those in patchSkyModel will be grouped into
    the patches defined in patchSkyModel. Sources that do not appear in
    patchSkyModel will be placed into separate patches (one per source).
    Patch positions are not transferred (as they may no longer be appropriate
    after transfer).

    Parameters
    ----------
    LSM : SkyModel object
        Input sky model
    patchSkyModel : str or SkyModel object
        Input sky model from which to transfer patches
    matchBy : str, optional
        Determines how matching sources are determined:
        - 'name' => matches are identified by name
        - 'position' => matches are identified by radius. Sources within the
            radius specified by the radius parameter are considered matches
    radius : float or str, optional
        Radius in degrees (if float) or 'value unit' (if str; e.g., '30 arcsec')
        for matching when matchBy='position'

    Examples
    --------
    Transfer patches from one sky model to another by matching to the source
    names and set their positions::

        >>> LSM = lsmtool.load('sky.model')
        >>> transfer(LSM, 'master_sky.model')
        >>> setPatchPositions(LSM, method='mid')

    Transfer patches by matching sources that lie within 10 arcsec of one
    another::

        >>> LSM = lsmtool.load('sky.model')
        >>> transfer(LSM, 'master_sky.model', matchBy='position', radius='10.0 arcsec')
        >>> setPatchPositions(LSM, method='mid')

    """
    from ..skymodel import SkyModel
    from ..operations_lib import matchSky

    if len(LSM) == 0:
        log.error('Sky model is empty.')
        return

    if type(patchSkyModel) is str:
        masterLSM = SkyModel(patchSkyModel)
    else:
        masterLSM = patchSkyModel

    # Group LSM by source. This ensures that any sources not in the master
    # sky model are given a patch of their own
    log.debug('Grouping master sky model by one patch per source...')
    LSM.group('every')
    patchNames = LSM.getColValues('Patch')
    masterPatchNames = masterLSM.getColValues('Patch')
    table = LSM.table.copy()

    if matchBy.lower() == 'name':
        log.debug('Transferring patches by matching names...')
        names = LSM.getColValues('Name')
        masterNames = masterLSM.getColValues('Name')

        lrIntr = lambda l, r: list(set(l).intersection(r))
        commonNames = lrIntr(names.tolist(), masterNames.tolist())
        nMissing = len(names) - len(commonNames)

        for name in commonNames:
            indx = LSM.getRowIndex(name)
            masterIndx = masterLSM.getRowIndex(name)
            table['Patch'][indx] = masterLSM.table['Patch'][masterIndx]

    elif matchBy.lower() == 'position':
        log.debug('Transferring patches by matching positions...')
        matches1, matches2 = matchSky(LSM, masterLSM, radius=radius)
        nMissing = len(LSM) - len(matches1)

        # Set patch names to be the same for the matches
        table['Patch'][matches1] = masterLSM.table['Patch'][matches2]

    log.debug('Number of sources not present in patchSkyModel: {0}'.format(
        nMissing))
    LSM.table = table
    LSM._updateGroups()
    LSM._addHistory('TRANSFER')
    LSM._info()
