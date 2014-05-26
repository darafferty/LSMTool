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
    alogrithm = parset.getString('.'.join(["LSMTool.Steps", step, "Alogrithm"]), '' )
    targetFlux = parset.getString('.'.join(["LSMTool.Steps", step, "MinFlux"]), '' )
    numClusters = parset.getString('.'.join(["LSMTool.Steps", step, "NumClusters"]), '' )
    radius = parset.getString('.'.join(["LSMTool.Steps", step, "Radius"]), '' )
    beamMS = parset.getString('.'.join(["LSMTool.Steps", step, "BeamMS"]), '' )
    method = parset.getString('.'.join(["LSMTool.Steps", step, "Method"]), '' )

    result = group(LSM, algorithm, targetFlux, beamMS, numClusters, method)

    # Write to outFile
    if outFile == '' or outFile is None:
        outFile = LSM._fileName
    LSM.writeFile(outFile, clobber=True)

    return result


def group(LSM, algorithm, targetFlux=None, beamMS=None, numClusters=100,
    method='mid'):
    """
    Groups sources into patches

    Parameters
    ----------
    algorithm : str
        Algorithm to use for grouping:
        - 'single'
        - 'every'
        - 'cluster'
        - 'tessellate'
    """
    import _tessellate
    import _cluster

    if algorithm.lower() == 'single':
        LSM.ungroup()
        addSingle(LSM, 'Patch_0')

    elif algorithm.lower() == 'every':
        LSM.ungroup()
        addEvery(LSM)

    elif algorithm.lower() == 'cluster':
        LSM.ungroup()
        patches = _cluster.compute_patch_center(LSM.table, beamMS)
        patchCol = _cluster.create_clusters(patches, numClusters)
        LSM.setColValues('Patch', patchCol, index=2)

    elif algorithm.lower() == 'tessellate':
        if targetFlux is None:
            logging.error('Please specify the targetFlux parameter in Jy.')
            return 1
        else:
            units = 'Jy'
            if type(targetFlux) is str:
                parts = targetFlux.split(' ')
                targetFlux = float(parts[0])
                if len(parts) == 2:
                    units = parts[1].strip()
        LSM.ungroup()
        RA = LSM.getColValues('RA')
        Dec = LSM.getColValues('Dec')
        x, y = _tessellate.radec2xy(RA, Dec)
        f = LSM.getColValues('I', units=units)
        if beamMS is not None:
            from ..operations_lib import applyBeam
            f = applyBeam(beamMS, f, RA, Dec)
        vobin = _tessellate.bin2D(x, y, f, target_flux=targetFlux)
        vobin.bin_voronoi()
        patchCol = _tessellate.bins2Patches(vobin)
        LSM.setColValues('Patch', patchCol, index=2)

    else:
        logging.error('Grouping alogrithm not understood.')
        return 1

    # Calculate/update patch positions
    LSM._updateGroups(method=method)
    return 0


def addSingle(LSM, patchName):
    """Add a Patch column with a single patch name"""
    import numpy as np

    uniformCol = np.array([patchName]*len(LSM.table))
    LSM.setColValues('Patch', uniformCol, index=2)


def addEvery(LSM):
    """Add a Patch column with a different name for each source"""
    import numpy as np

    names = LSM.table['Name'].filled().copy().data
    LSM.setColValues('Patch', names, index=2)
