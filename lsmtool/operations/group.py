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
    algorithm = parset.getString('.'.join(["LSMTool.Steps", step, "Algorithm"]), 'single' )
    targetFlux = parset.getString('.'.join(["LSMTool.Steps", step, "TargetFlux"]), '1.0 Jy' )
    numClusters = parset.getInt('.'.join(["LSMTool.Steps", step, "NumClusters"]), 10 )
    applyBeam = parset.getBool('.'.join(["LSMTool.Steps", step, "applyBeam"]), False )
    method = parset.getString('.'.join(["LSMTool.Steps", step, "Method"]), 'mid' )

    result = group(LSM, algorithm, targetFlux, numClusters, applyBeam, method)

    # Write to outFile
    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result


def group(LSM, algorithm, targetFlux=None, numClusters=100, applyBeam=False,
    method='mid'):
    """
    Groups sources into patches

    Parameters
    ----------
    LSM : SkyModel object
        Input sky model.
    algorithm : str
        Algorithm to use for grouping:
        - 'single' => all sources are grouped into a single patch
        - 'every' => every source gets a separate patch
        - 'cluster' => SAGECAL clustering algorithm that groups sources into
            specified number of clusters (specified by the numClusters parameter).
        - 'tessellate' => group into tiles whose total flux approximates
            the target flux (specified by the targetFlux parameter).
    targetFlux : str or float, optional
        Target flux for tessellation (the total flux of each tile will be close
        to this value). The target flux can be specified as either a float in Jy
        or as a string with units (e.g., '25.0 mJy').
    numClusters : int, optional
        Number of clusters for clustering. Sources are grouped around the
        numClusters brightest sources.
    applyBeam : bool, optional
        If True, fluxes will be attenuated by the beam.
    method : str, optional
        Method by which patch positions will be calculated:
        - 'mid' => use the midpoint of the patch
        - 'mean' => use the mean position
        - 'wmean' => use the flux-weighted mean position

    Examples
    --------
    Tesselate the sky model into patches with approximately 30 Jy total
    flux:

        >>> LSM = lsmtool.load('sky.model')
        >>> group(LSM, 'tessellate', targetFlux=30.0)

    """
    from . import _tessellate
    from . import _cluster
    import numpy as np

    if algorithm.lower() == 'single':
        LSM.ungroup()
        addSingle(LSM, 'Patch_0')

    elif algorithm.lower() == 'every':
        LSM.ungroup()
        addEvery(LSM)

    elif algorithm.lower() == 'cluster':
        LSM.ungroup()
        patches = _cluster.compute_patch_center(LSM.table, applyBeam=applyBeam)
        patchCol = _cluster.create_clusters(patches, numClusters)
        LSM.setColValues('Patch', patchCol, index=2)

    elif algorithm.lower() == 'tessellate':
        try:
            from ..operations_lib import radec2xy
        except:
            from .operations_lib import radec2xy
        if targetFlux is None:
            logging.error('Please specify the targetFlux parameter.')
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
        x, y = radec2xy(RA, Dec)
        f = LSM.getColValues('I', units=units, applyBeam=applyBeam)
        vobin = _tessellate.bin2D(np.array(x), np.array(y), f,
            target_flux=targetFlux)
        vobin.bin_voronoi()
        patchCol = _tessellate.bins2Patches(vobin)
        LSM.setColValues('Patch', patchCol, index=2)

    else:
        logging.error('Grouping alogrithm not understood.')
        return 1

    # Calculate/update patch positions
    LSM._updateGroups(method=method)
    LSM._info()
    return 0


def addSingle(LSM, patchName):
    """Add a Patch column with a single patch name"""
    import numpy as np

    uniformCol = np.array([patchName]*len(LSM.table))
    LSM.setColValues('Patch', uniformCol, index=2)


def addEvery(LSM):
    """Add a Patch column with a different name for each source"""
    import numpy as np

    names = LSM.getColValues('Name').copy()
    LSM.setColValues('Patch', names, index=2)
