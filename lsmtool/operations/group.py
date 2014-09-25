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
    root = parset.getString('.'.join(["LSMTool.Steps", step, "Root"]), 'Patch' )
    targetFlux = parset.getString('.'.join(["LSMTool.Steps", step, "TargetFlux"]), '1.0 Jy' )
    numClusters = parset.getInt('.'.join(["LSMTool.Steps", step, "NumClusters"]), 100 )
    applyBeam = parset.getBool('.'.join(["LSMTool.Steps", step, "ApplyBeam"]), False )

    try:
        group(LSM, algorithm, targetFlux, numClusters, applyBeam, root)
        result = 0
    except Exception as e:
        logging.error(e.message)
        result = 1

    # Write to outFile
    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result


def group(LSM, algorithm, targetFlux=None, numClusters=100, applyBeam=False,
    root='Patch'):
    """
    Groups sources into patches

    Parameters
    ----------
    LSM : SkyModel object
        Input sky model.
    algorithm : str
        Algorithm to use for grouping:
        - 'single' => all sources are grouped into a single patch
        - 'every' => every source gets a separate patch named 'source_patch'
        - 'cluster' => SAGECAL clustering algorithm that groups sources into
            specified number of clusters (specified by the numClusters parameter).
        - 'tessellate' => group into tiles whose total flux approximates
            the target flux (specified by the targetFlux parameter).
        - the filename of a mask image => group by masked regions (where mask =
            True). Source outside of masked regions are given patches of their
            own.
    targetFlux : str or float, optional
        Target flux for tessellation (the total flux of each tile will be close
        to this value). The target flux can be specified as either a float in Jy
        or as a string with units (e.g., '25.0 mJy').
    numClusters : int, optional
        Number of clusters for clustering. Sources are grouped around the
        numClusters brightest sources.
    applyBeam : bool, optional
        If True, fluxes will be attenuated by the beam.
    root : str, optional
        Root string from which patch names are constructed (when algorithm =
        'single', 'cluster', or 'tesselate'). Patch names will be 'root_INDX',
        where INDX is an integer ranging from (0:nPatches).

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
    import os
    from itertools import groupby

    if algorithm.lower() == 'single':
        LSM.ungroup()
        addSingle(LSM, root+'_0')

    elif algorithm.lower() == 'every':
        LSM.ungroup()
        addEvery(LSM)

    elif algorithm.lower() == 'cluster':
        LSM.ungroup()
        patches = _cluster.compute_patch_center(LSM, applyBeam=applyBeam)
        patchCol = _cluster.create_clusters(LSM, patches, numClusters,
            applyBeam=applyBeam, root=root)
        LSM.setColValues('Patch', patchCol, index=2)

    elif algorithm.lower() == 'tessellate':
        if targetFlux is None:
            raise ValueError('Please specify the targetFlux parameter.')
        else:
            units = 'Jy'
            if type(targetFlux) is str:
                parts = [''.join(g).strip() for _, g in groupby(targetFlux,
                    str.isalpha)]
                targetFlux = float(parts[0])
                if len(parts) == 2:
                    units = parts[1]
        LSM.ungroup()
        x, y, midRA, midDec  = LSM._getXY()
        f = LSM.getColValues('I', units=units, applyBeam=applyBeam)
        vobin = _tessellate.bin2D(np.array(x), np.array(y), f,
            target_flux=targetFlux)
        vobin.bin_voronoi()
        patchCol = _tessellate.bins2Patches(vobin, root=root)
        LSM.setColValues('Patch', patchCol, index=2)

    elif os.path.exists(algorithm):
        # Mask image
        mask = algorithm
        RARad = LSM.getColValues('Ra', units='radian')
        DecRad = LSM.getColValues('Dec', units='radian')
        patchCol = getPatchNamesFromMask(mask, RARad, DecRad)
        LSM.setColValues('Patch', patchCol, index=2)

    else:
        raise ValueError('Grouping alogrithm not understood.')

    # Update table grouping
    LSM._updateGroups()
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
    for i, name in enumerate(names):
        names[i] = name + '_patch'
    LSM.setColValues('Patch', names, index=2)


def getPatchNamesFromMask(mask, RARad, DecRad):
    """
    Returns an array of patch names for each (RA, Dec) pair in radians
    """
    import math
    import pyrap.images as pim
    import scipy.ndimage as nd
    import numpy as np

    maskdata = pim.image(mask)
    maskval = maskdata.getdata()[0][0]

    act_pixels = maskval
    rank = len(act_pixels.shape)
    connectivity = nd.generate_binary_structure(rank, rank)
    mask_labels, count = nd.label(act_pixels, connectivity)

    patchNums = []
    patchNames = []
    for raRad, decRad in zip(RARad, DecRad):
        (a, b, _, _) = maskdata.toworld([0, 0, 0, 0])
        (_, _, pixY, pixX) = maskdata.topixel([a, b, decRad, raRad])
        try:
            # != is a XOR for booleans
            patchNums.append(mask_labels[pixY, pixX])
        except:
            patchNums.append(0)

    # Check if there is a patch with id = 0. If so, this means there were
    # some Gaussians that fell outside of the regions in the patch
    # mask file.
    n = 0
    for p in patchNums:
        if p != 0:
            in_patch = np.where(patchNums == p)
            patchNames.append('mask_patch_'+str(p))
        else:
            patchNames.append('patch_'+str(n))
            n += 1

    return np.array(patchNames)
