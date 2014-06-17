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

    result = group(LSM, algorithm, targetFlux, numClusters, applyBeam, method)

    # Write to outFile
    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result


def group(LSM, algorithm, targetFlux=None, numClusters=100, applyBeam=False):
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
        patches = _cluster.compute_patch_center(LSM, applyBeam=applyBeam)
        patchCol = _cluster.create_clusters(LSM, patches, numClusters,
            applyBeam=applyBeam)
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
        x, y, midRA, midDec  = LSM._getXY()
        f = LSM.getColValues('I', units=units, applyBeam=applyBeam)
        vobin = _tessellate.bin2D(np.array(x), np.array(y), f,
            target_flux=targetFlux)
        vobin.bin_voronoi()
        patchCol = _tessellate.bins2Patches(vobin)
        LSM.setColValues('Patch', patchCol, index=2)

    elif os.path.exists(algorithm):
        # Mask image
        RARad = LSM.getColValues('Ra', units='radian')
        DecRad = LSM.getColValues('Dec', units='radian')
        patchCol = getPatchNamesFromMask(mask, RARad, DecRad)
        LSM.setColValues('Patch', patchCol, index=2)

    else:
        logging.error('Grouping alogrithm not understood.')
        return 1

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
    import pyrap
    import scipy.ndimage as nd

    try:
        maskdata = pyrap.images.image(mask)
        maskval = maskdata.getdata()[0][0]
    except:
        loggin.error("Error opening mask file '{0}'".format(mask))
        return None

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
            vals.append(0)

    # Check if there is a patch with id = 0. If so, this means there were
    # some Gaussians that fell outside of the regions in the patch
    # mask file.
    n = 0
    for p in patchNums:
        if p != 0:
            in_patch = N.where(patchnums == p)
            patchNames.append('mask_patch_'+str(p))
        else:
            patchNames.append('patch_'+str(n))
            n += 1

    return np.array(patchNames)
