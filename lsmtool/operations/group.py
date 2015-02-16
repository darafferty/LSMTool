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

log = logging.getLogger('LSMTool.GROUP')
log.debug('Loading GROUP module.')


def run(step, parset, LSM):

    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )
    algorithm = parset.getString('.'.join(["LSMTool.Steps", step, "Algorithm"]), 'single' )
    root = parset.getString('.'.join(["LSMTool.Steps", step, "Root"]), 'Patch' )
    targetFlux = parset.getString('.'.join(["LSMTool.Steps", step, "TargetFlux"]), '1.0 Jy' )
    numClusters = parset.getInt('.'.join(["LSMTool.Steps", step, "NumClusters"]), 100 )
    threshold = parset.getString('.'.join(["LSMTool.Steps", step, "Threshold"]), '1.0 Jy' )
    FWHM = parset.getString('.'.join(["LSMTool.Steps", step, "FWHM"]), '1.0 Jy' )
    applyBeam = parset.getBool('.'.join(["LSMTool.Steps", step, "ApplyBeam"]), False )
    method = parset.getString('.'.join(["LSMTool.Steps", step, "Method"]), 'mid' )

    try:
        group(LSM, algorithm, targetFlux, numClusters, applyBeam, root, method)
        result = 0
    except Exception as e:
        log.error(e.message)
        result = 1

    # Write to outFile
    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result


def group(LSM, algorithm, targetFlux=None, numClusters=100, FWHM=None,
    threshold=0.1, applyBeam=False, root='Patch', method='mid'):
    """
    Groups sources into patches.

    Parameters
    ----------
    LSM : SkyModel object
        Input sky model
    algorithm : str
        Algorithm to use for grouping:
        - 'single' => all sources are grouped into a single patch
        - 'every' => every source gets a separate patch named 'source_patch'
        - 'cluster' => SAGECAL clustering algorithm that groups sources into
            specified number of clusters (specified by the numClusters parameter)
        - 'tessellate' => group into tiles whose total flux approximates
            the target flux (specified by the targetFlux parameter)
        - 'threshold' => group by convolving the sky model with a Gaussian beam
            and then thresholding to find islands of emission (NOTE: all sources
            are currently considered to be point sources of flux unity)
        - the filename of a mask image => group by masked regions (where mask =
            True). Source outside of masked regions are given patches of their
            own
    targetFlux : str or float, optional
        Target flux for tessellation (the total flux of each tile will be close
        to this value). The target flux can be specified as either a float in Jy
        or as a string with units (e.g., '25.0 mJy')
    numClusters : int, optional
        Number of clusters for clustering. Sources are grouped around the
        numClusters brightest sources
    FWHM : str or float, optional
        FWHM of convolving Gaussian used for thresholding. The FWHM can
        be specified as either a float in degrees or as a string with units
        (e.g., '25.0 arcsec')
    threshold : float, optional
        Value between 0 and 1 above which emission is considered for thresholding
    applyBeam : bool, optional
        If True, fluxes will be attenuated by the beam
    root : str, optional
        Root string from which patch names are constructed (when algorithm =
        'single', 'cluster', or 'tesselate'). For 'single', the patch name
        will be set to root; for the other grouping algorithms, the patch
        names will be 'root_INDX', where INDX is an integer ranging from
        (0:nPatches).
    method : None or str, optional
        This parameter specifies the method used to set the patch positions:
        - 'mid' => the position is set to the midpoint of the patch
        - 'mean' => the positions is set to the mean RA and Dec of the patch
        - 'wmean' => the position is set to the flux-weighted mean RA and
        Dec of the patch
        - 'zero' => set all positions to [0.0, 0.0]

    Examples
    --------
    Tesselate the sky model into patches with approximately 30 Jy total
    flux:

        >>> LSM = lsmtool.load('sky.model')
        >>> group(LSM, 'tessellate', targetFlux=30.0)

    """
    from . import _tessellate
    from . import _cluster
    from . import _threshold
    import numpy as np
    import os
    from itertools import groupby

    if len(LSM) == 0:
        log.error('Sky model is empty.')
        return

    if algorithm.lower() == 'single':
        LSM.ungroup()
        addSingle(LSM, root)

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
        try:
            vobin.bin_voronoi()
            patchCol = _tessellate.bins2Patches(vobin, root=root)
            LSM.setColValues('Patch', patchCol, index=2)
        except ValueError:
            # Catch error in some cases with high target flux relative to
            # total model flux
            addSingle(LSM, root+'_0')

    elif algorithm.lower() == 'threshold':
        from astropy.coordinates import SkyCoord, Angle

        if threshold is None:
            threshold = 0.1
        if threshold < 0.01:
            threshold = 0.01
        if threshold > 1.0:
            threshold = 1.0
        if FWHM is None:
            raise ValueError('Please specify the FWHM parameter.')
        else:
            units = 'degree'
            if type(FWHM) is str:
                parts = [''.join(g).strip() for _, g in groupby(FWHM,
                    str.isalpha)]
                FWHM = float(parts[0])
                if len(parts) == 2:
                    units = parts[1]
            fwhmArcsec = Angle(FWHM, unit=units).to('arcsec').value

        patchCol = _threshold.getPatchNamesByThreshold(LSM, fwhmArcsec, threshold)
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

    # Update table grouping and set default patch positions
    LSM._updateGroups()
    history = "algorithm = '{0}'".format(algorithm)
    if algorithm.lower() == 'cluster':
        history += ', numClusters = {0}'.format(numClusters)
    elif algorithm.lower() == 'tessellate':
        history += ', targetFlux = {0}'.format(targetFlux)
    LSM._addHistory("GROUP ({0})".format(history))
    LSM.setPatchPositions(method=method, applyBeam=applyBeam)
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
