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
    FWHM = parset.getString('.'.join(["LSMTool.Steps", step, "FWHM"]), None )
    applyBeam = parset.getBool('.'.join(["LSMTool.Steps", step, "ApplyBeam"]), False )
    method = parset.getString('.'.join(["LSMTool.Steps", step, "Method"]), 'mid' )
    pad_index = parset.getBool('.'.join(["LSMTool.Steps", step, "PadIndex"]), False )
    byPatch = parset.getBool('.'.join(["LSMTool.Steps", step, "ByPatch"]), False )
    facet = parset.getString('.'.join(["LSMTool.Steps", step, "Facet"]), '' )
    kernelSize = parset.getString('.'.join(["LSMTool.Steps", step, "KernelSize"]), '0.1' )
    nIterations = parset.getInt('.'.join(["LSMTool.Steps", step, "NIterations"]), '100' )
    lookDistance = parset.getString('.'.join(["LSMTool.Steps", step, "LookDistance"]), '0.2' )
    groupingDistance = parset.getString('.'.join(["LSMTool.Steps", step, "GroupingDistance"]), '0.01' )

    try:
        group(LSM, algorithm, targetFlux, numClusters, FWHM, threshold, applyBeam, root,
              pad_index, method, facet, byPatch, float(kernelSize), nIterations,
              float(lookDistance), float(groupingDistance))
        result = 0
    except Exception as e:
        log.error(e.message)
        result = 1

    # Write to outFile
    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result


def group(LSM, algorithm, targetFlux=None, weightBySize=False, numClusters=100, FWHM=None,
          threshold=0.1, applyBeam=False, root='Patch', pad_index=False, method='mid',
          facet="", byPatch=False, kernelSize=0.1, nIterations=100, lookDistance=0.2,
          groupingDistance=0.01):
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
        - 'facet' => group by facets using as an input a fits file. It requires
            the use of the additional parameter 'facet' to enter the name of the
            fits file.
        - 'voronoi' => given a previously grouped sky model, voronoi tesselate
            using the patch positions for patches above the target flux (specified
            by the targetFlux parameter)
        - 'meanshift' => use the meanshift clustering algorithm
        - the filename of a mask image => group by masked regions (where mask =
            True). Sources outside of masked regions are given patches of their
            own
    targetFlux : str or float, optional
        Target flux for tessellation (the total flux of each tile will be close
        to this value) and voronoi algorithms. The target flux can be specified
        as either a float in Jy or as a string with units (e.g., '25.0 mJy')
    weightBySize : bool, optional
        If True, fluxes are weighted by patch size (as median_size / size) when
        the targetFlux criterion is applied. Patches with sizes below the median
        (flux-weighted) size are upweighted and those above the median are
        downweighted
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
        Root string from which patch names are constructed. For 'single', the
        patch name will be set to root; for the other grouping algorithms, the
        patch names will be 'root_INDX', where INDX is an integer ranging from
        (0:nPatches)
    pad_index : bool, optional
        If True, pad the INDX used in the patch names. E.g., facet_patch_001
        instead of facet_patch_1
    method : None or str, optional
        This parameter specifies the method used to set the patch positions:
        - 'mid' => the position is set to the midpoint of the patch
        - 'mean' => the positions is set to the mean RA and Dec of the patch
        - 'wmean' => the position is set to the flux-weighted mean RA and
        Dec of the patch
        - 'zero' => set all positions to [0.0, 0.0]
    facet : str, optional
        Facet fits file used with the algorithm 'facet'
    byPatch : bool, optional
        For the 'tessellate' or 'meanshift' algorithms, use patches instead of sources
    kernelSize : float, optional
        Kernel size in degrees for meanshift grouping
    nIterations : int, optional
        Number of iterations for meanshift grouping
    lookDistance : float, optional
        Look distance in degrees for meanshift grouping
    groupingDistance : float, optional
        Grouping distance in degrees for meanshift grouping

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
    try:
        from . import _meanshiftc as _meanshift
    except ImportError:
        from . import _meanshift
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
                                            applyBeam=applyBeam, root=root,
                                            pad_index=pad_index)
        LSM.setColValues('Patch', patchCol, index=2)

    elif algorithm.lower() == 'tessellate':
        if targetFlux is None:
            raise ValueError('Please specify the targetFlux parameter.')
        else:
            units = 'Jy'
            if type(targetFlux) is str:
                parts = [''.join(g).strip() for _, g in groupby(targetFlux, str.isalpha)]
                targetFlux = float(parts[0])
                if len(parts) == 2:
                    units = parts[1]
        if byPatch:
            if 'Patch' not in LSM.table.keys():
                raise ValueError('Sky model must be grouped before "byPatch" can be used.')
            x, y, midRA, midDec = LSM._getXY(byPatch=True)
            f = LSM.getColValues('I', units=units, applyBeam=applyBeam, aggregate='sum')
        else:
            LSM.ungroup()
            x, y, midRA, midDec = LSM._getXY()
            f = LSM.getColValues('I', units=units, applyBeam=applyBeam)
        vobin = _tessellate.bin2D(np.array(x), np.array(y), f, target_flux=targetFlux)
        try:
            vobin.bin_voronoi()
            patchCol = _tessellate.bins2Patches(vobin, root=root, pad_index=pad_index)
            if byPatch:
                newPatchNames = patchCol.copy()
                origPatchNames = LSM.getPatchNames()
                patchCol = np.zeros(len(LSM), dtype='U100')
                for newPatchName, origPatchName in zip(newPatchNames, origPatchNames):
                    ind = np.array(LSM.getRowIndex(origPatchName))
                    patchCol[ind] = newPatchName
                LSM.setColValues('Patch', patchCol, index=2)
            else:
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
                parts = [''.join(g).strip() for _, g in groupby(FWHM, str.isalpha)]
                FWHM = float(parts[0])
                if len(parts) == 2:
                    units = parts[1]
            fwhmArcsec = Angle(FWHM, unit=units).to('arcsec').value

        patchCol = _threshold.getPatchNamesByThreshold(LSM, fwhmArcsec, threshold,
            root=root, pad_index=pad_index)
        LSM.setColValues('Patch', patchCol, index=2)

    elif algorithm.lower() == 'voronoi':
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        if 'Patch' not in LSM.table.keys():
            raise ValueError('Sky model must be grouped before "voronoi" can be used.')
        else:
            dirs = LSM.getPatchPositions()

        if targetFlux is not None:
            # Select only those patches that lie above the target flux
            units = 'Jy'
            if type(targetFlux) is str:
                parts = [''.join(g).strip() for _, g in groupby(targetFlux, str.isalpha)]
                targetFlux = float(parts[0])
                if len(parts) == 2:
                    units = parts[1]
            dirs_names = []
            names = LSM.getPatchNames()
            fluxes = LSM.getColValues('I', aggregate='sum', units=units, applyBeam=applyBeam)
            if weightBySize:
                sizes = LSM.getPatchSizes(units='arcsec', weight=True, applyBeam=applyBeam)
                sizes[sizes < 1.0] = 1.0
                bright_ind = np.where(fluxes >= targetFlux)
                medianSize = np.median(sizes[bright_ind])
                weights = medianSize / sizes
                weights[weights > 1.0] = 1.0
                weights[weights < 0.5] = 0.5
                fluxes *= weights
            for name, flux in zip(names, fluxes):
                if flux >= targetFlux:
                    dirs_names.append(name)
            if len(dirs_names) == 0:
                log.warn('No patches meet specified targetFlux. All sources placed in a single patch.')
                LSM.ungroup()
                addSingle(LSM, root)
                return 0
        else:
            # Use all patches
            dirs_names = [name for name, d in dirs.items()]

        dirs_ras = []
        dirs_decs = []
        for name in dirs_names:
            d = dirs[name]
            dirs_ras.append(d[0])
            dirs_decs.append(d[1])
        RADeg = LSM.getColValues('Ra', units='degree')
        DecDeg = LSM.getColValues('Dec', units='degree')
        patchNames = []
        for r, d in zip(RADeg, DecDeg):
            dists = SkyCoord(r*u.degree, d*u.degree).separation(
                SkyCoord(dirs_ras*u.degree, dirs_decs*u.degree))
            patchNames.append(dirs_names[np.argmin(dists)])
        LSM.setColValues('Patch', patchNames, index=2)

    elif algorithm.lower() == 'meanshift':
        if byPatch:
            if 'Patch' not in LSM.table.keys():
                raise ValueError('Sky model must be grouped before "byPatch" can be used.')
            x, y, midRA, midDec = LSM._getXY(byPatch=True)
            f = LSM.getColValues('I', applyBeam=applyBeam, aggregate='sum')
        else:
            addEvery(LSM)
            x, y, midRA, midDec = LSM._getXY()
            f = LSM.getColValues('I', applyBeam=applyBeam)
        crdelt = 0.066667  # WCS delta in deg/pixel, as used by LSM._getXY()
        grouper = _meanshift.Grouper(list(zip(x, y)), f, kernelSize/crdelt, nIterations,
                                     lookDistance/crdelt, groupingDistance/crdelt)
        grouper.run()
        clusters = grouper.grouping()
        patchNames = LSM.getPatchNames()
        table = LSM.table.copy()
        for cindx, cluster in enumerate(clusters):
            if pad_index:
                name = '{0}_{1}'.format(root, str(int(cindx)).zfill(int(np.ceil(np.log10(len(cluster)+1)))))
            else:
                name = '{0}_{1}'.format(root, str(int(cindx)))
            patches = patchNames[cluster]
            for patchName in patches:
                indices = LSM.getRowIndex(patchName)
                table['Patch'][indices] = name
        LSM.table = table
        LSM._updateGroups()

    elif algorithm.lower() == 'facet':
        if os.path.exists(facet):
            RADeg = LSM.getColValues('Ra', units='degree')
            DecDeg = LSM.getColValues('Dec', units='degree')
            facet_col = get_facet_values(facet, RADeg, DecDeg, root=root, pad_index=pad_index)
            LSM.setColValues('Patch', facet_col, index=2)
        else:
            raise ValueError('Please enter the facet filename in the facet parameter.')

    elif os.path.exists(algorithm):
        # Mask image
        mask = algorithm
        RARad = LSM.getColValues('Ra', units='radian')
        DecRad = LSM.getColValues('Dec', units='radian')
        patchCol = getPatchNamesFromMask(mask, RARad, DecRad, root=root, pad_index=pad_index)
        LSM.setColValues('Patch', patchCol, index=2)

    else:
        raise ValueError('Grouping alogrithm not understood.')

    # Update table grouping and set default patch positions
    LSM._updateGroups()
    history = "algorithm = '{0}'".format(algorithm)
    if algorithm.lower() == 'cluster':
        history += ', numClusters = {0}'.format(numClusters)
    elif algorithm.lower() == 'tessellate' or algorithm.lower() == 'voronoi':
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
    names = LSM.getColValues('Name').copy()
    for i, name in enumerate(names):
        names[i] = name + '_patch'
    LSM.setColValues('Patch', names, index=2)


def getPatchNamesFromMask(mask, RARad, DecRad, root='mask', pad_index=False):
    """
    Returns an array of patch names for each (RA, Dec) pair in radians
    """
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
            patchNums.append(mask_labels[int(pixY), int(pixX)])
        except:
            patchNums.append(0)

    # Check if there is a patch with id = 0. If so, this means there were
    # some Gaussians that fell outside of the regions in the patch
    # mask file.
    n = 0
    for p in patchNums:
        if p != 0:
            if pad_index:
                patchNames.append('{0}_patch_'.format(root) +
                                  str(p).zfill(int(np.ceil(np.log10(len(set(patchNums))+1)))))
            else:
                patchNames.append('{0}_patch_'.format(root)+str(p))
        else:
            patchNames.append('patch_'+str(n))
            n += 1

    return np.array(patchNames)


def get_facet_values(facet, ra, dec, root="facet", default=0, pad_index=False):
    """
    Extract the value from a fits facet file
    """
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    # TODO: Check astropy version
    # TODO: Check facet is a fits file

    with fits.open(facet) as f:
        shape = f[0].data.shape

        w = WCS(f[0].header)
        if len(w.wcs.crval) == 4:
            freq = w.wcs.crval[2]
            stokes = w.wcs.crval[3]
            xe, ye, _1, _2 = w.all_world2pix(ra, dec, freq, stokes, 1)
        elif len(w.wcs.crval) == 2:
            xe, ye = w.all_world2pix(ra, dec, 1)
        else:
            raise ValueError('Input mask must have 2 axes (x, y) or 4 axes (x, y, freq, stokes).')
        x, y = np.round(xe).astype(int), np.round(ye).astype(int)

        # Dummy value for points out of the fits area
        x[(x < 0) | (x >= shape[-1])] = -1
        y[(y < 0) | (y >= shape[-2])] = -1

        if len(w.wcs.crval) == 4:
            data = f[0].data[0, 0, :, :]
        else:
            data = f[0].data[:, :]

        values = data[y, x]

        # Assign the default value to NaNs and points out of the fits area
        values[(x == -1) | (y == -1)] = default
        values[np.isnan(values)] = default

        if pad_index:
            return np.array(["{0}_{1}".format(root, str(int(val)).zfill(int(np.ceil(np.log10(len(set(values))+1))))) for val in values])
        else:
            return np.array(["{0}_{1}".format(root, int(val)) for val in values])
