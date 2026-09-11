# -*- coding: utf-8 -*-
#
# Defines threshold functions used by the group operation
#
# Copyright (C) 2013 -  David Rafferty
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

import numpy as np
import scipy.ndimage as nd


def getPatchNamesByThreshold(LSM, fwhmArcsec, threshold=0.1, root='threshold',
    pad_index=False):
    """
    Projects sky model to image plane, convolves with Gaussian, and finds islands
    of emission
    """
    LSM.ungroup()

    # Convolve with Gaussian of FWHM = 4 pixels
    fwhm = 4
    sigma = fwhm/2.35482
    truncate = 4.0
    padding = int(np.ceil(truncate * sigma))

    # Generate image grid with 1 pix = FWHM / 4
    x, y, _, _ = LSM._getXY(crdelt=fwhmArcsec/4.0/3600.0)
    xint = np.array(x, dtype=int)
    yint = np.array(y, dtype=int)
    xint -= min(xint)
    yint -= min(yint)
    size_x = max(xint) + 1 + 2 * padding  # Add padding on both sides.
    size_y = max(yint) + 1 + 2 * padding  # Add padding on both sides.
    image = np.zeros((size_x, size_y))
    xint += padding
    yint += padding

    # Set pixels with sources to one
    image[xint, yint] = 1.0

    # Blur the image with a Gaussian filter
    image = nd.gaussian_filter(image, [sigma, sigma], truncate=truncate)

    mask = image >= threshold
    return getPatchNamesFromMask(mask, xint, yint, root=root, pad_index=pad_index)


def getPatchNamesFromMask(mask, x, y, root='mask', pad_index=False):
    """
    Returns an array of patch names for each (x, y) pair
    """
    act_pixels = mask
    rank = len(act_pixels.shape)
    connectivity = nd.generate_binary_structure(rank, rank)
    mask_labels, _ = nd.label(act_pixels, connectivity)

    patchNums = []
    patchNames = []
    for xs, ys in zip(x, y):
        try:
            patchNums.append(mask_labels[xs, ys])
        except:
            patchNums.append(0)

    # Check if there is a patch with id = 0. If so, this means there were
    # some Gaussians that fell outside of the regions in the patch
    # mask file.
    n = 0
    for p in patchNums:
        if p != 0:
            if pad_index:
                patchNames.append('{0}_patch_'.format(root)+
                    str(p).zfill(int(np.ceil(np.log10(len(set(patchNums))+1)))))
            else:
                patchNames.append('{0}_patch_'.format(root)+str(p))
        else:
            patchNames.append('patch_'+str(n))
            n += 1

    return np.array(patchNames)
