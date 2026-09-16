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
    x_indices, y_indices = gridCoordinates(LSM, fwhmArcsec, padding)
    size_x = computeImageSize(x_indices, padding)
    size_y = computeImageSize(y_indices, padding)
    image = np.zeros((size_x, size_y))

    # Set pixels with sources to one
    image[x_indices, y_indices] = 1.0

    # Blur the image with a Gaussian filter
    image = nd.gaussian_filter(image, [sigma, sigma], truncate=truncate)

    mask = image >= threshold
    return getPatchNamesFromMask(mask, x_indices, y_indices, root=root, pad_index=pad_index)


def gridCoordinates(LSM, fwhmArcsec, padding):
    """Generate image grid coordinates with 1 pix = FWHM / 4"""
    x, y, _, _ = LSM._getXY(crdelt=fwhmArcsec/4.0/3600.0)
    # Convert to integer coordinates.
    x_indices = np.array(x, dtype=int)
    y_indices = np.array(y, dtype=int)
    # Shift coordinates so they start from zero.
    x_indices -= min(x_indices)
    y_indices -= min(y_indices)
    # Apply padding to the coordinates
    x_indices += padding
    y_indices += padding
    return x_indices, y_indices


def computeImageSize(indices, padding):
    """
    Computes the required size of the image using indices and padding.

    Parameters
    ----------
    indices : list of int
        Array of indices (either x or y) for which to compute the image size.
        The indices should already include padding for one side.
    padding : int
        The amount of padding to add.

    Returns
    -------
    int
        The required size of the image including padding.
    """
    # - Add 1 to the maximum index, since indices are zero-based.
    # - Add padding once, since the indices already include padding on one side.
    return max(indices) + 1 + padding


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
