# -*- coding: utf-8 -*-
#
# Defines threshold functions used by the group operation
#
# Copyright (C) 2013 - Reinout van Weeren
# Copyright (C) 2013 - Francesco de Gasperin
# Modified by David Rafferty as required for integration into LSMTool
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

def getPatchNamesByThreshold(LSM, fwhmArcsec, threshold=0.1, root=root):
    """
    Projects sky model to image plane, convolves with Gaussian, and finds islands
    of emission
    """
    import numpy as np

    LSM.ungroup()
    x, y, midRA, midDec  = LSM._getXY(crdelt=fwhmArcsec/2.0/3600.0)
    sizeX = int(1.2 * (max(x) - min(x)))
    sizeY = int(1.2 * (max(y) - min(y)))
    image = np.zeros((sizeX, sizeY))
    xint = np.array(x, dtype=int)
    xint += -1 * min(xint) + 1
    yint = np.array(y, dtype=int)
    yint += -1 * min(yint) + 1
    image[xint, yint] = 1.0
    image = blur_image(image, fwhmArcsec/240.0)

    mask = image / threshold >= 1.0
    patchCol = getPatchNamesFromMask(mask, xint, yint, root=root)

    return patchCol


def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    from scipy.ndimage import gaussian_filter

    sx = n
    if ny is not None:
        sy = ny
    else:
        sy = n
    improc = gaussian_filter(im, [sy, sx])

    return improc


def getPatchNamesFromMask(mask, x, y, root='mask'):
    """
    Returns an array of patch names for each (x, y) pair
    """
    import math
    import scipy.ndimage as nd
    import numpy as np

    act_pixels = mask
    rank = len(act_pixels.shape)
    connectivity = nd.generate_binary_structure(rank, rank)
    mask_labels, count = nd.label(act_pixels, connectivity)

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
            in_patch = np.where(patchNums == p)
            patchNames.append('{0}_patch_'.format(root)+str(p))
        else:
            patchNames.append('patch_'+str(n))
            n += 1

    return np.array(patchNames)

