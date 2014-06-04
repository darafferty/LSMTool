#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This operation implements plotting of sources
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

logging.debug('Loading PLOT module.')


def run( step, parset, LSM ):

    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )

    if outFile == '':
        outFile = None
    plot(LSM, outFile)

    return 0


def plot(LSM, fileName=None):
    """
    Shows a simple plot of the sky model.

    The circles in the plot are scaled with flux. If the sky model is grouped
    into patches, sources are colored by patch and the patch positions are
    indicated with stars.

    Parameters
    ----------
    fileName : str, optional
        If given, the plot is saved to a file instead of displayed.

    Examples:
    ---------
    Plot and display to the screen::

        >>> s.plot()

    Plot and save to a PDF file::

        >>>s.plot('sky_plot.pdf')

    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(1,figsize=(7,7))
    plt.clf()
    ax = plt.gca()
    if LSM._hasPatches:
        nsrc = len(LSM.getColValues('Patch', aggregate=True))
    else:
        nsrc = len(LSM)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Set3, norm=plt.Normalize(vmin=0,
        vmax=nsrc))
    sm._A = []

    # Set symbol sizes by flux, making sure no symbol is smaller than 50 or
    # larger than 1000
    s = []
    for flux in LSM.getColValues('I'):
        s.append(min(1000.0, max(flux*10.0, 50.0)))

    # Plot sources, colored by patch if possible
    c = []
    cp = []
    if LSM._hasPatches:
        for p, patchName in enumerate(LSM.getColValues('Patch', aggregate=True)):
            indices = LSM.getRowIndex(patchName)
            cp.append(sm.to_rgba(p))
            for ind in indices:
                c.append(sm.to_rgba(p))
    else:
        c = [sm.to_rgba(0)] * nsrc

    # Plot sources
    RA = LSM.getColValues('RA')
    Dec = LSM.getColValues('Dec')
    x, y, minx, miny, scale = radec2xy(RA, Dec)
    scaleRA = (np.max(RA) - np.min(RA))/(np.max(x) - np.min(x))
    scaleDec = (np.max(Dec) - np.min(Dec))/(np.max(y) - np.min(y))
    x = (x - np.min(x)) * scaleRA + np.min(RA)
    y = (y - np.min(y)) * scaleDec + min(Dec)

    plt.scatter(x, y, s=s, c=c)
    if LSM._hasPatches:
        posDict = LSM.getPatchPositions()
        RAp = []
        Decp = []
        for patchName in LSM.getColValues('Patch', aggregate=True):
            RAp.append(posDict[patchName][0])
            Decp.append(posDict[patchName][1])
        xp, yp, minx, miny, pscale = radec2xy(RAp, Decp, minx, miny)
        xp = (xp - np.min(xp)) * pscale / 3600.0 + min(RA)
        yp = (yp - np.min(yp)) * pscale / 3600.0 + min(Dec)
        plt.scatter(xp, yp, s=100, c=cp, marker='*')
    ax.set_xlim(ax.get_xlim()[::-1])
#     plt.axis('image')
    plt.xlabel("RA (degrees)")
    plt.ylabel("Dec (degrees)")

    if fileName is not None:
        plt.savefig(fileName)
    else:
        plt.show()
    plt.close(fig)


def radec2xy(RA, Dec, minx=None, miny=None):
    """Returns x, y for input ra, dec
    """
    from astropy.wcs import WCS
    import numpy as np

    y = []
    x = []

    # Make wcs object to handle transformation from ra and dec to pixel coords.
    w = WCS(naxis=2)
    w.wcs.crpix = [-234.75, 8.3393]
    w.wcs.cdelt = np.array([0.066667, 0.066667])
    w.wcs.crval = [0, -90]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.set_pv([(2, 1, 45.0)])
    arcsec_per_pix = abs(w.wcs.cdelt[0]) * 3600.0 # arcsec/pixel

    for ra_deg, dec_deg in zip(RA, Dec):
        ra_dec = np.array([[ra_deg, dec_deg]])
        x.append(w.wcs_world2pix(ra_dec, 1)[0][0])
        y.append(w.wcs_world2pix(ra_dec, 1)[0][1])

    if minx is None:
        minx = np.min(x)
    if miny is None:
        miny = np.min(y)

    x += abs(minx)
    y += abs(miny)
    return y, x, minx, miny, arcsec_per_pix
