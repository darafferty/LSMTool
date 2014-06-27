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

        >>> LSM = lsmtool.load('sky.model')
        >>> plot(LSM)

    Plot and save to a PDF file::

        >>> plot(LSM, 'sky_plot.pdf')

    """
    try:
        import matplotlib.pyplot as plt
    except:
        print('PyPlot could not be imported. Plotting is not available.')
        return
    from matplotlib.ticker import FuncFormatter
    import numpy as np
    try:
        from ..operations_lib import radec2xy, xy2radec
    except:
        from .operations_lib import radec2xy, xy2radec
    global midRA, midDec, ymin, xmin

    fig = plt.figure(1,figsize=(7,7))
    plt.clf()
    ax = plt.gca()
    if LSM.hasPatches:
        nsrc = len(LSM.getPatchNames())
    else:
        nsrc = len(LSM)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Set3, norm=plt.Normalize(vmin=0,
        vmax=nsrc))
    sm._A = []

    # Set symbol sizes by flux, making sure no symbol is smaller than 50 or
    # larger than 1000
    s = []
    minflux = np.min(LSM.getColValues('I'))
    for flux in LSM.getColValues('I'):
        s.append(min(1000.0, (1.0+2.0*np.log10(flux/minflux))*50.0))

    # Plot sources, colored by patch if grouped
    c = [0]*len(LSM)
    cp = []
    if LSM.hasPatches:
        for p, patchName in enumerate(LSM.getPatchNames()):
            indices = LSM.getRowIndex(patchName)
            cp.append(sm.to_rgba(p))
            for ind in indices:
                c[ind] = sm.to_rgba(p)
    else:
        c = [sm.to_rgba(0)] * nsrc

    # Plot sources
    x, y, midRA, midDec  = LSM._getXY()
    plt.scatter(x, y, s=s, c=c)

    if LSM.hasPatches:
        RAp, Decp = LSM.getPatchPositions(asArray=True)
        goodInd = np.where( (RAp != 0.0) & (Decp != 0.0) )
        xp, yp = radec2xy(RAp[goodInd], Decp[goodInd], midRA, midDec)
        plt.scatter(xp, yp, s=100, c=cp, marker='*')

    # Define coodinate formater to show RA and Dec under mouse pointer
    RAformatter = FuncFormatter(RAtickformatter)
    ax.format_coord = formatCoord
    plt.xlabel("RA (arb. units)")
    plt.ylabel("Dec (arb. units)")

    if fileName is not None:
        plt.savefig(fileName)
    else:
        plt.show()
    plt.close(fig)

def formatCoord(x, y):
    """Custom coordinate format"""
    try:
        from ..operations_lib import xy2radec
    except:
        from .operations_lib import xy2radec
    global midRA, midDec
    RA, Dec = xy2radec([x], [y], midRA, midDec)
    return '{0:.2f} {1:.2f}'.format(RA[0], Dec[0])


def RAtickformatter(x, pos):
    """Changes x tick labels from pixels to RA in degrees"""
    try:
        from ..operations_lib import xy2radec
    except:
        from .operations_lib import xy2radec
    global ymin, midRA, midDec
    ratick = xy2radec([x], [ymin], midRA, midDec)[0][0]
    rastr = '{0:.2f}'.format(ratick)
    return rastr


def Dectickformatter(y, pos):
    """Changes y tick labels from pixels to Dec in degrees"""
    try:
        from ..operations_lib import xy2radec
    except:
        from .operations_lib import xy2radec

    global xmin, midRA, midDec
    dectick = xy2radec([xmin], [y], midRA, midDec)[1][0]
    decstr = '{0:.2f}'.format(dectick)
    return decstr

