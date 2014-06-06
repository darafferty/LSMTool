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
    from matplotlib.ticker import FuncFormatter
    import numpy as np
    from lsmtool.operations_lib import radec2xy, xy2radec
    global maxRA, minDec, ymin, xmin

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
    maxRA = np.max(RA)
    minDec = np.min(Dec)
    x, y  = radec2xy(RA, Dec)
    plt.scatter(x, y, s=s, c=c)

    if LSM._hasPatches:
        posDict = LSM.getPatchPositions()
        RAp = []
        Decp = []
        for patchName in LSM.getColValues('Patch', aggregate=True):
            RAp.append(posDict[patchName][0])
            Decp.append(posDict[patchName][1])
        xp, yp = radec2xy(RAp, Decp, maxRA, minDec)
        plt.scatter(xp, yp, s=100, c=cp, marker='*')

    # Define tick formatter to translate axis labels from x, y to ra, dec
    RAformatter = FuncFormatter(RAtickformatter)
    Decformatter = FuncFormatter(Dectickformatter)
    xmin = min(ax.get_xlim())
    ymin = min(ax.get_ylim())
    ax.xaxis.set_major_formatter(RAformatter)
    ax.yaxis.set_major_formatter(Decformatter)
    plt.xlabel("RA (degrees)")
    plt.ylabel("Dec (degrees)")

    if fileName is not None:
        plt.savefig(fileName)
    else:
        plt.show()
    plt.close(fig)

def RAtickformatter(x, pos):
    from lsmtool.operations_lib import xy2radec

    global ymin, maxRA, minDec
    ratick = xy2radec([x], [ymin], maxRA, minDec)[0][0]
    rastr = '{0:.2f}'.format(ratick)
    return rastr


def Dectickformatter(y, pos):
    from lsmtool.operations_lib import xy2radec

    global xmin, maxRA, minDec
    dectick = xy2radec([xmin], [y], maxRA, minDec)[1][0]
    decstr = '{0:.2f}'.format(dectick)
    return decstr

