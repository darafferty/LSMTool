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

log = logging.getLogger('LSMTool.PLOT')
log.debug('Loading PLOT module.')


def run(step, parset, LSM):

    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '')
    labelBy = parset.getString('.'.join(["LSMTool.Steps", step, "LabelBy"]), '')

    if outFile == '':
        outFile = None
    if labelBy == '':
        labelBy = None

    try:
        plot(LSM, fileName=outFile, labelBy=labelBy)
        result = 0
    except Exception as e:
        log.error(e.message)
        result = 1

    return 0


def plot(LSM, fileName=None, labelBy=None):
    """
    Shows a simple plot of the sky model.

    The circles in the plot are scaled with flux. If the sky model is grouped
    into patches, sources are colored by patch and the patch positions are
    indicated with stars.

    Parameters
    ----------
    LSM : SkyModel object
        Input sky model
    fileName : str, optional
        If given, the plot is saved to a file instead of displayed
    labelBy : str, optional
        One of 'source' or 'patch': label points using source names ('source') or
        patch names ('patch')

    Examples:
    ---------
    Plot and display to the screen::

        >>> LSM = lsmtool.load('sky.model')
        >>> plot(LSM)

    Plot and save to a PDF file::

        >>> plot(LSM, 'sky_plot.pdf')

    """
    try:
        import os
        if 'DISPLAY' not in os.environ:
            import matplotlib
            if matplotlib.get_backend() is not 'Agg':
                matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception as e:
        raise ImportError('PyPlot could not be imported. Plotting is not '
            'available: {0}'.format(e.message))
    try:
        try:
            from astropy.visualization.wcsaxes import WCSAxes
            hasWCSaxes = True
        except:
            from wcsaxes import WCSAxes
            hasWCSaxes = True
    except:
        hasWCSaxes = False
    import numpy as np
    from ..operations_lib import radec2xy, makeWCS
    global midRA, midDec, ymin, xmin

    if len(LSM) == 0:
        log.error('Sky model is empty.')
        return

    fig = plt.figure(1,figsize=(7.66,7))
    plt.clf()
    x, y, midRA, midDec  = LSM._getXY()
    if hasWCSaxes:
        wcs = makeWCS(midRA, midDec)
        ax = WCSAxes(fig, [0.16, 0.1, 0.8, 0.8], wcs=wcs)
        fig.add_axes(ax)
    else:
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
    fluxes = LSM.getColValues('I')
    if len(fluxes[fluxes > 0.0]) == 0:
        minflux = 0.0
    else:
        minflux = np.min(fluxes[fluxes > 0.0])
    for flux in LSM.getColValues('I'):
        if flux > 0.0:
            s.append(min(1000.0, (1.0+2.0*np.log10(flux/minflux))*50.0))
        else:
            s.append(50.0)

    # Color sources by patch if grouped
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
    if hasWCSaxes:
        RA = LSM.getColValues('Ra')
        Dec = LSM.getColValues('Dec')
        ax.set_xlim(np.min(x)-20, np.max(x)+20)
        ax.set_ylim(np.min(y)-20, np.max(y)+20)
    plt.scatter(x, y, s=s, c=c)

    if LSM.hasPatches:
        RAp, Decp = LSM.getPatchPositions(asArray=True)
        goodInd = np.where( (RAp != 0.0) & (Decp != 0.0) )
        if len(goodInd[0]) < len(RAp):
            log.info('Some patch positions are unset. Run setPatchPositions() '
                'before plotting to see patch positions and patch names.')
        xp, yp = radec2xy(RAp[goodInd], Decp[goodInd], midRA, midDec)
        plt.scatter(xp, yp, s=100, c=cp, marker='*')

    # Set axis labels, etc.
    if hasWCSaxes:
        RAAxis = ax.coords['ra']
        RAAxis.set_axislabel('RA', minpad=0.75)
        RAAxis.set_major_formatter('hh:mm:ss')
        DecAxis = ax.coords['dec']
        DecAxis.set_axislabel('Dec', minpad=0.75)
        DecAxis.set_major_formatter('dd:mm:ss')
        ax.coords.grid(color='black', alpha=0.5, linestyle='solid')
    else:
        plt.xlabel("RA (arb. units)")
        plt.ylabel("Dec (arb. units)")

    if labelBy is not None:
        if labelBy.lower() == 'source':
            labels = LSM.getColValues('name')
            xls = x
            yls = y
        elif labelBy.lower() == 'patch':
            if LSM.hasPatches:
                labels = LSM.getPatchNames()
                xls = xp
                yls = yp
            else:
                labels = LSM.getColValues('name')
                xls = x
                yls = y
        else:
            raise ValueError("The lableBy parameter must be one of 'source' or "
                "'patch'.")
        for label, xl, yl in zip(labels, xls, yls):
            plt.annotate(label, xy = (xl, yl), xytext = (-2, 2), textcoords=
                'offset points', ha='right', va='bottom')

    # Define coodinate formater to show RA and Dec under mouse pointer
    RAformatter = FuncFormatter(RAtickformatter)
    ax.format_coord = formatCoord

    if fileName is not None:
        plt.savefig(fileName)
    else:
        plt.show()
    plt.close(fig)


def formatCoord(x, y):
    """Custom coordinate format"""
    from ..operations_lib import xy2radec

    global midRA, midDec
    RA, Dec = xy2radec([x], [y], midRA, midDec)
    return 'RA = {0:.2f} Dec = {1:.2f}'.format(RA[0], Dec[0])


def RAtickformatter(x, pos):
    """Changes x tick labels from pixels to RA in degrees"""
    from ..operations_lib import xy2radec

    global ymin, midRA, midDec
    ratick = xy2radec([x], [ymin], midRA, midDec)[0][0]
    rastr = '{0:.2f}'.format(ratick)
    return rastr


def Dectickformatter(y, pos):
    """Changes y tick labels from pixels to Dec in degrees"""
    from ..operations_lib import xy2radec

    global xmin, midRA, midDec
    dectick = xy2radec([xmin], [y], midRA, midDec)[1][0]
    decstr = '{0:.2f}'.format(dectick)
    return decstr

