#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This operation compares two sky models
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

log = logging.getLogger('LSMTool.COMPARE')
log.debug('Loading COMPARE module.')


def run(step, parset, LSM):

    outDir = parset.getString('.'.join(["LSMTool.Steps", step, "OutDir"]), '' )
    skyModel2 = parset.getString('.'.join(["LSMTool.Steps", step, "Skymodel2"]), '' )
    radius = parset.getString('.'.join(["LSMTool.Steps", step, "Radius"]), '0.1' )
    labelBy = parset.getString('.'.join(["LSMTool.Steps", step, "LabelBy"]), '' )

    if outDir == '':
        outDir = None
    if labelBy == '':
        labelBy = None

    try:
        compare(LSM, skyModel2, radius, outDir, labelBy)
        result = 0
    except Exception as e:
        log.error(e.message)
        result = 1

    return result


def compare(LSM1, LSM2, radius='10 arcsec', outDir=None, labelBy=None,
    ignoreSpec=None, excludeMultiple=True):
    """
    Compare two sky models

    Parameters
    ----------
    LSM1 : SkyModel object
        Parent sky model
    LSM2 : SkyModel object
        Sky model to compare to the parent sky model
    radius : float or str, optional
        Radius in degrees (if float) or 'value unit' (if str; e.g., '30 arcsec')
        for matching
    outDir : str, optional
        If given, the plots and stats are saved to this directory instead of
        displayed
    labelBy : str, optional
        One of 'source' or 'patch': label points using source names ('source') or
        patch names ('patch')
    ignoreSpec : float, optional
        Ignore sources with this spectral index
    excludeMultiple : bool, optional
        If True, sources with multiple matches are excluded

    Examples
    --------
    Compare two sky models and save plots::

        >>> LSM1 = lsmtool.load('sky1.model')
        >>> LSM2 = lsmtool.load('sky2.model')
        >>> compare(LSM1, LSM2, outDir='comparison_results/')

    """
    from astropy.table import vstack, Column
    from ..operations_lib import matchSky, radec2xy
    import numpy as np
    import os

    if type(LSM2) is str:
        LSM2 = skymodel.SkyModel(LSM2)

    if len(LSM1) == 0:
        log.info('Parent sky model is empty. No comparison possible.')
        return
    if len(LSM2) == 0:
        log.info('Secondary sky model is empty. No comparison possible.')
        return

    byPatch = False
    if (LSM1.hasPatches and not LSM2.hasPatches):
         LSM2.group('every')
    if (LSM2.hasPatches and not LSM1.hasPatches):
         LSM1.group('every')
    if (LSM2.hasPatches and LSM1.hasPatches):
         byPatch = True

    # Cross match the tables
    if excludeMultiple:
        nearestOnly = False
    else:
        nearestOnly = True
    matches11, matches21 = matchSky(LSM1, LSM2, radius=radius, byPatch=byPatch,
        nearestOnly=nearestOnly)
    matches12, matches22 = matchSky(LSM2, LSM1, radius=radius, byPatch=byPatch,
        nearestOnly=nearestOnly)
    if len(matches11) == 0:
        log.info('No matches found.')
        return

    # Get reference frequencies
    if byPatch:
        aggregate = 'wmean'
    else:
        aggregate = None
    if 'ReferenceFrequency' in LSM1.getColNames():
        refFreq1 = LSM1.getColValues('ReferenceFrequency', aggregate=aggregate)
    else:
        refFreq1 = np.array([LSM1.table.meta['ReferenceFrequency']]*len(LSM1))
    if 'ReferenceFrequency' in LSM2.getColNames():
        refFreq2 = LSM2.getColValues('ReferenceFrequency', aggregate=aggregate)
    else:
        refFreq2 = np.array([LSM2.table.meta['ReferenceFrequency']]*len(LSM2))

    # Get spectral indices
    try:
        alphas2 = LSM2.getColValues('SpectralIndex', aggregate=aggregate)
    except IndexError:
        alphas2 = np.array([-0.8]*len(LSM2))
    try:
        nterms = alphas2.shape[1]
    except IndexError:
        nterms = 1

    # Select sources that match up with only a single source and filter by spectral
    # index if desired
    filter = []
    for i in range(len(matches11)):
        nMatches = len(np.where(matches21 == matches21[i])[0])
        if nMatches == 1:
            # This source has a single match
            if ignoreSpec is not None:
                if nterms > 1:
                    spec = alphas2[matches21[i]][0]
                else:
                    spec = alphas2[matches21[i]]
                if spec != ignoreSpec:
                    filter.append(i)
            else:
                filter.append(i)
    good1 = set(matches11[filter])
    filter = []
    for i in range(len(matches12)):
        nMatches = len(np.where(matches22 == matches22[i])[0])
        if nMatches == 1:
            # This source has a single match
            if ignoreSpec is not None:
                if nterms > 1:
                    spec = alphas2[matches12[i]][0]
                else:
                    spec = alphas2[matches12[i]]
                if spec != ignoreSpec:
                    filter.append(i)
            else:
                filter.append(i)
    good2 = set(matches22[filter])
    good = good1.intersection(good2)
    matches1 = []
    matches2 = []
    for i in range(len(matches11)):
        if matches11[i] in good:
            matches1.append(matches11[i])
            matches2.append(matches21[i])
    if len(matches1) == 0:
        log.info('No suitable sources found for comparison.')
        return

    # Apply the filters
    if byPatch:
        fluxes1 =  LSM1.getColValues('I', aggregate='sum')[matches1]
        fluxes2 =  LSM2.getColValues('I', aggregate='sum')[matches2]
        RA, Dec = LSM1.getPatchPositions(asArray=True)
        RA = RA[matches1]
        Dec = Dec[matches1]
        RA2, Dec2 = LSM2.getPatchPositions(asArray=True)
        RA2 = RA2[matches2]
        Dec2 = Dec2[matches2]
    else:
        fluxes1 =  LSM1.getColValues('I', aggregate=aggregate)[matches1]
        fluxes2 =  LSM2.getColValues('I', aggregate=aggregate)[matches2]
        RA = LSM1.getColValues('Ra')[matches1]
        Dec = LSM1.getColValues('Dec')[matches1]
        RA2 = LSM2.getColValues('Ra')[matches2]
        Dec2 = LSM2.getColValues('Dec')[matches2]

    # Calculate predicted LSM2 fluxes at frequencies of LSM1
    predFlux = fluxes2
    if nterms > 1:
        for i in range(nterms):
            predFlux *= 10.0**(alphas2[:, i][matches2] *
                (np.log10(refFreq1[matches1] / refFreq2[matches2]))**(i+1))
    else:
        predFlux *= 10.0**(alphas2[matches2] *
            np.log10(refFreq1[matches1] / refFreq2[matches2]))

    # Find reference RA and Dec for center of LSM1
    x, y, refRA, refDec = LSM1._getXY()
    if byPatch:
        RAp, Decp = LSM1.getPatchPositions(asArray=True)
        RAp = RAp[matches1]
        Decp = Decp[matches1]
        x, y = radec2xy(RAp, Decp, refRA, refDec)
    else:
        x = x[matches1]
        y = y[matches1]

    if labelBy is not None:
        if labelBy.lower() == 'source':
            labels = LSM1.getColValues('name')[matches1]
        elif labelBy.lower() == 'patch':
            if LSM1.hasPatches:
                labels = LSM1.getPatchNames()[matches1]
            else:
                labels = LSM1.getColValues('name')[matches1]
        else:
            raise ValueError("The lableBy parameter must be one of 'source' or "
                "'patch'.")
    else:
        labels = None

    # Make plots
    if outDir is None:
        outDir = '.'
    if outDir[-1] != '/':
        outDir += '/'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    plotFluxRatiosDist(predFlux, fluxes1, RA, Dec, refRA, refDec, labels, outDir)
    plotFluxRatioPos(predFlux, fluxes1, x, y, RA, Dec, refRA, refDec, labels, outDir)
    plotFluxRatiosFlux(predFlux, fluxes1, labels, outDir)
    plotOffsets(RA, Dec, RA2, Dec2, labels, outDir)


def plotFluxRatiosDist(predFlux, measFlux, RA, Dec, refRA, refDec, labels, outDir):
    """
    Makes plot of measured-to-predicted flux ratio vs. distance from center
    """
    import numpy as np
    from ..operations_lib import calculateSeparation
    try:
        import os
        if 'DISPLAY' not in os.environ:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception as e:
        raise ImportError('PyPlot could not be imported. Plotting is not '
            'available: {0}'.format(e.message))

    ratio = measFlux / predFlux
    separation = np.zeros(len(measFlux))
    for i in range(len(measFlux)):
        separation[i] = calculateSeparation(RA[i], Dec[i], refRA, refDec).value

    fig = plt.figure(figsize=(7.0, 5.0))
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(separation, ratio, 'o')
    ax1.set_ylim(0, 2)
    plt.title('Flux Ratios (Model 1 / Model 2)')
    plt.ylabel('Flux ratio')
    plt.xlabel('Distance from center (deg)')

    # Calculate mean ratio and std dev.
    mean_ratio = np.mean(ratio)
    std = np.std(ratio)
    xmin, xmax, ymin, ymax = plt.axis()
    ax1.plot([0.0, xmax], [mean_ratio, mean_ratio], '--g')
    ax1.plot([0.0, xmax], [mean_ratio+std, mean_ratio+std], '-.g')
    ax1.plot([0.0, xmax], [mean_ratio-std, mean_ratio-std], '-.g')

    if labels is not None:
        xls = separation
        yls = ratio
        for label, xl, yl in zip(labels, xls, yls):
            plt.annotate(label, xy = (xl, yl), xytext = (-2, 2), textcoords=
                'offset points', ha='right', va='bottom')

    plt.savefig(outDir+'flux_ratio_vs_distance.pdf', format='pdf')


def plotFluxRatiosFlux(predFlux, measFlux, labels, outDir):
    """
    Makes plot of measured-to-predicted flux ratio vs. flux
    """
    import os
    import numpy as np
    from ..operations_lib import calculateSeparation
    try:
        import os
        if 'DISPLAY' not in os.environ:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception as e:
        raise ImportError('PyPlot could not be imported. Plotting is not '
            'available: {0}'.format(e.message))

    ratio = measFlux / predFlux

    fig = plt.figure(figsize=(7.0, 5.0))
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(measFlux, ratio, 'o')
    ax1.set_ylim(0, 2)
    plt.title('Flux Ratios (Model 1 / Model 2)')
    plt.ylabel('Flux ratio')
    plt.xlabel('Model 1 flux (Jy)')

    # Calculate mean ratio and std dev.
    mean_ratio = np.mean(ratio)
    std = np.std(ratio)
    xmin, xmax, ymin, ymax = plt.axis()
    ax1.plot([0.0, xmax], [mean_ratio, mean_ratio], '--g')
    ax1.plot([0.0, xmax], [mean_ratio+std, mean_ratio+std], '-.g')
    ax1.plot([0.0, xmax], [mean_ratio-std, mean_ratio-std], '-.g')

    if labels is not None:
        xls = measFlux
        yls = ratio
        for label, xl, yl in zip(labels, xls, yls):
            plt.annotate(label, xy = (xl, yl), xytext = (-2, 2), textcoords=
                'offset points', ha='right', va='bottom')

    plt.savefig(outDir+'flux_ratio_vs_flux.pdf', format='pdf')


def plotFluxRatioPos(predFlux, measFlux, x, y, RA, Dec, midRA, midDec, labels, outDir):
    """
    Makes sky plot of measured-to-predicted flux ratio
    """
    import os
    import numpy as np
    from ..operations_lib import calculateSeparation, makeWCS
    try:
        import os
        if 'DISPLAY' not in os.environ:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception as e:
        raise ImportError('PyPlot could not be imported. Plotting is not '
            'available: {0}'.format(e.message))
    try:
        from wcsaxes import WCSAxes
        hasWCSaxes = True
    except:
        hasWCSaxes = False

    ratio = measFlux / predFlux

    fig = plt.figure(figsize=(7.0, 5.0))
    if hasWCSaxes:
        wcs = makeWCS(midRA, midDec)
        ax1 = WCSAxes(fig, [0.12, 0.12, 0.8, 0.8], wcs=wcs)
        fig.add_axes(ax1)
    else:
        ax1 = plt.gca()
    plt.title('Flux Ratios (Model 1 / Model 2)')

    # Set symbol color by ratio
    vmin = np.min(ratio)
    vmax = np.max(ratio)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,
        norm=plt.normalize(vmin=vmin, vmax=vmax))
    sm.set_array(ratio)
    sm._A = []
    c = []
    for r in ratio:
        c.append(sm.to_rgba(r))

    if hasWCSaxes:
        ax1.set_xlim(np.min(x)-20, np.max(x)+20)
        ax1.set_ylim(np.min(y)-20, np.max(y)+20)
    plot = plt.scatter(x, y, c=c)
    cbar = plt.colorbar(sm)

    # Set axis labels, etc.
    if hasWCSaxes:
        RAAxis = ax1.coords['ra']
        DecAxis = ax1.coords['dec']
        RAAxis.set_axislabel('RA')
        DecAxis.set_axislabel('Dec')
        ax1.coords.grid(color='black', alpha=0.5, linestyle='solid')
    else:
        plt.xlabel("RA (arb. units)")
        plt.ylabel("Dec (arb. units)")

    if labels is not None:
        xls = x
        yls = y
        for label, xl, yl in zip(labels, xls, yls):
            plt.annotate(label, xy = (xl, yl), xytext = (-2, 2), textcoords=
                'offset points', ha='right', va='bottom')

    plt.savefig(outDir+'flux_ratio_sky.pdf', format='pdf')


def plotOffsets(RA, Dec, refRA, refDec, labels, outDir):
    """
    Makes plot of measured - predicted RA and DEC offsets
    """
    from ..operations_lib import calculateSeparation
    import os
    import numpy as np
    try:
        import os
        if 'DISPLAY' not in os.environ:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception as e:
        raise ImportError('PyPlot could not be imported. Plotting is not '
            'available: {0}'.format(e.message))

    RAOffsets = np.zeros(len(RA))
    DecOffsets = np.zeros(len(Dec))
    for i in range(len(RA)):
        if RA[i] >= refRA[i]:
            sign = 1.0
        else:
            sign = -1.0
        RAOffsets[i] = sign * calculateSeparation(RA[i], Dec[i], refRA[i], Dec[i]).value * 3600.0 # arcsec
        if Dec[i] >= refDec[i]:
            sign = 1.0
        else:
            sign = -1.0
        DecOffsets[i] = sign * calculateSeparation(RA[i], Dec[i], RA[i], refDec[i]).value * 3600.0 # arcsec

    fig = plt.figure(figsize=(7.0, 5.0))
    ax1 = plt.subplot(1, 1, 1)
    plt.title('Positional offsets (Model 1 - Model 2)')
    ax1.plot(RAOffsets, DecOffsets, 'o')
    xmin, xmax, ymin, ymax = plt.axis()
    ax1.plot([xmin, xmax], [0.0, 0.0], '--g')
    ax1.plot([0.0, 0.0], [ymin, ymax], '--g')
    plt.xlabel('RA Offset (arcsec)')
    plt.ylabel('Dec Offset (arcsec)')

    if labels is not None:
        xls = RAOffsets
        yls = DecOffsets
        for label, xl, yl in zip(labels, xls, yls):
            plt.annotate(label, xy = (xl, yl), xytext = (-2, 2), textcoords=
                'offset points', ha='right', va='bottom')

    plt.savefig(outDir+'postional_offsets.pdf', format='pdf')
