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
    skyModel2 = parset.getString('.'.join(["LSMTool.Steps", step, "SkyModel2"]), '' )
    radius = parset.getString('.'.join(["LSMTool.Steps", step, "Radius"]), '10 arcsec' )
    labelBy = parset.getString('.'.join(["LSMTool.Steps", step, "LabelBy"]), '' )
    excludeMultiple = parset.getBool('.'.join(["LSMTool.Steps", step, "ExcludeMultiple"]), True )
    ignoreSpec = parset.getString('.'.join(["LSMTool.Steps", step, "IgnoreSpec"]), '' )

    if outDir == '':
        outDir = '.'
    if labelBy == '':
        labelBy = None
    if ignoreSpec == '':
        ignoreSpec = None
    else:
        ignoreSpec = float(ignoreSpec)

    try:
        compare(LSM, skyModel2, radius=radius, outDir=outDir, labelBy=labelBy,
            excludeMultiple=excludeMultiple, ignoreSpec=ignoreSpec)
        result = 0
    except Exception as e:
        log.error(e.message)
        result = 1

    return result


def compare(LSM1, LSM2, radius='10 arcsec', outDir='.', labelBy=None,
    ignoreSpec=None, excludeMultiple=True, excludeByFlux=True, name1=None, name2=None,
    format='pdf'):
    """
    Compare two sky models

    Comparison plots and a text file with statistics are written out to the
    an output directory. Plots are made for:
        - flux ratio vs. radius from sky model center
        - flux ratio vs. sky position
        - flux ratio vs flux
        - position offsets
    The following statistics are saved to 'stats.txt' in the output directory:
        - mean and standard deviation of flux ratio
        - mean and standard deviation of RA offsets (in degrees)
        - mean and standard deviation of Dec offsets (in degrees)

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
        Plots are saved to this directory
    labelBy : str, optional
        One of 'source' or 'patch': label points using source names ('source') or
        patch names ('patch')
    ignoreSpec : float, optional
        Ignore sources with this spectral index
    excludeMultiple : bool, optional
        If True, sources with multiple matches are excluded. If False, the
        nearest of the multiple matches will be used for comparison
    excludeByFlux : bool, optional
        If True, matches whose predicted fluxes differ from the parent model
        fluxes by 25% are excluded from the positional offset plot.
    name1 : str, optional
        Name to use in the plots for LSM1. If None, 'Model 1' is used.
    name2 : str, optional
        Name to use in the plots for LSM2. If None, 'Model 2' is used.
    format : str, optional
        Format of plot files.

    Examples
    --------
    Compare two sky models and save plots and stats to 'comparison_results/'::

        >>> LSM1 = lsmtool.load('sky1.model')
        >>> LSM2 = lsmtool.load('sky2.model')
        >>> compare(LSM1, LSM2, outDir='comparison_results/')

    Compare a LOFAR sky model to a global sky model made from VLSS+TGSS+NVSS (where
    refRA and refDec are the approximate center of the LOFAR sky model coverage)::

        >>> LSM1 = lsmtool.load('lofar_sky.model')
        >>> LSM2 = lsmtool.load('GSM', VOPosition=[refRA, refDec], VORadius='5 deg')
        >>> compare(LSM1, LSM2, radius='30 arcsec', excludeMultiple=True,
            outDir='comparison_results/', name1='LOFAR', name2='GSM', format='png')


    """
    from astropy.table import vstack, Column
    from ..operations_lib import matchSky, radec2xy
    from ..skymodel import SkyModel
    import numpy as np
    import os

    if type(LSM2) is str:
        LSM2 = SkyModel(LSM2)

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
        alphas2 = LSM2.getColValues('SpectralIndex', aggregate=aggregate).squeeze(axis=0)
    except (IndexError, ValueError):
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
        x, y = radec2xy(RA, Dec, refRA, refDec)
    else:
        x = x[matches1]
        y = y[matches1]
    refx, refy = radec2xy(RA2, Dec2, refRA, refDec)

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
    plotFluxRatiosDist(predFlux, fluxes1, RA, Dec, refRA, refDec, labels, outDir, name1, name2, format)
    plotFluxRatioSky(predFlux, fluxes1, x, y, RA, Dec, refRA, refDec, labels, outDir, name1, name2, format)
    plotFluxRatiosFlux(predFlux, fluxes1, labels, outDir, name1, name2, format)
    retstatus = plotOffsets(RA, Dec, RA2, Dec2, x, y, refx, refy, labels,
        outDir, predFlux, fluxes1, excludeByFlux, name1, name2, format)
    if retstatus == 1:
        log.warn('No matches found within +/- 25% of predicted flux density. Skipping offset plot.')
    argInfo = 'Used radius = {0}, ignoreSpec = {1}, and excludeMultiple = {2}'.format(
        radius, ignoreSpec, excludeMultiple)
    stats = findStats(predFlux, fluxes1, RA, Dec, RA2, Dec2, outDir, argInfo,
        LSM1._info(), LSM2._info(), name1, name2)

    return stats


def plotFluxRatiosDist(predFlux, measFlux, RA, Dec, refRA, refDec, labels,
    outDir, name1, name2, format, clip=True):
    """
    Makes plot of measured-to-predicted flux ratio vs. distance from center
    """
    import numpy as np
    from ..operations_lib import calculateSeparation
    try:
        from astropy.stats.funcs import sigma_clip
    except ImportError:
        from astropy.stats import sigma_clip
    try:
        import matplotlib
        if matplotlib.get_backend() is not 'Agg':
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception as e:
        raise ImportError('PyPlot could not be imported. Plotting is not '
            'available: {0}'.format(e.message))

    if name1 is None:
        name1 = 'Model 1'
    if name2 is None:
        name2 = 'Model 2'

    ratio = measFlux / predFlux
    separation = np.zeros(len(measFlux))
    for i in range(len(measFlux)):
        separation[i] = calculateSeparation(RA[i], Dec[i], refRA, refDec).value

    fig = plt.figure(figsize=(7.0, 5.0))
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(separation, ratio, 'o')
    plt.title('Flux Density Ratios ({0} / {1})'.format(name1, name2))
    plt.ylabel('Flux density ratio')
    plt.xlabel('Distance from center (deg)')

    # Calculate mean ratio and std dev.
    if clip:
        mean_ratio = np.mean(sigma_clip(ratio))
        std = np.std(sigma_clip(ratio))
    else:
        mean_ratio = np.mean(ratio)
        std = np.std(ratio)
    ax1.set_ylim(0, 2.0*mean_ratio)
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

    plt.savefig(outDir+'flux_ratio_vs_distance.{}'.format(format), format=format)


def plotFluxRatiosFlux(predFlux, measFlux, labels, outDir, name1, name2, format, clip=True):
    """
    Makes plot of measured-to-predicted flux ratio vs. flux
    """
    import os
    import numpy as np
    from ..operations_lib import calculateSeparation
    try:
        from astropy.stats.funcs import sigma_clip
    except ImportError:
        from astropy.stats import sigma_clip
    try:
        import matplotlib
        if matplotlib.get_backend() is not 'Agg':
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception as e:
        raise ImportError('PyPlot could not be imported. Plotting is not '
            'available: {0}'.format(e.message))

    if name1 is None:
        name1 = 'Model 1'
    if name2 is None:
        name2 = 'Model 2'

    ratio = measFlux / predFlux

    fig = plt.figure(figsize=(7.0, 5.0))
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(measFlux, ratio, 'o')
    ax1.set_xscale('log')
    plt.title('Flux Density Ratios ({0} / {1})'.format(name1, name2))
    plt.ylabel('Flux density ratio')
    plt.xlabel('{0} flux density (Jy)'.format(name1))

    # Calculate mean ratio and std dev.
    if clip:
        mean_ratio = np.mean(sigma_clip(ratio))
        std = np.std(sigma_clip(ratio))
    else:
        mean_ratio = np.mean(ratio)
        std = np.std(ratio)
    ax1.set_ylim(0, 2.0*mean_ratio)
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

    plt.savefig(outDir+'flux_ratio_vs_flux.{}'.format(format), format=format)


def plotFluxRatioSky(predFlux, measFlux, x, y, RA, Dec, midRA, midDec, labels,
    outDir, name1, name2, format):
    """
    Makes sky plot of measured-to-predicted flux ratio
    """
    import os
    import numpy as np
    from ..operations_lib import calculateSeparation, makeWCS
    try:
        from astropy.stats.funcs import sigma_clip
    except ImportError:
        from astropy.stats import sigma_clip
    try:
        import matplotlib
        if matplotlib.get_backend() is not 'Agg':
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        from matplotlib.ticker import FuncFormatter
    except Exception as e:
        raise ImportError('PyPlot could not be imported. Plotting is not '
            'available: {0}'.format(e.message))
    try:
        from wcsaxes import WCSAxes
        hasWCSaxes = True
    except:
        hasWCSaxes = False

    if name1 is None:
        name1 = 'Model 1'
    if name2 is None:
        name2 = 'Model 2'

    ratio = measFlux / predFlux

    fig = plt.figure(figsize=(7.0, 5.0))
    if hasWCSaxes:
        wcs = makeWCS(midRA, midDec)
        ax1 = WCSAxes(fig, [0.12, 0.12, 0.8, 0.8], wcs=wcs)
        fig.add_axes(ax1)
    else:
        ax1 = plt.gca()
    plt.title('Flux Density Ratios ({0} / {1})'.format(name1, name2))

    # Set symbol color by ratio
    vmin = np.min(ratio) - 0.1
    vmax = np.max(ratio) + 0.1
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,
        norm=colors.Normalize(vmin=vmin, vmax=vmax))
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

    plt.savefig(outDir+'flux_ratio_sky.{}'.format(format), format=format)


def plotOffsets(RA, Dec, refRA, refDec, x, y, refx, refy, labels, outDir,
    predFlux, measFlux, excludeByFlux, name1, name2, format, plot_imcoords=False):
    """
    Makes plot of measured - predicted RA and DEC and x and y offsets
    """
    from ..operations_lib import calculateSeparation
    import os
    import numpy as np
    try:
        from astropy.stats.funcs import sigma_clip
    except ImportError:
        from astropy.stats import sigma_clip
    try:
        import matplotlib
        if matplotlib.get_backend() is not 'Agg':
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception as e:
        raise ImportError('PyPlot could not be imported. Plotting is not '
            'available: {0}'.format(e.message))

    if excludeByFlux:
        ratio = measFlux / predFlux
        goodInd = np.where((ratio > 0.75) & (ratio < 1.25))[0]
        if len(goodInd) == 0:
            return 1
        RA = RA[goodInd]
        Dec = Dec[goodInd]
        refRA = refRA[goodInd]
        refDec = refDec[goodInd]

    if name1 is None:
        name1 = 'Model 1'
    if name2 is None:
        name2 = 'Model 2'

    RAOffsets = np.zeros(len(RA))
    DecOffsets = np.zeros(len(Dec))
    xOffsets = np.zeros(len(RA))
    yOffsets = np.zeros(len(Dec))
    for i in range(len(RA)):
        if RA[i] >= refRA[i]:
            sign = 1.0
        else:
            sign = -1.0
        RAOffsets[i] = sign * calculateSeparation(RA[i], Dec[i], refRA[i], Dec[i]).value * 3600.0 # arcsec
        xOffsets[i] = x[i] - refx[i]
        if Dec[i] >= refDec[i]:
            sign = 1.0
        else:
            sign = -1.0
        DecOffsets[i] = sign * calculateSeparation(RA[i], Dec[i], RA[i], refDec[i]).value * 3600.0 # arcsec
        yOffsets[i] = y[i] - refy[i]

    fig = plt.figure(figsize=(7.0, 5.0))
    ax1 = plt.subplot(1, 1, 1)
    plt.title('Positional offsets ({0} - {1})'.format(name1, name2))
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
    plt.savefig(outDir+'positional_offsets_sky.{}'.format(format), format=format)

    if plot_imcoords:
        fig = plt.figure(figsize=(7.0, 5.0))
        ax1 = plt.subplot(1, 1, 1)
        plt.title('Positional offsets ({0} - {1})'.format(name1, name2))
        ax1.plot(xOffsets, yOffsets, 'o')
        xmin, xmax, ymin, ymax = plt.axis()
        ax1.plot([xmin, xmax], [0.0, 0.0], '--g')
        ax1.plot([0.0, 0.0], [ymin, ymax], '--g')
        plt.xlabel('Image-plane X Offset (arb. units)')
        plt.ylabel('Image-plane Y Offset (arb. units)')

        if labels is not None:
            xls = RAOffsets
            yls = DecOffsets
            for label, xl, yl in zip(labels, xls, yls):
                plt.annotate(label, xy = (xl, yl), xytext = (-2, 2), textcoords=
                    'offset points', ha='right', va='bottom')
        plt.savefig(outDir+'positional_offsets_im.{}'.format(format), format=format)
    return 0


def findStats(predFlux, measFlux, RA, Dec, refRA, refDec, outDir, info0, info1,
    info2, name1, name2):
    """
    Calculates statistics and saves them to 'stats.txt'
    """
    import os
    import numpy as np
    from ..operations_lib import calculateSeparation
    try:
        from astropy.stats.funcs import sigma_clip
    except ImportError:
        from astropy.stats import sigma_clip

    ratio = measFlux / predFlux
    meanRatio = np.mean(ratio)
    stdRatio = np.std(ratio)
    clippedRatio = sigma_clip(ratio)
    meanClippedRatio = np.mean(clippedRatio)
    stdClippedRatio = np.std(clippedRatio)

    RAOffsets = np.zeros(len(RA))
    DecOffsets = np.zeros(len(Dec))
    for i in range(len(RA)):
        if RA[i] >= refRA[i]:
            sign = 1.0
        else:
            sign = -1.0
        RAOffsets[i] = sign * calculateSeparation(RA[i], Dec[i], refRA[i], Dec[i]).value # deg
        if Dec[i] >= refDec[i]:
            sign = 1.0
        else:
            sign = -1.0
        DecOffsets[i] = sign * calculateSeparation(RA[i], Dec[i], RA[i], refDec[i]).value # deg

    meanRAOffset = np.mean(RAOffsets)
    stdRAOffset = np.std(RAOffsets)
    clippedRAOffsets = sigma_clip(RAOffsets)
    meanClippedRAOffset = np.mean(clippedRAOffsets)
    stdClippedRAOffset = np.std(clippedRAOffsets)

    meanDecOffset = np.mean(DecOffsets)
    stdDecOffset = np.std(DecOffsets)
    clippedDecOffsets = sigma_clip(DecOffsets)
    meanClippedDecOffset = np.mean(clippedDecOffsets)
    stdClippedDecOffset = np.std(clippedDecOffsets)

    stats = {'meanRatio':meanRatio,
             'stdRatio':stdRatio,
             'meanClippedRatio':meanClippedRatio,
             'stdClippedRatio':stdClippedRatio,
             'meanRAOffsetDeg':meanRAOffset,
             'stdRAOffsetDeg':stdRAOffset,
             'meanClippedRAOffsetDeg':meanClippedRAOffset,
             'stdClippedRAOffsetDeg':stdClippedRAOffset,
             'meanDecOffsetDeg':meanDecOffset,
             'stdDecOffsetDeg':stdDecOffset,
             'meanClippedDecOffsetDeg':meanClippedDecOffset,
             'stdClippedDecOffsetDeg':stdClippedDecOffset}

    outLines = ['Statistics from sky model comparison\n']
    outLines.append('------------------------------------\n\n')

    outLines.append('Sky model 1 ({}):\n'.format(name1))
    outLines.append(info1+'\n\n')
    outLines.append('Sky model 2 ({}):\n'.format(name2))
    outLines.append(info2+'\n\n')

    outLines.append(info0+'\n')
    outLines.append('Number of matches found for comparison: {0}\n\n'.format(len(predFlux)))

    outLines.append('Mean flux density ratio (1 / 2): {0}\n'.format(meanRatio))
    outLines.append('Std. dev. flux density ratio (1 / 2): {0}\n'.format(stdRatio))
    outLines.append('Mean 3-sigma-clipped flux density ratio (1 / 2): {0}\n'.format(meanClippedRatio))
    outLines.append('Std. dev. 3-sigma-clipped flux density ratio (1 / 2): {0}\n\n'.format(stdClippedRatio))

    outLines.append('Mean RA offset (1 - 2): {0} degrees\n'.format(meanRAOffset))
    outLines.append('Std. dev. RA offset (1 - 2): {0} degrees\n'.format(stdRAOffset))
    outLines.append('Mean 3-sigma-clipped RA offset (1 - 2): {0} degrees\n'.format(meanClippedRAOffset))
    outLines.append('Std. dev. 3-sigma-clipped RA offset (1 - 2): {0} degrees\n\n'.format(stdClippedRAOffset))

    outLines.append('Mean Dec offset (1 - 2): {0} degrees\n'.format(meanDecOffset))
    outLines.append('Std. dev. Dec offset (1 - 2): {0} degrees\n'.format(stdDecOffset))
    outLines.append('Mean 3-sigma-clipped Dec offset (1 - 2): {0} degrees\n'.format(meanClippedDecOffset))
    outLines.append('Std. dev. 3-sigma-clipped Dec offset (1 - 2): {0} degrees\n\n'.format(stdClippedDecOffset))

    fileName = outDir+'stats.txt'
    statsFile = open(fileName, 'w')
    statsFile.writelines(outLines)
    statsFile.close()

    return stats


