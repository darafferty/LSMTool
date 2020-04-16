# -*- coding: utf-8 -*-
#
# This module defines functions used for more than one operation.
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


def apply_beam_star(inputs):
    """
    Simple multiprocessing helper function for apply_beam()
    """
    return apply_beam(*inputs)


def apply_beam(RA, Dec, spectralIndex, flux, referenceFrequency, beamMS, time, ant1,
               numchannels, startfreq, channelwidth, invert):
    import numpy as np
    import lofar.stationresponse as lsr

    # Use ant1, times, and n channel to compute the beam, where n is determined by
    # the order of the spectral index polynomial
    sr = lsr.stationresponse(beamMS, inverse=invert, useElementResponse=False,
                             useArrayFactor=True, useChanFreq=False)
    sr.setDirection(RA*np.pi/180., Dec*np.pi/180.)

    # Correct the source spectrum if needed
    if spectralIndex is not None:
        nspec = len(spectralIndex)
        s = max(1, int(numchannels/nspec))
        ch_indices = list(range(0, numchannels, s))
        ch_indices.append(numchannels)
    else:
        ch_indices = [int(numchannels/2)]
    freqs_new = []
    fluxes_new = []
    for ind in ch_indices:
        # Evaluate beam and take XX only (XX and YY should be equal) and square
        beam = abs(sr.evaluateChannel(time, ant1, ind))
        beam = beam[0][0]**2
        if spectralIndex is not None:
            nu = (startfreq + ind*channelwidth) / referenceFrequency - 1
            freqs_new.append(nu)
            fluxes_new.append(polynomial(flux, spectralIndex, nu) * beam)
        else:
            fluxes_new.append(flux * beam)
    if spectralIndex is not None:
        fit = np.polyfit(freqs_new, fluxes_new, len(spectralIndex-1)).tolist()
        fit.reverse()
        return fit[0], fit[1:]
    else:
        return fluxes_new[0], None


def attenuate(beamMS, fluxes, RADeg, DecDeg, spectralIndex=None, referenceFrequency=None,
              timeIndx=0.5, invert=False):
    """
    Returns flux attenuated by primary beam.

    Parameters
    ----------
    beamMS : str
        Measurement set for which the beam model is made
    fluxes : list
        List of fluxes to attenuate
    RADeg : list
        List of RA values in degrees
    DecDeg : list
        List of Dec values in degrees
    spectralIndex : list, optional
        List of spectral indices to adjust
    referenceFrequency : list, optional
        List of reference frequency of polynomial fit
    timeIndx : float (between 0 and 1), optional
        Time as fraction of that covered by the beamMS for which the beam is
        calculated
    invert : bool, optional

    Returns
    -------
    attFluxes : numpy array
        Attenuated fluxes
    adjSpectralIndex : numpy array
        Adjusted spectral indices. Returned only if spectralIndex is not None

    """
    import numpy as np
    import pyrap.tables as pt
    import multiprocessing
    import itertools

    t = pt.table(beamMS, ack=False)
    time = None
    ant1 = -1
    ant2 = 1
    while time is None:
        ant1 += 1
        ant2 += 1
        tt = t.query('ANTENNA1=={0} AND ANTENNA2=={1}'.format(ant1, ant2), columns='TIME')
        time = tt.getcol("TIME")
    t.close()
    if timeIndx < 0.0:
        timeIndx = 0.0
    if timeIndx > 1.0:
        timeIndx = 1.0
    time = min(time) + (max(time) - min(time)) * timeIndx
    sw = pt.table(beamMS+'::SPECTRAL_WINDOW', ack=False)
    numchannels = sw.col('NUM_CHAN')[0]
    startfreq = np.min(sw.col('CHAN_FREQ')[0])
    channelwidth = sw.col('CHAN_WIDTH')[0][0]
    sw.close()

    attFluxes = []
    adjSpectralIndex = []
    if type(fluxes) is not list:
        fluxes = list(fluxes)
    if type(RADeg) is not list:
        RADeg = list(RADeg)
    if type(DecDeg) is not list:
        DecDeg = list(DecDeg)
    if spectralIndex is not None:
        spectralIndex_list = spectralIndex
    else:
        spectralIndex_list = [None] * len(fluxes)
    if referenceFrequency is not None:
        referenceFrequency_list = referenceFrequency
    else:
        referenceFrequency_list = [None] * len(fluxes)

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    args_list = list(zip(RADeg, DecDeg, spectralIndex_list, fluxes, referenceFrequency_list, itertools.repeat(beamMS),
                         itertools.repeat(time), itertools.repeat(ant1), itertools.repeat(numchannels),
                         itertools.repeat(startfreq), itertools.repeat(channelwidth), itertools.repeat(invert)))
    results = pool.map(apply_beam_star, args_list)
    pool.close()
    pool.join()
    for fl, si in results:
        attFluxes.append(fl)
        adjSpectralIndex.append(si)

    if spectralIndex is not None:
        return np.array(attFluxes), np.array(adjSpectralIndex)
    else:
        return np.array(attFluxes)


def polynomial(stokesI, coeff, nu):
    result = stokesI
    for i in range(len(coeff)):
        result += coeff[i] * nu**(i+1)

    return result


def radec2xy(RA, Dec, refRA=None, refDec=None, crdelt=None):
    """
    Returns x, y for input ra, dec.

    Note that the reference RA and Dec must be the same in calls to both
    radec2xy() and xy2radec() if matched pairs of (x, y) <=> (RA, Dec) are
    desired.

    Parameters
    ----------
    RA : list
        List of RA values in degrees
    Dec : list
        List of Dec values in degrees
    refRA : float, optional
        Reference RA in degrees.
    refDec : float, optional
        Reference Dec in degrees
    crdelt: float, optional
        Delta in degrees for sky grid

    Returns
    -------
    x, y : list, list
        Lists of x and y pixel values corresponding to the input RA and Dec
        values

    """
    import numpy as np

    x = []
    y = []
    if refRA is None:
        refRA = RA[0]
    if refDec is None:
        refDec = Dec[0]

    # Make wcs object to handle transformation from ra and dec to pixel coords.
    w = makeWCS(refRA, refDec, crdelt=crdelt)

    for ra_deg, dec_deg in zip(RA, Dec):
        ra_dec = np.array([[ra_deg, dec_deg]])
        x.append(w.wcs_world2pix(ra_dec, 0)[0][0])
        y.append(w.wcs_world2pix(ra_dec, 0)[0][1])

    return x, y


def xy2radec(x, y, refRA=0.0, refDec=0.0, crdelt=None):
    """
    Returns x, y for input ra, dec.

    Note that the reference RA and Dec must be the same in calls to both
    radec2xy() and xy2radec() if matched pairs of (x, y) <=> (RA, Dec) are
    desired.

    Parameters
    ----------
    x : list
        List of x values in pixels
    y : list
        List of y values in pixels
    refRA : float, optional
        Reference RA in degrees
    refDec : float, optional
        Reference Dec in degrees
    crdelt: float, optional
        Delta in degrees for sky grid

    Returns
    -------
    RA, Dec : list, list
        Lists of RA and Dec values corresponding to the input x and y pixel
        values

    """
    import numpy as np

    RA = []
    Dec = []

    # Make wcs object to handle transformation from ra and dec to pixel coords.
    w = makeWCS(refRA, refDec, crdelt=crdelt)

    for xp, yp in zip(x, y):
        x_y = np.array([[xp, yp]])
        RA.append(w.wcs_pix2world(x_y, 0)[0][0])
        Dec.append(w.wcs_pix2world(x_y, 0)[0][1])

    return RA, Dec


def makeWCS(refRA, refDec, crdelt=None):
    """
    Makes simple WCS object.

    Parameters
    ----------
    refRA : float
        Reference RA in degrees
    refDec : float
        Reference Dec in degrees
    crdelt: float, optional
        Delta in degrees for sky grid

    Returns
    -------
    w : astropy.wcs.WCS object
        A simple TAN-projection WCS object for specified reference position

    """
    from astropy.wcs import WCS
    import numpy as np

    w = WCS(naxis=2)
    w.wcs.crpix = [1000, 1000]
    if crdelt is None:
        crdelt = 0.066667  # 4 arcmin
    w.wcs.cdelt = np.array([-crdelt, crdelt])
    w.wcs.crval = [refRA, refDec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.set_pv([(2, 1, 45.0)])

    return w


def matchSky(LSM1, LSM2, radius=0.1, byPatch=False, nearestOnly=False):
    """
    Matches two sky models by position.

    Parameters
    ----------
    LSM1 : SkyModel object
        Sky model for which match indices are desired
    LSM2 : SkyModel object
        Sky model to match against
    radius : float or str, optional
        Radius in degrees (if float) or 'value unit' (if str; e.g., '30 arcsec')
        for matching when matchBy='position'
    byPatch : bool, optional
        If True, matching is done by patches
    nearestOnly : bool, optional
        If True, only the nearest of multiple matches is returned

    Returns
    -------
    matches1, matches2 : np.array, np.array
        matches1 is the array of indices of LSM1 that have matches in LSM2
        within the specified radius. matches2 is the array of indices of LSM2
        for the same sources.

    """

    from astropy.coordinates import SkyCoord, Angle
    from astropy import units as u
    from distutils.version import StrictVersion
    import numpy as np
    import scipy

    log = logging.getLogger('LSMTool')
    if StrictVersion(scipy.__version__) < StrictVersion('0.11.0'):
        log.debug('The installed version of SciPy contains a bug that affects catalog matching. '
                  'Falling back on (slower) matching script.')
        from operations._matching import match_coordinates_sky
    else:
        from astropy.coordinates.matching import match_coordinates_sky

    if byPatch:
        RA, Dec = LSM1.getPatchPositions(asArray=True)
        catalog1 = SkyCoord(RA, Dec, unit=(u.degree, u.degree), frame='fk5')
        RA, Dec = LSM2.getPatchPositions(asArray=True)
        catalog2 = SkyCoord(RA, Dec, unit=(u.degree, u.degree), frame='fk5')
    else:
        catalog1 = SkyCoord(LSM1.getColValues('Ra'), LSM1.getColValues('Dec'),
                            unit=(u.degree, u.degree))
        catalog2 = SkyCoord(LSM2.getColValues('Ra'), LSM2.getColValues('Dec'),
                            unit=(u.degree, u.degree))
    idx, d2d, d3d = match_coordinates_sky(catalog1, catalog2)

    try:
        radius = float(radius)
    except ValueError:
        pass
    if type(radius) is float:
        radius = '{0} degree'.format(radius)
    radius = Angle(radius).degree
    matches1 = np.where(d2d.value <= radius)[0]
    matches2 = idx[matches1]

    if nearestOnly:
        filter = []
        for i in range(len(matches1)):
            mind = np.where(matches2 == matches2[i])[0]
            nMatches = len(mind)
            if nMatches > 1:
                mradii = d2d.value[matches1][mind]
                if d2d.value[matches1][i] == np.min(mradii):
                    filter.append(i)
            else:
                filter.append(i)
        matches1 = matches1[filter]
        matches2 = matches2[filter]

    return matches1, matches2


def calculateSeparation(ra1, dec1, ra2, dec2):
    """
    Returns angular separation between two coordinates (all in degrees).

    Parameters
    ----------
    ra1 : float or numpy array
        RA of coordinate 1 in degrees
    dec1 : float or numpy array
        Dec of coordinate 1 in degrees
    ra2 : float
        RA of coordinate 2 in degrees
    dec2 : float
        Dec of coordinate 2 in degrees

    Returns
    -------
    separation : astropy Angle or numpy array
        Angular separation in degrees

    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coord1 = SkyCoord(ra1, dec1, unit=(u.degree, u.degree), frame='fk5')
    coord2 = SkyCoord(ra2, dec2, unit=(u.degree, u.degree), frame='fk5')

    return coord1.separation(coord2)


def getFluxAtSingleFrequency(LSM, targetFreq=None, aggregate=None):
    """
    Returns flux density at given target frequency, adjusted according to the
    spectral index.

    Parameters
    ----------
    LSM : sky model object
        RA of coordinate 1 in degrees
    targetFreq : float, optional
        Frequency in Hz. If None, the median is used
    aggregate : str, optional
        Aggregation to use

    Returns
    -------
    fluxes : numpy array
        Flux densities in Jy

    """
    import numpy as np

    # Calculate flux densities
    if targetFreq is None:
        if 'ReferenceFrequency' in LSM.getColNames():
            refFreq = LSM.getColValues('ReferenceFrequency', aggregate=aggregate)
        else:
            refFreq = np.array([LSM.table.meta['ReferenceFrequency']]*len(LSM))
        targetFreq = np.median(refFreq)
    fluxes = LSM.getColValues('I', aggregate=aggregate)

    try:
        alphas = LSM.getColValues('SpectralIndex', aggregate=aggregate).squeeze(axis=0)
    except (IndexError, ValueError):
        alphas = np.array([-0.8]*len(fluxes))
    nterms = alphas.shape[1]

    if 'LogarithmicSI' in LSM.table.meta:
        logSI = LSM.table.meta['LogarithmicSI']
    else:
        logSI = True

    if nterms > 1:
        for i in range(nterms):
            if logSI:
                fluxes *= 10.0**(alphas[:, i] * (np.log10(refFreq / targetFreq))**(i+1))
            else:
                # stokesI + term0 (nu/refnu - 1) + term1 (nu/refnu - 1)^2 + ...
                fluxes += alphas[:, i] * ((refFreq / targetFreq) - 1.0)**(i+1)
    else:
        if logSI:
            fluxes *= 10.0**(alphas * np.log10(refFreq / targetFreq))
        else:
            fluxes += alphas * ((refFreq / targetFreq) - 1.0)

    return fluxes


def make_template_image(image_name, reference_ra_deg, reference_dec_deg, reference_freq,
                        ximsize=512, yimsize=512, cellsize_deg=0.000417, fill_val=0):
    """
    Make a blank image and save it to disk

    Parameters
    ----------
    image_name : str
        Filename of output image
    reference_ra_deg : float
        RA for center of output image
    reference_dec_deg : float
        Dec for center of output image
    reference_freq  : float
        Ref freq of output image
    ximsize : int, optional
        Size of output image
    yimsize : int, optional
        Size of output image
    cellsize_deg : float, optional
        Size of a pixel in degrees
    fill_val : int, optional
        Value with which to fill the image
    """
    import numpy as np
    from astropy.io import fits as pyfits

    # Make fits hdu
    # Axis order is [STOKES, FREQ, DEC, RA]
    shape_out = [1, 1, yimsize, ximsize]
    hdu = pyfits.PrimaryHDU(np.ones(shape_out, dtype=np.float32)*fill_val)
    hdulist = pyfits.HDUList([hdu])
    header = hdulist[0].header

    # Add RA, Dec info
    i = 1
    header['CRVAL{}'.format(i)] = reference_ra_deg
    header['CDELT{}'.format(i)] = -cellsize_deg
    header['CRPIX{}'.format(i)] = ximsize / 2.0
    header['CUNIT{}'.format(i)] = 'deg'
    header['CTYPE{}'.format(i)] = 'RA---SIN'
    i += 1
    header['CRVAL{}'.format(i)] = reference_dec_deg
    header['CDELT{}'.format(i)] = cellsize_deg
    header['CRPIX{}'.format(i)] = yimsize / 2.0
    header['CUNIT{}'.format(i)] = 'deg'
    header['CTYPE{}'.format(i)] = 'DEC--SIN'
    i += 1

    # Add STOKES info
    header['CRVAL{}'.format(i)] = 1.0
    header['CDELT{}'.format(i)] = 1.0
    header['CRPIX{}'.format(i)] = 1.0
    header['CUNIT{}'.format(i)] = ''
    header['CTYPE{}'.format(i)] = 'STOKES'
    i += 1

    # Add frequency info
    del_freq = 1e8
    header['RESTFRQ'] = reference_freq
    header['CRVAL{}'.format(i)] = reference_freq
    header['CDELT{}'.format(i)] = del_freq
    header['CRPIX{}'.format(i)] = 1.0
    header['CUNIT{}'.format(i)] = 'Hz'
    header['CTYPE{}'.format(i)] = 'FREQ'
    i += 1

    # Add equinox
    header['EQUINOX'] = 2000.0

    # Add telescope
    header['TELESCOP'] = 'LOFAR'

    hdulist[0].header = header
    hdulist.writeto(image_name, overwrite=True)
    hdulist.close()


def gaussian_fcn(g, x1, x2, const=False):
    """
    Evaluate a Gaussian on the given grid.

    Parameters
    ----------
    g: list
        List of Gaussian parameters:
        [peak_flux, xcen, ycen, FWHMmaj, FWHMmin, PA_E_of_N]
    x1, x2: grid (as produced by numpy.mgrid)
        Grid coordinates on which to evaluate the Gaussian
    const : bool, optional
        If True, all values are set to the peak_flux

    Returns
    -------
    img : array
        Image of Gaussian
    """
    from math import radians, sin, cos
    import numpy as np

    A, C1, C2, S1, S2, Th = g
    fwsig = 2.35482  # FWHM = fwsig * sigma
    S1 = S1 / fwsig
    S2 = S2 / fwsig
    Th = 360.0 - Th  # theta is angle E of N (ccw from +y axis)
    th = radians(Th)
    cs = cos(th)
    sn = sin(th)
    f1 = ((x1-C1)*cs + (x2-C2)*sn)/S1
    f2 = (-(x1-C1)*sn + (x2-C2)*cs)/S2
    gimg = A * np.exp(-(f1*f1 + f2*f2)/2)

    if const:
        mask = np.where(gimg/A > 1e-5)
        cimg = np.zeros(x1.shape)
        cimg[mask] = A
        return cimg
    else:
        return gimg


def tessellate(x_pix, y_pix, w, dist_pix):
    """
    Returns Voronoi tessellation vertices

    Parameters
    ----------
    x_pix : array
        Array of x pixel values for tessellation centers
    y_pix : array
        Array of y pixel values for tessellation centers
    w : WCS object
        WCS for transformation from pix to world coordinates
    dist_pix : float
        Distance in pixels from center to outer boundary of facets

    Returns
    -------
    verts : list
        List of facet vertices in (RA, Dec)
    """
    import numpy as np
    from scipy.spatial import Voronoi
    import shapely.geometry
    import shapely.ops

    # Get x, y coords for directions in pixels. We use the input calibration sky
    # model for this, as the patch positions written to the h5parm file by DPPP may
    # be different
    xy = []
    for RAvert, Decvert in zip(x_pix, y_pix):
        xy.append((RAvert, Decvert))

    # Generate array of outer points used to constrain the facets
    nouter = 64
    means = np.ones((nouter, 2)) * np.array(xy).mean(axis=0)
    offsets = []
    angles = [np.pi/(nouter/2.0)*i for i in range(0, nouter)]
    for ang in angles:
        offsets.append([np.cos(ang), np.sin(ang)])
    scale_offsets = dist_pix * np.array(offsets)
    outer_box = means + scale_offsets

    # Tessellate and clip
    points_all = np.vstack([xy, outer_box])
    vor = Voronoi(points_all)
    lines = [
        shapely.geometry.LineString(vor.vertices[line])
        for line in vor.ridge_vertices
        if -1 not in line
    ]
    polygons = [poly for poly in shapely.ops.polygonize(lines)]
    verts = []
    for poly in polygons:
        verts_xy = poly.exterior.xy
        verts_deg = []
        for x, y in zip(verts_xy[0], verts_xy[1]):
            x_y = np.array([[y, x, 0.0, 0.0]])
            ra_deg, dec_deg = w.wcs_pix2world(x_y, 0)[0][0], w.wcs_pix2world(x_y, 0)[0][1]
            verts_deg.append((ra_deg, dec_deg))
        verts.append(verts_deg)

    # Reorder to match the initial ordering
    ind = []
    for poly in polygons:
        for j, (xs, ys) in enumerate(zip(x_pix, y_pix)):
            if poly.contains(shapely.geometry.Point(xs, ys)):
                ind.append(j)
                break
    verts = [verts[i] for i in ind]

    return verts
