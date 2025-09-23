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

from astropy.coordinates import Angle
from collections import namedtuple
from math import floor, ceil
import numpy as np
import scipy as sp

from lsmtool.constants import WCS_ORIGIN

NormalizedRADec = namedtuple('NormalizedRADec', ['ra', 'dec'])


def normalize_ra_dec(ra, dec):
    """
    Normalize RA to be in the range [0, 360) and Dec to be in the
    range [-90, 90].

    Parameters
    ----------
    ra : float or astropy.coordinates.Angle
        The RA in degrees to be normalized.
    dec : float or astropy.coordinates.Angle
        The Dec in degrees to be normalized.

    Returns
    -------
    normalized_ra_dec : NormalizedRADec
        The normalized RA in degrees in the range [0, 360) and the
        Dec in degrees in the range [-90, 90], with the following
        elements:

            - NormalizedRADec.ra: RA in degrees
            - NormalizedRADec.dec: Dec in degrees
    """
    ra = ra.value if type(ra) is Angle else ra
    dec = dec.value if type(dec) is Angle else dec
    normalized_dec = (dec + 180) % 360 - 180
    normalized_ra = ra % 360
    if abs(normalized_dec) > 90:
        normalized_dec = 180 - normalized_dec
        normalized_ra = normalized_ra + 180
        normalized_dec = (normalized_dec + 180) % 360 - 180
        normalized_ra = normalized_ra % 360

    return NormalizedRADec(normalized_ra, normalized_dec)


def radec_to_xyz(ra, dec, time):
    """
    Convert RA and Dec ICRS coordinates to ITRS cartesian coordinates.

    Parameters
    ----------
    ra : astropy.coordinates.Angle
        Right ascension
    dec: astropy.coordinates.Angle
        Declination
    time: float
        MJD time in seconds

    Returns
    -------
    pointing_xyz: numpy.ndarray
        NumPy array containing the ITRS X, Y and Z coordinates
    """
    from astropy.coordinates import SkyCoord, ITRS
    from astropy.time import Time
    import numpy as np

    obstime = Time(time/3600/24, scale='utc', format='mjd')

    dir_pointing = SkyCoord(ra, dec)
    dir_pointing_itrs = dir_pointing.transform_to(ITRS(obstime=obstime))

    return np.asarray(dir_pointing_itrs.cartesian.xyz.transpose())


def apply_beam(beamMS, fluxes, RADeg, DecDeg, timeIndx=0.5, invert=False):
    """
    Returns flux attenuated by primary beam.

    Note: the attenuation is approximated using the array factor beam from the
    first station in the beam MS only (and it is assumed that this station is at
    LOFAR core). This approximation has been found to produce reasonable results
    for a typical LOFAR observation but may not work well for atypical observations.

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
    timeIndx : float (between 0 and 1), optional
        Time as fraction of that covered by the beamMS for which the beam is
        calculated
    invert : bool, optional
        If True, invert the beam (i.e. to un-attenuate the flux)

    Returns
    -------
    attFluxes : numpy.ndarray
        Attenuated fluxes

    """
    import everybeam as eb
    import numpy as np
    import casacore.tables as pt
    from astropy.coordinates import Angle
    import astropy.units as u

    # Determine a time stamp (in MJD) for later use, betweeen the start and end
    # times of the Measurement Set, using `timeIndx` as fractional indicator.
    tmin, tmax, ant1 = pt.taql(f"select gmin(TIME), gmax(TIME), gmin(ANTENNA1) from {beamMS}")[0].values()

    # Constrain `timeIndx` between 0 and 1.
    timeIndx = max(0., min(1., timeIndx))
    time = tmin + (tmax - tmin) * timeIndx

    # Get frequency information from the Measurement Set.
    with pt.table(beamMS+'::SPECTRAL_WINDOW', ack=False) as sw:
        numchannels = sw.col('NUM_CHAN')[0]
        startfreq = np.min(sw.col('CHAN_FREQ')[0])
        channelwidth = sw.col('CHAN_WIDTH')[0][0]

    # Get the pointing direction from the Measurement Set, and convert to local
    # xyz coordinates at the LOFAR core.
    with pt.table(beamMS+'::FIELD', ack=False) as obs:
        pointing_ra = Angle(float(obs.col('REFERENCE_DIR')[0][0][0]), unit=u.rad)
        pointing_dec = Angle(float(obs.col('REFERENCE_DIR')[0][0][1]), unit=u.rad)
    pointing_xyz = radec_to_xyz(pointing_ra, pointing_dec, time)

    # Convert the source directions to local xyz coordinates at the LOFAR core.
    source_ra = Angle(RADeg, unit=u.deg)
    source_dec = Angle(DecDeg, unit=u.deg)
    source_xyz = radec_to_xyz(source_ra, source_dec, time)

    # Load the beam model.
    sr = eb.load_telescope(beamMS)

    # Evaluate beam for the center frequency only, ...
    freq = startfreq + (numchannels // 2) * channelwidth
    beam = abs(sr.array_factor(time, ant1, freq, source_xyz, pointing_xyz))
    # ...take XX only (XX and YY should be equal) and square it, ...
    beam = beam[:, 0, 0] ** 2
    # ...and invert if necessary.
    if invert:
        beam = 1 / beam

    # Return fluxes attenuated by the beam.
    return fluxes * beam


def make_wcs(refRA, refDec, crdelt=None):
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
    w : astropy.wcs.WCS
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
    return w


def matchSky(LSM1, LSM2, radius=0.1, byPatch=False, nearestOnly=False):
    """
    Matches two sky models by position.

    Parameters
    ----------
    LSM1 : lsmtool.skymodel.SkyModel
        Sky model for which match indices are desired
    LSM2 : lsmtool.skymodel.SkyModel
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
    matches1, matches2 : numpy.ndarray
        matches1 is the array of indices of LSM1 that have matches in LSM2
        within the specified radius. matches2 is the array of indices of LSM2
        for the same sources.

    """

    from astropy.coordinates import SkyCoord, Angle
    from astropy.coordinates.matching import match_coordinates_sky
    from astropy import units as u
    import numpy as np

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
    ra1 : float or numpy.ndarray
        RA of coordinate 1 in degrees
    dec1 : float or numpy.ndarray
        Dec of coordinate 1 in degrees
    ra2 : float
        RA of coordinate 2 in degrees
    dec2 : float
        Dec of coordinate 2 in degrees

    Returns
    -------
    separation : astropy.coordinates.Angle or numpy.ndarray
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
    LSM : lsmtool.skymodel.SkyModel
        RA of coordinate 1 in degrees
    targetFreq : float, optional
        Frequency in Hz. If None, the median is used
    aggregate : str, optional
        Aggregation to use

    Returns
    -------
    fluxes : numpy.ndarray
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
    x1, x2: numpy.ndarray
        Grid coordinates on which to evaluate the Gaussian (as produced by
        :py:data:`numpy.mgrid`)
    const : bool, optional
        If True, all values are set to the peak_flux

    Returns
    -------
    img : numpy.ndarray
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


def tessellate(ra_cal, dec_cal, ra_mid, dec_mid, width_ra, width_dec):
    """
    Makes a Voronoi tessellation and returns the resulting facet centers
    and polygons.

    Parameters
    ----------
    ra_cal : numpy.ndarray
        RA values in degrees of calibration directions.
    dec_cal : numpy.ndarray
        Dec values in degrees of calibration directions.
    ra_mid : float
        RA in degrees of bounding box center.
    dec_mid : float
        Dec in degrees of bounding box center.
    width_ra : float
        Width of bounding box in RA in degrees, corrected to Dec = 0.
    width_dec : float
        Width of bounding box in Dec in degrees.

    Returns
    -------
    facet_points : list of tuple
        List of facet points (centers) as (RA, Dec) tuples in degrees.
    facet_polys : list of numpy.ndarray
        List of facet polygons (vertices) as [RA, Dec] arrays in degrees
        (each of shape N x 2, where N is the number of vertices in a given
        facet).
    """
    # Build the bounding box corner coordinates
    if width_ra <= 0.0 or width_dec <= 0.0:
        raise ValueError('The RA/Dec width cannot be zero or less')

    wcs_pixel_scale = 20.0 / 3600.0  # 20"/pixel
    wcs = make_wcs(ra_mid, dec_mid, wcs_pixel_scale)
    x_cal, y_cal = wcs.wcs_world2pix(ra_cal, dec_cal, WCS_ORIGIN)
    x_mid, y_mid = wcs.wcs_world2pix(ra_mid, dec_mid, WCS_ORIGIN)
    width_x = width_ra / wcs_pixel_scale / 2.0
    width_y = width_dec / wcs_pixel_scale / 2.0
    bounding_box = np.array([x_mid - width_x, x_mid + width_x,
                             y_mid - width_y, y_mid + width_y])

    # Tessellate and convert resulting facet polygons from (x, y) to (RA, Dec)
    vor = voronoi(np.stack((x_cal, y_cal)).T, bounding_box)
    facet_polys = []
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        ra, dec = wcs.wcs_pix2world(vertices[:, 0], vertices[:, 1], WCS_ORIGIN)
        vertices = np.stack((ra, dec)).T
        facet_polys.append(vertices)
    facet_points = list(map(
        tuple, wcs.wcs_pix2world(vor.filtered_points, WCS_ORIGIN)
    ))
    return facet_points, facet_polys


def voronoi(cal_coords, bounding_box):
    """
    Produces a Voronoi tessellation for the given coordinates and bounding box

    Parameters
    ----------
    cal_coords : numpy.ndarray
        Array of x, y coordinates
    bounding_box : numpy.ndarray
        Array defining the bounding box as [minx, maxx, miny, maxy]

    Returns
    -------
    vor : scipy.spatial.Voronoi
        The resulting Voronoi object
    """
    eps = 1e-6

    # Select calibrators inside the bounding box
    inside_ind = np.logical_and(np.logical_and(bounding_box[0] <= cal_coords[:, 0],
                                               cal_coords[:, 0] <= bounding_box[1]),
                                np.logical_and(bounding_box[2] <= cal_coords[:, 1],
                                               cal_coords[:, 1] <= bounding_box[3]))
    points_center = cal_coords[inside_ind, :]

    # Mirror points
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)

    # Compute Voronoi, sorting the output regions to match the order of the
    # input coordinates
    vor = sp.spatial.Voronoi(points)
    sorted_regions = np.array(vor.regions, dtype=object)[np.array(vor.point_region)]
    vor.regions = sorted_regions.tolist()

    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not (bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
                        bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
                    flag = False
                    break
        if region and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions

    return vor
