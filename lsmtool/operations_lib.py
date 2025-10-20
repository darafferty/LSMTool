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

from collections import namedtuple

from astropy.coordinates import Angle

NormalizedRADec = namedtuple("NormalizedRADec", ["ra", "dec"])


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
    import numpy as np
    from astropy.coordinates import ITRS, SkyCoord
    from astropy.time import Time

    obstime = Time(time / 3600 / 24, scale="utc", format="mjd")

    dir_pointing = SkyCoord(ra, dec)
    dir_pointing_itrs = dir_pointing.transform_to(ITRS(obstime=obstime))

    return np.asarray(dir_pointing_itrs.cartesian.xyz.transpose())


def apply_beam(beamMS, fluxes, RADeg, DecDeg, timeIndx=0.5, invert=False):
    """
    Returns flux attenuated by primary beam.

    Note: the attenuation is approximated using the array factor beam from the
    first station in the beam MS only (and it is assumed that this station is
    at LOFAR core). This approximation has been found to produce reasonable
    results for a typical LOFAR observation but may not work well for atypical
    observations.

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
    import astropy.units as u
    import casacore.tables as pt
    import everybeam as eb
    import numpy as np
    from astropy.coordinates import Angle

    # Determine a time stamp (in MJD) for later use, betweeen the start and end
    # times of the Measurement Set, using `timeIndx` as fractional indicator.
    tmin, tmax, ant1 = pt.taql(
        f"select gmin(TIME), gmax(TIME), gmin(ANTENNA1) from {beamMS}"
    )[0].values()

    # Constrain `timeIndx` between 0 and 1.
    timeIndx = max(0.0, min(1.0, timeIndx))
    time = tmin + (tmax - tmin) * timeIndx

    # Get frequency information from the Measurement Set.
    with pt.table(f"{beamMS}::SPECTRAL_WINDOW", ack=False) as sw:
        numchannels = sw.col("NUM_CHAN")[0]
        startfreq = np.min(sw.col("CHAN_FREQ")[0])
        channelwidth = sw.col("CHAN_WIDTH")[0][0]

    # Get the pointing direction from the Measurement Set, and convert to local
    # xyz coordinates at the LOFAR core.
    with pt.table(f"{beamMS}::FIELD", ack=False) as obs:
        pointing_ra = Angle(
            float(obs.col("REFERENCE_DIR")[0][0][0]), unit=u.rad
        )
        pointing_dec = Angle(
            float(obs.col("REFERENCE_DIR")[0][0][1]), unit=u.rad
        )
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
    import numpy as np
    from astropy.wcs import WCS

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

    import numpy as np
    from astropy import units as u
    from astropy.coordinates import Angle, SkyCoord
    from astropy.coordinates.matching import match_coordinates_sky

    if byPatch:
        RA, Dec = LSM1.getPatchPositions(asArray=True)
        catalog1 = SkyCoord(RA, Dec, unit=(u.degree, u.degree), frame="fk5")
        RA, Dec = LSM2.getPatchPositions(asArray=True)
        catalog2 = SkyCoord(RA, Dec, unit=(u.degree, u.degree), frame="fk5")
    else:
        catalog1 = SkyCoord(
            LSM1.getColValues("Ra"),
            LSM1.getColValues("Dec"),
            unit=(u.degree, u.degree),
        )
        catalog2 = SkyCoord(
            LSM2.getColValues("Ra"),
            LSM2.getColValues("Dec"),
            unit=(u.degree, u.degree),
        )
    idx, d2d, d3d = match_coordinates_sky(catalog1, catalog2)

    try:
        radius = float(radius)
    except ValueError:
        pass
    if type(radius) is float:
        radius = f"{radius} degree"
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
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    coord1 = SkyCoord(ra1, dec1, unit=(u.degree, u.degree), frame="fk5")
    coord2 = SkyCoord(ra2, dec2, unit=(u.degree, u.degree), frame="fk5")

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
        if "ReferenceFrequency" in LSM.getColNames():
            refFreq = LSM.getColValues(
                "ReferenceFrequency", aggregate=aggregate
            )
        else:
            refFreq = np.array(
                [LSM.table.meta["ReferenceFrequency"]] * len(LSM)
            )
        targetFreq = np.median(refFreq)
    fluxes = LSM.getColValues("I", aggregate=aggregate)

    try:
        alphas = LSM.getColValues("SpectralIndex", aggregate=aggregate).squeeze(
            axis=0
        )
    except (IndexError, ValueError):
        alphas = np.array([-0.8] * len(fluxes))
    nterms = alphas.shape[1]

    if "LogarithmicSI" in LSM.table.meta:
        logSI = LSM.table.meta["LogarithmicSI"]
    else:
        logSI = True

    if nterms > 1:
        for i in range(nterms):
            if logSI:
                fluxes *= 10.0 ** (
                    alphas[:, i] * (np.log10(refFreq / targetFreq)) ** (i + 1)
                )
            else:
                # stokesI + term0 (nu/refnu - 1) + term1 (nu/refnu - 1)^2 + ...
                fluxes += alphas[:, i] * ((refFreq / targetFreq) - 1.0) ** (
                    i + 1
                )
    elif logSI:
        fluxes *= 10.0 ** (alphas * np.log10(refFreq / targetFreq))
    else:
        fluxes += alphas * ((refFreq / targetFreq) - 1.0)

    return fluxes


def make_template_image(
    image_name,
    reference_ra_deg,
    reference_dec_deg,
    reference_freq,
    ximsize=512,
    yimsize=512,
    cellsize_deg=0.000417,
    fill_val=0,
):
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
    hdu = pyfits.PrimaryHDU(np.ones(shape_out, dtype=np.float32) * fill_val)
    hdulist = pyfits.HDUList([hdu])
    header = hdulist[0].header

    # Add RA, Dec info
    i = 1
    header[f"CRVAL{i}"] = reference_ra_deg
    header[f"CDELT{i}"] = -cellsize_deg
    header[f"CRPIX{i}"] = ximsize / 2.0
    header[f"CUNIT{i}"] = "deg"
    header[f"CTYPE{i}"] = "RA---SIN"
    i += 1
    header[f"CRVAL{i}"] = reference_dec_deg
    header[f"CDELT{i}"] = cellsize_deg
    header[f"CRPIX{i}"] = yimsize / 2.0
    header[f"CUNIT{i}"] = "deg"
    header[f"CTYPE{i}"] = "DEC--SIN"
    i += 1

    # Add STOKES info
    header[f"CRVAL{i}"] = 1.0
    header[f"CDELT{i}"] = 1.0
    header[f"CRPIX{i}"] = 1.0
    header[f"CUNIT{i}"] = ""
    header[f"CTYPE{i}"] = "STOKES"
    i += 1

    # Add frequency info
    del_freq = 1e8
    header["RESTFRQ"] = reference_freq
    header[f"CRVAL{i}"] = reference_freq
    header[f"CDELT{i}"] = del_freq
    header[f"CRPIX{i}"] = 1.0
    header[f"CUNIT{i}"] = "Hz"
    header[f"CTYPE{i}"] = "FREQ"
    i += 1

    # Add equinox
    header["EQUINOX"] = 2000.0

    # Add telescope
    header["TELESCOP"] = "LOFAR"

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
    from math import cos, radians, sin

    import numpy as np

    A, C1, C2, S1, S2, Th = g
    fwsig = 2.35482  # FWHM = fwsig * sigma
    S1 = S1 / fwsig
    S2 = S2 / fwsig
    Th = 360.0 - Th  # theta is angle E of N (ccw from +y axis)
    th = radians(Th)
    cs = cos(th)
    sn = sin(th)
    f1 = ((x1 - C1) * cs + (x2 - C2) * sn) / S1
    f2 = (-(x1 - C1) * sn + (x2 - C2) * cs) / S2
    gimg = A * np.exp(-(f1 * f1 + f2 * f2) / 2)

    if const:
        mask = np.where(gimg / A > 1e-5)
        cimg = np.zeros(x1.shape)
        cimg[mask] = A
        return cimg
    return gimg
