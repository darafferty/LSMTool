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
import sys
import os


def attenuate(beamMS, fluxes, RADeg, DecDeg):
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

    Returns
    -------
    attFluxes : numpy array

    """
    import numpy as np

    try:
        import pyrap.tables as pt
    except ImportError:
        logging.error('Could not import pyrap.tables')
        return None
    try:
        import lofar.stationresponse as lsr
    except ImportError:
        logging.error('Could not import lofar.stationresponse')

    try:
        t = pt.table(beamMS, ack=False)
    except:
        raise Exception('Could not open {0}'.format(beamMS))

    time = None
    ant1 = -1
    ant2 = 1
    while time is None:
        ant1 += 1
        ant2 += 1
        tt = t.query('ANTENNA1=={0} AND ANTENNA2=={1}'.format(ant1, ant2), columns='TIME')
        time = tt.getcol("TIME")
    t.close()
    time = min(time) + ( max(time) - min(time) ) / 2.

    attFluxes = []
    sr = lsr.stationresponse(beamMS, inverse=False, useElementResponse=False,
        useArrayFactor=True, useChanFreq=False)
    if type(fluxes) is not list:
        fluxes = list(fluxes)
    if type(RADeg) is not list:
        RADeg= list(RADeg)
    if type(DecDeg) is not list:
        DecDeg = list(DecDeg)

    for flux, RA, Dec in zip(fluxes, RADeg, DecDeg):
        # Use ant1, mid time, and mid channel to compute the beam
        sr.setDirection(RA*np.pi/180., Dec*np.pi/180.)
        beam = sr.evaluateStation(time, ant1)
        r = abs(beam[int(len(beam)/2.)])
        beam = ( r[0][0] + r[1][1] ) / 2.
        attFluxes.append(flux * beam)

    return np.array(attFluxes)


def radec2xy(RA, Dec, refRA=None, refDec=None):
    """Returns x, y for input ra, dec.

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

    Returns
    -------
    x, y : list, list
        Lists of x and y pixel values corresponding to the input RA and Dec
        values

    """
    from astropy.wcs import WCS
    import numpy as np

    x = []
    y = []
    if refRA is None:
        refRA = RA[0]
    if refDec is None:
        refDec = Dec[0]

    # Make wcs object to handle transformation from ra and dec to pixel coords.
    w = WCS(naxis=2)
    w.wcs.crpix = [1000, 1000]
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.crval = [refRA, refDec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.set_pv([(2, 1, 45.0)])

    for ra_deg, dec_deg in zip(RA, Dec):
        ra_dec = np.array([[ra_deg, dec_deg]])
        x.append(w.wcs_world2pix(ra_dec, 0)[0][0])
        y.append(w.wcs_world2pix(ra_dec, 0)[0][1])

    return x, y


def xy2radec(x, y, refRA=0.0, refDec=0.0):
    """Returns x, y for input ra, dec.

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

    Returns
    -------
    RA, Dec : list, list
        Lists of RA and Dec values corresponding to the input x and y pixel
        values

     """
    from astropy.wcs import WCS
    import numpy as np

    RA = []
    Dec = []

    # Make wcs object to handle transformation from ra and dec to pixel coords.
    w = WCS(naxis=2)
    w.wcs.crpix = [1000, 1000]
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.crval = [refRA, refDec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.set_pv([(2, 1, 45.0)])

    for xp, yp in zip(x, y):
        x_y = np.array([[xp, yp]])
        RA.append(w.wcs_pix2world(x_y, 0)[0][0])
        Dec.append(w.wcs_pix2world(x_y, 0)[0][1])

    return RA, Dec
