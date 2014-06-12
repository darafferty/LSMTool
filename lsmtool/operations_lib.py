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


def attenuate(beamMS, fluxes, RADeg, DecDeg):
    """
    Returns flux attenuated by primary beam.
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

    tt = t.query('ANTENNA1==0 AND ANTENNA2==1', columns='TIME')
    time = tt.getcol("TIME")
    time = min(time) + ( max(time) - min(time) ) / 2.
    t.close()

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
        # Use station 0 to compute the beam and get mid channel
        sr.setDirection(RA*np.pi/180., Dec*np.pi/180.)
        beam = sr.evaluateStation(time, 0)
        r = abs(beam[int(len(beam)/2.)])
        beam = ( r[0][0] + r[1][1] ) / 2.
        attFluxes.append(flux * beam)

    return np.array(attFluxes)


def radec2xy(RA, Dec, maxRA=None, minDec=None):
    """Returns x, y for input ra, dec
    """
    from astropy.wcs import WCS
    import numpy as np

    x = []
    y = []
    if maxRA is None:
        maxRA = np.max(RA)
    if minDec is None:
        minDec = np.min(Dec)

    # Make wcs object to handle transformation from ra and dec to pixel coords.
    w = WCS(naxis=2)
    w.wcs.crpix = [0, 0]
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.crval = [maxRA, minDec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.set_pv([(2, 1, 45.0)])

    for ra_deg, dec_deg in zip(RA, Dec):
        ra_dec = np.array([[ra_deg, dec_deg]])
        x.append(w.wcs_world2pix(ra_dec, 0)[0][0])
        y.append(w.wcs_world2pix(ra_dec, 0)[0][1])

    return x, y


def xy2radec(x, y, maxRA=0.0, minDec=0.0):
    """Returns x, y for input ra, dec
    """
    from astropy.wcs import WCS
    import numpy as np

    RA = []
    Dec = []

    # Make wcs object to handle transformation from ra and dec to pixel coords.
    w = WCS(naxis=2)
    w.wcs.crpix = [0, 0]
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.crval = [maxRA, minDec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.set_pv([(2, 1, 45.0)])

    for xp, yp in zip(x, y):
        x_y = np.array([[xp, yp]])
        RA.append(w.wcs_pix2world(x_y, 0)[0][0])
        Dec.append(w.wcs_pix2world(x_y, 0)[0][1])

    return RA, Dec
