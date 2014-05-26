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


def applyBeam(beamMS, fluxes, RADeg, DecDeg):
    """
    Returns flux attenuated by primary beam.
    """
    import numpy as np

    try:
        import pyrap.tables as pt
    except ImportError:
        logger.error('Could not import pyrap.tables')
    try:
        import lofar.stationresponse as lsr
    except ImportError:
        logger.error('Could not import lofar.stationresponse')

    t = pt.table(beamMS, ack=False)
    tt = t.query('ANTENNA1==0 AND ANTENNA2==1', columns='TIME')
    time = tt.getcol("TIME")
    time = min(time) + ( max(time) - min(time) ) / 2.
    t.close()

    attFluxes = []
    sr = lsr.stationresponse(beamMS, False, True)
    for flux, RA, Dec in zip(fluxes, RADeg, DecDeg):
        # Use station 0 to compute the beam and get mid channel
        sr.setDirection(RA*np.pi/180., Dec*np.pi/180.)
        beam = sr.evaluateStation(time, 0)
        r = abs(beam[int(len(beam)/2.)])
        beam = ( r[0][0] + r[1][1] ) / 2.
        attFluxes.append(flux * beam)

    return np.array(attFluxes)
