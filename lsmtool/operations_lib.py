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
    """Returns x, y for input ra, dec
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
    """Returns x, y for input ra, dec
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


class StatusBar():
    # class variables:
    # max:  number of total items to be completed
    # pos:  number of completed items
    # spin_pos: current position in array of busy_chars
    # inc:  amount of items to increment completed 'pos' by
    #           (shared resource)
    # comp: amount of '=' to display in the progress bar
    # started: whether or not the statusbar has been started
    # color: color of text
    def __init__(self, text, pos=0, max=100, color='\033[0m'):
        self.text = text
        self.pos = pos
        self.max = max
        self.busy_char = '|'
        self.spin_pos = 0
        self.inc =  0
        self.started = 0
        self.color = color
        self.__getsize()
        if max > 0:
            self.comp = int(float(self.pos) / self.max * self.columns)
        else:
            self.comp = 0

    # find number of columns in terminal
    def __getsize(self):
        try:
            rows, columns = getTerminalSize()
        except ValueError:
            rows = columns = 0
        if int(columns) > self.max + 2 + 44 + (len(str(self.max))*2 + 2):
            self.columns = self.max
        else:
            # note: -2 is for brackets, -44 for 'Fitting islands...' text, rest is for pos/max text
            self.columns = int(columns) - 2 - 44 - (len(str(self.max))*2 + 2)
        return

    # redraw progress bar
    def __print(self):
        self.__getsize()

        sys.stdout.write('\x1b[1G')
        if self.max == 0:
            sys.stdout.write(self.color + self.text + '[] 0/0\033[0m\n')
        else:
            sys.stdout.write(self.color + self.text + '[' + '=' * self.comp + self.busy_char + '-'*(self.columns - self.comp - 1) + '] ' + str(self.pos) + '/' + str(self.max) + '\033[0m')
            sys.stdout.write('\x1b[' + str(self.comp + 2 + 44) + 'G')
        sys.stdout.flush()
        return

    # spin the spinner by one increment
    def spin(self):
        busy_chars = ['|','/','-','\\']
        self.spin_pos += 1
        if self.spin_pos >= len(busy_chars):
            self.spin_pos = 0
        # display the busy spinning icon
        self.busy_char = busy_chars[self.spin_pos]
        sys.stdout.write(self.color + busy_chars[self.spin_pos] + '\x1b[1D' + '\033[0m')
        sys.stdout.flush()

    # increment number of completed items
    def increment(self):
        self.inc = 1
        if (self.pos + self.inc) >= self.max:
            self.pos = self.max
            self.comp = self.columns
            self.busy_char = ''
            self.__print()
            return 0
        else:
            self.pos += self.inc
            self.inc = 0
            self.spin()
            self.comp = int(float(self.pos) / self.max \
                * self.columns)
            self.__print()
        return 1

    def start(self):
        self.started = 1
        self.__print()

    def stop(self):
        if self.started:
            self.pos = self.max
            self.comp = self.columns
            self.busy_char = ''
            self.__print()
            sys.stdout.write('\n')
            self.started = 0
            return 0


def getTerminalSize():
    """
    returns (lines:int, cols:int)
    """
    import os, struct
    def ioctl_GWINSZ(fd):
        import fcntl, termios
        return struct.unpack("hh", fcntl.ioctl(fd, termios.TIOCGWINSZ, "1234"))
    # try stdin, stdout, stderr
    for fd in (0, 1, 2):
        try:
            return ioctl_GWINSZ(fd)
        except:
            pass
    # try os.ctermid()
    try:
        fd = os.open(os.ctermid(), os.O_RDONLY)
        try:
            return ioctl_GWINSZ(fd)
        finally:
            os.close(fd)
    except:
        pass
    # try `stty size`
    try:
        return tuple(int(x) for x in os.popen("stty size", "r").read().split())
    except:
        pass
    # try environment variables
    try:
        return tuple(int(os.getenv(var)) for var in ("LINES", "COLUMNS"))
    except:
        pass
    # Give up. return 0.
    return (0, 0)
