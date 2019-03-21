# -*- coding: utf-8 -*-
#
# This module initializes the LSMTool module
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

"""The load() convenience function is used to load a sky model file into a
SkyModel object.

.. moduleauthor:: David Rafferty <drafferty@hs.uni-hamburg.de>

"""

from ._version import changelog
from . import _logging as logger
logger.setLevel('info')


def load(fileName, beamMS=None, VOPosition=None, VORadius=None):
    """
    Loads a sky model from a file or VO service and returns a SkyModel object.

    Parameters
    ----------
    fileName : str
        Input ASCII file from which the sky model is read (must respect the
        makesourcedb format), name of VO service to query (must be one of
        'WENSS', 'NVSS', 'TGSS', or 'GSM'), or dict (single source only)
    beamMS : str, optional
        Measurement set from which the primary beam will be estimated. A
        column of attenuated Stokes I fluxes will be added to the table
    VOPosition : list of floats
        A list specifying a new position as [RA, Dec] in either makesourcedb
        format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
        [123.2312, 23.3422]) for a cone search
    VORadius : float or str, optional
        Radius in degrees (if float) or 'value unit' (if str; e.g.,
        '30 arcsec') for cone search region in degrees

    Returns
    -------
    SkyModel object
        A SkyModel object that stores the sky model and provides methods for
        accessing it.

    Examples
    --------
    Load a sky model from a makesourcedb-formated file::

        >>> import lsmtool
        >>> s = lsmtool.load('sky.model')

    Load a sky model with a beam MS so that apparent fluxes will
    be available (in addition to intrinsic fluxes)::

        >>> s = lsmtool.load('sky.model', 'SB100.MS')

    Load a sky model from the WENSS using all sources within 5 degrees of the
    position RA = 212.8352792, Dec = 52.202644::

        >>> s = lsmtool.load('WENSS', VOPosition=[212.8352792, 52.202644],
            VOradius=5.0)

    Load a sky model from a dictionary defining the source::

        >>> source = {'Name':'src1', 'Type':'POINT', 'Ra':'12:32:10.1',
            'Dec':'23.43.21.21', 'I':2.134}
        >>> s = lsmtool.load(source)

    """
    from .skymodel import SkyModel

    return SkyModel(fileName, beamMS=beamMS, VOPosition=VOPosition,
        VORadius=VORadius)
