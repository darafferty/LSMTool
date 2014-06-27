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
from .skymodel import SkyModel


def load(fileName, beamMS=None):
    """
    Loads a sky model from a file and returns a SkyModel object.

    Parameters
    ----------
    fileName : str
        Input ASCII file from which the sky model is read. Must
        respect the makesourcedb format
    beamMS : str, optional
        Measurement set from which the primary beam will be estimated. A
        column of attenuated Stokes I fluxes will be added to the table.

    Returns
    -------
    SkyModel object
        A SkyModel object that stores the sky model and provides methods for
        accessing it.

    Examples
    --------
    Load a sky model into a SkyModel object::

        >>> import lsmtool
        >>> s = lsmtool.load('sky.model')

    Load a sky model with a beam MS so that apparent fluxes will
    be available (in addition to intrinsic fluxes)::

        >>> s = lsmtool.load('sky.model', 'SB100.MS')

    """
    return SkyModel(fileName, beamMS)

