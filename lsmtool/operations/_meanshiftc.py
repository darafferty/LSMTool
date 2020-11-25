# -*- coding: utf-8 -*-
#
# Defines meanshift grouper functions used by the group operation. Based on
# implementation of Francesco de Gasperin (see
# https://github.com/revoltek/LiLF/blob/master/lib_dd.py)
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
import numpy as np
from . import _grouper
import logging
log = logging.getLogger('LSMTool.Group')


class Grouper(object):
    """
    Class to group a list of coordinates and fluxes by the meanshift algorighm.
    Based on: http://www.chioka.in/meanshift-algorithm-for-the-rest-of-us-python/
    Parameters
    ----------
    coords : list
        List of (x, y) pixel coordinates for source positions
    fluxes: list
        total flux density for each source
    kernel_size : float
        Kernel size in pixels
    n_iterations : int
        Number of iterations
    look_distance : float
        Look distance in pixels
    grouping_distance : float
        Grouping distance in pixels
    """
    def __init__(self, coords, fluxes, kernel_size, n_iterations, look_distance,
                 grouping_distance):
        self.coords = np.array(coords)
        self.fluxes = fluxes
        self.kernel_size = kernel_size
        self.n_iterations = n_iterations
        self.look_distance = look_distance
        self.grouping_distance = grouping_distance
        self.past_coords = [np.copy(self.coords)]
        self.clusters = []
        self.g = _grouper.Grouper()
        self.g.readCoordinates(self.coords, self.fluxes)
        self.g.setKernelSize(self.kernel_size)
        self.g.setNumberOfIterations(self.n_iterations)
        self.g.setLookDistance(self.look_distance)
        self.g.setGroupingDistance(self.grouping_distance)


    def run(self):
        self.g.run()

    def grouping(self):
        """
        Take the last coords set and group sources nearby, then return a list of lists.
        Each list has the index of one cluster.
        """

        self.g.group(self.clusters)

        log.info('Creating %i groups.' % len(self.clusters))
        return self.clusters
