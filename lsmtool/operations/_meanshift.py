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

    def euclid_distance(self, coord, coords):
        """
        Simple distance from coord to all coords
        """
        return np.sqrt(np.sum((coord - coords)**2, axis=1))

    def neighbourhood_points(self, centroid, coords, max_distance):
        """
        Find close points, this reduces the load
        """
        distances = self.euclid_distance(centroid, coords)
        return np.where(distances < max_distance)

    def gaussian_kernel(self, distance):
        """
        Defines a Gaussian kernel
        """
        return (1/(self.kernel_size*np.sqrt(2*np.pi))) * np.exp(-0.5*((distance / self.kernel_size))**2)

    def run(self):
        """
        Run the algorithm
        """
        for it in range(self.n_iterations):
            log.info("Starting iteration %i" % it)

            for i, x in enumerate(self.coords):
                # Step 1. For each datapoint x in X, find the neighbouring points N(x) of x.
                idx_neighbours = self.neighbourhood_points(x, self.coords, max_distance=self.look_distance)

                # Step 2. For each datapoint x in X, calculate the mean shift m(x).
                distances = self.euclid_distance(self.coords[idx_neighbours], x)
                weights = self.gaussian_kernel(distances)
                weights *= self.fluxes[idx_neighbours]
                numerator = np.sum(weights[:, np.newaxis] * self.coords[idx_neighbours], axis=0)
                denominator = np.sum(weights)
                new_x = numerator / denominator

                # Step 3. For each datapoint x in X, update x <- m(x).
                self.coords[i] = new_x

            self.past_coords.append(np.copy(self.coords))

            # if things change little, break
            if it > 1 and np.max(self.euclid_distance(self.coords, self.past_coords[-2])) < self.grouping_distance/2.0:
                break

    def grouping(self):
        """
        Take the last coords set and group sources nearby, then return a list of lists.
        Each list has the index of one cluster.
        """
        coords_to_check = np.copy(self.coords)
        while len(coords_to_check) > 0:
            idx_cluster = self.neighbourhood_points(coords_to_check[0], self.coords, max_distance=self.grouping_distance)
            idx_cluster_to_remove = self.neighbourhood_points(coords_to_check[0], coords_to_check, max_distance=self.grouping_distance)

            # remove all coords of this clusters from the global list
            mask = np.ones(coords_to_check.shape[0], dtype=bool)
            mask[idx_cluster_to_remove] = False
            coords_to_check = coords_to_check[mask]

            # save this cluster indexes
            self.clusters.append(idx_cluster)

        log.info('Creating %i groups.' % len(self.clusters))
        return self.clusters
