# -*- coding: utf-8 -*-
#
# Defines cluster functions used by the group operation
#
# Copyright (C) 2013 - Reinout van Weeren
# Copyright (C) 2013 - Francesco de Gasperin
# Modified by David Rafferty as required for integration into LSMTool
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

import os
import sys
import numpy as np
import logging
import itertools

class Patch():
    def __init__(self, name, ra, dec, flux):
        self.name = name
        self.ra = ra
        self.dec = dec
        self.flux = flux

class Cluster():
    def __init__(self, name, patch):
        self.name = name
        self.init_patch = patch
        self.patches = []
        self.update_cluster_coords()

    def add_patch(self, patch):
        """
        Add a patch to this cluster
        """
        self.patches.append(patch)

    def total_cluster_flux(self):
        """
        Return total cluster flux
        """
        cluster_flux = self.init_patch.flux
        for patch in self.patches:
            cluster_flux += patch.flux

        return cluster_flux

    def update_cluster_coords(self):
        """
        update the self.centroid_ra, self.centroid_dec, self.mid_ra, self.mid_dec
        """
        self.centroid_ra = self.init_patch.ra*self.init_patch.flux
        self.centroid_dec = self.init_patch.dec*self.init_patch.flux
        min_ra = np.inf
        max_ra = -np.inf
        min_dec = np.inf
        max_dec = -np.inf

        for patch in self.patches:
            self.centroid_ra += patch.ra*patch.flux
            self.centroid_dec += patch.dec*patch.flux
            if patch.ra < min_ra: min_ra = patch.ra
            if patch.ra > max_ra: max_ra = patch.ra
            if patch.dec < min_dec: min_dec = patch.dec
            if patch.dec > max_dec: max_dec = patch.dec

        self.centroid_ra /= self.total_cluster_flux()
        self.centroid_dec /= self.total_cluster_flux()
        self.mid_ra = min_ra + (max_ra - min_ra)/2.
        self.mid_dec = min_dec + (max_dec - min_dec)/2.


def ratohms_string(ra):
    rah, ram, ras = ratohms(ra)
    return str(rah) + ':' + str(ram) + ':' +str(round(ras,2))


def dectodms_string(dec):
    decd, decm, decs = dectodms(dec)
    return str(decd) + '.' + str(decm) + '.' +str(round(decs,2))


def compute_patch_center(LSM, applyBeam=False):
    """
    Return the patches names, central (weighted) RA and DEC and total flux
    """
    data = LSM.table
    patch_names = np.unique(data['Name'])
    fluxes = LSM.getColValues('I', applyBeam=applyBeam)
    patches = []

    for patch_name in patch_names:
        comp_ids = np.where(data['Name'] == patch_name)[0]

        patch_ra    = 0.
        patch_dec   = 0.
        weights_ra  = 0.
        weights_dec = 0.
        patch_flux  = 0.
        for comp_id in comp_ids:

            comp_ra   = data['Ra'][comp_id]
            comp_dec  = data['Dec'][comp_id]
            comp_flux = fluxes[comp_id]

            # calculate the average weighted patch center, and patch flux
            patch_flux  += comp_flux
            patch_ra    += comp_ra * comp_flux
            patch_dec   += comp_dec * comp_flux
            weights_ra  += comp_flux
            weights_dec += comp_flux

        patches.append(Patch(patch_name, patch_ra/weights_ra, patch_dec/weights_dec, patch_flux))

    return patches


def angsep2(ra1, dec1, ra2, dec2):
    """Returns angular separation between two coordinates (all in degrees)"""
    from astropy.coordinates import FK5
    import astropy.units as u

    coord1 = FK5(ra1, dec1, unit=(u.degree, u.degree))
    coord2 = FK5(ra2, dec2, unit=(u.degree, u.degree))

    return coord1.separation(coord2)


def create_clusters(LSM, patches_orig, Q, applyBeam=False, root='Patch', pad_index=False):
    """
    Clusterize all the patches of the skymodel iteratively around the brightest patches
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from distutils.version import StrictVersion
    import scipy
    log = logging.getLogger('LSMTool.Cluster')
    if StrictVersion(scipy.__version__) < StrictVersion('0.11.0'):
        log.debug('The installed version of SciPy contains a bug that affects catalog matching. '
            'Falling back on (slower) matching script.')
        from ._matching import match_coordinates_sky
    else:
        from astropy.coordinates.matching import match_coordinates_sky

    # sort the patches by brightest first
    idx = np.argsort([patch.flux for patch in patches_orig])[::-1] # -1 to reverse sort
    patches = list(np.array(patches_orig)[idx])

    # initialize clusters with the brightest patches
    clusters = []
    for i, patch in enumerate(patches[0:Q]):
        if pad_index:
            clusters.append(Cluster(root+'_'+str(i).zfill(int(np.ceil(np.log10(Q)))), patch))
        else:
            clusters.append(Cluster(root+'_'+str(i), patch))

    # Iterate until no changes in which patch belongs to which cluster
    count = 1
    patches_seq_old = []
    patchRAs = []
    patchDecs = []
    for patch in patches:
        patchRAs.append(patch.ra)
        patchDecs.append(patch.dec)

    while True:
        clusterRAs = []
        clusterDecs = []
        if LSM.hasPatches:
            clusterRA, clusterDec = LSM.getPatchPositions(method='wmean',
                asArray=True, applyBeam=applyBeam, perPatchProjection=False)
            clusterNames = LSM.getPatchNames()
            patches_orig = LSM.getColValues('Name')
        else:
            clusterRA = [cluster.centroid_ra for cluster in clusters]
            clusterDec = [cluster.centroid_dec for cluster in clusters]
            clusterNames = [cluster.name for cluster in clusters]
        for cluster in clusters:
            # reset patches
            if type(clusterNames) is not list:
                clusterNames = clusterNames.tolist()
            cindx = clusterNames.index(cluster.name)
            cluster.patches = []
            clusterRAs.append(clusterRA[cindx])
            clusterDecs.append(clusterDec[cindx])

        catalog1 = SkyCoord(clusterRAs, clusterDecs,
            unit=(u.degree, u.degree), frame='fk5')
        catalog2 = SkyCoord(patchRAs, patchDecs,
            unit=(u.degree, u.degree), frame='fk5')
        matchIdx, d2d, d3d = match_coordinates_sky(catalog2, catalog1)

        for i, patch in zip(matchIdx, patches):
            cluster = clusters[i]
            cluster.add_patch(patch)

        patches_seq = []
        for cluster in clusters:
            patches_seq.extend(cluster.patches)

        count += 1
        if patches_seq == patches_seq_old:
            break
        patches_seq_old = patches_seq

        # Make output patch column
        patchNames = [''] * len(patches)
        patchNames_orig = LSM.getColValues('Name').tolist()
        for c in clusters:
            for p in c.patches:
                patchNames[patchNames_orig.index(p.name)] = c.name

        LSM.setColValues('Patch', patchNames, index=2)
        LSM._updateGroups()
    return np.array(patchNames)

