"""
Module that holds functions and classes related to faceting.
"""

import numpy as np
import scipy as sp

from lsmtool.constants import WCS_ORIGIN, WCS_PIXEL_SCALE
from lsmtool.operations_lib import make_wcs


def tessellate(
    ra_cal,
    dec_cal,
    ra_mid,
    dec_mid,
    width_ra,
    width_dec,
    wcs_pixel_scale=WCS_PIXEL_SCALE,
):
    """
    Make a Voronoi tessellation.

    This function partitions an image region using Voronoi tessellation seeded
    with the input calibration directions. It filters points that fall
    outside the given dimensions of the bounding box and returns the facet
    centers and polygons that enscribe these points in celestial coordinates.

    Parameters
    ----------
    ra_cal : numpy.ndarray
        RA values in degrees of calibration directions.
    dec_cal : numpy.ndarray
        Dec values in degrees of calibration directions.
    ra_mid : float
        RA in degrees of bounding box center.
    dec_mid : float
        Dec in degrees of bounding box center.
    width_ra : float
        Width of bounding box in RA in degrees, corrected to Dec = 0.
    width_dec : float
        Width of bounding box in Dec in degrees.
    wcs_pixel_scale : float
        The pixel scale to use for the conversion to pixel coordinates in
        degrees per pixel.

    Returns
    -------
    facet_points : list of tuple
        List of facet points (centers) as (RA, Dec) tuples in degrees.
    facet_polys : list of numpy.ndarray
        List of facet polygons (vertices) as [RA, Dec] arrays in degrees
        (each of shape N x 2, where N is the number of vertices in a given
        facet).
    """
    # Build the bounding box corner coordinates
    if width_ra <= 0.0 or width_dec <= 0.0:
        raise ValueError("The RA/Dec width cannot be zero or less")

    wcs = make_wcs(ra_mid, dec_mid, wcs_pixel_scale)
    x_cal, y_cal = wcs.wcs_world2pix(ra_cal, dec_cal, WCS_ORIGIN)
    x_mid, y_mid = wcs.wcs_world2pix(ra_mid, dec_mid, WCS_ORIGIN)
    width_x = width_ra / wcs_pixel_scale / 2.0
    width_y = width_dec / wcs_pixel_scale / 2.0
    bounding_box = np.array(
        [x_mid - width_x, x_mid + width_x, y_mid - width_y, y_mid + width_y]
    )

    # Tessellate and convert resulting facet polygons from (x, y) to (RA, Dec)
    vor = voronoi(np.stack((x_cal, y_cal)).T, bounding_box)
    facet_polys = []
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        ra, dec = wcs.wcs_pix2world(vertices[:, 0], vertices[:, 1], WCS_ORIGIN)
        vertices = np.stack((ra, dec)).T
        facet_polys.append(vertices)
    facet_points = list(map(tuple, wcs.wcs_pix2world(vor.filtered_points, WCS_ORIGIN)))
    return facet_points, facet_polys


def voronoi(cal_coords, bounding_box):
    """
    Produces a Voronoi tessellation for the given coordinates and bounding box

    Parameters
    ----------
    cal_coords : numpy.ndarray
        Array of x, y coordinates
    bounding_box : numpy.ndarray
        Array defining the bounding box as [minx, maxx, miny, maxy]

    Returns
    -------
    vor : scipy.spatial.Voronoi
        The resulting Voronoi object
    """
    eps = 1e-6

    # Select calibrators inside the bounding box
    inside_ind = np.logical_and(
        np.logical_and(
            bounding_box[0] <= cal_coords[:, 0], cal_coords[:, 0] <= bounding_box[1]
        ),
        np.logical_and(
            bounding_box[2] <= cal_coords[:, 1], cal_coords[:, 1] <= bounding_box[3]
        ),
    )
    points_center = cal_coords[inside_ind, :]

    # Mirror points
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(
        points_center,
        np.append(
            np.append(points_left, points_right, axis=0),
            np.append(points_down, points_up, axis=0),
            axis=0,
        ),
        axis=0,
    )

    # Compute Voronoi, sorting the output regions to match the order of the
    # input coordinates
    vor = sp.spatial.Voronoi(points)
    sorted_regions = np.array(vor.regions, dtype=object)[np.array(vor.point_region)]
    vor.regions = sorted_regions.tolist()

    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not (
                    bounding_box[0] - eps <= x
                    and x <= bounding_box[1] + eps
                    and bounding_box[2] - eps <= y
                    and y <= bounding_box[3] + eps
                ):
                    flag = False
                    break
        if region and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions

    return vor
