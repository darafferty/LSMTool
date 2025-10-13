"""
Module that holds functions and classes related to faceting.
"""

import numpy as np
import scipy

from lsmtool.constants import WCS_ORIGIN, WCS_PIXEL_SCALE
from lsmtool.operations_lib import make_wcs

INDEX_OUTSIDE_DIAGRAM = -1
BBOX_SHAPE_FOR_XY_RANGES = (2, 2)


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
    points, vertices, regions = voronoi(
        np.stack((x_cal, y_cal)).T, bounding_box
    )
    facet_polys = []
    for region in regions:
        polygon = vertices[region + [region[0]], :]
        ra, dec = wcs.wcs_pix2world(polygon[:, 0], polygon[:, 1], WCS_ORIGIN)
        polygon = np.stack((ra, dec)).T
        facet_polys.append(polygon)

    facet_points = list(map(tuple, wcs.wcs_pix2world(points, WCS_ORIGIN)))
    return facet_points, facet_polys


def in_box(cal_coords, bounding_box):
    """
    Check if coordinates are inside the bounding box.

    Parameters
    ----------
    cal_coords : numpy.ndarray
        Array of x, y coordinates with shape (n, 2).
    bounding_box : numpy.ndarray
        Array defining the bounding box as [minx, maxx, miny, maxy].

    Returns
    -------
    inside : numpy.ndarray
        Boolean array with True for inside and False if not.
    """
    minx, maxx, miny, maxy = bounding_box
    minx, maxx = sorted([minx, maxx])
    miny, maxy = sorted([miny, maxy])
    x, y = cal_coords[..., 0], cal_coords[..., 1]
    return (minx <= x) & (x <= maxx) & (miny <= y) & (y <= maxy)


def voronoi(cal_coords, bounding_box, eps=1e-6):
    """
    Produce a Voronoi tessellation for the given coordinates and bounding box.

    Parameters
    ----------
    cal_coords : numpy.ndarray
        Array of x, y coordinates with shape (n, 2).
    bounding_box : numpy.ndarray
        Array defining the bounding box as [minx, maxx, miny, maxy].
    eps : float
        Numerical tolerance value, used to expand the bounding box slightly to
        avoid issues related to numeric precision.

    Returns
    -------
    points_centre : numpy.ndarray
        Centre points of the Voronoi cells.
    vertices : numpy.ndarray
        Vertices of the Voronoi grid. To obtain the vertices of the polygon that
        encloses any particular point, use the indices provided in the return
        value `filtered_regions` to select the corresponding vertices for a
        given cell.
    filtered_regions : list of list of int
        For each cell in the tesselation, a list of index points for the
        vertices that enclose the cell. For example
        `vertices[filtered_regions[0]]` are the vertices of the first cell.
        Only points that fall within the `bounding_box` are retained.
    """

    points_centre, points = prepare_points_for_tessellate(
        cal_coords, bounding_box
    )

    # Compute Voronoi, sorting the output regions to match the order of the
    # input coordinates
    vor = scipy.spatial.Voronoi(points)
    sorted_regions = np.array(vor.regions, dtype=object)[vor.point_region]

    # Add
    bounding_box = np.ravel(
        np.reshape(bounding_box, BBOX_SHAPE_FOR_XY_RANGES) + (-eps, eps)
    )
    # Filter regions
    filtered_regions = [
        region
        for region in sorted_regions
        if region
        and (INDEX_OUTSIDE_DIAGRAM not in region)
        and all(in_box(vor.vertices[region], bounding_box))
    ]
    return points_centre, vor.vertices, filtered_regions


def prepare_points_for_tessellate(cal_coords, bounding_box):
    """
    Select calibration points inside the bounding box and generates mirrored
    points for Voronoi tessellation.

    This function filters the input coordinates to those within the bounding
    box and creates mirrored points to ensure proper tessellation at the
    boundaries.

    Parameters
    ----------
    cal_coords : numpy.ndarray
        Array of x, y coordinates with shape (n, 2).
    bounding_box : list or numpy.ndarray
        Array defining the bounding box as [minx, maxx, miny, maxy].

    Returns
    -------
    points_centre : numpy.ndarray
        Calibration points inside the bounding box.
    points : numpy.ndarray
        Array of calibration points and their mirrored counterparts for
        tessellation.
    """
    # Select calibrators inside the bounding box
    points_centre = cal_coords[in_box(cal_coords, bounding_box)]

    # Mirror points
    points_mirror = np.tile(points_centre, (2, 2, 1, 1))
    intervals = np.reshape(bounding_box, (2, 1, 2))
    xy = 2 * intervals - points_centre.T[..., None]
    points_mirror[0, ..., 0] = xy[0].T
    points_mirror[1, ..., 1] = xy[1].T

    points = np.vstack([points_centre, points_mirror.reshape(-1, 2)])
    return points_centre, points
