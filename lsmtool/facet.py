"""
Module that holds functions and classes related to faceting.
"""

import numpy as np
import scipy

from lsmtool.constants import WCS_ORIGIN, WCS_PIXEL_SCALE
from lsmtool.operations_lib import make_wcs

INDEX_OUTSIDE_DIAGRAM = -1


def tessellate(
    directions,
    bbox_midpoint,
    bbox_size,
    wcs_pixel_scale=WCS_PIXEL_SCALE,
):
    """
    Make a Voronoi tessellation.

    This function partitions an image region using Voronoi tessellation seeded
    with the input calibration directions. It filters points that fall
    outside the given dimensions of the bounding box and returns the facet
    centres and polygons that enscribe these points in celestial coordinates.

    Parameters
    ----------
    directions : astropy.coordinates.SkyCoord
        Coordinates of input calibration directions.
    bbox_midpoint : astropy.coordinates.SkyCoord
        Coordinates of bounding box centre.
    bbox_size : tuple of float
        Size of bounding box (RA, Dec). Should be a 2-tuple of numbers in
        degrees.
    wcs_pixel_scale : float
        The pixel scale to use for the conversion to pixel coordinates in
        degrees per pixel.

    Returns
    -------
    facet_points : numpy.ndarray
        Array of facet points centres with (RA, Dec) in degrees along the
        columns.
    facet_polys : list of numpy.ndarray
        Array of facet polygons (vertices) with (RA, Dec) in degrees along the
        columns (each array has shape (n, 2), where n is the number of vertices
        in a given facet).
    """
    width_ra, width_dec = bbox_size
    if width_ra <= 0.0 or width_dec <= 0.0:
        raise ValueError("The RA/Dec width cannot be zero or less")

    # Build the bounding box corner coordinates
    coords_sky = np.column_stack([directions.ra.deg, directions.dec.deg])
    ra_mid, dec_mid = bbox_midpoint.ra.deg, bbox_midpoint.dec.deg

    wcs = make_wcs(ra_mid, dec_mid, wcs_pixel_scale)
    coords_pixel = wcs.wcs_world2pix(coords_sky, WCS_ORIGIN)
    x_mid, y_mid = wcs.wcs_world2pix(ra_mid, dec_mid, WCS_ORIGIN)
    width_x = width_ra / wcs_pixel_scale / 2.0
    width_y = width_dec / wcs_pixel_scale / 2.0
    bounding_box = [
        x_mid - width_x,
        x_mid + width_x,
        y_mid - width_y,
        y_mid + width_y,
    ]

    # Tessellate and convert resulting facet polygons from (x, y) to (RA, Dec)
    points, vertices, regions = voronoi(coords_pixel, bounding_box)

    # Close each region's polygon by adding the first point to the end.
    # Convert to celestial coordinates. In general, each polygon may consist of
    # a different number of vertices.
    facet_polys = [
        wcs.wcs_pix2world(vertices[[*region, region[0]]], WCS_ORIGIN)
        for region in regions
    ]

    facet_points = wcs.wcs_pix2world(points, WCS_ORIGIN)
    return facet_points, facet_polys


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

    # Add
    minx, maxx, miny, maxy = bounding_box
    bounding_box = (minx - eps, maxx + eps, miny - eps, maxy + eps)

    # Add
    bounding_box = np.ravel(
        np.reshape(bounding_box, BBOX_SHAPE_FOR_XY_RANGES) + (-eps, eps)
    )
    # Filter regions
    regions = vor.regions
    vertices = vor.vertices
    filtered_regions = [
        region
        for index in vor.point_region
        if is_valid_region((region := regions[index]), vertices, bounding_box)
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

    if len(points_centre) == 0:
        return points_centre, points_centre

    # Extract bounding box coordinates
    minx, maxx, miny, maxy = bounding_box

    # Create mirrored points more efficiently
    x_coords, y_coords = points_centre[..., 0], points_centre[..., 1]

    # Mirror across each boundary
    mirror_x_min = np.column_stack((2 * minx - x_coords, y_coords))
    mirror_x_max = np.column_stack((2 * maxx - x_coords, y_coords))
    mirror_y_min = np.column_stack((x_coords, 2 * miny - y_coords))
    mirror_y_max = np.column_stack((x_coords, 2 * maxy - y_coords))

    # Combine all points
    points = np.vstack(
        [points_centre, mirror_x_min, mirror_x_max, mirror_y_min, mirror_y_max]
    )

    return points_centre, points


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
    x, y = np.transpose(cal_coords)
    return (minx <= x) & (x <= maxx) & (miny <= y) & (y <= maxy)


def is_valid_region(region, vertices, bounding_box):
    """
    Check if a Voronoi region is valid by verifying all its vertices are within
    the bounding box.

    Parameters
    ----------
    region : list of int
        Indices of the vertices forming the region.
    vertices : numpy.ndarray
        Array of vertex coordinates with shape (n, 2).
    bounding_box : list
        Bounding box defined as [minx, maxx, miny, maxy].

    Returns
    -------
    bool
        True if all vertices are inside the bounding box and the region is not
        empty, False otherwise.
    """
    minx, maxx, miny, maxy = bounding_box

    for index in region:
        if index == INDEX_OUTSIDE_DIAGRAM:
            return False

        if not (
            (minx < vertices[index, 0] < maxx)
            and (miny < vertices[index, 1] < maxy)
        ):
            return False

    return bool(region)
