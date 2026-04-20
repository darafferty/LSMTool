"""
Module that holds functions and classes related to faceting.
"""

import logging
import tempfile

import numpy as np
import scipy
from astropy.coordinates import Angle
from matplotlib import patches
from shapely.geometry import Polygon

from . import tableio
from .constants import WCS_ORIGIN, WCS_PIXEL_SCALE
from .download_skymodel import download_skymodel
from .operations_lib import make_wcs, normalize_ra_dec
from .io import load



INDEX_OUTSIDE_DIAGRAM = -1


class Facet(object):
    """
    Base Facet class

    Parameters
    ----------
    name : str
        Name of facet
    ra : float or str
        RA of reference coordinate in degrees (if float) or as a string in a
        format supported by astropy.coordinates.Angle
    dec : float or str
        Dec of reference coordinate in degrees (if float) or as a string in a
        format supported by astropy.coordinates.Angle
    vertices : list of tuples
        List of (RA, Dec) tuples, one for each vertex of the facet
    """

    def __init__(self, name, ra, dec, vertices):
        self.name = name
        self.log = logging.getLogger("rapthor:{0}".format(self.name))
        if type(ra) is str:
            ra = Angle(ra).to("deg").value
        if type(dec) is str:
            dec = Angle(dec).to("deg").value
        self.ra, self.dec = normalize_ra_dec(ra, dec)
        self.vertices = np.array(vertices)

        # Convert input (RA, Dec) vertices to (x, y) polygon
        self.wcs = make_wcs(self.ra, self.dec, WCS_PIXEL_SCALE)
        self.polygon_ras = [radec[0] for radec in self.vertices]
        self.polygon_decs = [radec[1] for radec in self.vertices]
        x_values, y_values = self.wcs.wcs_world2pix(
            self.polygon_ras, self.polygon_decs, WCS_ORIGIN
        )
        polygon_vertices = [(x, y) for x, y in zip(x_values, y_values)]
        self.polygon = Polygon(polygon_vertices)

        # Find the size and center coordinates of the facet
        xmin, ymin, xmax, ymax = self.polygon.bounds
        self.size = min(
            0.5, max(xmax - xmin, ymax - ymin) * abs(self.wcs.wcs.cdelt[0])
        )  # degrees
        self.x_center = xmin + (xmax - xmin) / 2
        self.y_center = ymin + (ymax - ymin) / 2
        self.ra_center, self.dec_center = map(
            float,
            self.wcs.wcs_pix2world(
                self.x_center, self.y_center, WCS_ORIGIN
            ),
        )

        # Find the centroid of the facet
        self.ra_centroid, self.dec_centroid = map(
            float,
            self.wcs.wcs_pix2world(
                self.polygon.centroid.x,
                self.polygon.centroid.y,
                WCS_ORIGIN,
            ),
        )

    def set_skymodel(self, skymodel):
        """
        Sets the facet's sky model

        The input sky model is filtered to contain only those sources that lie
        inside the facet's polygon. The filtered sky model is stored in
        self.skymodel

        Parameters
        ----------
        skymodel : LSMTool skymodel object
            Input sky model
        """
        self.skymodel = filter_skymodel(self.polygon, skymodel, self.wcs)

    def download_panstarrs(self, max_search_cone_radius=0.5):
        """
        Returns a Pan-STARRS sky model for the area around the facet

        Note: the resulting sky model may contain sources outside the facet's
        polygon

        Parameters
        ----------
        max_search_cone_radius : float, optional
            The maximum radius in degrees to use in the cone search. The smaller
            of this radius and the minimum radius that covers the facet is used

        Returns
        -------
        skymodel : LSMTool skymodel object
            The Pan-STARRS sky model
        """
        try:
            with tempfile.NamedTemporaryFile() as fp:
                skymodel_cone_params = {
                    "ra": self.ra_center,
                    "dec": self.dec_center,
                    "radius": min(max_search_cone_radius, self.size / 2),
                }
                download_skymodel(
                    skymodel_cone_params,
                    skymodel_path=fp.name,
                    overwrite=True,
                    survey="PANSTARRS",
                )
                skymodel = load(fp.name)
                skymodel.group("every")
        except IOError:
            # Comparison catalog not downloaded successfully
            self.log.warning(
                "The Pan-STARRS catalog could not be successfully downloaded"
            )
            skymodel = tableio.makeEmptyTable()

        return skymodel

    def find_astrometry_offsets(self, comparison_skymodel=None, min_number=5):
        """
        Finds the astrometry offsets for sources in the facet

        The offsets are calculated as (LOFAR model value) - (comparison model
        value); e.g., a positive Dec offset indicates that the LOFAR sources
        are on average North of the comparison source positions.

        The offsets are stored in self.astrometry_diagnostics, a dict with
        the following keys (see LSMTool's compare operation for details of the
        diagnostics):

            'meanRAOffsetDeg', 'stdRAOffsetDeg', 'meanClippedRAOffsetDeg',
            'stdClippedRAOffsetDeg', 'meanDecOffsetDeg', 'stdDecOffsetDeg',
            'meanClippedDecOffsetDeg', 'stdClippedDecOffsetDeg'

        Note: if the comparison is unsuccessful, self.astrometry_diagnostics is
        an empty dict

        Parameters
        ----------
        comparison_skymodel : LSMTool skymodel object, optional
            Comparison sky model. If not given, the Pan-STARRS catalog is
            used
        min_number : int, optional
            Minimum number of sources required for comparison
        """
        self.astrometry_diagnostics = {}
        if comparison_skymodel is None:
            comparison_skymodel = self.download_panstarrs()

        # Find the astrometry offsets between the facet's sky model and the
        # comparison sky model
        #
        # Note: If there are no successful matches, the compare() method
        # returns None
        if len(comparison_skymodel) >= min_number:
            result = self.skymodel.compare(
                comparison_skymodel,
                radius="5 arcsec",
                excludeMultiple=True,
                make_plots=False,
            )
            # Save offsets
            if result is not None:
                self.astrometry_diagnostics.update(result)
        else:
            self.log.warning(
                "Too few matches to determine astrometry offsets "
                "(min_number = %i but number of matches = %i)",
                min_number,
                len(comparison_skymodel),
            )

    def get_matplotlib_patch(self, wcs=None):
        """
        Returns a matplotlib patch for the facet polygon

        Parameters
        ----------
        wcs : WCS object, optional
            WCS object defining (RA, Dec) <-> (x, y) transformation. If not given,
            the facet's transformation is used

        Returns
        -------
        patch : matplotlib patch object
            The patch for the facet polygon
        """
        if wcs is not None:
            x, y = wcs.wcs_world2pix(
                self.polygon_ras, self.polygon_decs, WCS_ORIGIN
            )
        else:
            x, y = self.polygon.exterior.coords.xy
        xy = np.vstack([x, y]).transpose()
        patch = patches.Polygon(xy=xy, edgecolor="black", facecolor="white")

        return patch


class SquareFacet(Facet):
    """
    Wrapper class for a square facet

    Parameters
    ----------
    name : str
        Name of facet
    ra : float or str
        RA of reference coordinate in degrees (if float) or as a string in a
        format supported by astropy.coordinates.Angle
    dec : float or str
        Dec of reference coordinate in degrees (if float) or as a string in a
        format supported by astropy.coordinates.Angle
    width : float
        Width in degrees of facet
    """

    def __init__(self, name, ra, dec, width):
        if type(ra) is str:
            ra = Angle(ra).to("deg").value
        if type(dec) is str:
            dec = Angle(dec).to("deg").value
        ra, dec = normalize_ra_dec(ra, dec)
        wcs = make_wcs(ra, dec, WCS_PIXEL_SCALE)

        # Make the vertices.
        xmin = wcs.wcs.crpix[0] - width / 2 / abs(wcs.wcs.cdelt[0])
        xmax = wcs.wcs.crpix[0] + width / 2 / abs(wcs.wcs.cdelt[0])
        ymin = wcs.wcs.crpix[1] - width / 2 / abs(wcs.wcs.cdelt[1])
        ymax = wcs.wcs.crpix[1] + width / 2 / abs(wcs.wcs.cdelt[1])
        # Corner order: lower-left, top-left, top-right and lower-right.
        corners_ra, corners_dec = wcs.wcs_pix2world(
            [xmin, xmin, xmax, xmax], [ymin, ymax, ymax, ymin], WCS_ORIGIN
        )

        vertices = list(zip(corners_ra, corners_dec))

        super().__init__(name, ra, dec, vertices)


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
