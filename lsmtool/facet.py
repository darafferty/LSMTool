"""
Module that holds functions and classes related to faceting.
"""

import ast
import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import scipy
import shapely
from astropy.coordinates import Angle, SkyCoord
from astropy.wcs import WCS
from matplotlib import patches
from mocpy import MOC
from shapely.geometry import Polygon
from shapely.ops import split as split_shape

from . import tableio
from .constants import WCS_ORIGIN, WCS_PIXEL_SCALE
from .download_skymodel import download_skymodel
from .io import check_file_exists, load
from .operations_lib import make_wcs, normalize_ra_dec
from .skymodel import SkyModel

# Module constants
# ---------------------------------------------------------------------------- #
INDEX_OUTSIDE_DIAGRAM = -1

FACET_NAME_REGEX = re.compile(
    r"""(?x)                    # verbose mode
        ^[^#]*                  # any text preceding the comment character
        \#.*?                   # comment character and any following text
        text\s*=\s*             # the text= keyword with optional whitespace
        (
            (?P<quote>["'])     # opening quote
        |                       # or
            (?P<brace>\{)       # opening brace
        |                       # or empty (no quotes or braces)
        )
        (?(quote)               # if opening quote was found
            (?P<text0>[^"\n]+)  # match any text that is not a quote or newline
            (?P=quote)          # match the previously matched quote character
        |                       # or
            (?(brace)           # if opening brace was found
                (?P<text1>[^\}\n]+) # match anything except closing brace
                \}              # match the closing brace
            |
                (?P<text2>[^"'\{\}\n]+)
            )
        )
        .*                      # any trailing text
        $                       # end of line
    """
)

# ---------------------------------------------------------------------------- #


def resolve_coordinates(ra, dec):
    """
    Resolve the given RA and Dec coordinates to a SkyCoord object.

    Parameters
    ----------
    ra : float or str
        Right Ascension in degrees (if float) or as a string in a format
        supported by astropy.coordinates.Angle
    dec : float or str
        Declination in degrees (if float) or as a string in a format
        supported by astropy.coordinates.Angle

    Returns
    -------
    astropy.coordinates.SkyCoord
        The resolved SkyCoord object.
    """
    if isinstance(ra, str):
        ra = Angle(ra).to("deg").value

    if isinstance(dec, str):
        dec = Angle(dec).to("deg").value

    ra, dec = normalize_ra_dec(ra, dec)
    return SkyCoord(ra, dec, unit="deg")


class Facet(object):
    """
    Base class for representing an image facet.

    A facet is a named region of the sky defined by a polygon in celestial
    coordinates (RA, Dec) and a reference point (RA, Dec). The transformation
    between celestial and image coordinates is handled by a `astropy.wcs.WCS`
    object. A facet can have a sky model containing the sources that lie inside
    it and can also be visualized as a matplotlib patch.
    """

    @classmethod
    def from_polygon(cls, polygon, wcs, name="from_polygon"):
        """
        Creates a Facet object from a polygon. The first point in the polygon
        is used as the reference point for the facet.

        Parameters
        ----------
        polygon : Shapely polygon object.
            Polygon object defining the facet region.
        wcs : WCS object
            WCS object defining image to sky transformations.
        name : str, optional
            Name of the facet. Defaults to "from_polygon".

        Returns
        -------
        facet : Facet object
            Facet object created from the polygon.
        """
        x, y = polygon.exterior.xy
        ra, dec = wcs.wcs_pix2world(x, y, WCS_ORIGIN)
        vertices = list(zip(ra, dec, strict=True))
        return Facet(name, ra[0], dec[0], vertices)

    def __init__(self, name, ra, dec, vertices, *, wcs=None):
        """
        Create a Facet object with a given name, located at the coordinates
        (ra, dec) and defined by the vertices in celestial coordinates
        (RA, Dec).

        Parameters
        ----------
        name : str
            Name of facet
        ra : float or str
            RA of reference coordinate in degrees (if float) or as a string in
            a format supported by astropy.coordinates.Angle
        dec : float or str
            Dec of reference coordinate in degrees (if float) or as a string in
            a format supported by astropy.coordinates.Angle
        vertices : list of tuples
            List of (RA, Dec) tuples, one for each vertex of the facet
        wcs : astropy.wcs.WCS, optional
            The WCS object that defines the world coordinate system to use. If
            not given, a WCS object is created using the reference RA and Dec
            and the default pixel scale from
            `lsmtool.constants.WCS_PIXEL_SCALE`
        """
        self.name = name
        self.log = logging.getLogger("lsmtool:{0}".format(self.name))

        self.coords = resolve_coordinates(ra, dec)

        if wcs is None:
            wcs = make_wcs(self.ra, self.dec, WCS_PIXEL_SCALE)

        self.wcs = wcs
        self.vertices = np.array(vertices)

        # Find the size and center coordinates of the facet
        xmin, ymin, xmax, ymax = self.polygon.bounds
        self.size = min(
            0.5,
            max(
                (xmax - xmin) * abs(self.wcs.wcs.cdelt[0]),
                (ymax - ymin) * abs(self.wcs.wcs.cdelt[1]),
            ),
        )  # degrees

        # skymodel is set in the `set_skymodel` method
        self.skymodel = None

    @property
    def ra(self):
        """Right Ascension of the facet's reference coordinate in degrees."""
        return self.coords.ra.deg

    @property
    def dec(self):
        """Declination of the facet's reference coordinate in degrees."""
        return self.coords.dec.deg

    @property
    def wcs(self):
        """WCS object defining the world coordinate system to use."""
        return self._wcs

    @wcs.setter
    def wcs(self, wcs):
        if not isinstance(wcs, WCS):
            raise TypeError("wcs must be an astropy.wcs.WCS object")
        self._wcs = wcs

    @property
    def vertices_xy(self):
        """Polygon vertices in image coordinates (x, y) as a numpy array."""
        return self.wcs.world_to_pixel_values(self.vertices)

    @property
    def polygon(self):
        """Shapely Polygon object representing the facet's polygon in image
        coordinates."""
        return Polygon(self.vertices_xy)

    @property
    def x_center(self):
        """The x-coordinate of the facet's center in the image."""
        xmin, _, xmax, _ = self.polygon.bounds
        return xmin + (xmax - xmin) / 2

    @property
    def y_center(self):
        """The y-coordinate of the facet's center in the image."""
        _, ymin, _, ymax = self.polygon.bounds
        return ymin + (ymax - ymin) / 2

    @property
    def center(self):
        """Centre of the facet in celestial coordinates (RA, Dec) as a SkyCoord
        object."""
        return self.wcs.pixel_to_world(self.x_center, self.y_center)

    @property
    def centroid(self):
        """Centroid of the facet in celestial coordinates (RA, Dec) as a
        SkyCoord object."""
        centroid = self.polygon.centroid
        return self.wcs.pixel_to_world(centroid.x, centroid.y)

    @property
    def moc(self):
        """
        Returns a MOC object for the facet's polygon.

        Returns
        -------
        moc : mocpy.MOC
            The MOC object for the facet's polygon.
        """
        polygon_sky = SkyCoord(*self.vertices.T, unit="deg")
        return MOC.from_polygon_skycoord(polygon_sky)

    def set_skymodel(self, skymodel):
        """
        Sets the facet's sky model

        The input sky model is filtered to contain only those sources that lie
        inside the facet's polygon. The filtered sky model is stored in
        self.skymodel

        Parameters
        ----------
        skymodel : lsmtool.skymodel.SkyModel
            Input sky model
        """
        if not isinstance(skymodel, SkyModel):
            raise TypeError(
                "skymodel must be an lsmtool.skymodel.SkyModel object"
            )
        self.skymodel = self.filter_skymodel(skymodel)

    def get_contained_sources(self, skymodel):
        """
        Check whether the sources in a skymodel are contained within the facet.

        Parameters
        ----------
        skymodel : lsmtool.skymodel.SkyModel
            Input sky model - sources inside the facet will be marked as
            True.

        Returns
        -------
        np.ndarray
            Boolean array indicating which sources in the input skymodel are
            contained within the facet's polygon.
        """
        # Make list of sources
        ra = skymodel.getColValues("Ra")
        dec = skymodel.getColValues("Dec")
        coords = SkyCoord(ra, dec, unit="deg")
        return self.moc.contains_skycoords(coords)

    def filter_skymodel(self, skymodel, invert=False):
        """
        Filters input skymodel to select only sources that lie inside the facet.

        Parameters
        ----------
        skymodel : lsmtool.skymodel.SkyModel
            Input sky model to be filtered.
        invert : bool, optional
            If True, select sources outside the facet instead of inside, by
            default False.

        Returns
        -------
        lsmtool.skymodel.SkyModel
            Filtered sky model containing only sources inside the facet.
        """

        sources_inside_facet = self.get_contained_sources(skymodel)

        if invert:
            skymodel.remove(sources_inside_facet)
        else:
            skymodel.select(sources_inside_facet)

        return skymodel

    def download_panstarrs(self, max_search_cone_radius=2.0):
        """
        Returns a Pan-STARRS sky model for the area around the facet

        Note: the resulting sky model may contain sources outside the facet's
        polygon

        Parameters
        ----------
        max_search_cone_radius : float, optional
            The maximum radius in degrees to use in the cone search. The
            smaller of this radius and the minimum radius that covers the facet
            is used

        Returns
        -------
        skymodel : lsmtool.skymodel.SkyModel
            The Pan-STARRS sky model
        """
        try:
            with tempfile.NamedTemporaryFile() as fp:
                skymodel_cone_params = {
                    "ra": self.center.ra,
                    "dec": self.center.dec,
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
        comparison_skymodel : lsmtool.skymodel.SkyModel, optional
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
                self.astrometry_diagnostics |= result
        else:
            self.log.warning(
                "Too few matches to determine astrometry offsets "
                "(min_number = %i but number of matches = %i)",
                min_number,
                len(comparison_skymodel),
            )

    def get_matplotlib_patch(self, wcs=None):
        """
        Returns a matplotlib patch for the facet polygon.

        Parameters
        ----------
        wcs : astropy.wcs.WCS, optional
            WCS object defining the celestial coordinate (RA, Dec) to image
            (x, y) transformation. If not given, the facet's WCS object is
            used.

        Returns
        -------
        patch : matplotlib patch object
            The patch for the facet polygon.
        """
        if wcs is not None:
            x, y = wcs.wcs_world2pix(*self.vertices.T, WCS_ORIGIN)
        else:
            x, y = self.polygon.exterior.coords.xy
        xy = np.vstack([x, y]).transpose()
        return patches.Polygon(xy=xy, edgecolor="black", facecolor="white")


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
    wcs : astropy.wcs.WCS, optional
        WCS object that defines the world coordinate system to use. If None, a
        generic WCS is used
    """

    def __init__(self, name, ra, dec, width, *, wcs=None):
        self.coords = resolve_coordinates(ra, dec)

        if wcs is None:
            wcs = make_wcs(self.ra, self.dec, WCS_PIXEL_SCALE)

        # Make the vertices.
        xmin = wcs.wcs.crpix[0] - width / 2 / abs(wcs.wcs.cdelt[0])
        xmax = wcs.wcs.crpix[0] + width / 2 / abs(wcs.wcs.cdelt[0])
        ymin = wcs.wcs.crpix[1] - width / 2 / abs(wcs.wcs.cdelt[1])
        ymax = wcs.wcs.crpix[1] + width / 2 / abs(wcs.wcs.cdelt[1])
        # Corner order: lower-left, top-left, top-right and lower-right.
        vertices = wcs.wcs_pix2world(
            [
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin),
            ],
            WCS_ORIGIN,
        )
        super().__init__(name, ra, dec, vertices, wcs=wcs)


def tessellate(
    directions,
    bbox_midpoint,
    bbox_size,
    *,
    wcs=None,
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
    wcs : astropy.wcs.WCS, optional
        The WCS object to use for the conversion to pixel coordinates. If not
        given, a WCS object is created using the reference RA and Dec and the
        default pixel scale from `lsmtool.constants.WCS_PIXEL_SCALE`

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

    if wcs is None:
        wcs = make_wcs(ra_mid, dec_mid, WCS_PIXEL_SCALE)

    coords_pixel = wcs.wcs_world2pix(coords_sky, WCS_ORIGIN)
    x_mid, y_mid = wcs.wcs_world2pix(ra_mid, dec_mid, WCS_ORIGIN)
    half_width_x = width_ra / abs(wcs.wcs.cdelt[0]) / 2.0
    half_width_y = width_dec / abs(wcs.wcs.cdelt[1]) / 2.0
    bounding_box = [
        x_mid - half_width_x,
        x_mid + half_width_x,
        y_mid - half_width_y,
        y_mid + half_width_y,
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
        Vertices of the Voronoi grid. To obtain the vertices of the polygon
        that encloses any particular point, use the indices provided in the
        return value `filtered_regions` to select the corresponding vertices
        for a given cell.
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


def _facet_pca_directions(polygon):
    """
    Return the minor and major principal-axis directions of a polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon in a local planar coordinate system.

    Returns
    -------
    minor_direction : numpy.ndarray
        Unit vector along the minor axis of the polygon.
    major_direction : numpy.ndarray
        Unit vector along the major axis of the polygon.
    """
    vertices = np.asarray(polygon.exterior.coords)[:-1]
    centered = vertices - vertices.mean(axis=0)
    covariance = np.cov(centered.T)
    _, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors[:, 0], eigenvectors[:, 1]


def _split_polygon_at_line(polygon, point, direction):
    """
    Split a polygon with an infinite line through a point.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon to split.
    point : array-like
        Point on the splitting line.
    direction : array-like
        Direction of the splitting line.

    Returns
    -------
    list of shapely.geometry.Polygon or None
        The two resulting polygon pieces, or None if the split did not
        produce exactly two polygons.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    diagonal = np.hypot(max_x - min_x, max_y - min_y)
    if diagonal <= 0:
        return None

    half_length = 2.0 * diagonal + 1.0
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    line = shapely.LineString(
        [
            point - half_length * direction,
            point + half_length * direction,
        ]
    )
    pieces = [
        piece
        for piece in split_shape(polygon, line).geoms
        if piece.geom_type == "Polygon" and piece.area > 0
    ]
    return pieces if len(pieces) == 2 else None


def _polygon_to_sky_vertices(polygon, wcs):
    """
    Convert a polygon in pixel coordinates to an array of sky vertices.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon in pixel coordinates.
    wcs : astropy.wcs.WCS
        WCS object used for the conversion.

    Returns
    -------
    numpy.ndarray
        Array of (RA, Dec) vertices in degrees. The polygon is closed.
    """
    coordinates = np.asarray(polygon.exterior.coords)
    ra, dec = wcs.pixel_to_world_values(coordinates[:, 0], coordinates[:, 1])
    ra, dec = normalize_ra_dec(ra, dec)
    vertices = np.column_stack((ra, dec))
    if not np.array_equal(vertices[0], vertices[-1]):
        vertices = np.vstack((vertices, vertices[0]))
    return vertices

def _split_facet(  # noqa: PLR0913, PLR0917
    facet_point,
    facet_vertices,
    facet_source_mask,
    source_ra,
    source_dec,
    source_flux,
    *,
    flux_threshold,
    min_area_fraction,
    min_separation_fraction,
):
    """
    Try to split a single large facet using bright sources.

    The facet is split perpendicular to its major axis, midway between the
    brightest source and a second source. Candidate sources that are closer
    to the brightest source than `min_separation_fraction` times the square
    root of the facet area are ignored, so that a source and its side-lobe
    are not chosen as the two anchors. If either new facet would be smaller
    than `min_area_fraction` times the original area, the next brightest
    candidate source is tried, and so on.

    Parameters
    ----------
    facet_point : array-like
        Reference position of the facet in degrees.
    facet_vertices : numpy.ndarray
        Facet polygon vertices in (RA, Dec) degrees.
    facet_source_mask : numpy.ndarray
        Boolean mask marking the sources contained in the facet.
    source_ra, source_dec : numpy.ndarray
        Source positions in degrees.
    source_flux : numpy.ndarray
        Source Stokes I fluxes in Jy.
    flux_threshold : float
        Minimum Stokes I flux in Jy for a source to be used.
    min_area_fraction : float
        Minimum area of each new facet as a fraction of the original area.
    min_separation_fraction : float
        Minimum separation between the brightest source and a candidate
        source, relative to the square root of the facet area. Candidates
        closer than this are ignored.

    Returns
    -------
    tuple or None
        Tuple of (new_points, new_polygons), or None if the facet could not
        be split. Each point lies in the polygon at the same position in the
        returned lists.
    """
    log = logging.getLogger(__name__)
    if facet_source_mask.sum() < 2:
        log.debug("Facet has fewer than two sources; not splitting it.")
        return None

    source_flux_facet = source_flux[facet_source_mask]

    eligible_sources_mask = source_flux_facet >= flux_threshold
    if eligible_sources_mask.sum() < 2:
        log.debug(
            "Facet has fewer than two sources above the flux threshold; "
            "not splitting it."
        )
        return None

    # Work only with sources above the specified flux threshold, sorted by
    # increasing flux so that the brightest source is last.
    source_ra_facet = source_ra[facet_source_mask][eligible_sources_mask]
    source_dec_facet = source_dec[facet_source_mask][eligible_sources_mask]
    source_flux_facet = source_flux_facet[eligible_sources_mask]

    sorted_ids = np.argsort(source_flux_facet)
    source_ra_facet_sorted = source_ra_facet[sorted_ids]
    source_dec_facet_sorted = source_dec_facet[sorted_ids]

    # Turn polygon coordinates into Shapely Polygon
    wcs = make_wcs(
        float(facet_point[0]), float(facet_point[1]), WCS_PIXEL_SCALE
    )
    vertices = np.asarray(facet_vertices, dtype=float)
    x_vertices, y_vertices = wcs.world_to_pixel_values(
        vertices[:, 0], vertices[:, 1]
    )
    polygon = shapely.Polygon(np.column_stack((x_vertices, y_vertices)))
    if not polygon.is_valid or polygon.area <= 0:
        log.debug("Facet polygon is invalid; not splitting it.")
        return None

    minor_direction, major_direction = _facet_pca_directions(polygon)

    # Project all eligible sources onto the major axis. The sources are
    # sorted by increasing flux, so the brightest source is the last one.
    source_x, source_y = wcs.world_to_pixel_values(
        source_ra_facet_sorted, source_dec_facet_sorted
    )
    source_xy = np.column_stack((source_x, source_y))
    centroid = np.asarray(polygon.centroid.coords[0], dtype=float)
    projections = (source_xy - centroid) @ major_direction

    # Split midway between the brightest source and a candidate source,
    # trying candidates in order of decreasing brightness. Candidates too
    # close to the brightest source (e.g. its side-lobes) are ignored. Keep
    # the first split for which both halves are large enough.
    anchor_index = len(source_xy) - 1
    min_separation = min_separation_fraction * np.sqrt(polygon.area)
    separations = np.hypot(
        source_xy[:, 0] - source_xy[anchor_index, 0],
        source_xy[:, 1] - source_xy[anchor_index, 1],
    )
    min_area = min_area_fraction * polygon.area
    split_pieces = None
    candidate_index = None
    for index in range(anchor_index - 1, -1, -1):
        if separations[index] < min_separation:
            continue
        cut_projection = 0.5 * (
            projections[anchor_index] + projections[index]
        )
        cut_point = centroid + cut_projection * major_direction
        pieces = _split_polygon_at_line(
            polygon, cut_point, minor_direction
        )
        if pieces is None:
            continue
        if min(piece.area for piece in pieces) < min_area:
            continue
        split_pieces = pieces
        candidate_index = index
        break

    if split_pieces is None:
        log.debug(
            "No candidate source gives an acceptable split; not splitting "
            "it."
        )
        return None

    # Order the polygons to match the anchor and candidate source positions.
    anchor_in_first_piece = shapely.contains_xy(
        split_pieces[0],
        source_xy[anchor_index, 0],
        source_xy[anchor_index, 1],
    )
    if anchor_in_first_piece:
        anchor_piece, candidate_piece = split_pieces
    else:
        candidate_piece, anchor_piece = split_pieces

    new_polys = [
        _polygon_to_sky_vertices(anchor_piece, wcs),
        _polygon_to_sky_vertices(candidate_piece, wcs),
    ]
    new_points = [
        (
            float(source_ra_facet_sorted[anchor_index]),
            float(source_dec_facet_sorted[anchor_index]),
        ),
        (
            float(source_ra_facet_sorted[candidate_index]),
            float(source_dec_facet_sorted[candidate_index]),
        ),
    ]
    return new_points, new_polys


def split_large_facets(  # noqa: PLR0913
    facet_points,
    facet_polys,
    sky_model,
    flux_threshold,
    *,
    area_factor_threshold=1.4,
    min_area_fraction=0.2,
    min_separation_fraction=0.05,
):
    """
    Split large facets so that both new facets have a bright source.

    A facet is considered large when its area exceeds
    `area_factor_threshold` times the median facet area. Each large facet is
    split at most once. The split line runs along the minor axis through the
    midpoint, along the major axis, between the brightest source and a
    candidate source. Candidates are tried in order of decreasing brightness
    until both halves are at least `min_area_fraction` of the original area.

    Parameters
    ----------
    facet_points : array-like
        Facet reference positions in degrees, with shape (n, 2).
    facet_polys : list of numpy.ndarray
        Facet polygon vertices in (RA, Dec) degrees.
    sky_model : lsmtool.skymodel.SkyModel
        Sky model containing the source positions and fluxes.
    flux_threshold : float
        Minimum Stokes I flux in Jy for a source to be used.
    area_factor_threshold : float, optional
        Minimum facet area, relative to the median facet area, for a facet to
        be split. Default is 1.4.
    min_area_fraction : float, optional
        Minimum area of each new facet as a fraction of the original area.
        Default is 0.2.
    min_separation_fraction : float, optional
        Minimum separation between the original source and a candidate,
        relative to the square root of the facet area. Default is 0.05.

    Returns
    -------
    facet_points : numpy.ndarray or list
        Facet reference positions after splitting.
    facet_polys : list of numpy.ndarray
        Facet polygon vertices after splitting.
    """
    log = logging.getLogger(__name__)
    if len(facet_polys) == 0:
        return facet_points, facet_polys

    polys = [shapely.Polygon(facet_coords) for facet_coords in facet_polys]
    areas = np.asarray([poly.area for poly in polys], dtype=float)
    median_area = float(np.median(areas))
    if not np.isfinite(median_area) or median_area <= 0:
        return facet_points, facet_polys

    large_area_mask = areas > (area_factor_threshold * median_area)
    if not large_area_mask.any():
        return facet_points, facet_polys

    source_ra = np.asarray(sky_model.getColValues("Ra"), dtype=float)
    source_dec = np.asarray(sky_model.getColValues("Dec"), dtype=float)
    source_flux = np.asarray(sky_model.getColValues("I"), dtype=float)
    if source_ra.size == 0:
        return facet_points, facet_polys

    # Find the sources that lie in each facet. A MOC is used for the
    # containment test so that facets that cross the RA = 0 meridian or
    # contain a pole are handled correctly. A high MOC depth is used to keep
    # the test accurate close to the facet boundaries.
    source_coords = SkyCoord(source_ra, source_dec, unit="deg")
    facet_sources = [
        MOC.from_polygon_skycoord(
            SkyCoord(*np.asarray(facet_coords).T, unit="deg"),
            max_depth=20,
        ).contains_skycoords(source_coords)
        for facet_coords in facet_polys
    ]

    new_points = []
    new_polys = []
    number_split = 0
    for index, facet_point in enumerate(facet_points):
        if not large_area_mask[index]:
            new_points.append(facet_point)
            new_polys.append(facet_polys[index])
            continue

        result = _split_facet(
            facet_point,
            facet_polys[index],
            facet_sources[index],
            source_ra,
            source_dec,
            source_flux,
            flux_threshold=flux_threshold,
            min_area_fraction=min_area_fraction,
            min_separation_fraction=min_separation_fraction,
        )
        if result is None:
            new_points.append(facet_point)
            new_polys.append(facet_polys[index])
        else:
            split_points, split_polys = result
            new_points.extend(split_points)
            new_polys.extend(split_polys)
            number_split += 1

    large_count = int(np.count_nonzero(large_area_mask))
    log.info(
        "Split %d of %d large facets (out of %d facets; %.2f x median "
        "area cut-off).",
        number_split,
        large_count,
        len(areas),
        area_factor_threshold,
    )
    if isinstance(facet_points, np.ndarray):
        new_points = np.asarray(new_points)
    return new_points, new_polys


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


def make_ds9_region_file(
    facets, outfile, enclose_names=True, associate_names_with_polygons=True
):
    """
    Make a ds9 region file for given facet polygons and centers

    Parameters
    ----------
    facets : list of Facet objects
        List of Facet objects to include.
    outfile : str
        Name of output region file.
    enclose_names : bool, optional
        If True, enclose patch names in curly brackets for full compatibility
        with ds9. Curly brackets may cause issues with other tools that use the
        region file, such as DP3, in which case they can be excluded by setting
        this option to False.
    associate_names_with_polygons : optional
        If True, the facet names are associated with the "polygon" entries.
        This convention matches that used by WSClean (see
        https://wsclean.readthedocs.io/en/latest/ds9_facet_file.html#adding-a-text-label).
        If False, the names are associated with the "point" entries instead
        (required by some DP3 steps).
    """
    with open(outfile, "w") as stream:
        stream.write(
            "# Region file format: DS9 version 4.0\n"
            'global color=green font="helvetica 10 normal" select=1 highlite=1'
            " edit=1  move=1 delete=1 include=1 fixed=0 source=1\n"
            "fk5\n"
        )
        for facet in facets:
            polygon_string = ", ".join(map(str, facet.vertices.ravel()))
            lines = [
                f"polygon({polygon_string})",
                f"point({facet.ra}, {facet.dec})",
            ]
            facet_name = f"{{{facet.name}}}" if enclose_names else facet.name
            append_name_to_line = 0 if associate_names_with_polygons else 1
            lines[append_name_to_line] += f" # text={facet_name}"
            stream.write("\n".join(lines) + "\n")


def read_ds9_region_file(region_file, wcs=None):
    """
    Read a ds9 facet region file and return facets.

    Parameters
    ----------
    region_file : str
        Filename of input ds9 region file.
    wcs : astropy.wcs.WCS, optional
        WCS object that defines the world coordinate system to use for the
        conversion to pixel coordinates. If None, a generic WCS object is
        created using the reference point of the facet and the default pixel
        scale from `lsmtool.constants.WCS_PIXEL_SCALE`.

    Returns
    -------
    facets : list
        List of Facet objects.
    """

    region_file = check_file_exists(region_file)

    facets = []
    for index, (polygon, *_, points) in enumerate(
        parse_ds9_facets(region_file)
    ):
        ra, dec = ast.literal_eval(points.split("point")[1])
        vertices = ast.literal_eval(polygon.split("polygon")[1])
        vertices = np.reshape(vertices, (-1, 2))

        facet_name = parse_facet_name((polygon, points))
        facet_name = facet_name or f"facet_{index}"

        # Lastly, add the facet to the list
        facets.append(Facet(facet_name, ra, dec, vertices, wcs=wcs))

    return facets


def parse_ds9_facets(region_file):
    """
    Parse facet definitions from a ds9 region file.

    Each facet in the region file is defined by a polygon line that starts
    with 'polygon' and gives the (RA, Dec) vertices.

    Each facet polygon line may be followed by a line giving the reference
    point that starts with 'point' and gives the reference (RA, Dec).

    The facet name may be set in the text property of either line
    (see https://wsclean.readthedocs.io/en/latest/ds9_facet_file.html).
    """

    with Path(region_file).open("r") as stream:
        buffer = []
        for line in stream:
            if not line.startswith(("polygon", "point")):
                continue

            if line.startswith("polygon") and buffer:
                yield buffer
                buffer = []

            elif line.startswith("point") and not buffer:
                raise ValueError(
                    f'Error parsing region file "{region_file}": "point" '
                    'line found without a preceding "polygon" line'
                )

            buffer.append(line)

        if buffer:
            yield buffer


def parse_facet_name(lines):
    """
    Parse the facet name from the polygon / point definition string.

    In the region file, the name is defined using the 'text' property. E.g.:
        'polygon(3.6, 60.9, 3.4, 58.9, 3.1, 59.2) # text = {Patch_1} width = 2'
        'point(0.1, 1.2) # text = {Patch_1} width = 2'

    Note: ds9 format allows strings to be quoted with " or ' or {}
    (see https://ds9.si.edu/doc/ref/region.html#RegionProperties),
    so we match everything between "", '', or {}, if the line contains
    anything like `... # text = ...`

    Note: if a name is defined for both the facet polygon and the facet
    reference point, the one for the point takes precedence.
    """

    for line in sorted(lines):
        if match := FACET_NAME_REGEX.search(line):
            facet_name = match["text0"] or match["text1"] or match["text2"]
            if not (facet_name := facet_name.strip()):
                return None

            # Replace characters that are potentially problematic for Rapthor,
            # DP3, etc. with an underscore
            for invalid_char in [" ", "{", "}", '"', "'"]:
                facet_name = facet_name.replace(invalid_char, "_")

            return facet_name

    return None


def read_from_skymodel(  # noqa: PLR0913
    skymodel,
    ra_mid,
    dec_mid,
    width_ra,
    width_dec,
    *,
    wcs=None,
):
    """
    Reads a sky model file and returns facets

    Parameters
    ----------
    skymodel : str
        Filename of the sky model (must have patches), in makesourcedb format
    ra_mid : float
        RA in degrees of bounding box center
    dec_mid : float
        Dec in degrees of bounding box center
    width_ra : float
        Width of bounding box in RA in degrees, corrected to Dec = 0
    width_dec : float
        Width of bounding box in Dec in degrees
    wcs : astropy.wcs.WCS, optional
        The world coordinate system (WCS) object to use for the conversion
        between celestial and pixel coordinate systems. If None, a generic WCS
        object is created using the reference point of the facet and the
        default pixel scale from `lsmtool.constants.WCS_PIXEL_SCALE`.

    Returns
    -------
    facets : list
        List of Facet objects
    """
    skymod = load(skymodel)
    if not skymod.hasPatches:
        raise ValueError("The sky model must be grouped into patches")

    # Set the position of the calibration patches to those of
    # the input sky model
    source_dict = skymod.getPatchPositions()

    # Make sure RA is between [0, 360) deg and Dec between [-90, 90]
    coordinates = [
        normalize_ra_dec(ra.value, dec.value)
        for (ra, dec) in source_dict.values()
    ]
    patch_coords = SkyCoord(coordinates, unit="deg")

    # Do the tessellation
    facet_points, facet_polys = tessellate(
        patch_coords,
        SkyCoord(ra_mid, dec_mid, unit="deg"),
        [width_ra, width_dec],
        wcs=wcs,
    )

    # For each facet, match the correct patch name (i.e., the name of the patch
    # closest to the facet reference point). This step is needed because some
    # patches in the sky model may not appear in the facet list if they lie
    # outside the bounding box.
    names = np.array(list(source_dict.keys()))
    facet_coords = SkyCoord(facet_points, unit="deg")
    facet_names = names[
        facet_coords[:, None].separation(patch_coords).argmin(1)
    ]

    # Create the facets
    return [
        Facet(name, *center_coords, vertices, wcs=wcs)
        for name, center_coords, vertices in zip(
            facet_names, facet_points, facet_polys, strict=True
        )
    ]


def filter_skymodel(polygon, skymodel, wcs, invert=False):
    """
    Filters input skymodel to select only sources that lie inside the input
    region defined by a polygon in celestial coordinates.

    Parameters
    ----------
    polygon : Shapely polygon object.
        Polygon object to use for filtering.
    skymodel : lsmtool.skymodel.SkyModel
        Input sky model to be filtered.
    wcs : WCS object
        WCS object defining image to sky transformations.
    invert : bool, optional
        If True, invert the selection (so select only sources that lie outside
        the facet).

    Returns
    -------
    filtered_skymodel : lsmtool.skymodel.SkyModel
        Skymodel object with only sources inside the facet either retained
        (invert=False, the default) or removed (invert=True).
    """
    facet = Facet.from_polygon(polygon, wcs, "filter_skymodel")
    return facet.filter_skymodel(skymodel, invert)
