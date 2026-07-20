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
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from lsmtool.io import check_file_exists

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
        \#.*?                   # comment character maybe followed by other text
        text\s*=\s*             # the text= keyword with optional whitespace
        (
            (?P<quote>["'])     # opening quote
        |                       # or
            (?P<brace>\{)       # opening brace
        |                       # or empty (no quotes or braces)
        )
        (?(quote)               # if opening quote was found
            (?P<text0>[^"\n]+)   # match any text that is not a quote or newline
            (?P=quote)          # match the previously matched quote character
        |                       # or 
            (?(brace)           # if opening brace was found
                (?P<text1>[^\}\n]+) # match any text that is not a closing brace
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

    def __init__(self, name, ra, dec, vertices, wcs=None):
        """
        Create a Facet object with a given name, located at the coordinates
        (ra, dec) and defined by the vertices in celestial coordinates (RA,
        Dec).

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
        wcs : astropy.wcs.WCS, optional
            The WCS object that defines the world coordinate system to use. If
            not given, a WCS object is created using the reference RA and Dec
            and the default pixel scale from `lsmtool.constants.WCS_PIXEL_SCALE`
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
            0.5, max(xmax - xmin, ymax - ymin) * abs(self.wcs.wcs.cdelt[0])
        )  # degrees

        # skymodel is set in the `set_skymodel` method
        self.skymodel = None

    @property
    def ra(self):
        return self.coords.ra.deg

    @property
    def dec(self):
        return self.coords.dec.deg

    @property
    def wcs(self):
        return self._wcs

    @wcs.setter
    def wcs(self, wcs):
        if not isinstance(wcs, WCS):
            raise TypeError("wcs must be an astropy.wcs.WCS object")
        self._wcs = wcs

    @property
    def vertices_xy(self):
        return self.wcs.world_to_pixel_values(self.vertices)

    @property
    def polygon(self):
        return Polygon(self.vertices_xy)

    @property
    def x_center(self):
        xmin, _, xmax, _ = self.polygon.bounds
        return xmin + (xmax - xmin) / 2

    @property
    def y_center(self):
        _, ymin, _, ymax = self.polygon.bounds
        return ymin + (ymax - ymin) / 2

    @property
    def center(self):
        return self.wcs.pixel_to_world(self.x_center, self.y_center)

    @property
    def centroid(self):
        centroid = self.polygon.centroid
        return self.wcs.pixel_to_world(centroid.x, centroid.y)

    @property
    def moc(self):
        """
        Returns a MOC object for the facet's polygon

        Returns
        -------
        moc : mocpy.MOC
            The MOC object for the facet's polygon
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
        # Make list of sources
        ra = skymodel.getColValues("Ra")
        dec = skymodel.getColValues("Dec")
        coords = SkyCoord(ra, dec, unit="deg")
        return self.moc.contains_skycoords(coords)

    def filter_skymodel(self, skymodel, invert=False):

        sources_inside_facet = self.get_contained_sources(skymodel)

        if invert:
            skymodel.remove(sources_inside_facet)
        else:
            skymodel.select(sources_inside_facet)

        if len(skymodel) == 0:
            return skymodel

        # Now check the actual boundary against filtered sky model. We first do a
        # quick (but coarse) check using ImageDraw with a padding of at least a few
        # pixels to ensure the quick check does not remove sources spuriously. We
        # then do a slow (but precise) check using Shapely
        ra = skymodel.getColValues("Ra")
        dec = skymodel.getColValues("Dec")
        x, y = self.wcs.wcs_world2pix(ra, dec, WCS_ORIGIN)

        if any(np.isnan(x) | np.isnan(y)):
            raise ValueError(
                "Source coordinates contains NaN values in pixel coordinates. "
                "This may be due to invalid RA/Dec values in the sky model or "
                "an issue with the WCS transformation."
            )

        # Keep only those sources inside the bounding box
        polygon = self.polygon
        xmin, ymin, xmax, ymax = polygon.bounds
        inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

        xy = np.array([x, y])[:, inside]
        xy_ranges = np.ptp(xy, 1)
        xy_padding = (0.1 * xy_ranges).astype(int).clip(3, None)
        xy_bottom_left = xy.min(1).astype(int) - xy_padding
        xy_sizes = tuple(np.ceil(xy_ranges).astype(int) + 2 * xy_padding)
        xy -= xy_bottom_left[:, None]

        # Unmask everything outside of the polygon + its border (outline)
        mask = Image.new("1", xy_sizes, 0)
        verts = (
            polygon.exterior.coords.xy - xy_bottom_left[:, None]
        ).T.tolist()
        ImageDraw.Draw(mask).polygon(verts, outline=1, fill=1)
        inside = np.array(mask)[tuple(xy.astype(int))[::-1]]

        # Now check sources in the border precisely
        mask = Image.new("1", xy_sizes, 0)
        ImageDraw.Draw(mask).polygon(verts, outline=1, fill=0)
        border = np.array(mask)[tuple(xy.astype(int))[::-1]]

        if border.any():
            (border_indices,) = np.nonzero(border)
            border_pixels_contain = shapely.contains_xy(polygon, *xy[:, border])
            inside[border_indices[~border_pixels_contain]] = False

        if invert:
            skymodel.remove(inside)
        else:
            skymodel.select(inside)

        return skymodel

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
        Returns a matplotlib patch for the facet polygon

        Parameters
        ----------
        wcs : astropy.wcs.WCS, optional
            WCS object defining the celestial coordinate (RA, Dec) to image
            (x, y) transformation. If not given, the facet's WCS object is used.

        Returns
        -------
        patch : matplotlib patch object
            The patch for the facet polygon
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
        super().__init__(name, ra, dec, vertices, wcs)


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
            'global color=green font="helvetica 10 normal" select=1 highlite=1 '
            "edit=1  move=1 delete=1 include=1 fixed=0 source=1\n"
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
                return

            # Replace characters that are potentially problematic for Rapthor,
            # DP3, etc. with an underscore
            for invalid_char in [" ", "{", "}", '"', "'"]:
                facet_name = facet_name.replace(invalid_char, "_")

            return facet_name


def read_from_skymodel(
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
