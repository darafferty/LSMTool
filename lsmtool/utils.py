"""
Module that holds miscellaneous utility functions and classes.

This file was adapted from the `original`_ in the Rapthor repository:

Additional functions for supporting skymodel filtering based on SoFiA was added
for the SKA self calibration pipeline in this `merge request`_.

Some functions were removed or combined when migrating the module to LSMTools.

.. _original:
    https://git.astron.nl/RD/rapthor/-/tree/master/rapthor/lib/miscellaneous.py
.. _merge request:
    https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-wflow-selfcal/-/blob/3be896/src/ska_sdp_wflow_selfcal/pipeline/support/miscellaneous.py
"""

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

# Always use a 0-based origin in wcs_pix2world and wcs_world2pix calls.
WCS_ORIGIN = 0


def format_coordinates(ra, dec, precision=6):
    """
    Format RA and Dec coordinates to strings in the makesourcedb format using
    astropy.

    Converts RA and Dec values (in degrees) to string representations according
    to the BBS makesourcedb sky model format. The format specification can be
    found in the `lofar documentation
    <https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb#angle_specification>`_.

    Parameters
    ----------
    ra : float or list of float
        Right ascension values in degrees.
    dec : float or list of float
        Declination values in degrees.
    precision : int, optional
        The number of decimal places for seconds in RA and Dec strings.
        Default is 6.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two arrays: formatted RA strings and formatted Dec
        strings.
    """
    coords = SkyCoord(ra, dec, unit="deg")
    return (
        coords.ra.to_string("hourangle", sep=":", precision=precision),
        coords.dec.to_string(sep=".", precision=precision, alwayssign=True),
    )


def convert_coordinates_to_pixels(coordinates, wcs):
    """
    Convert celestial coordinates (RA, Dec) to image pixel coordinates.

    This function transforms an array of shape (N ,2), with RA and Dec
    coordinates as columns, into pixel coordinates using the provided WCS
    object, handling extra axes as needed.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Array of shape (N, 2) containing RA and Dec values.
    wcs : astropy.wcs.WCS
        WCS object used for the coordinate transformation.

    Returns
    -------
    list of tuple
        List of (x, y) pixel coordinate tuples.
    """

    # NOTE: In case the wcs has four axes (ra, dec, freq, pol), we need to add
    # two extra (dummy) elements to the celestial coordinates, then ignore them.
    null_coordinates = [0] * (wcs.naxis - 2)
    vertices_x, vertices_y, *_ = wcs.wcs_world2pix(
        *coordinates.T, *null_coordinates, WCS_ORIGIN
    )
    # Convert to a list of (x, y) tuples.
    return list(zip(vertices_x, vertices_y))


def rasterize(vertices, data, blank_value=0):
    """
    Rasterize a polygon into a data array, setting any data outside the polygon
    to `blank_value`.

    .. note:: This function modifies data in-place.

    Parameters
    ----------
    vertices : list of tuples
        List of input vertices of polygon to rasterize. Each item in the list
        should be a (x, y) coordinate point, where x and y are float or int.
    data : np.ndarray
        A 2-D numpy array into which to rasterize the polygon. Note that the
        data are updated in-place.
    blank_value : int or float, optional
        Value to use for filling regions outside the polygon. The data type of
        the fill value should be compatible with the dtype of the data array.

    Returns
    -------
    data : np.ndarray
        2-D array containing the rasterized polygon.
    """
    if isinstance(vertices, np.ndarray):
        vertices = vertices.tolist()

    poly = Polygon(vertices)
    prepared_polygon = prep(poly)

    # Mask everything outside of the polygon plus its border (outline) with
    # zeros (inside polygon plus border are ones)
    mask = Image.new("L", data.shape[1::-1], 0)
    ImageDraw.Draw(mask).polygon(vertices, outline=1, fill=1)
    data *= mask

    # Now check the border precisely
    mask = Image.new("L", data.shape[1::-1], 0)
    ImageDraw.Draw(mask).polygon(vertices, outline=1, fill=0)
    masked_ind = np.where(np.array(mask).transpose())

    points = [Point(xm, ym) for xm, ym in zip(masked_ind[0], masked_ind[1])]
    outside_points = [v for v in points if prepared_polygon.disjoint(v)]
    for outside_point in outside_points:
        data[int(outside_point.y), int(outside_point.x)] = 0

    if blank_value != 0:
        data[data == 0] = blank_value

    return data


def rasterize_polygon_mask_exterior(
    fits_file, vertices_file, output_file=None, precision=None
):
    """
    Rasterize the image data in `fits_file` using the polygon defined by the
    `vertices_file`, mask any data outside the polygon, and write the result to
    an `output_fits` file.

    Parameters
    ----------
    fits_file : str or pathlib.Path
        Path to the input FITS file.
    vertices_file : str or pathlib.Path
        Path to the file containing polygon vertices in RA, DEC coordinates.
    output_file : str or pathlib.Path, optional
        Path to the output FITS file. If None, the default, the input file will
        be overwritten.
    precision : int, optional
        A integer specifying the numeric precision for the converted
        vertices. If provided, the vertex points will be rounded to this number
        of significant figures. This is useful on occasion since there may be
        some imprecision in the result due to rounding errors.
    """
    from lsmtool.io import read_vertices_x_y

    hdulist = (hdu,) = fits.open(fits_file, memmap=False)
    vertices = read_vertices_x_y(vertices_file, wcs.WCS(hdu.header))

    if precision is not None:
        vertices = np.round(vertices, precision)

    # Rasterize the polygon and mask exterior pixels. Data is modified inplace.
    rasterize(vertices, hdu.data[0, 0, :, :])

    # Write output
    hdulist.writeto(output_file or fits_file, overwrite=True)
    hdulist.close()


def rotation_matrix_2d(theta):
    """
    Computes the transformation matrix for rotating vectors with two spatial
    dimensions by angle theta in radians.

    Parameters
    ----------
    theta: float or np.ndarray
        The angle of rotation in radians. If theta is a number (float or int),
        the resulting array will have shape (2, 2). If theta is a numpy array
        of shape (n, m, ...), the resulting array will have dimensions
        (2, 2, n, m, ...).

    Returns:
        A NumPy array containing the rotation matrix (or matrices) along the
        first two array dimensions.
    """
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos, -sin], [sin, cos]])


def table_to_array(table, dtype=float):
    """
    Convert an astropy Table into a NumPy array.

    Converts a two-column astropy Table to a NumPy array of the specified
    dtype. This function assumes each row in the table has a uniform data type.

    Parameters
    ----------
    table : astropy.table.Table
        The input astropy Table with two columns.
    dtype : type, optional
        The desired data type for the NumPy array. Defaults to float.

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape (n, 2) where n is the number of rows in the
        input table, and the dtype is as specified.
    """
    return table.as_array().view(dtype).reshape(-1, len(table.colnames))


def transfer_patches(from_skymodel, to_skymodel, patch_dict=None):
    """
    Transfers the patches defined in from_skymodel to to_skymodel.

    Parameters
    ----------
    from_skymodel : LSMTool skymodel.SkyModel object
        Sky model from which to transfer patches.
    to_skymodel : LSMTool skymodel.SkyModel object
        Sky model to which to transfer patches.
    patch_dict : dict, optional
        Dict of patch positions.
    """
    if not from_skymodel.hasPatches:
        raise ValueError(
            "Cannot transfer patches since from_skymodel is not grouped "
            "into patches."
        )

    names_from = from_skymodel.getColValues("Name").tolist()
    names_to = to_skymodel.getColValues("Name").tolist()
    names_from_set = set(names_from)
    names_to_set = set(names_to)

    if not to_skymodel.hasPatches:
        to_skymodel.group("single")

    input_patches = from_skymodel.table["Patch"]
    output_patches = to_skymodel.table["Patch"]

    if names_from_set == names_to_set:
        # Both sky models have the same sources, so use indexing
        ind_ss = np.argsort(names_from)
        ind_ts = np.argsort(names_to)
        output_patches[ind_ts] = input_patches[ind_ss]
    elif names_to_set.issubset(names_from_set):
        # The to_skymodel is a subset of from_skymodel, so use slower matching
        # algorithm
        for ind_ts, name in enumerate(names_to):
            output_patches[ind_ts] = input_patches[names_from.index(name)]
    elif names_from_set.issubset(names_to_set):
        # The from_skymodel is a subset of to_skymodel, so use slower matching
        # algorithm, leaving non-matching sources in their initial patches
        for ind_ss, name in enumerate(names_from):
            output_patches[names_to.index(name)] = input_patches[ind_ss]
    else:
        # Skymodels don't match, raise error
        raise ValueError(
            "Cannot transfer patches since neither sky model is a "
            "subset of the other."
        )

    to_skymodel._updateGroups()
    if patch_dict is not None:
        to_skymodel.setPatchPositions(patchDict=patch_dict)
