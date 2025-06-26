"""
Module that holds miscellaneous utility functions and classes.
"""

# This file is adapted from the original in the Rapthor repository:
# https://git.astron.nl/RD/rapthor/-/tree/master/rapthor/lib/miscellaneous.py

import errno
import logging
import os
import pickle
import shutil

import numpy as np
from astropy.wcs import WCS
from casacore.tables import taql
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon
from shapely.prepared import prep



def read_vertices_ra_dec(filename):
    """
    Returns facet vertices stored in input file
    """
    with open(filename, "rb") as vertices_file:
        vertices = pickle.load(vertices_file)

        vertices_ra = vertices[0]
        vertices_dec = vertices[1]

        return zip(vertices_ra, vertices_dec)


def read_vertices_pixel_coordinates(pickle_filename, wcs):
    """Load vertices from file and convert ra/dec coordinates to pixel
    coordinates"""

    with open(pickle_filename, "rb") as pickle_file:
        # Load vertices from file and convert ra/dec coordinates to pixel
        # coordinates
        vertices = np.array(pickle.load(pickle_file))
        vertices_x, vertices_y = radec2xy(
            wcs,
            vertices[0].tolist(),
            vertices[1].tolist(),
        )

        return zip(vertices_x, vertices_y)


def rasterize(verts, data, blank_value=0):
    """
    Rasterize a polygon into a data array

    Parameters
    ----------
    verts : list of (x, y) tuples
        List of input vertices of polygon to rasterize
    data : 2-D array
        Array into which rasterize polygon
    blank_value : int or float, optional
        Value to use for blanking regions outside the poly

    Returns
    -------
    data : 2-D array
        Array with rasterized polygon
    """
    poly = Polygon(verts)
    prepared_polygon = prep(poly)

    # Mask everything outside of the polygon plus its border (outline) with
    # zeros (inside polygon plus border are ones)
    mask = Image.new("L", (data.shape[0], data.shape[1]), 0)
    ImageDraw.Draw(mask).polygon(verts, outline=1, fill=1)
    data *= mask

    # Now check the border precisely
    mask = Image.new("L", (data.shape[0], data.shape[1]), 0)
    ImageDraw.Draw(mask).polygon(verts, outline=1, fill=0)
    masked_ind = np.where(np.array(mask).transpose())
    points = [Point(xm, ym) for xm, ym in zip(masked_ind[0], masked_ind[1])]
    outside_points = [v for v in points if prepared_polygon.disjoint(v)]
    for outside_point in outside_points:
        data[int(outside_point.y), int(outside_point.x)] = 0

    if blank_value != 0:
        data[data == 0] = blank_value

    return data


def ra2hhmmss(degrees):
    """
    Convert RA coordinate (in degrees) to hours / minutes / seconds.

    Inputs and outputs can be single values or numpy arrays.

    Parameters
    ----------
    degrees : float or numpy.array of float
        The RA coordinate in degrees

    Returns
    -------
    hours : int or numpy.array of int
        The hour (HH) part.
    minutes : int  or numpy.array of int
        The minute (MM) part.
    seconds : float  or numpy.array of float
        The second (SS) part.
    """
    degrees = degrees % 360
    fraction, hours = np.modf(degrees / 15)
    fraction, minutes = np.modf(fraction * 60)
    seconds = fraction * 60

    return (hours.astype(int), minutes.astype(int), seconds)


def dec2ddmmss(deg):
    """
    Convert Dec coordinate (in degrees) to DD MM SS

    Inputs and outputs can be single values or numpy arrays

    Parameters
    ----------
    deg : float
        The Dec coordinate in degrees

    Returns
    -------
    dd : int
        The degree (DD) part
    mm : int
        The arcminute (MM) part
    ss : float
        The arcsecond (SS) part
    sign : int
        The sign (+/-)
    """
    sign = np.where(deg < 0.0, -1, 1)
    x, dd = np.modf(np.abs(deg))  # pylint: disable=C0103
    x, mm = np.modf(x * 60)  # pylint: disable=C0103
    ss = x * 60  # pylint: disable=C0103

    return (dd.astype(int), mm.astype(int), ss, sign)


def normalize_ra_dec(ra, dec):
    """
    Normalize ra to be in the range [0, 360).
    Normalize dec to be in the range [-90, 90].

    Parameters
    ----------
    ra, dec : float, float
        The ra in degrees to be normalized.
        The dec in degrees to be normalized.

    Returns
    -------
    normalized_ra, normalized_dec : float, float
        normalized_ra in degrees in the range [0, 360).
        normalized_dec in degrees in the range [-90, 90].
    """

    normalized_dec = (dec + 180) % 360 - 180
    normalized_ra = ra % 360
    if abs(normalized_dec) > 90:
        normalized_dec = 180 - normalized_dec
        normalized_ra = normalized_ra + 180
        normalized_dec = (normalized_dec + 180) % 360 - 180
        normalized_ra = normalized_ra % 360
    return normalized_ra, normalized_dec


def _float_approx_equal(x, y, tol=None, rel=None):
    if tol is rel is None:
        raise TypeError(
            "cannot specify both absolute and relative errors are None"
        )
    tests = []
    if tol is not None:
        tests.append(tol)
    if rel is not None:
        tests.append(rel * abs(x))
    assert tests
    return abs(x - y) <= max(tests)


def approx_equal(x, y, *args, **kwargs):
    """
    Return True if x and y are approximately equal, otherwise False

    If x and y are floats, return True if y is within either absolute error
    tol or relative error rel of x. You can disable either the absolute or
    relative check by passing None as tol or rel (but not both).

    Parameters
    ----------
    x : float
        First value to be compared
    y : float
        Second value to be compared
    """
    if not type(x) is type(y) is float:
        # Skip checking for __approx_equal__ in the common case of two floats.
        methodname = "__approx_equal__"
        # Allow the objects to specify what they consider "approximately
        # equal", giving precedence to x. If either object has the appropriate
        # method, we pass on any optional arguments untouched.
        for a, b in ((x, y), (y, x)):  # pylint: disable=C0103
            try:
                method = getattr(a, methodname)
            except AttributeError:
                continue
            else:
                result = method(b, *args, **kwargs)
                if result is NotImplemented:
                    continue
                return bool(result)
    # If we get here without returning, then neither x nor y knows how to do an
    # approximate equal comparison (or are both floats). Fall back to a numeric
    # comparison.
    return _float_approx_equal(x, y, *args, **kwargs)


def delete_directory(dirname):
    """
    Recursively delete a directory tree, without failing if it does not exist

    Parameters
    ----------
    dirname : str
        Path of directory
    """
    try:
        shutil.rmtree(dirname)
    except OSError as e:  # pylint: disable=C0103
        if e.errno != errno.ENOENT:
            raise e


def concatenate_ms(ms_list, output_ms_name, delete_input=False):
    """
    Concatenates multiple Measurement Sets.

    Args:
        ms_list (list): List of input MS paths to be concatenated.
        output_ms_name (str): Output MS path.
        delete_input (bool): Toggles deleting the input MSs afterwards.
    """

    ms_string_list = ",".join(map(str, ms_list))

    taql(f'select from [{ms_string_list}] giving "{output_ms_name}" AS PLAIN')

    if delete_input:
        for input_ms in ms_list:
            shutil.rmtree(input_ms, ignore_errors=True)


def deg2asec(value: float = 1.0) -> float:
    """Converts a float in degrees to arcseconds.

    Parameters
    ----------
    value : float
        Number in degrees.

    Returns
    -------
    asec : float
        Value in arcseconds.
    """
    return value * 3600.0


def rotation_matrix_2d(theta):
    """Rotation matrix."""
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos, -sin], [sin, cos]])


def table_to_array(table, dtype=float):
    """Convert an astropy Table into a NumPy array.

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
