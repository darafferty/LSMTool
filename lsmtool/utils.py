"""
Module that holds miscellaneous utility functions and classes.
"""

# This file is adapted from the original in the Rapthor repository:
# https://git.astron.nl/RD/rapthor/-/tree/master/rapthor/lib/miscellaneous.py

from pathlib import Path
import pickle

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from astropy.coordinates import SkyCoord


def format_coordinates(ra, dec, unit="deg", precision=6):
    """
    Format RA and Dec coordinates to strings in the makesourcedb format using
    astropy.

    Converts RA and Dec values (in degrees) to string representations according
    to the BBS makesourcedb sky model format. The format specification can be
    found at:
    https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb#angle_specification

    Parameters
    ----------
    ra : numbers.Real or numpy.ndarray
        Right ascension values in degrees.
    dec : numbers.Real or numpy.ndarray
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
    coords = SkyCoord(ra, dec, unit=unit)
    return (
        coords.ra.to_string("hourangle", sep=":", precision=precision),
        coords.dec.to_string(sep=".", precision=precision, alwayssign=True),
    )


def read_vertices_ra_dec(filename):
    """
    Read facet vertices from a pickle file where the data are stored as
    tuples of RA and Dec values.

    Parameters
    ----------
    filename : str or pathlib.Path
        The path to the pickle file.

    Returns
    -------
    tuple of iterables
        A tuple containing two iterables: RA vertices and Dec vertices.
    """
    return tuple(zip(*pickle.loads(Path(filename).read_bytes())))


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
