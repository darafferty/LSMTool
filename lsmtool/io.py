"""
Module for file read / write tasks.
"""

import contextlib as ctx
import numbers
import os
import pickle
import tempfile
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from .skymodel import SkyModel

# Always use a 0-based origin in wcs_pix2world and wcs_world2pix calls.
WCS_ORIGIN = 0

# save original tmp path if defined
ORIGINAL_TMPDIR = os.environ.get("TMPDIR")
TRIAL_TMP_PATHS = [tempfile.gettempdir()]


# Type aliases for paths-like objects
PathLike = Union[str, Path]
PathLikeOptional = Union[PathLike, None]
ListOfPathLike = Sequence[PathLike]
ListOfPathLikeOptional = Union[ListOfPathLike, None]
PathLikeOrListOptional = Union[PathLikeOptional, ListOfPathLike]


@ctx.contextmanager
def temp_storage(trial_paths: PathLikeOrListOptional = TRIAL_TMP_PATHS):
    """
    Context manager for setting a temporary storage path.

    This context manager attempts to set the TMPDIR environment variable
    to a short path to avoid issues with path length limitations,
    particularly for socket paths used by the multiprocessing module in PyBDSF
    calls.

    Parameters
    ----------
    trial_paths : tuple of str, optional
        A tuple of paths to try setting as the TMPDIR environment variable.
        The first existing path in the tuple will be used. Defaults to
        TRIAL_TMP_PATHS, which uses the same locations used by the
        :py::mod:`tempfile` library.
    """
    if isinstance(trial_paths, (str, Path)):
        trial_paths = [trial_paths]

    try:
        yield _set_tmpdir(trial_paths)
    finally:
        _restore_tmpdir()


def _set_tmpdir(trial_paths: PathLikeOrListOptional = TRIAL_TMP_PATHS):
    """Sets a temporary directory to avoid path length issues."""
    trial_paths = trial_paths or []
    for tmpdir in trial_paths:
        path = Path(tmpdir)
        if path.exists():
            os.environ["TMPDIR"] = tmpdir
            return path

    raise NotADirectoryError(
        f"None of the trial paths exist: {', '.join(trial_paths)}."
    )


def _restore_tmpdir():
    """Restores the original temporary directory."""
    if ORIGINAL_TMPDIR is None:
        os.environ.pop("TMPDIR", None)
    else:
        os.environ["TMPDIR"] = ORIGINAL_TMPDIR


def check_file_exists(path: PathLike):
    """
    Check if a file exists at the given path.

    This function checks if a file exists at the specified path. It raises
    exceptions if the path is not a string or pathlib.Path object, or if the file
    does not exist.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to the file.

    Returns
    -------
    Path
        The path to the file.

    Raises
    ------
    TypeError
        If the input path is not a string or pathlib.Path object.
    FileNotFoundError
        If the file does not exist at the given path.
    """

    if not isinstance(path, (str, Path)):
        raise TypeError(
            f"Object of type {type(path).__name__} cannot be interpreted as a "
            "system path."
        )

    path = Path(path)
    if not path.exists() or not path.is_file():
        # Fails if the path does not exits or is not a file.
        raise FileNotFoundError(f"Not able to find file: '{path}'.")

    return path


def validate_paths(required: bool = True, **filenames):
    """
    Checks if provided filenames exist, raising an exception if a file is
    non-existant and required.

    Parameters
    ----------
    required : bool, optional
        If True, all filenames must correspond to existing files. If False,
        non truthy objects (None or "" etc) will not raise an exception.
    **filenames
        Keyword arguments where keys are parameter names and values are
        filenames to validate.

    Raises
    ------
    TypeError
        If a filename is not a string or pathlib.Path object. The corresponding
        parameter name will be shown in the exception message.
    FileNotFoundError
        If a required file does not exist. The corresponding parameter name
        will be shown in the exception message.
    """
    for name, path in filenames.items():
        try:
            if path or required:
                check_file_exists(path)
        except (TypeError, FileNotFoundError) as err:
            raise type(err)(f"Invalid filename for {name!r}: {err}") from None


def load(
    fileName: PathLike,
    beamMS: PathLikeOptional = None,
    VOPosition: Sequence[numbers.Real] = None,
    VORadius: Union[numbers.Real, str] = None,
) -> SkyModel:
    """
    Load a sky model from a file or VO service and return a SkyModel object.

    Parameters
    ----------
    fileName : str
        Input ASCII file from which the sky model is read (must respect the
        makesourcedb format), name of VO service to query (must be one of
        'GSM', 'LOTSS', 'NVSS', 'TGSS', 'VLSSR', or 'WENSS'), or dict (single
        source only).
    beamMS : str, optional
        Measurement set from which the primary beam will be estimated. A
        column of attenuated Stokes I fluxes will be added to the table.
    VOPosition : list of floats
        A list specifying a new position as [RA, Dec] in either makesourcedb
        format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
        [123.2312, 23.3422]) for a cone search.
    VORadius : float or str, optional
        Radius in degrees (if float) or 'value unit' (if str; e.g.,
        '30 arcsec') for cone search region in degrees.

    Returns
    -------
    SkyModel object
        A SkyModel object that stores the sky model and provides methods for
        accessing it.

    Examples
    --------
    Load a sky model from a makesourcedb-formated file::

        >>> import lsmtool
        >>> s = lsmtool.load('sky.model')

    Load a sky model with a beam MS so that apparent fluxes will
    be available (in addition to intrinsic fluxes)::

        >>> s = lsmtool.load('sky.model', 'SB100.MS')

    Load a sky model from the WENSS using all sources within 5 degrees of the
    position RA = 212.8352792, Dec = 52.202644::

        >>> s = lsmtool.load('WENSS', VOPosition=[212.8352792, 52.202644],
            VORadius=5.0)

    Load a sky model from a dictionary defining the source::

        >>> source = {'Name':'src1', 'Type':'POINT', 'Ra':'12:32:10.1',
            'Dec':'23.43.21.21', 'I':2.134}
        >>> s = lsmtool.load(source)

    """

    return SkyModel(
        str(fileName),
        beamMS=(str(beamMS) if beamMS else None),
        VOPosition=VOPosition,
        VORadius=VORadius,
    )


def read_vertices_ra_dec(filename: PathLike):
    """
    Read facet vertices from a pickle file.

    Parameters
    ----------
    filename : str or pathlib.Path
        The path to the pickle file where facet vertices are stored as
        tuples of RA and Dec values.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 2) containing RA and Dec vertices as columns.
    """
    data = pickle.loads(Path(filename).read_bytes())

    if isinstance(data, list) and len(data) == 2:
        ra, dec = data
        if (
            (type(ra) is type(dec) is np.ndarray)
            and (ra.dtype == dec.dtype == "float64")
            and ra.shape == dec.shape
        ):
            return np.transpose(data)

    raise ValueError(
        f"Unexpected data in file: {filename}."
        "Expected two equally-shaped arrays with RA and Dec coordinates."
    )


def read_vertices_x_y(filename, wcs):
    """
    Read facet vertices from a file and convert them to pixel coordinates.

    Parameters
    ----------
    filename: str or pathlib.Path
        Path to file containing the vertices to read.
    wcs : astropy.wcs.WCS object
        WCS object for converting the vertices to pixel coordinates.

    Returns
    -------
    vertices: list of (x, y) tuples of float
        The converted coordinates.
    """
    # The input file always contains vertices as RA,Dec coordinates.
    vertices_celestial = read_vertices_ra_dec(filename)
    return convert_coordinates_to_pixels(vertices_celestial, wcs)


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

    vertices_x, vertices_y = wcs.celestial.wcs_world2pix(
        *coordinates.T, WCS_ORIGIN
    )
    # Convert to a list of (x, y) tuples.
    return list(zip(vertices_x, vertices_y, strict=True))
