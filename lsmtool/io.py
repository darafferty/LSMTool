"""
Module for file read / write tasks.
"""

import contextlib as ctx
import os
import pickle
from pathlib import Path

import numpy as np

from .skymodel import SkyModel

# save original tmp path if defined
ORIGINAL_TMPDIR = os.environ.get("TMPDIR")
TRIAL_TMP_PATHS = ("/tmp", "/var/tmp", "/usr/tmp")


@ctx.contextmanager
def temp_storage(trial_paths=TRIAL_TMP_PATHS):
    """Context manager for setting a temporary storage path.

    This context manager attempts to set the TMPDIR environment variable
    to a short path to avoid issues with path length limitations,
    particularly for socket paths used by the multiprocessing module in PyBDSF
    calls.

    Parameters
    ----------
    trial_paths : tuple of str, optional
        A tuple of paths to try setting as the TMPDIR environment variable.
        The first existing path in the tuple will be used. Defaults to
        TRIAL_TMP_PATHS, which includes the same locations used in the
        tempfile Python library.
    """
    if isinstance(trial_paths, (str, Path)):
        trial_paths = [trial_paths]

    try:
        _set_tmpdir(trial_paths)
        yield
    finally:
        _restore_tmpdir()


def _set_tmpdir(trial_paths=TRIAL_TMP_PATHS):
    """Sets a temporary directory to avoid path length issues."""
    trial_paths = trial_paths or []
    for tmpdir in trial_paths:
        if Path(tmpdir).exists():
            os.environ["TMPDIR"] = tmpdir
            return

    raise NotADirectoryError(
        f"None of the trial paths exist: {', '.join(trial_paths)}."
    )


def _restore_tmpdir():
    """Restores the original temporary directory."""
    if ORIGINAL_TMPDIR is not None:
        os.environ["TMPDIR"] = ORIGINAL_TMPDIR


def load(fileName, beamMS=None, VOPosition=None, VORadius=None):
    """
    Load a sky model from a file or VO service and return a SkyModel object.

    Parameters
    ----------
    fileName : str
        Input ASCII file from which the sky model is read (must respect the
        makesourcedb format), name of VO service to query (must be one of
        'GSM', 'LOTSS', 'NVSS', 'TGSS', 'VLSSR', or 'WENSS'), or dict (single
        source only)
    beamMS : str, optional
        Measurement set from which the primary beam will be estimated. A
        column of attenuated Stokes I fluxes will be added to the table
    VOPosition : list of floats
        A list specifying a new position as [RA, Dec] in either makesourcedb
        format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
        [123.2312, 23.3422]) for a cone search
    VORadius : float or str, optional
        Radius in degrees (if float) or 'value unit' (if str; e.g.,
        '30 arcsec') for cone search region in degrees

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
        fileName, beamMS=beamMS, VOPosition=VOPosition, VORadius=VORadius
    )


def read_vertices_ra_dec(filename):
    """
    Read facet vertices from a pickle file.

    Parameters
    ----------
    filename : str or pathlib.Path
        The path to the pickle file where facet vertices are stored as
        tuples of RA and Dec values.

    Returns
    -------
    tuple of iterables
        A tuple containing two iterables: RA vertices and Dec vertices.
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
        f"Unexpected number of data columns ({len(data)}) in file: "
        f"{filename}. Expected vertices to be a sequence of 2-tuples for "
        "RA and Dec coordinates."
    )
