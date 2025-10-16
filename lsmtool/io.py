"""
Module for file read / write tasks.
"""

import contextlib as ctx
import logging
import numbers
import os
import tempfile
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import requests

import lsmtool
from lsmtool.constants import WCS_ORIGIN
from lsmtool.skymodel import SkyModel

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
        :mod:`tempfile` library.
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
    exceptions if the path is not a string or pathlib.Path object, or if the
    file does not exist.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to the file.

    Returns
    -------
    path : pathlib.Path
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
    fileName : str or pathlib.Path
        Input ASCII file from which the sky model is read (must respect the
        makesourcedb format), name of VO service to query (must be one of
        'GSM', 'LOTSS', 'NVSS', 'TGSS', 'VLSSR', or 'WENSS'), or dict (single
        source only).
    beamMS : str or pathlib.Path, optional
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
    skymodel : lsmtool.skymodel.SkyModel
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
    Read facet vertices from a stored numpy array in .npy format.

    Parameters
    ----------
    filename : str or pathlib.Path
        The path to the file where facet vertices are stored as an array with
        RA and Dec values along columns.

    Returns
    -------
    vertices : numpy.ndarray
        Array of shape (N, 2) containing RA and Dec vertices as columns.
    """
    data = np.load(Path(filename), allow_pickle=False)

    if data.ndim != 2 or data.shape[1] != 2 or data.dtype != "float64":
        raise ValueError(
            f"Unexpected data in file: {filename}."
            "Expected an array with 2 columns of RA and Dec coordinates."
        )

    return data


def read_vertices_x_y(filename, wcs):
    """
    Read facet vertices from a file and convert them to pixel coordinates.

    Parameters
    ----------
    filename: str or pathlib.Path
        Path to file containing the vertices to read.
    wcs : astropy.wcs.WCS
        WCS object for converting the vertices to pixel coordinates.

    Returns
    -------
    vertices: list of tuple of float
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
    vertices : list of tuple
        List of (x, y) pixel coordinate tuples.
    """

    vertices_x, vertices_y = wcs.celestial.wcs_world2pix(*coordinates.T, WCS_ORIGIN)
    # Convert to a list of (x, y) tuples.
    return list(zip(vertices_x, vertices_y, strict=True))


def download_skymodel(
    ra,
    dec,
    skymodel_path,
    radius=5.0,
    overwrite=False,
    source="TGSS",
    targetname="Patch",
):
    """
    Downloads a skymodel for the given position and radius

    Parameters
    ----------
    ra : float
        Right ascension in degrees of the skymodel centre
    dec : float
        Declination in degrees of the skymodel centre
    skymodel_path : str
        Full name (with path) to the output skymodel
    radius : float, optional
        Radius for the cone search in degrees. For Pan-STARRS, the radius must be
        <= 0.5 degrees
    source : str, optional
        Source where to obtain a skymodel from. Can be one of: TGSS, GSM, LOTSS, or
        PANSTARRS. Note: the PANSTARRS sky model is only suitable for use in
        astrometry checks and should not be used for calibration
    overwrite : bool, optional
        Overwrite the existing skymodel pointed to by skymodel_path
    target_name : str, optional
        Give the patch a certain name
    """
    logger = logging.getLogger("LSMTool")

    file_exists = os.path.isfile(skymodel_path)
    if file_exists and not overwrite:
        logger.warning(
            'Sky model "%s" exists and overwrite is set to False! '
            'Not downloading sky model.', skymodel_path
        )
        return

    if not file_exists and os.path.exists(skymodel_path):
        raise ValueError('Path "%s" exists but is not a file!' % skymodel_path)

    # Empty strings are False. Only attempt directory creation if there is a
    # directory path involved.
    if (
        not file_exists
        and os.path.dirname(skymodel_path)
        and not os.path.exists(os.path.dirname(skymodel_path))
    ):
        os.makedirs(os.path.dirname(skymodel_path))

    if file_exists and overwrite:
        logger.warning(
            'Found existing sky model "{}" and overwrite is True. Deleting '
            "existing sky model!".format(skymodel_path)
        )
        os.remove(skymodel_path)

    # Check the radius for Pan-STARRS (it must be <= 0.5 degrees)
    source = source.upper().strip()
    if source == "PANSTARRS" and radius > 0.5:
        raise ValueError("The radius for Pan-STARRS must be <= 0.5 deg")

    # Check if LoTSS has coverage
    if source == "LOTSS":
        logger.info("Checking LoTSS coverage for the requested centre and radius.")
        mocpath = os.path.join(os.path.dirname(skymodel_path), "dr2-moc.moc")
        subprocess.run(
            [
                "wget",
                "https://lofar-surveys.org/public/DR2/catalogues/dr2-moc.moc",
                "-O",
                mocpath,
            ],
            capture_output=True,
            check=True,
        )
        moc = mocpy.MOC.from_fits(mocpath)
        covers_centre = moc.contains(ra * u.deg, dec * u.deg)

        # Checking single coordinates, so get rid of the array
        covers_left = moc.contains(ra * u.deg - radius * u.deg, dec * u.deg)[0]
        covers_right = moc.contains(ra * u.deg + radius * u.deg, dec * u.deg)[0]
        covers_bottom = moc.contains(ra * u.deg, dec * u.deg - radius * u.deg)[0]
        covers_top = moc.contains(ra * u.deg, dec * u.deg + radius * u.deg)[0]
        if covers_centre and not (
            covers_left and covers_right and covers_bottom and covers_top
        ):
            logger.warning(
                "Incomplete LoTSS coverage for the requested centre and radius! "
                "Please check the field coverage in plots/field_coverage.png!"
            )
        elif not covers_centre and (
            covers_left or covers_right or covers_bottom or covers_top
        ):
            logger.warning(
                "Incomplete LoTSS coverage for the requested centre and radius! "
                "Please check the field coverage in plots/field_coverage.png!"
            )
        elif not covers_centre and not (
            covers_left and covers_right and covers_bottom and covers_top
        ):
            raise ValueError("No LoTSS coverage for the requested centre and radius!")
        else:
            logger.info("Complete LoTSS coverage for the requested centre and radius.")

    logger.info("Downloading skymodel for the target into " + skymodel_path)
    max_tries = 5
    for tries in range(1, 1 + max_tries):
        retry = False
        if source == "LOTSS" or source == "TGSS" or source == "GSM":
            try:
                skymodel = SkyModel(source, VOPosition=[ra, dec], VORadius=radius)
                skymodel.write(skymodel_path)
                if len(skymodel) > 0:
                    break
            except ConnectionError:
                retry = True
        elif source == "PANSTARRS":
            baseurl = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"
            release = "dr1"  # the release with the mean data
            table = "mean"  # the main catalog, with the mean data
            cat_format = "csv"  # use csv format for the intermediate file
            url = f"{baseurl}/{release}/{table}.{cat_format}"
            search_params = {
                "ra": ra,
                "dec": dec,
                "radius": radius,
                "nDetections.min": "5",  # require detection in at least 5 epochs
                "columns": ["objID", "ramean", "decmean"],  # get only the info we need
            }
            try:
                result = requests.get(url, params=search_params, timeout=300)
                if result.ok:
                    # Convert the result to makesourcedb format and write to the output file
                    lines = result.text.split("\n")[1:]  # split and remove header line
                    out_lines = [
                        "FORMAT = Name, Ra, Dec, Type, I, ReferenceFrequency=1e6\n"
                    ]
                    for line in lines:
                        # Add entries for type and Stokes I flux density
                        if line.strip():
                            out_lines.append(line.strip() + ",POINT,0.0,\n")
                    with open(skymodel_path, "w") as f:
                        f.writelines(out_lines)
                    break
                else:
                    retry = True
            except requests.exceptions.Timeout:
                retry = True
        else:
            raise ValueError(
                "Unsupported sky model source specified! Please use LOTSS, TGSS, "
                "GSM, or PANSTARRS."
            )

        if retry:
            if tries == max_tries:
                logger.error(
                    "Attempt #{0:d} to download {1} sky model failed.".format(
                        tries, source
                    )
                )
                raise IOError(
                    "Download of {0} sky model failed after {1} attempts.".format(
                        source, max_tries
                    )
                )
            else:
                suffix = "s" if max_tries - tries > 1 else ""
                logger.error(
                    "Attempt #{0:d} to download {1} sky model failed. Attempting "
                    "{2:d} more time{3}.".format(
                        tries, source, max_tries - tries, suffix
                    )
                )
                time.sleep(5)

    if not os.path.isfile(skymodel_path):
        raise IOError(
            'Sky model file "{}" does not exist after trying to download the '
            "sky model.".format(skymodel_path)
        )

    # Treat all sources as one group (direction)
    skymodel = lsmtool.load(skymodel_path)
    skymodel.group("single", root=targetname)
    skymodel.write(clobber=True)
