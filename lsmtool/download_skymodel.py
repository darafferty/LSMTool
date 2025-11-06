"""
Module for functions to download sky models.
"""

import logging
import os
import time
from pathlib import Path

import astropy.units as u
import mocpy
import requests

import lsmtool
from lsmtool.skymodel import SkyModel

logger = logging.getLogger("LSMTool")


def download_skymodel(
    cone_params,
    skymodel_path,
    overwrite=False,
    survey="TGSS",
    targetname="Patch",
):
    """
    Download a skymodel for the given position and radius.

    Parameters
    ----------
    cone_params : dict
        Dictionary containing the cone search parameters:
        'ra': Right ascension of the target position.
        'dec': Declination of the target position.
        'radius': Search radius in degrees.
    skymodel_path : str
        Full name (with path) to the output skymodel.
    overwrite : bool, optional
        Overwrite the existing skymodel pointed to by skymodel_path.
    survey : str, optional
        Survey to obtain a skymodel from. Can be one of: TGSS, GSM,
        LOTSS, or PANSTARRS. Note: the PANSTARRS sky model is only suitable
        for use in astrometry checks and should not be used for calibration.
    target_name : str, default="Patch"
        Give the patch a certain name.
    """
    skymodel_exists = _sky_model_exists(skymodel_path)

    if _download_not_required(skymodel_exists, overwrite):
        return

    _validate_skymodel_path(skymodel_path)

    if _overwrite_required(skymodel_exists, overwrite):
        os.remove(skymodel_path)

    if _new_directory_required(skymodel_path):
        Path(skymodel_path).parent.mkdir(parents=True, exist_ok=True)

    download_skymodel_from_survey(cone_params, survey, skymodel_path)

    if not Path(skymodel_path).is_file():
        raise IOError(
            f'Sky model file "{skymodel_path}" does not exist after trying to '
            "download the sky model."
        )

    # Treat all sources as one group (direction)
    skymodel = lsmtool.load(skymodel_path)
    skymodel.group("single", root=targetname)
    skymodel.write(clobber=True)


def download_skymodel_from_survey(
    cone_params, survey, skymodel_path, max_tries=5
):
    """
    Download a skymodel from the specified source.

    Parameters
    ----------
    cone_params : dict
        Dictionary containing the cone search parameters:
            'ra': Right ascension of the target position.
            'dec': Declination of the target position.
            'radius': Search radius in degrees.
    survey : str
        Source of the skymodel (e.g. "LOTSS", "TGSS", "GSM", "PANSTARRS").
    skymodel_path : str
        Path to the output skymodel file.
    max_tries : int, default=5
        Maximum number of attempts to download the skymodel.

    Raises
    ------
    IOError
        If the download fails after the maximum number of attempts.
    ValueError
        If an unsupported sky model source is specified.
    """
    survey = survey.upper().strip()
    if survey == "LOTSS":
        check_lotss_coverage(cone_params, skymodel_path)

    logger.info("Downloading skymodel for the target into %s", skymodel_path)
    for tries in range(1, 1 + max_tries):
        if survey in ("LOTSS", "TGSS", "GSM"):
            success = download_skymodel_catalog(
                cone_params, survey, skymodel_path
            )
        elif survey == "PANSTARRS":
            url, search_params = get_panstarrs_request(cone_params)
            success = download_skymodel_panstarrs(
                url, search_params, skymodel_path
            )
        else:
            raise ValueError(
                "Unsupported sky model survey specified! "
                "Please use LOTSS, TGSS, GSM, or PANSTARRS."
            )
        if success:
            break
        if tries == max_tries:
            logger.error(
                "Attempt #%d to download %s sky model failed.", tries, survey
            )
            raise IOError(
                f"Download of {survey} sky model failed after "
                f"{max_tries} attempts."
            )
        suffix = "s" if max_tries - tries > 1 else ""
        logger.error(
            "Attempt #%d to download %s sky model failed. "
            "Attempting %d more time%s.",
            tries,
            survey,
            max_tries - tries,
            suffix,
        )
        time.sleep(5)


def download_skymodel_catalog(cone_params, survey, skymodel_path):
    """
    Download a skymodel from the specified source catalog.
    Parameters
    ----------
    cone_params : dict
        Dictionary containing the cone search parameters:
        'ra': Right ascension of the target position.
        'dec': Declination of the target position.
        'radius': Search radius in degrees.
    skymodel_path : str
        Path to the output skymodel file.
    survey : str
        Source of the skymodel (must be one of "LOTSS", "TGSS", "GSM").

    Returns
    -------
    bool
        True if download was successful, False otherwise.
    """
    logger.info("Downloading skymodel from %s into %s", survey, skymodel_path)
    try:
        skymodel = SkyModel(
            survey,
            VOPosition=[cone_params["ra"], cone_params["dec"]],
            VORadius=cone_params["radius"],
        )
        skymodel.write(skymodel_path)
        if len(skymodel) > 0:
            return True
    except ConnectionError:
        return False
    return False


def get_panstarrs_request(cone_params):
    """
    Create a Pan-STARRS request URL and parameters.

    Parameters
    ----------
    cone_params : dict
        Dictionary containing the cone search parameters:
        'ra': Right ascension of the target position.
        'dec': Declination of the target position.
        'radius': Search radius in degrees.

    Returns
    -------
    url : str
        The Pan-STARRS API URL.
    search_params : dict
        The search parameters for the request.

    Raises
    ------
    ValueError
        If the radius is greater than 0.5 degrees.
    """
    if cone_params["radius"] > 0.5:
        raise ValueError("The radius for Pan-STARRS must be <= 0.5 deg")
    baseurl = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"
    release = "dr1"  # the release with the mean data
    table = "mean"  # the main catalog, with the mean data
    cat_format = "csv"  # use csv format for the intermediate file
    url = f"{baseurl}/{release}/{table}.{cat_format}"
    search_params = {
        "ra": cone_params["ra"],
        "dec": cone_params["dec"],
        "radius": cone_params["radius"],
        "nDetections.min": "5",  # require detection in at least 5 epochs
        "columns": ["objID", "ramean", "decmean"],  # get only the info we need
    }
    return url, search_params


def download_skymodel_panstarrs(url, search_params, skymodel_path):
    """
    Download a skymodel from the Pan-STARRS source.

    Parameters
    ----------
    url : str
        The Pan-STARRS API URL.
    search_params : dict
        The search parameters for the request.
    skymodel_path : str
        Path to the output skymodel file.

    Returns
    -------
    bool
        True if download was successful, False otherwise.

    Raises
    ------
    requests.exceptions.Timeout
        If the request times out.
    """
    logger.info("Downloading skymodel from Pan-STARRS into %s", skymodel_path)
    timeout = 300
    try:
        result = requests.get(url, params=search_params, timeout=timeout)
        if result.ok:
            # Convert the result to makesourcedb format and write to
            # the output file. Split and remove header line.
            lines = result.text.split("\n")[1:]
            out_lines = [
                "FORMAT = Name, Ra, Dec, Type, I, ReferenceFrequency=1e6\n"
            ]
            # Add entries for type and Stokes I flux density
            out_lines.extend(
                [f"{l},POINT,0.0,\n" for line in lines if (l := line.strip())]
            )
            with open(skymodel_path, "w", encoding="utf-8") as f:
                f.writelines(out_lines)
            return True
        return False
    except requests.exceptions.Timeout:
        logger.warning("Request timed out after %d seconds", timeout)
        return False


def check_lotss_coverage(cone_params, skymodel_path):
    """
    Check if LoTSS has coverage for the given position and radius.

    Parameters
    ----------
    cone_params : dict
        Dictionary containing the cone search parameters:
            'ra': Right ascension of the target position.
            'dec': Declination of the target position.
            'radius': Search radius in degrees.
    skymodel_path : str
        Full name (with path) to the output skymodel.

    Raises
    ------
    ValueError
        If there is no LoTSS coverage for the requested centre and radius.
    ConnectionError
        If the LoTSS MOC file cannot be downloaded.
    """
    logger.info("Checking LoTSS coverage for the requested centre and radius.")

    moc = _get_lotss_moc(skymodel_path)
    _check_coverage(cone_params, moc)


def _get_lotss_moc(skymodel_path):
    """
    Download and return the LoTSS MOC (Multi-Order Coverage map).

    Parameters
    ----------
    skymodel_path : str
        Full name (with path) to the output skymodel.

    Returns
    -------
    mocpy.MOC
        The LoTSS MOC object.

    Raises
    ------
    ConnectionError
        If the LoTSS MOC file cannot be downloaded.
    """
    mocpath = Path(skymodel_path).parent / "dr2-moc.moc"
    # Securely download the MOC file without spawning an external process.
    # (Fix for security lint S607: avoid subprocess with partial
    # executable path.)
    moc_url = "https://lofar-surveys.org/public/DR2/catalogues/dr2-moc.moc"
    try:
        response = requests.get(moc_url, timeout=300)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Failed to download LoTSS MOC file from {moc_url}: {exc}"
        ) from exc
    with open(mocpath, "wb") as fh:
        fh.write(response.content)

    return mocpy.MOC.from_fits(mocpath)


def _check_coverage(
    cone_params,
    moc,
):
    """
    Check if the MOC has coverage for the given position and radius.

    Parameters
    ----------
    cone_params : dict
        Dictionary containing the cone search parameters:
            'ra': Right ascension of the target position.
            'dec': Declination of the target position.
            'radius': Search radius in degrees.
    moc : mocpy.MOC
        The MOC object to check coverage against.

    Raises
    ------
    ValueError
        If there is no LoTSS coverage for the requested centre and radius.
    """
    ra = cone_params["ra"] * u.deg
    dec = cone_params["dec"] * u.deg
    radius = cone_params["radius"] * u.deg

    # Checking single coordinates, so get rid of the array
    covers_centre = moc.contains_lonlat(ra, dec)[0]
    covers_left = moc.contains_lonlat(ra - radius, dec)[0]
    covers_right = moc.contains_lonlat(ra + radius, dec)[0]
    covers_bottom = moc.contains_lonlat(ra, dec - radius)[0]
    covers_top = moc.contains_lonlat(ra, dec + radius)[0]

    covers_all = all(
        [covers_centre, covers_left, covers_right, covers_bottom, covers_top]
    )

    covers_none = not any(
        [covers_centre, covers_left, covers_right, covers_bottom, covers_top]
    )

    covers_partial = not (covers_all or covers_none)

    if covers_partial:
        logger.warning(
            "Incomplete LoTSS coverage for the requested centre and radius! "
            "Please check the field coverage in plots/field_coverage.png!"
        )
    elif covers_none:
        raise ValueError(
            "No LoTSS coverage for the requested centre and radius!"
        )
    else:
        logger.info(
            "Complete LoTSS coverage for the requested centre and radius."
        )


def _download_not_required(skymodel_exists: bool, overwrite: bool):
    """
    Check if sky model exists and should not be overwritten.

    Parameters
    ----------
    skymodel_exists : bool
        Whether the sky model exists.
    overwrite : bool, optional
        Overwrite the existing skymodel pointed to by skymodel_path.

    Returns
    -------
    bool
        True if sky model download not required, False otherwise.
    """
    if skymodel_exists and not overwrite:
        logger.warning(
            "Download skipped! "
            "Sky model already exists and overwrite is set to False."
        )
        return True
    return False


def _sky_model_exists(skymodel_path: str):
    """
    Check if the sky model file exists.

    Parameters
    ----------
    skymodel_path : str
        Full name (with path) to the output skymodel.

    Returns
    -------
    bool
        True if the sky model file exists, False otherwise.
    """
    file_exists = Path(skymodel_path).is_file()
    if file_exists:
        logger.warning('Sky model "%s" exists!', skymodel_path)
    return file_exists


def _new_directory_required(skymodel_path: str):
    """
    Check if the given path is a new directory that needs to be created.

    Parameters
    ----------
    skymodel_path : str
        The path to check.

    Returns
    -------
    bool
        True if the skymodel_path is a valid directory, False otherwise.
    """
    # Empty strings are False. Only attempt directory creation if there is a
    # directory path involved.
    return (
        not Path(skymodel_path).is_file()
        and not Path(skymodel_path).parent.exists()
    )


def _overwrite_required(skymodel_exists: bool, overwrite: bool):
    """
    Check if existing sky model file should be removed.

    Parameters
    ----------
    skymodel_exists : bool
        Whether the sky model exists
    overwrite : bool, optional
        Whether to overwrite the existing skymodel.

    Returns
    -------
    bool
        True if existing sky model should be removed, False otherwise.
    """
    if skymodel_exists and overwrite:
        logger.warning(
            "Found existing sky model and overwrite is True. Deleting "
            "existing sky model!"
        )
        return True
    return False


def _validate_skymodel_path(skymodel_path: str):
    """Validate the sky model path.

    Parameters
    ----------
    skymodel_path : str
        Full name (with path) to the output skymodel.

    Raises
    ------
    ValueError
        If the skymodel_path exists but is not a file.
    """
    if not Path(skymodel_path).is_file() and Path(skymodel_path).exists():
        raise ValueError(f'Path "{skymodel_path}" exists but is not a file!')
