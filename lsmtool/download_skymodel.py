"""
Module for functions to download sky models.
"""

import logging
import os
import time
from contextlib import suppress
from pathlib import Path

import astropy.units as u
import mocpy
import pyvo
import requests

import lsmtool
from lsmtool.skymodel import SkyModel

logger = logging.getLogger("LSMTool")

REQUEST_CONNECT_TIMEOUT = 10
REQUEST_READ_TIMEOUT = 300


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
        Survey to obtain a skymodel from. Can be one of:
        'GSM': Global Sky Model
        'LOTSS': LOFAR Two-Meter Sky Survey
        'NVSS': NRAO VLA Sky Survey
        'PANSTARRS': Pan-STARRS optical survey (only suitable for use in
            astrometry checks and should not be used for calibration)
        'TGSS': TIFR GMRT Sky Survey
        'VLSSR': VLA Low-Frequency Sky Survey Redux
        'WENSS': Westerbork Northern Sky Survey
    target_name : str, default="Patch"
        Give the patch a certain name.
    """
    skymodel_exists = _sky_model_already_exists(skymodel_path)

    if _download_not_required(skymodel_exists, overwrite):
        return

    _prepare_path_for_download(skymodel_path, skymodel_exists, overwrite)

    download_skymodel_from_survey(cone_params, survey, skymodel_path)

    _verify_download(skymodel_path)

    _group_sources_into_single_direction(skymodel_path, targetname)


def download_skymodel_from_survey(
    cone_params, survey, skymodel_path, retries=4, time_between_retries=5
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
        Source of the skymodel (e.g. "LOTSS", "TGSS", "GSM", "NVSS", "VLSSR",
        "WENSS", or "PANSTARRS").
    skymodel_path : str
        Path to the output skymodel file.
    retries : int, default=4
        Number of repeat attempts to download the skymodel.
    time_between_retries : int, default=5
        Time to wait between retries in seconds.

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

    for attempt in range(retries + 1):
        match survey:
            case "LOTSS" | "TGSS" | "GSM" | "NVSS" | "VLSSR" | "WENSS":
                success = download_skymodel_catalog(
                    cone_params, survey, skymodel_path
                )
            case "PANSTARRS":
                success = download_skymodel_panstarrs(
                    cone_params, skymodel_path
                )
            case _:
                raise ValueError(
                    "Unsupported sky model survey specified! "
                    "Please use LOTSS, TGSS, GSM, NVSS, VLSSR, WENSS, or "
                    "PANSTARRS."
                )
        if success:
            logger.info(
                "Download of %s sky model completed successfully.", survey
            )
            return

        if attempt < retries:
            retries_left = retries - attempt
            logger.error(
                "Attempt #%d to download %s sky model failed. "
                "Attempting %d more time%s.",
                attempt + 1,
                survey,
                retries_left,
                "s" if retries_left > 1 else "",
            )
            time.sleep(time_between_retries)

    logger.error(
        "Attempt #%d to download %s sky model failed.", attempt + 1, survey
    )
    raise IOError(
        f"Download of {survey} sky model failed after {retries + 1} attempts."
    )


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
        Source of the skymodel (must be one of "LOTSS", "TGSS", "GSM", "NVSS",
        "VLSSR", or "WENSS").

    Returns
    -------
    bool
        True if download was successful, False otherwise.
    """
    logger.info("Downloading skymodel from %s into %s", survey, skymodel_path)
    with suppress(ConnectionError):
        skymodel = SkyModel(
            survey,
            VOPosition=[cone_params["ra"], cone_params["dec"]],
            VORadius=cone_params["radius"],
        )
        skymodel.write(skymodel_path)
        if len(skymodel) > 0:
            return True
    return False


def get_panstarrs_request():
    """
    Create a Pan-STARRS VO URL.

    Returns
    -------
    url : str
        The Pan-STARRS VO URL.
    """
    url = "https://vizier.cds.unistra.fr/viz-bin/votable/"  # VO service URL
    url += "-A?-source=II/389/ps1_dr2&amp;"  # Pan-STARRS DR2 catalog
    url += "-out.max=unlimited&amp;"  # unlimited number of output lines
    url += "-out=objID&amp;"  # output objID
    url += "-out=RAJ2000&amp;-out=DEJ2000&amp;"  # output RA, Dec
    url += "Nd=5&amp;"  # require detection in at least 5 epochs
    return url


def download_skymodel_panstarrs(cone_params, skymodel_path):
    """
    Download a skymodel from the Pan-STARRS source.

    Parameters
    ----------
    cone_params : dict
        Dictionary containing the cone search parameters:
        'ra': Right ascension of the target position.
        'dec': Declination of the target position.
        'radius': Search radius in degrees.
    skymodel_path : str
        Path to the output skymodel file.

    Returns
    -------
    bool
        True if download was successful, False otherwise.
    """
    logger.info("Downloading skymodel from Pan-STARRS into %s", skymodel_path)
    try:
        url = get_panstarrs_request()
        result = pyvo.conesearch(
            url, [cone_params["ra"], cone_params["dec"]], cone_params["radius"]
        )
        if result.status[0] == "OK":
            # Convert the result to makesourcedb format and write to
            # the output file
            lines = [
                f"{row['objID']}, {row['RAJ2000']}, {row['DEJ2000']}"
                for row in result.to_table()
            ]
            out_lines = [
                "FORMAT = Name, Ra, Dec, Type, I, ReferenceFrequency=1e6\n"
            ]
            # Add entries for type and Stokes I flux density
            out_lines.extend(
                [
                    f"{clean_line},POINT,0.0,\n"
                    for raw_line in lines
                    if (clean_line := raw_line.strip())
                ]
            )
            with open(skymodel_path, "w", encoding="utf-8") as f:
                f.writelines(out_lines)
            return True
        return False
    except (pyvo.dal.exceptions.DALQueryError, pyvo.dal.DALServiceError) as exc:
        logger.warning("Pan-STARRS request failed: %s", exc)
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
    _check_moc_coverage(cone_params, moc)


def _prepare_path_for_download(skymodel_path, skymodel_exists, overwrite):
    """
    Prepare the path for downloading the sky model.

    Parameters
    ----------
    skymodel_path : str
        Full name (with path) to the output skymodel.
    overwrite : bool
        Whether to overwrite the existing sky model file.
    """
    if _overwrite_required(skymodel_exists, overwrite):
        os.remove(skymodel_path)

    _validate_skymodel_path(skymodel_path)

    Path(skymodel_path).parent.mkdir(parents=True, exist_ok=True)


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
    moc_url = "https://lofar-surveys.org/public/DR2/catalogues/dr2-moc.moc"
    try:
        response = requests.get(
            moc_url,
            timeout=(REQUEST_CONNECT_TIMEOUT, REQUEST_READ_TIMEOUT),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Failed to download LoTSS MOC file from {moc_url}: {exc}"
        ) from exc
    with open(mocpath, "wb") as fh:
        fh.write(response.content)

    return mocpy.MOC.from_fits(mocpath)


def _check_moc_coverage(
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


def _sky_model_already_exists(skymodel_path: str):
    """
    Check if the sky model file already exists and log a warning.

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
    if Path(skymodel_path).exists() and not Path(skymodel_path).is_file():
        raise ValueError(f'Path "{skymodel_path}" exists but is not a file!')


def _verify_download(skymodel_path: str):
    """
    Verify that the sky model file was downloaded successfully.

    Parameters
    ----------
    skymodel_path : str
        Full name (with path) to the output skymodel.

    Raises
    ------
    IOError
        If the sky model file does not exist after the download attempt.
    """
    if not Path(skymodel_path).is_file():
        raise IOError(
            f'Sky model file "{skymodel_path}" does not exist after trying to '
            "download the sky model."
        )


def _group_sources_into_single_direction(skymodel_path: str, target_name: str):
    """
    Group all sources in the sky model into a single direction.

    Parameters
    ----------
    skymodel_path : str
        Full name (with path) to the output skymodel.
    target_name : str
        Name to give to the single direction group.
    """
    skymodel = lsmtool.load(skymodel_path)
    skymodel.group("single", root=target_name)
    skymodel.write(clobber=True)
