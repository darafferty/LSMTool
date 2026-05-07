"""
Configuration for python tests.
"""

import contextlib
import shutil
import tarfile
from pathlib import Path

import astropy.units as u
import mocpy
import pytest
import requests
from astropy.coordinates import Latitude, Longitude
from astropy.io import fits
from astropy.wcs import WCS

from lsmtool.io import PathLike, PathLikeOptional, check_file_exists, load

# ---------------------------------------------------------------------------- #\
# Module constants

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "resources"

# Path to the LOFAR HBA mock measurement set
LOFAR_HBA_URL = "https://support.astron.nl/software/ci_data/EveryBeam/L258627-one-timestep.tar.bz2"


# ---------------------------------------------------------------------------- #


def pytest_configure(config):
    config.resource_dir = TEST_DATA_PATH


# ---------------------------------------------------------------------------- #
# Helper functions


def untar(
    filename: PathLike,
    destination: PathLikeOptional = None,
    remove_archive: bool = False,
):
    """
    Uncompress the measurement set in the tgz file.

    Parameters
    ----------
    filename:  str or pathlib.Path
        Name of the tar file.
    destination:  str or pathlib.Path
        Path to extract the tar file to.
    """

    path = check_file_exists(filename)

    # Default output folder is the same as the input folder.
    destination = destination or path.parent

    # Uncompress the tgz file.
    with tarfile.open(path, "r:gz") as file:
        file.extractall(destination, filter="data")

    # Remove the compressed archive if requested
    if remove_archive:
        path.unlink()


def copy_test_data(files_to_copy, target):
    """
    Copy a single file to a target path, or a list of files into target folder.
    The function emulates GNU's `cp` command-line program.
    The source files are assumed to be in the test data folder.
    """

    if isinstance(files_to_copy, (str, Path)):
        # copy single file
        files_to_copy = [files_to_copy]
    elif not target.is_dir():
        # copy multiple files - ensure target is a directory
        raise NotADirectoryError("Copy target is not a valid directory.")

    for filename in files_to_copy:
        path = check_file_exists(TEST_DATA_PATH / filename)
        shutil.copy(path, target)


def get_context(expected, **kws):
    """
    Get the appropriate runtime context for executing test code based on
    whether the expected result is an exception or not.

    Parameters
    ----------
    expected : Exception or object
        The expected result of the test. If this object is an exception class,
        the context manager will be `pytest.raises(expected, **kws)`. Otherwise,
        it will be a null context manager.

    Examples
    --------
    For tests that are expected to succeed:
    >>> @pytest.mark.parametrize("expected", [1])
    ... def test_success(expected):
    ...     with get_context(expected):
    ...         assert expected == 1

    For tests that are expected to fail:
    The following example will raise an IndexError, which will get caught by
    the `pytest.raises` context manager, leading to a successful test
    >>> @pytest.mark.parametrize("expected", [LookupError])
    ... def test_expected_failure(expected):
    ...     with get_context(expected):
    ...         [][1]

    Returns
    -------
    contextlib.AbstractContextManager
    """
    if isinstance(expected, type):
        if isinstance(expected, contextlib.AbstractContextManager):
            return expected

        if issubclass(expected, BaseException):
            return pytest.raises(expected, **kws)

    return contextlib.nullcontext(expected)


# ---------------------------------------------------------------------------- #
# Fixtures


@pytest.fixture(scope="module")
def test_data_path(request):
    """Path to the test data subfolder for the test module."""

    test_module_name = request.node.module.__name__
    test_data_path = request.config.resource_dir / test_module_name
    return (
        test_data_path
        if test_data_path.exists()
        else request.config.resource_dir
    )


@pytest.fixture
def midbands_ms(tmp_path, test_data_path):
    """Uncompresses test_midbands.ms into a temporary directory."""
    ms_name = "test_midbands.ms"
    untar(test_data_path / f"{ms_name}.tgz", tmp_path)
    return tmp_path / ms_name


@pytest.fixture(scope="module")
def test_ms_lofar_hba(test_data_path):
    """
    Fixture that provides the path to the LOFAR HBA mock measurement set. If
    the file does not exist, it will be downloaded and extracted from the
    specified URL.
    """
    path = test_data_path / "LOFAR_HBA_MOCK.ms"
    if path.exists():
        return path

    def filter_(member, _):
        return member.replace(name=Path(*Path(member.path).parts[1:]))

    with requests.get(LOFAR_HBA_URL, stream=True) as req:
        with tarfile.open(fileobj=req.raw, mode="r|bz2") as tarobj:
            tarobj.extractall(path=path, filter=filter_)

    return path


@pytest.fixture(scope="module")
def lofar_hba_skymodel(test_data_path):
    """Fixture that loads the skymodel data from the test data path."""
    return load(test_data_path / "LOFAR_HBA_MOCK.sky")


@pytest.fixture(scope="module")
def test_image_wcs(test_data_path):
    return WCS(fits.getheader(test_data_path / "test_image.fits"))


@pytest.fixture
def mock_moc():
    """Fixture that provides a mock MOC object for testing."""
    lon = Longitude([5, -5, -5, 5], u.deg)
    lat = Latitude([5, 5, -5, -5], u.deg)
    return mocpy.MOC.from_polygon(lon, lat)


@pytest.fixture
def cone_params():
    """Fixture that provides cone search parameters for testing."""
    return {"ra": 190.0, "dec": 44.0, "radius": 1.0}
