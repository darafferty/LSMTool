"""
Configuration for python tests.
"""

import contextlib
import inspect
import shutil
import tarfile
from pathlib import Path

import astropy.units as u
import mocpy
import pytest
from astropy.coordinates import Latitude, Longitude
from astropy.io import fits
from astropy.wcs import WCS

from lsmtool.io import PathLike, PathLikeOptional, check_file_exists

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "resources"


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

    test_module = inspect.getmodule(request._pyfuncitem.parent._obj)
    test_data_path = request.config.resource_dir / test_module.__name__
    return (
        test_data_path
        if test_data_path.exists()
        else request.config.resource_dir
    )


@pytest.fixture
def midbands_ms(tmp_path):
    """Uncompresses test_midbands.ms into a temporary directory."""
    ms_name = "test_midbands.ms"
    untar(TEST_DATA_PATH / f"{ms_name}.tgz", tmp_path)
    return tmp_path / ms_name


@pytest.fixture(scope="module")
def test_image_wcs():
    return WCS(fits.getheader(TEST_DATA_PATH / "test_image.fits"))


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
