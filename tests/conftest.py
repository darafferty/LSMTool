"""
Configuration for python tests.
"""

import shutil
import tarfile
from pathlib import Path
from astropy.coordinates import Latitude, Longitude
import astropy.units as u
import pytest
import mocpy
from lsmtool.io import check_file_exists, PathLike, PathLikeOptional
from astropy.io import fits
from astropy.wcs import WCS

from lsmtool.io import PathLike, PathLikeOptional, check_file_exists

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "resources"


@pytest.fixture
def midbands_ms(tmp_path):
    """Uncompresses test_midbands.ms into a temporary directory."""
    ms_name = "test_midbands.ms"
    untar(TEST_DATA_PATH / f"{ms_name}.tgz", tmp_path)
    return tmp_path / ms_name


@pytest.fixture(scope="session")
def test_image_wcs():
    return WCS(fits.getheader(TEST_DATA_PATH / "test_image.fits"))


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


@pytest.fixture
def existing_skymodel_filepath(tmp_path):
    """Fixture that provides a path to an existing sky model file."""
    file_path = tmp_path / "existing_sky.model"
    file_path.write_text("This is a test sky model.")
    return file_path


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
