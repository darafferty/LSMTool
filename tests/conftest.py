"""
Configuration for python tests.
"""

import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import lsmtool

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "resources"


@pytest.fixture
def wsrt_test_ms(tmp_path):
    """Uncompresses test_midbands.ms into a temporary directory."""
    ms_name = "test_midbands.ms"
    untar(TEST_DATA_PATH / f"{ms_name}.tgz", tmp_path)
    return tmp_path / ms_name


def get_wsrt_measures():
    """
    Downloads and extracts WSRT measurement set if it does not exist locally.
    """
    measures_path = Path(TEST_DATA_PATH / "WSRT_Measures")
    if not measures_path.exists():
        print(f"Downloading WSRT measures to {measures_path}.")
        measures_path.mkdir()
        measures_tar = measures_path.with_suffix(".ztar")
        urllib.request.urlretrieve(
            f"https://www.astron.nl/iers/{measures_tar.name}",
            str(measures_tar),
        )

        # Extract the measurement set and delete compressed archive.
        untar(measures_tar, remove_archive=True)

    yield


def untar(filename, destination=None, remove_archive=False):
    """
    Uncompress the measurement set in the tgz file.

    Parameters
    ----------
    filename:  str or Path
        Name of the tar file.
    destination:  str or Path
        Path to extract the tar file to.
    """

    path = _check_file_exists(filename)

    # Default output folder in the same as input folder
    destination = destination or path.parent / path.stem

    # Uncompress the tgz file.
    with tarfile.open(path, "r:gz") as file:
        file.extractall(destination)

    # Remove the compressed archive if requested
    if remove_archive:
        path.unlink()


def assert_skymodels_are_equal(
    left_filename, right_filename, check_patch_names_sizes=True
):
    """
    Compares the contents of two skymodels.

    This function loads two skymodels and compares their contents, ignoring
    comments since they contain log messages which vary depending on run time.

    Parameters
    ----------
    left_filename : str or Path
        Path to the first skymodel file.
    right_filename : str or Path
        Path to the second skymodel file.
    check_patch_names_sizes : bool
        Whether to check patch names and sizes.
    """
    left = lsmtool.load(str(left_filename))
    right = lsmtool.load(str(right_filename))

    assert left.getDefaultValues() == right.getDefaultValues()
    assert left.getPatchPositions() == right.getPatchPositions()
    assert left.getColNames() == right.getColNames()
    for name in left.getColNames():
        left_values = left.getColValues(name)
        right_values = right.getColValues(name)
        if np.issubdtype(left_values.dtype, np.inexact):
            assert np.isclose(left_values, right_values).all()
        else:
            assert (left_values == right_values).all()

    if check_patch_names_sizes:
        assert (left.getPatchNames() == right.getPatchNames()).all()
        assert (left.getPatchSizes() == right.getPatchSizes()).all()


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
        path = _check_file_exists(TEST_DATA_PATH / filename)
        shutil.copy(path, target)


def _check_file_exists(path):
    path = Path(path)
    if not path.is_absolute():
        path = TEST_DATA_PATH / path

    if not path.exists() or not path.is_file():
        # Fails if the path does not exits or is not a file.
        raise FileNotFoundError(f"Not able to find file: '{path}'.")

    return path
