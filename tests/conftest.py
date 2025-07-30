"""
Configuration for python tests.
"""

import shutil
import tarfile
from pathlib import Path

import pytest
from lsmtool.io import check_file_exists, PathLike, PathLikeOptional


TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "resources"


@pytest.fixture
def midbands_ms(tmp_path):
    """Uncompresses test_midbands.ms into a temporary directory."""
    ms_name = "test_midbands.ms"
    untar(TEST_DATA_PATH / f"{ms_name}.tgz", tmp_path)
    return tmp_path / ms_name


def untar(
    filename: PathLike,
    destination: PathLikeOptional = None,
    remove_archive: bool = False,
):
    """
    Uncompress the measurement set in the tgz file.

    Parameters
    ----------
    filename:  str or Path
        Name of the tar file.
    destination:  str or Path
        Path to extract the tar file to.
    """

    path = check_file_exists(filename)

    # Default output folder is the same as the input folder.
    destination = destination or path.parent

    # Uncompress the tgz file.
    with tarfile.open(path, "r:gz") as file:
        file.extractall(destination)

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
