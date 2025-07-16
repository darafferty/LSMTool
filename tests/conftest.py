"""
Configuration for python tests.
"""

import shutil
from pathlib import Path

import pytest
from lsmtool.io import check_file_exists, untar


TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "resources"


@pytest.fixture
def midband_ms(tmp_path):
    """Uncompresses test_midbands.ms into a temporary directory."""
    ms_name = "test_midbands.ms"
    untar(TEST_DATA_PATH / f"{ms_name}.tgz", tmp_path)
    return tmp_path / ms_name


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
