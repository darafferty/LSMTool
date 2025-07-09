"""
Configuration for python tests.
"""

import shutil
from pathlib import Path

import numpy as np

import lsmtool

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "resources"


def assert_skymodels_are_equal(
    left_filename, right_filename, check_patch_names_sizes=True
):
    """
    Compares the contents of two skymodels.

    This function loads two skymodels and compares their contents, ignoring
    comments since they contain log messages which vary depending on run time.

    Args:
        left_filename (str): Path to the first skymodel file.
        right_filename (str): Path to the second skymodel file.
        check_patch_names_sizes (bool): Whether to check patch names and sizes.
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

    if not path.is_file():
        raise FileNotFoundError(f"Not able to find file: '{path}'.")

    return path
