"""
Utility functions used for testing.
"""

import numpy as np

from .io import load


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
    left = load(str(left_filename))
    right = load(str(right_filename))

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
