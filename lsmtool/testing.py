"""
Utility functions used for testing.
"""

import numpy as np
from astropy.coordinates import Angle

from .io import load


def check_skymodels_equal(
    left_filename, right_filename, check_patch_names_sizes=True
):
    """
    Compares the contents of two skymodels to check for equality.

    This function loads two skymodels and compares their contents, ignoring
    comments since they contain log messages which vary depending on run time.

    Parameters
    ----------
    left_filename : str or pathlib.Path
        Path to the first skymodel file.
    right_filename : str or pathlib.Path
        Path to the second skymodel file.
    check_patch_names_sizes : bool
        Whether to check patch names and sizes.
    """
    left = load(str(left_filename))
    right = load(str(right_filename))

    # Check the default (static) values
    if left.getDefaultValues() != right.getDefaultValues():
        return False

    # Check column names (ignoring the Patch column if needed)
    ignore = set() if check_patch_names_sizes else {"Patch"}
    left_column_names = set(left.getColNames()) - ignore
    right_column_names = set(right.getColNames()) - ignore
    if left_column_names != right_column_names:
        return False

    # Check patch positions. If they are defined (not None), check if they are
    # the same shape, and if so, if they are approximately equal to within some
    # tolerance
    left_patch_pos = left.getPatchPositions()
    right_patch_pos = right.getPatchPositions()
    if left_patch_pos and right_patch_pos:
        if len(left_patch_pos) != len(right_patch_pos):
            return False

        if not np.allclose(
            # Need to convert list of lists of Angle object to Angle array for
            # element-wise comparison to work as expected here
            Angle(left_patch_pos.values()),
            Angle(right_patch_pos.values()),
        ):
            return False

    for name in left_column_names:
        left_values = left.getColValues(name)
        right_values = right.getColValues(name)
        equals = (
            np.isclose
            if np.issubdtype(left_values.dtype, np.inexact)
            else np.equal
        )
        if not equals(left_values, right_values).all():
            return False

    if check_patch_names_sizes and (
        np.any(left.getPatchNames() != right.getPatchNames())
        or np.any(left.getPatchSizes() != right.getPatchSizes())
    ):
        return False

    return True
