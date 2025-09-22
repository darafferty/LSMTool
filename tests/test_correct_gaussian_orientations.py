from math import pi

import astropy.units as u
import numpy as np
import pytest

from lsmtool.correct_gaussian_orientation import compute_absolute_orientation


@pytest.mark.parametrize(
    "relative_orientation, ra_source, dec_source, ra_center, dec_center,"
    "expected",
    [
        # source and image center are the same
        pytest.param(
            *np.radians([0, 150, 20, 150, 20, 0]),
            id="source_equals_center_zero_orientation",
        ),
        # source and image center are the same, non-zero relative orientation
        pytest.param(
            *[45, 120, 15, 120, 15, 45] * u.deg,
            id="source_equals_center_non_zero_orientation",
        ),
        # realistic values
        pytest.param(
            *np.radians([45, 150, 20, 160, 25, 41.58065681548678]),
            id="realistic_values",
        ),
        # edge case: source and center at opposite RA, on equator
        pytest.param(
            pi / 2,
            0,
            0,
            pi,
            0,
            pi / 2,
            id="source_and image_center_on_equator_opposing",
        ),
        # edge case: source at NCP, center on equator
        pytest.param(
            pi / 2,
            0,
            pi / 2,
            0,
            0,
            pi / 2,
            id="source_at_ncp",
        ),
    ],
)
def test_compute_absolute_orientation(
    relative_orientation,
    ra_source,
    dec_source,
    ra_center,
    dec_center,
    expected,
):
    # Act
    result = compute_absolute_orientation(
        relative_orientation, ra_source, dec_source, ra_center, dec_center
    )

    # Assert
    assert result == expected
