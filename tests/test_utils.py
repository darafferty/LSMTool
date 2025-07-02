"""
Test utility function.
"""

import numpy as np
import pytest

from lsmtool.utils import format_coordinates


@pytest.mark.parametrize(
    "ra, dec, precision, expected_ra, expected_dec",
    [
        pytest.param(
            0, 0, 1, "0:00:00.0", "+0.00.00.0", id="first_point_of_aries"
        ),
        pytest.param(
            0, 90, 2, "0:00:00.00", "+90.00.00.00", id="north_celestial_pole"
        ),
        pytest.param(
            270.0,
            -90.0,
            6,
            "18:00:00.000000",
            "-90.00.00.000000",
            id="south_celestial_pole",
        ),
        pytest.param(
            180.0,
            -45.0,
            3,
            "12:00:00.000",
            "-45.00.00.000",
            id="west_hemisphere",
        ),
        pytest.param(
            266.416816625,
            -29.007824972,
            4,
            "17:45:40.0360",
            "-29.00.28.1699",
            id="Sgr_A*",
        ),
        pytest.param(
            375, 0, 1, "1:00:00.0", "+0.00.00.0", id="ra_normalised_over"
        ),
        pytest.param(
            -15, 0, 1, "23:00:00.0", "+0.00.00.0", id="ra_normalised_under"
        ),
        pytest.param(
            [0, 30, 60, 180],
            [10, 20, -40, -80],
            1,
            np.array(["0:00:00.0", "2:00:00.0", "4:00:00.0", "12:00:00.0"]),
            np.array(
                ["+10.00.00.0", "+20.00.00.0", "-40.00.00.0", "-80.00.00.0"]
            ),
            id="array_input",
        ),
    ],
)
def test_format_coordinates_nominal(
    ra, dec, precision, expected_ra, expected_dec
):
    # Act
    ra_str, dec_str = format_coordinates(ra, dec, precision=precision)

    # Assert
    assert np.all(ra_str == expected_ra)
    assert np.all(dec_str == expected_dec)


@pytest.mark.parametrize(
    "ra, dec, precision, exception",
    [
        pytest.param("a", 0.0, 6, ValueError, id="invalid_ra_value"),
        pytest.param(0.0, "b", 6, ValueError, id="invalid_dec_value"),
        pytest.param(0.0, 0.0, "c", TypeError, id="invalid_precision_type"),
        pytest.param(
            np.array([0.0, "a"]),
            0.0,
            6,
            ValueError,
            id="invalid_ra_element_value",
        ),
        pytest.param(
            0.0,
            np.array([0.0, "b"]),
            6,
            ValueError,
            id="invalid_dec_element_value",
        ),
        pytest.param(0.0, 100.0, 6, ValueError, id="dec_above_range"),
        pytest.param(0.0, -100.0, 6, ValueError, id="dec_below_range"),
    ],
)
def test_format_coordinates_error_cases(ra, dec, precision, exception):
    # Act and Assert
    with pytest.raises(exception):
        format_coordinates(ra, dec, precision=precision)
