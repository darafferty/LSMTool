"""
Test utility functions.
"""

import numpy as np
import pytest
from conftest import TEST_DATA_PATH

from lsmtool.utils import format_coordinates, read_vertices_ra_dec


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


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param(
            (path := TEST_DATA_PATH / "expected_sector_1_vertices.pkl"),
            id="path_input",
        ),
        pytest.param(str(path), id="string_input"),
    ],
)
def test_read_pickled_vertices(filename):
    """Test reading vertices from pickle file."""
    verts = read_vertices_ra_dec(filename)
    expected = (
        (265.2866140036157, 53.393467021582275),
        (266.78226621292583, 61.02229999320357),
        (250.90915045307418, 61.02229999320357),
        (252.40480266238433, 53.393467021582275),
        (265.2866140036157, 53.393467021582275),
    )
    assert verts == expected


import numpy as np
import pytest
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from lsmtool.utils import rasterize


@pytest.mark.parametrize(
    "verts, data_shape, blank_value, expected_array",
    [
        pytest.param(
            [(0, 0), (0, 1), (1, 1), (1, 0)],
            (4, 4),
            0,
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            id="simple_square",
        ),
        pytest.param(
            [(0, 0), (0, 2), (1, 1), (2, 2), (2, 0)],
            (4, 4),
            -1,
            [
                [1.0, 1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
            ],
            id="irregular_shape",
        ),
        pytest.param(
            [(0, 0), (1, 1), (2, 0)],
            (4, 4),
            0,
            [
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            id="triangle",
        ),
    ],
)
def test_rasterize_nominal(verts, data_shape, blank_value, expected_array):
    # Arrange
    data = np.ones(data_shape)

    # Act
    result = rasterize(verts, data, blank_value=blank_value)

    # Assert
    assert np.all(result == expected_array)


@pytest.mark.parametrize(
    "verts, data_shape, blank_value, expected_error",
    [
        pytest.param(
            [(0, 0), (0, "a")], (2, 2), 0, ValueError, id="invalid_vertex_type"
        ),
        pytest.param(
            "invalid", (2, 2), 0, ValueError, id="invalid_verts_type"
        ),
        pytest.param(
            [(0, 0), (0, 1)], "invalid", 0, ValueError, id="invalid_data"
        ),
        pytest.param(
            [(0, 0), (0, 1)],
            (2, 2),
            "invalid",
            ValueError,
            id="invalid_blank_value_type",
        ),
        pytest.param(
            [(0, 0), (0, 1)], (2,), 0, ValueError, id="invalid_data_shape"
        ),
    ],
)
def test_rasterize_error_cases(verts, data_shape, blank_value, expected_error):
    # Arrange
    data = np.ones(data_shape) if isinstance(data_shape, tuple) else data_shape

    # Act and Assert
    with pytest.raises(expected_error):
        rasterize(verts, data, blank_value=blank_value)
