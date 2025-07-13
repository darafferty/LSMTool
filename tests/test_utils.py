"""
Test utility functions.
"""

import sys

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from conftest import TEST_DATA_PATH

from lsmtool.skymodel import SkyModel
from lsmtool.utils import (
    format_coordinates,
    rasterize,
    rotation_matrix_2d,
    table_to_array,
    transfer_patches,
)


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
    "verts, data_shape, blank_value, expected_array",
    [
        pytest.param(
            [(0, 0), (0, 1), (1, 1), (1, 0)],
            (4, 4),
            0,
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            id="square",
        ),
        pytest.param(
            [(0, 0), (0, 2), (1, 1), (2, 2), (2, 0)],
            (4, 4),
            -1,
            [
                [1, 1, 1, -1],
                [1, 1, 1, -1],
                [1, -1, 1, -1],
                [-1, -1, -1, -1],
            ],
            id="irregular_shape",
        ),
        pytest.param(
            [(3.0, 0.0), (2, 1), (1, 0)],
            (4, 4),
            0,
            [
                [0, 1, 1, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            id="triangle",
        ),
        pytest.param(
            [(0, 0), (1, 1), (0, 0)],
            (2, 2),
            0,
            [[1, 0], [0, 1]],
            id="line",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9), reason="Fails in python<=3.8"
            ),
        ),
        pytest.param(
            [(0, 0), (0, 0), (0, 0)],
            (2, 2),
            0,
            [[1, 0], [0, 0]],
            id="point",
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
    "theta, expected_matrix",
    [
        pytest.param(0, [[1, 0], [0, 1]], id="zero"),
        pytest.param(np.pi / 2, [[0, -1], [1, 0]], id="pi_over_2"),
        pytest.param(np.pi, [[-1, 0], [0, -1]], id="pi"),
        pytest.param(-np.pi / 2, [[0, 1], [-1, 0]], id="minus_pi_over_2"),
        pytest.param(2 * np.pi, [[1, 0], [0, 1]], id="two_pi"),
        pytest.param(
            np.pi / 4,
            [
                [np.sqrt(2) / 2, -np.sqrt(2) / 2],
                [np.sqrt(2) / 2, np.sqrt(2) / 2],
            ],
            id="pi_over_4",
        ),
    ],
)
def test_rotation_matrix_2d(theta, expected_matrix):
    """Test cases for valid theta values."""

    # Act
    result = rotation_matrix_2d(theta)

    # Assert
    np.testing.assert_allclose(result, expected_matrix, atol=1e-12)


@pytest.mark.parametrize(
    "table_data, dtype, expected_shape, expected_dtype",
    [
        (  # two_rows_float
            {"col1": [1, 2], "col2": [3, 4]},
            int,
            (2, 2),
            int,
        ),
        (  # two_rows_int
            {"col1": [1.0, 2.0], "col2": [3.0, 4.0]},
            float,
            (2, 2),
            float,
        ),
        (  # one_row_float
            {"col1": [1.0], "col2": [2.0]},
            float,
            (1, 2),
            float,
        ),
    ],
)
def test_table_to_array(table_data, dtype, expected_shape, expected_dtype):
    # Arrange
    table = Table(table_data)

    # Act
    result = table_to_array(table, dtype=dtype)

    expected_result = np.transpose(list(table_data.values()))

    # Assert
    assert result.shape == expected_shape
    assert result.dtype == expected_dtype
    assert np.all(result == expected_result)


def test_transfer_patches():
    # Arrange
    from_skymodel = SkyModel(str(TEST_DATA_PATH / "transfer_patches_from.sky"))
    to_skymodel = SkyModel(str(TEST_DATA_PATH / "transfer_patches_to.sky"))

    # Act
    transfer_patches(from_skymodel, to_skymodel)

    # Assert
    expected_patches = np.char.add(["Patch"], np.arange(1, 7).astype(str))
    assert all(to_skymodel.table["Patch"] == expected_patches)


def test_transfer_patches_no_patches():
    # Arrange
    from_skymodel = SkyModel(str(TEST_DATA_PATH / "no_patches.sky"))
    to_skymodel = SkyModel(str(TEST_DATA_PATH / "transfer_patches_to.sky"))

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="Cannot transfer patches since from_skymodel is not grouped into"
        " patches.",
    ):
        transfer_patches(from_skymodel, to_skymodel)


def test_transfer_patches_non_matching_skymodels():
    # Arrange
    from_skymodel = SkyModel(str(TEST_DATA_PATH / "transfer_patches_from.sky"))
    to_skymodel = SkyModel(
        str(TEST_DATA_PATH / "transfer_patches_no_overlap.sky")
    )

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="Cannot transfer patches since neither sky model is a subset of "
        "the other.",
    ):
        transfer_patches(from_skymodel, to_skymodel)


def test_transfer_patches_with_patch_dict():
    # Arrange
    from_skymodel = SkyModel(str(TEST_DATA_PATH / "transfer_patches_from.sky"))
    to_skymodel = SkyModel(str(TEST_DATA_PATH / "transfer_patches_to.sky"))

    patch_dict = {
        "Patch1": ["17:23:39.8208", "+52.36.48.8520"],
        "Patch2": ["17:20:06.5856", "+52.34.22.6200"],
        "Patch3": ["17:17:20.3688", "+52.29.08.7000"],
        "Patch4": ["17:17:08.5320", "+52.29.09.8520"],
        "Patch5": ["17:42:37.3608", "+54.11.24.1080"],
        "Patch6": ["17:41:21.3432", "+54.43.53.1480"],
    }

    # Act
    transfer_patches(from_skymodel, to_skymodel, patch_dict=patch_dict)

    # Assert
    expected_patches = list(patch_dict.keys())
    assert np.all(to_skymodel.table["Patch"] == expected_patches)

    # Get the patch positions from the sky model
    pos = to_skymodel.getPatchPositions()
    coords = SkyCoord(list(pos.values()))
    ra, dec = format_coordinates(coords.ra, coords.dec, precision=4)

    # Check that the patch positions match
    ra_in, dec_in = zip(*patch_dict.values())
    assert np.all([ra.ravel() == ra_in, dec.ravel() == dec_in])
