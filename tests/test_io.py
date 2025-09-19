import contextlib as ctx
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from astropy.wcs import WCS
from conftest import TEST_DATA_PATH

from lsmtool.io import (
    _restore_tmpdir,
    _set_tmpdir,
    convert_coordinates_to_pixels,
    read_vertices_ra_dec,
    read_vertices_x_y,
    temp_storage,
)


@pytest.mark.parametrize(
    "trial_paths, expected_tmpdir, context",
    [
        # Nominal test cases
        pytest.param(["/tmp"], "/tmp", None, id="path_tmp"),
        pytest.param(
            ["/var/tmp", "/tmp"], "/var/tmp", None, id="path_var_tmp"
        ),
        pytest.param(
            ["/usr/tmp", "/var/tmp", "/tmp"],
            "/usr/tmp" if Path("/usr/tmp").exists() else "/var/tmp",
            None,
            id="path_usr_tmp",
        ),
        # Error test cases
        pytest.param(
            None,
            None,
            pytest.raises(NotADirectoryError),
            id="invalid_trial_paths_none",
        ),
        pytest.param(
            [],
            [],
            pytest.raises(NotADirectoryError),
            id="invalid_trial_paths_empty",
        ),
    ],
)
@patch("lsmtool.io._restore_tmpdir", wraps=_restore_tmpdir)
@patch("lsmtool.io._set_tmpdir", wraps=_set_tmpdir)
def test_temp_storage(
    mock_set_tmpdir, mock_restore_tmpdir, trial_paths, expected_tmpdir, context
):
    """Test the temp_storage context manager."""

    # Act
    with context or ctx.nullcontext():
        with temp_storage(trial_paths) as tmp:
            assert os.environ["TMPDIR"] == expected_tmpdir == str(tmp)

    # Assert
    mock_set_tmpdir.assert_called_once_with(trial_paths)
    mock_restore_tmpdir.assert_called_once()


# ---------------------------------------------------------------------------- #
EXPECTED_VERTICES_XY = [
    (23.750160560517628, 23.750160560516235),
    (23.75016056051743, 476.24983943948354),
    (476.24983943948394, 476.24983943948325),
    (476.24983943948195, 23.75016056051578),
    (23.750160560517628, 23.750160560516235),
]
EXPECTED_VERTICES_RA_DEC = (
    (265.2866140036157, 53.393467021582275),
    (266.78226621292583, 61.02229999320357),
    (250.90915045307418, 61.02229999320357),
    (252.40480266238433, 53.393467021582275),
    (265.2866140036157, 53.393467021582275),
)


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param(
            (TEST_DATA_PATH / "expected_sector_1_vertices.npy"),
            id="path_input",
        ),
        pytest.param(
            str(TEST_DATA_PATH / "expected_sector_1_vertices.npy"),
            id="string_input",
        ),
    ],
)
def test_read_vertices_ra_dec(filename):
    """Test reading vertices from npy file."""
    verts = read_vertices_ra_dec(filename)
    np.testing.assert_allclose(verts, EXPECTED_VERTICES_RA_DEC)


def test_read_vertices_x_y(test_image_wcs):
    filename = TEST_DATA_PATH / "expected_sector_1_vertices.npy"
    vertices_pixel = read_vertices_x_y(filename, test_image_wcs)
    np.testing.assert_allclose(vertices_pixel, EXPECTED_VERTICES_XY)


@pytest.fixture(params=["Invalid content", ["Invalid", "content"]])
def invalid_vertices_file(request, tmp_path):
    """Generate vertices file with invalid content."""
    path = tmp_path / "invalid_vertices.npy"
    path.unlink(missing_ok=True)
    np.save(path, request.param, allow_pickle=False)
    return path


@pytest.mark.parametrize(
    "reader, wcs", [(read_vertices_ra_dec, ()), (read_vertices_x_y, (WCS(),))]
)
def test_read_vertices_invalid(reader, wcs, invalid_vertices_file):
    """Test reading vertices from npy file with invalid content."""
    with pytest.raises(ValueError):
        reader(invalid_vertices_file, *wcs)


@pytest.mark.parametrize(
    "reader, wcs", [(read_vertices_ra_dec, ()), (read_vertices_x_y, (WCS(),))]
)
def test_read_vertices_non_existent(reader, wcs):
    non_existent_file = "/path/to/vertices.file"
    with pytest.raises(FileNotFoundError):
        reader(non_existent_file, *wcs)


@pytest.fixture(
    params=[
        wcs_params_2d := {
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CRVAL1": (RA := -101.154291667),
            "CRVAL2": (DEC := 57.4111944444),
            "CRPIX1": (CRPIX := 251.0),
            "CRPIX2": CRPIX,
            "CDELT1": -(CDELT := 0.01694027),
            "CDELT2": CDELT,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
        },
        {
            **wcs_params_2d,
            "CTYPE3": "FREQ",
            "CRPIX3": 1.0,
            "CRVAL3": 143650817.871094,
            "CDELT3": 11718750.0,
            "CUNIT3": "Hz",
            "CTYPE4": "STOKES",
            "CRPIX4": 1.0,
            "CRVAL4": 1.0,
            "CDELT4": 1.0,
            "CUNIT4": "",
        },
    ],
    ids=["2d", "4d"],
)
def wcs(request):
    return WCS(request.param)


@pytest.mark.parametrize(
    "coordinates, pixels_expected", [(np.array([[RA, DEC]]), [(250.0, 250.0)])]
)
def test_convert_coordinates_to_pixels(coordinates, pixels_expected, wcs):
    result = convert_coordinates_to_pixels(coordinates, wcs)
    np.testing.assert_equal(result, pixels_expected)
