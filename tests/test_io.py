import contextlib as ctx
import os
from pathlib import Path
from unittest.mock import patch

import lsmtool
import numpy as np
import pytest
from astropy.wcs import WCS
from conftest import TEST_DATA_PATH, copy_test_data

from lsmtool.io import (
    _restore_tmpdir,
    _set_tmpdir,
    _sky_model_exists,
    _new_directory_required,
    _validate_skymodel_path,
    _overwrite_required,
    _download_not_required,
    convert_coordinates_to_pixels,
    download_skymodel,
    download_skymodel_panstarrs,
    get_panstarrs_request,
    download_skymodel_catalog,
    download_skymodel_from_source,
    read_vertices_ra_dec,
    read_vertices_x_y,
    temp_storage,
    check_lotss_coverage,
)


@pytest.mark.parametrize(
    "trial_paths, expected_tmpdir, context",
    [
        # Nominal test cases
        pytest.param(["/tmp"], "/tmp", None, id="path_tmp"),
        pytest.param(["/var/tmp", "/tmp"], "/var/tmp", None, id="path_var_tmp"),
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


@pytest.fixture(
    params=[
        "Invalid content",
        ["Invalid", "content"],
        np.array([np.array([1, 2]), np.array([2, 3, 4])], object),
    ]
)
def invalid_vertices_file(request, tmp_path):
    """Generate vertices file with invalid content."""
    path = tmp_path / "invalid_vertices.npy"
    path.unlink(missing_ok=True)
    np.save(path, request.param)
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


@pytest.mark.parametrize("ra", (10.75,))
@pytest.mark.parametrize("dec", (5.34,))
@pytest.mark.parametrize("radius", (5.0,))
@pytest.mark.parametrize("overwrite", (False,))
@pytest.mark.parametrize("source", ("TGSS",))
@pytest.mark.parametrize("targetname", ("Patch",))
def test_download_skymodel(ra, dec, tmp_path, radius, overwrite, source, targetname):
    """Test downloading a sky model."""

    # Arrange
    copy_test_data("expected.tgss.sky.model", tmp_path)
    downloaded_skymodel_path = tmp_path / "sky.model"
    expected_skymodel_path = tmp_path / "expected.tgss.sky.model"
    skymodel_expected = lsmtool.load(str(expected_skymodel_path))
    cone_params = {"ra": ra, "dec": dec, "radius": radius}

    # Act
    download_skymodel(
        cone_params, str(downloaded_skymodel_path), overwrite, source, targetname
    )
    skymodel_downloaded = lsmtool.load(str(downloaded_skymodel_path))

    # Assert
    assert list(skymodel_downloaded.table.columns) == list(
        skymodel_expected.table.columns
    )
    assert len(skymodel_downloaded) == len(skymodel_expected)
    for col in skymodel_expected.table.columns:
        assert col in skymodel_downloaded.table.columns
        assert all(
            skymodel_downloaded.getColValues(col) == skymodel_expected.getColValues(col)
        )

    # Test that attempting to download again without overwrite logs two warnings
    # First for existing sky model, second for skipping download
    with patch("lsmtool.io.logging.Logger.warning") as mock_warning:
        download_skymodel(
            cone_params,
            str(downloaded_skymodel_path),
            overwrite,
            source,
            targetname,
        )
        assert mock_warning.call_count == 2

    # Test that attempting to download again with overwrite logs a warning
    # First that sky model exists, second that it is being overwritten
    with patch("lsmtool.io.logging.Logger.warning") as mock_warning:
        download_skymodel(
            cone_params,
            str(downloaded_skymodel_path),
            overwrite,
            source,
            targetname,
        )
        assert mock_warning.call_count == 2

    # Test that attempting to download again with overwrite logs a warning
    # First that sky model exists, second that it is being overwritten
    with patch("lsmtool.io.logging.Logger.warning") as mock_warning:
        download_skymodel(
            cone_params,
            str(downloaded_skymodel_path),
            True,
            source,
            targetname,
        )
        assert mock_warning.call_count == 2
    assert downloaded_skymodel_path.is_file()


def test_sky_model_exists_existing_skymodel(existing_skymodel_filepath):
    """Test the _sky_model_exists function when the sky model exists."""

    with patch("lsmtool.io.logging.Logger.warning") as mock_warning:
        result = _sky_model_exists(str(existing_skymodel_filepath))
        mock_warning.assert_called_once()
    assert result is True


def test_sky_model_exists_no_existing_skymodel(tmp_path):
    """Test the _sky_model_exists function when the sky model does not exist."""

    skymodel_path = tmp_path / "non_existent_sky.model"
    with patch("lsmtool.io.logging.Logger.warning") as mock_warning:
        result = _sky_model_exists(str(skymodel_path))
        mock_warning.assert_not_called()
    assert result is False


def test_new_directory_required_existing_directory(tmp_path):
    """Test the _new_directory_required function."""

    existing_dir_path = tmp_path / "new_directory"
    existing_dir_path.mkdir()
    assert _new_directory_required(str(existing_dir_path)) is False


def test_new_directory_required_non_existent_directory(tmp_path):
    """Test the _new_directory_required function."""

    non_existent_path = tmp_path / "non_existent_directory"
    assert _new_directory_required(str(non_existent_path)) is False


def test_new_directory_required_file_in_existing_directory(tmp_path):
    """Test the _new_directory_required function."""

    file_in_existing_dir = tmp_path / "existing_directory" / "file.model"
    file_in_existing_dir.parent.mkdir()
    assert _new_directory_required(str(file_in_existing_dir)) is False


def test_new_directory_required_file_in_non_existent_directory(tmp_path):
    """Test the _new_directory_required function."""

    file_in_non_existent_dir = tmp_path / "non_existent_directory" / "file.model"
    assert _new_directory_required(str(file_in_non_existent_dir)) is True


def test_validate_skymodel_path_existing_file(tmp_path):
    """Test the _validate_skymodel_path function when the sky model file exists."""

    existing_file_path = tmp_path / "existing_sky.model"
    existing_file_path.touch()
    _validate_skymodel_path(str(existing_file_path))


def test_validate_skymodel_path_not_a_file(tmp_path):
    """Test the _validate_skymodel_path function with invalid file."""

    existing_dir_path = tmp_path / "existing_directory"
    existing_dir_path.mkdir()
    with pytest.raises(ValueError):
        _validate_skymodel_path(str(existing_dir_path))


@pytest.mark.parametrize(
    "overwrite, skymodel_exists, expected",
    [
        (True, True, True),
        (False, True, False),
        (True, False, False),
        (False, False, False),
    ],
)
def test_overwrite_required_existing_file(overwrite, skymodel_exists, expected):
    """Test the _overwrite_required function when the sky model file exists."""

    assert _overwrite_required(skymodel_exists, overwrite) is expected


@pytest.mark.parametrize(
    "overwrite, skymodel_exists, expected",
    [
        (True, True, False),
        (False, True, True),
        (True, False, False),
        (False, False, False),
    ],
)
def test_download_not_required(overwrite, skymodel_exists, expected):
    """Test the _download_not_required function when the sky model file exists."""

    assert _download_not_required(skymodel_exists, overwrite) is expected


def test_check_lotss_coverage_within_coverage(tmp_path):
    """Test the check_lotss_coverage function for coordinates within LoTSS coverage."""
    ra_within = 190.0  # RA within LoTSS coverage
    dec_within = 44.0  # DEC within LoTSS coverage
    radius = 1.0  # radius in degrees
    cone_params = {"ra": ra_within, "dec": dec_within, "radius": radius}
    check_lotss_coverage(cone_params, tmp_path)


def test_check_lotss_coverage_outside_coverage(tmp_path):
    """Test the check_lotss_coverage function for coordinates outside LoTSS coverage."""
    ra_outside = 30.0  # RA outside LoTSS coverage
    dec_outside = -30.0  # DEC outside LoTSS coverage
    radius = 1.0  # radius in degrees
    cone_params = {"ra": ra_outside, "dec": dec_outside, "radius": radius}
    with pytest.raises(ValueError):
        assert check_lotss_coverage(cone_params, tmp_path) is False


def test_get_panstarrs_request():
    """Test the get_panstarrs_request function."""

    # Arrange
    ra = 10.75
    dec = 5.34
    radius = 0.1
    cone_params = {"ra": ra, "dec": dec, "radius": radius}

    expected_url = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr1/mean.csv"
    expected_search_params = {
        "ra": ra,
        "dec": dec,
        "radius": radius,
        "nDetections.min": "5",
        "columns": ["objID", "ramean", "decmean"],
    }

    # Act
    request_url, search_params = get_panstarrs_request(cone_params)

    # Assert
    assert request_url == expected_url
    assert search_params == expected_search_params


def test_get_panstarrs_request_raises_error_large_radius():
    """Test the get_panstarrs_request function."""

    radius_limit = 0.5
    cone_params = {"ra": 10.0, "dec": 10.0, "radius": radius_limit + 0.001}
    with pytest.raises(ValueError):
        _, _ = get_panstarrs_request(cone_params)


def test_download_skymodel_panstarrs(tmp_path):
    """Test downloading a sky model from Pan-STARRS."""

    # Arrange
    url = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr1/mean.csv"
    search_params = {
        "ra": 10.75,
        "dec": 5.34,
        "radius": 0.1,
        "nDetections.min": "5",
        "columns": ["objID", "ramean", "decmean"],
    }
    skymodel_path = tmp_path / "panstarrs_sky.model"

    # Act
    download_skymodel_panstarrs(url, search_params, str(skymodel_path))

    # Assert
    assert skymodel_path.is_file()


@pytest.mark.parametrize(
    "source,ra,dec,radius",
    [
        ("LOTSS", 190.0, 30.0, 1.0),
        ("TGSS", 12.34, 56.78, 1.0),
        ("GSM", 123.23, 23.34, 1.0),
    ],
)
def test_download_skymodel_catalog(source, ra, dec, radius, tmp_path):
    """Test downloading a sky model from a catalog source."""

    # Arrange
    skymodel_path = tmp_path / f"catalog_sky_{source}.model"
    cone_params = {"ra": ra, "dec": dec, "radius": radius}

    # Act
    download_skymodel_catalog(cone_params, source, str(skymodel_path))

    # Assert
    assert skymodel_path.is_file()


@pytest.mark.parametrize(
    "source,ra,dec,radius",
    [
        ("LOTSS", 190.0, 30.0, 1.0),
        ("TGSS", 12.34, 56.78, 1.0),
        ("GSM", 123.23, 23.34, 1.0),
        ("PANSTARRS", 10.75, 5.34, 0.1),
    ],
)
def test_download_skymodel_from_source(source, ra, dec, radius, tmp_path):
    """Test downloading a sky model from a source."""

    # Arrange
    skymodel_path = tmp_path / f"source_sky_{source}.model"
    cone_params = {"ra": ra, "dec": dec, "radius": radius}

    # Act
    download_skymodel_from_source(cone_params, source, str(skymodel_path))

    # Assert
    assert skymodel_path.is_file()
