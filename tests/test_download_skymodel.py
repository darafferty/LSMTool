"""
Unit tests for the download_skymodel module.
"""

from unittest.mock import patch

import mocpy
import pytest
from conftest import copy_test_data

import lsmtool
from lsmtool.download_skymodel import (
    _check_coverage,
    _download_not_required,
    _get_lotss_moc,
    _new_directory_required,
    _overwrite_required,
    _sky_model_exists,
    _validate_skymodel_path,
    check_lotss_coverage,
    download_skymodel,
    download_skymodel_catalog,
    download_skymodel_from_source,
    download_skymodel_panstarrs,
    get_panstarrs_request,
)


@pytest.mark.parametrize("ra", (10.75,))
@pytest.mark.parametrize("dec", (5.34,))
@pytest.mark.parametrize("radius", (5.0,))
@pytest.mark.parametrize("overwrite", (False,))
@pytest.mark.parametrize("source", ("TGSS",))
@pytest.mark.parametrize("targetname", ("Patch",))
def test_download_skymodel(
    ra, dec, tmp_path, radius, overwrite, source, targetname
):
    """Test downloading a sky model."""

    # Arrange
    copy_test_data("expected.tgss.sky.model", tmp_path)
    downloaded_skymodel_path = tmp_path / "sky.model"
    expected_skymodel_path = tmp_path / "expected.tgss.sky.model"
    skymodel_expected = lsmtool.load(str(expected_skymodel_path))
    cone_params = {"ra": ra, "dec": dec, "radius": radius}

    # Act
    download_skymodel(
        cone_params,
        str(downloaded_skymodel_path),
        overwrite,
        source,
        targetname,
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
            skymodel_downloaded.getColValues(col)
            == skymodel_expected.getColValues(col)
        )

    # Test that attempting to download again without overwrite logs two
    # warnings: first for existing sky model, second for skipping download
    with patch(
        "lsmtool.download_skymodel.logging.Logger.warning"
    ) as mock_warning:
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
    with patch(
        "lsmtool.download_skymodel.logging.Logger.warning"
    ) as mock_warning:
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
    with patch(
        "lsmtool.download_skymodel.logging.Logger.warning"
    ) as mock_warning:
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

    with patch(
        "lsmtool.download_skymodel.logging.Logger.warning"
    ) as mock_warning:
        result = _sky_model_exists(str(existing_skymodel_filepath))
        mock_warning.assert_called_once()
    assert result is True


def test_sky_model_exists_no_existing_skymodel(tmp_path):
    """Test the _sky_model_exists function when sky model does not exist."""

    skymodel_path = tmp_path / "non_existent_sky.model"
    with patch(
        "lsmtool.download_skymodel.logging.Logger.warning"
    ) as mock_warning:
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

    file_in_non_existent_dir = (
        tmp_path / "non_existent_directory" / "file.model"
    )
    assert _new_directory_required(str(file_in_non_existent_dir)) is True


def test_validate_skymodel_path_existing_file(tmp_path):
    """Test the _validate_skymodel_path function when sky model file exists."""

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
    """Test the _download_not_required function when sky model file exists."""

    assert _download_not_required(skymodel_exists, overwrite) is expected


def test_check_lotss_coverage_within_coverage(tmp_path):
    """Test the check_lotss_coverage function within LoTSS coverage."""
    ra_within = 190.0  # RA within LoTSS coverage
    dec_within = 44.0  # DEC within LoTSS coverage
    radius = 1.0  # radius in degrees
    cone_params = {"ra": ra_within, "dec": dec_within, "radius": radius}
    check_lotss_coverage(cone_params, tmp_path)


def test_check_lotss_coverage_outside_coverage(tmp_path):
    """Test the check_lotss_coverage function outside LoTSS coverage."""
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

    expected_url = (
        "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr1/mean.csv"
    )
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


def test_get_lotss_moc(tmp_path):
    """Test the _get_lotss_moc function."""

    skymodel_path = tmp_path / "lotss_sky.model"
    expected_moc_path = tmp_path / "dr2-moc.moc"
    moc = _get_lotss_moc(skymodel_path)

    # Assert
    assert moc is not None
    assert isinstance(moc, mocpy.MOC)
    assert expected_moc_path.is_file()


def test_check_coverage_within_coverage(tmp_path):
    """Test the _check_coverage function within LoTSS coverage."""
    ra_within = 190.0  # RA within LoTSS coverage
    dec_within = 44.0  # DEC within LoTSS coverage
    radius = 1.0  # radius in degrees
    cone_params = {"ra": ra_within, "dec": dec_within, "radius": radius}
    moc = _get_lotss_moc(tmp_path / "lotss_sky.model")
    _check_coverage(cone_params, moc)


def test_check_coverage_outside_coverage(tmp_path):
    """Test the _check_coverage function outside LoTSS coverage."""
    ra_outside = 30.0  # RA outside LoTSS coverage
    dec_outside = -30.0  # DEC outside LoTSS coverage
    radius = 1.0  # radius in degrees
    cone_params = {"ra": ra_outside, "dec": dec_outside, "radius": radius}
    moc = _get_lotss_moc(tmp_path / "lotss_sky.model")
    with pytest.raises(ValueError):
        _check_coverage(cone_params, moc)


@pytest.mark.parametrize(
    "contains_lonlat_return",
    (
        [[False], [True], [True], [True], [True]],
        [[True], [False], [True], [True], [True]],
        [[True], [True], [False], [True], [True]],
        [[True], [True], [True], [False], [True]],
        [[True], [True], [True], [True], [False]],
    ),
)
def test_check_coverage_partial(
    mocker, mock_moc, cone_params, caplog, contains_lonlat_return
):
    """Test the _check_coverage function for some coordinates within MOC."""

    mocker.patch.object(
        mock_moc, "contains_lonlat", side_effect=contains_lonlat_return
    )
    with caplog.at_level("WARNING"):
        _check_coverage(cone_params, mock_moc)
    assert "Incomplete LoTSS coverage" in caplog.text


def test_check_coverage_full(mocker, mock_moc, cone_params, caplog):
    """Test the _check_coverage function for all coordinates within MOC."""

    mocker.patch.object(
        mock_moc,
        "contains_lonlat",
        side_effect=[[True], [True], [True], [True], [True]],
    )
    with caplog.at_level("INFO"):
        _check_coverage(cone_params, mock_moc)
    assert "Complete LoTSS coverage" in caplog.text


def test_check_coverage_zero(mocker, mock_moc, cone_params):
    """Test the _check_coverage function for all coordinates outside MOC."""

    mocker.patch.object(
        mock_moc,
        "contains_lonlat",
        side_effect=[[False], [False], [False], [False], [False]],
    )
    with pytest.raises(ValueError):
        _check_coverage(cone_params, mock_moc)
