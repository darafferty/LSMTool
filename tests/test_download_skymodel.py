"""
Unit tests for the download_skymodel module.
"""

import shutil
from pathlib import Path

import mocpy
import pytest
import requests
from conftest import copy_test_data

import lsmtool
from lsmtool.download_skymodel import (
    _check_moc_coverage,
    _download_not_required,
    _get_lotss_moc,
    _group_sources_into_single_direction,
    _new_directory_required,
    _overwrite_required,
    _prepare_path_for_download,
    _sky_model_exists,
    _validate_skymodel_path,
    check_lotss_coverage,
    download_skymodel,
    download_skymodel_catalog,
    download_skymodel_from_survey,
    download_skymodel_panstarrs,
    get_panstarrs_request,
)


@pytest.fixture
def mock_sky_model_class(mocker):
    """Fixture that provides a mock SkyModel constructor."""

    def _mock_sky_model(*_args, **_kwargs):
        mock_model = mocker.MagicMock()
        mock_model.__len__.return_value = 1
        mock_model.write.side_effect = lambda out_path: Path(
            out_path
        ).write_text(
            "FORMAT = Name, Type, Ra, Dec, I\n",
            encoding="utf-8",
        )
        return mock_model

    return _mock_sky_model


@pytest.fixture
def patch_download_skymodel_from_survey(mocker):
    """Patch download_skymodel_from_survey using data from a source file."""

    def _patch(source_skymodel_path):
        mocker.patch(
            "lsmtool.download_skymodel.download_skymodel_from_survey",
            side_effect=lambda _, __, out_skymodel_path: shutil.copyfile(
                source_skymodel_path,
                out_skymodel_path,
            ),
        )

    return _patch


@pytest.fixture
def cone_params():
    """Fixture that provides example cone search parameters for PAN-STARRS."""
    return {"ra": 10.75, "dec": 5.34, "radius": 0.01}


def test_download_skymodel(
    tmp_path,
    patch_download_skymodel_from_survey,
    mocker,
    cone_params,
):
    """Test downloading a sky model."""

    # Arrange
    copy_test_data("expected.tgss.sky", tmp_path)
    overwrite = False
    survey = "TGSS"
    targetname = "Patch"
    downloaded_skymodel_path = tmp_path / "downloaded.sky"
    expected_skymodel_path = tmp_path / "expected.tgss.sky"
    skymodel_expected = lsmtool.load(str(expected_skymodel_path))
    patch_download_skymodel_from_survey(expected_skymodel_path)

    # Act
    download_skymodel(
        cone_params,
        str(downloaded_skymodel_path),
        overwrite,
        survey,
        targetname,
    )
    skymodel_downloaded = lsmtool.load(str(downloaded_skymodel_path))

    # Assert
    assert len(skymodel_downloaded) == len(skymodel_expected)
    for col in skymodel_expected.table.columns:
        assert col in skymodel_downloaded.table.columns
        assert all(
            skymodel_downloaded.getColValues(col)
            == skymodel_expected.getColValues(col)
        )

    # Test that attempting to download again without overwrite logs two
    # warnings: first for existing sky model, second for skipping download
    mock_warning = mocker.patch(
        "lsmtool.download_skymodel.logging.Logger.warning"
    )
    download_skymodel(
        cone_params,
        str(downloaded_skymodel_path),
        overwrite,
        survey,
        targetname,
    )
    assert mock_warning.call_count == 2

    # Test that attempting to download again with overwrite logs a warning
    # First that sky model exists, second that it is being overwritten
    mock_warning = mocker.patch(
        "lsmtool.download_skymodel.logging.Logger.warning"
    )
    download_skymodel(
        cone_params,
        str(downloaded_skymodel_path),
        True,
        survey,
        targetname,
    )
    assert mock_warning.call_count == 2
    assert downloaded_skymodel_path.is_file()


def test_sky_model_exists_existing_skymodel(tmp_path, mocker):
    """Test the _sky_model_exists function when the sky model exists."""

    skymodel_path = tmp_path / "test.skymodel"
    skymodel_path.touch()

    mock_warning = mocker.patch(
        "lsmtool.download_skymodel.logging.Logger.warning"
    )
    result = _sky_model_exists(str(skymodel_path))
    mock_warning.assert_called_once()
    assert result is True


def test_sky_model_exists_no_existing_skymodel(tmp_path, mocker):
    """Test the _sky_model_exists function when sky model does not exist."""

    skymodel_path = tmp_path / "non_existent_sky.model"
    mock_warning = mocker.patch(
        "lsmtool.download_skymodel.logging.Logger.warning"
    )
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


def test_check_lotss_coverage_within_coverage(tmp_path, mocker):
    """Test the check_lotss_coverage function within LoTSS coverage."""
    ra_within = 190.0  # RA within LoTSS coverage
    dec_within = 44.0  # DEC within LoTSS coverage
    radius = 1.0  # radius in degrees
    cone_params = {"ra": ra_within, "dec": dec_within, "radius": radius}
    moc = mocker.Mock()
    moc.contains_lonlat.side_effect = [
        [True],
        [True],
        [True],
        [True],
        [True],
    ]
    mocker.patch("lsmtool.download_skymodel._get_lotss_moc", return_value=moc)
    check_lotss_coverage(cone_params, tmp_path)


def test_check_lotss_coverage_outside_coverage(tmp_path, mocker):
    """Test the check_lotss_coverage function outside LoTSS coverage."""
    ra_outside = 30.0  # RA outside LoTSS coverage
    dec_outside = -30.0  # DEC outside LoTSS coverage
    radius = 1.0  # radius in degrees
    cone_params = {"ra": ra_outside, "dec": dec_outside, "radius": radius}
    moc = mocker.Mock()
    moc.contains_lonlat.side_effect = [
        [False],
        [False],
        [False],
        [False],
        [False],
    ]
    mocker.patch("lsmtool.download_skymodel._get_lotss_moc", return_value=moc)
    with pytest.raises(ValueError):
        check_lotss_coverage(cone_params, tmp_path)


def test_get_panstarrs_request(cone_params):
    """Test the get_panstarrs_request function."""

    # Arrange
    expected_url = (
        "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr1/mean.csv"
    )
    expected_search_params = {
        "ra": cone_params["ra"],
        "dec": cone_params["dec"],
        "radius": cone_params["radius"],
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


def test_download_skymodel_panstarrs(cone_params, tmp_path, mocker):
    """Test downloading a sky model from Pan-STARRS."""

    # Arrange
    skymodel_path = tmp_path / "panstarrs.sky"

    mock_response = mocker.Mock()
    mock_response.ok = True
    mock_response.text = "objID,ramean,decmean\n1,10.75,5.34\n"
    mocker.patch(
        "lsmtool.download_skymodel.requests.get", return_value=mock_response
    )

    # Act
    download_skymodel_panstarrs(cone_params, str(skymodel_path))

    # Assert
    assert skymodel_path.is_file()
    lines = skymodel_path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "FORMAT = Name, Ra, Dec, Type, I, ReferenceFrequency=1e6",
        "1,10.75,5.34,POINT,0.0,",
    ]


def test_download_skymodel_panstarrs_not_ok(cone_params, tmp_path, mocker):
    """Test Pan-STARRS download returns False when response is not OK."""

    # Arrange
    skymodel_path = tmp_path / "panstarrs.sky"

    mock_response = mocker.Mock()
    mock_response.ok = False
    mock_response.text = ""
    mocker.patch(
        "lsmtool.download_skymodel.requests.get", return_value=mock_response
    )

    # Act
    success = download_skymodel_panstarrs(cone_params, str(skymodel_path))

    # Assert
    assert success is False
    assert not skymodel_path.exists()


def test_download_skymodel_panstarrs_request_exception(
    cone_params, tmp_path, mocker
):
    """Test Pan-STARRS download handles request exceptions."""

    # Arrange
    skymodel_path = tmp_path / "panstarrs.sky"

    mocker.patch(
        "lsmtool.download_skymodel.requests.get",
        side_effect=requests.exceptions.RequestException("network error"),
    )
    mock_warning = mocker.patch("lsmtool.download_skymodel.logger.warning")

    # Act
    success = download_skymodel_panstarrs(cone_params, str(skymodel_path))

    # Assert
    assert success is False
    mock_warning.assert_called_once()
    assert not skymodel_path.exists()


@pytest.mark.parametrize(
    "survey,ra,dec,radius",
    [
        ("LOTSS", 190.0, 30.0, 0.5),
        ("TGSS", 12.34, 56.78, 0.6),
        ("GSM", 123.23, 23.34, 0.6),
    ],
)
def test_download_skymodel_catalog(
    survey, ra, dec, radius, tmp_path, mock_sky_model_class, mocker
):
    """Test downloading a sky model from a survey."""

    # Arrange
    skymodel_path = tmp_path / f"catalog_sky_{survey}.model"
    cone_params = {"ra": ra, "dec": dec, "radius": radius}

    # Act
    mocker.patch("lsmtool.download_skymodel.SkyModel", new=mock_sky_model_class)
    success = download_skymodel_catalog(cone_params, survey, str(skymodel_path))

    # Assert
    assert success is True
    assert skymodel_path.is_file()


def test_download_skymodel_catalog_empty_result(cone_params, tmp_path, mocker):
    """Test catalog download returns False when no sources are found."""

    # Arrange
    skymodel_path = tmp_path / "catalog_sky_empty.model"
    mock_model = mocker.MagicMock()
    mock_model.__len__.return_value = 0
    mock_model.write.side_effect = lambda out_path: Path(out_path).write_text(
        "FORMAT = Name, Type, Ra, Dec, I\n",
        encoding="utf-8",
    )
    mocker.patch(
        "lsmtool.download_skymodel.SkyModel",
        return_value=mock_model,
    )

    # Act
    success = download_skymodel_catalog(cone_params, "TGSS", str(skymodel_path))

    # Assert
    assert success is False


@pytest.mark.parametrize(
    "survey",
    [
        "LOTSS",
        "TGSS",
        "GSM",
        "PANSTARRS",
    ],
)
def test_download_skymodel_from_survey(
    cone_params, survey, tmp_path, mocker, caplog
):
    """Test downloading a sky model from a survey."""

    # Arrange
    skymodel_path = tmp_path / f"survey_sky_{survey}.model"

    # Mock sucessful download attempt on first try
    mocker.patch(
        "lsmtool.download_skymodel.download_skymodel_catalog",
        side_effect=[True],
    )
    mocker.patch(
        "lsmtool.download_skymodel.download_skymodel_panstarrs",
        side_effect=[True],
    )
    mocker.patch(
        "lsmtool.download_skymodel.check_lotss_coverage", side_effect=[True]
    )
    # Act
    with caplog.at_level("INFO"):
        download_skymodel_from_survey(cone_params, survey, str(skymodel_path))

    # Assert
    assert (
        f"Download of {survey} sky model completed successfully." in caplog.text
    )


@pytest.mark.parametrize(
    "survey,ra,dec,radius",
    [
        ("LOTSS", 190.0, 30.0, 1.0),
        ("PANSTARRS", 10.75, 5.34, 0.01),
    ],
)
def test_download_skymodel_from_survey_retries(
    survey, ra, dec, radius, tmp_path, mocker, caplog
):
    """Test downloading a sky model from a survey retries on failure."""

    # Arrange
    skymodel_path = tmp_path / f"survey_sky_{survey}.model"
    cone_params = {"ra": ra, "dec": dec, "radius": radius}
    retries = 2

    # Mock failed download attempts for the first (max_tries - 1) tries
    mocker.patch(
        "lsmtool.download_skymodel.download_skymodel_catalog",
        side_effect=[False] * retries + [True],
    )
    mocker.patch(
        "lsmtool.download_skymodel.download_skymodel_panstarrs",
        side_effect=[False] * retries + [True],
    )
    # Act
    with caplog.at_level("INFO"):
        download_skymodel_from_survey(
            cone_params,
            survey,
            str(skymodel_path),
            retries=retries,
            time_between_retries=0,
        )

    # Assert
    assert (
        f"Attempt #1 to download {survey} sky model failed. Attempting 2 more times."
        in caplog.text
    )
    assert (
        f"Attempt #2 to download {survey} sky model failed. Attempting 1 more time."
        in caplog.text
    )
    assert "Attempting 1 more time." in caplog.text
    assert "Attempting 2 more times." in caplog.text
    assert (
        f"Download of {survey} sky model completed successfully." in caplog.text
    )


@pytest.mark.parametrize(
    "survey,ra,dec,radius",
    [
        ("LOTSS", 190.0, 30.0, 1.0),
        ("PANSTARRS", 10.75, 5.34, 0.01),
    ],
)
def test_download_skymodel_from_survey_all_retries_fail(
    survey, ra, dec, radius, tmp_path, mocker, caplog
):
    """Test downloading a sky model from a survey retries on failure."""

    # Arrange
    skymodel_path = tmp_path / f"survey_sky_{survey}.model"
    cone_params = {"ra": ra, "dec": dec, "radius": radius}
    retries = 2

    # Mock failed download attempts for all tries
    mocker.patch(
        "lsmtool.download_skymodel.download_skymodel_catalog",
        side_effect=[False] * (retries + 1),
    )
    mocker.patch(
        "lsmtool.download_skymodel.download_skymodel_panstarrs",
        side_effect=[False] * (retries + 1),
    )
    # Act
    with caplog.at_level("ERROR"):
        with pytest.raises(IOError):
            download_skymodel_from_survey(
                cone_params,
                survey,
                str(skymodel_path),
                retries=retries,
                time_between_retries=0,
            )

    # Assert
    for retry in range(retries):
        assert (
            f"Attempt #{retry + 1} to download {survey} sky model failed."
            in caplog.text
        )


def test_download_skymodel_from_survey_unsupported_survey(tmp_path):
    """Test unsupported survey value raises ValueError."""

    cone_params = {"ra": 10.75, "dec": 5.34, "radius": 0.5}
    skymodel_path = tmp_path / "survey_sky_BAD.model"

    with pytest.raises(ValueError):
        download_skymodel_from_survey(cone_params, "BAD", str(skymodel_path))


def test_get_lotss_moc(tmp_path, mocker, mock_moc):
    """Test the _get_lotss_moc function."""

    skymodel_path = tmp_path / "lotss_sky.model"
    expected_moc_path = tmp_path / "dr2-moc.moc"

    mock_response = mocker.Mock()
    mock_response.content = b"fake-moc-content"
    mock_response.raise_for_status.return_value = None
    mocker.patch(
        "lsmtool.download_skymodel.requests.get", return_value=mock_response
    )
    mocker.patch(
        "lsmtool.download_skymodel.mocpy.MOC.from_fits", return_value=mock_moc
    )

    moc = _get_lotss_moc(skymodel_path)

    # Assert
    assert moc is not None
    assert isinstance(moc, mocpy.MOC)
    assert expected_moc_path.is_file()


def test_get_lotss_moc_download_failure(tmp_path, mocker):
    """Test _get_lotss_moc raises ConnectionError on request failure."""

    skymodel_path = tmp_path / "lotss_sky.model"
    mocker.patch(
        "lsmtool.download_skymodel.requests.get",
        side_effect=requests.exceptions.RequestException("download failed"),
    )

    with pytest.raises(ConnectionError):
        _get_lotss_moc(skymodel_path)


def test_download_skymodel_raises_if_output_missing(tmp_path, mocker):
    """Test top-level download raises if output file is not produced."""

    cone_params = {"ra": 10.75, "dec": 5.34, "radius": 0.5}
    skymodel_path = tmp_path / "missing_sky.model"
    mocker.patch("lsmtool.download_skymodel.download_skymodel_from_survey")

    with pytest.raises(IOError):
        download_skymodel(
            cone_params,
            str(skymodel_path),
            overwrite=False,
            survey="TGSS",
            targetname="Patch",
        )


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
def test_check_moc_coverage_partial(
    mocker, mock_moc, cone_params, caplog, contains_lonlat_return
):
    """Test the _check_moc_coverage function for some coordinates within MOC."""

    mocker.patch.object(
        mock_moc, "contains_lonlat", side_effect=contains_lonlat_return
    )
    with caplog.at_level("WARNING"):
        _check_moc_coverage(cone_params, mock_moc)
    assert "Incomplete LoTSS coverage" in caplog.text


def test_check_moc_coverage_full(mocker, mock_moc, cone_params, caplog):
    """Test the _check_moc_coverage function for all coordinates within MOC."""

    mocker.patch.object(
        mock_moc,
        "contains_lonlat",
        side_effect=[[True], [True], [True], [True], [True]],
    )
    with caplog.at_level("INFO"):
        _check_moc_coverage(cone_params, mock_moc)
    assert "Complete LoTSS coverage" in caplog.text


def test_check_moc_coverage_zero(mocker, mock_moc, cone_params):
    """Test the _check_moc_coverage function for all coordinates outside MOC."""

    mocker.patch.object(
        mock_moc,
        "contains_lonlat",
        side_effect=[[False], [False], [False], [False], [False]],
    )
    with pytest.raises(ValueError):
        _check_moc_coverage(cone_params, mock_moc)


def test_group_sources_into_single_direction(tmp_path, mocker):
    """Test sky model is grouped into a single direction and overwritten."""

    skymodel_path = tmp_path / "sky.model"
    target_name = "Patch"

    mock_skymodel = mocker.Mock()
    mocker.patch(
        "lsmtool.download_skymodel.lsmtool.load", return_value=mock_skymodel
    )

    _group_sources_into_single_direction(str(skymodel_path), target_name)

    mock_skymodel.group.assert_called_once_with("single", root=target_name)
    mock_skymodel.write.assert_called_once_with(clobber=True)


@pytest.mark.parametrize(
    "skymodel_exists_before, overwrite, skymodel_exists_after",
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
def test_prepare_path_for_download(
    skymodel_exists_before, overwrite, skymodel_exists_after, tmp_path
):
    """Test the _prepare_path_for_download function."""

    skymodel_path = tmp_path / "fake.sky"
    if skymodel_exists_before:
        skymodel_path.touch()

    _prepare_path_for_download(
        str(skymodel_path), skymodel_exists_before, overwrite
    )
    assert skymodel_path.is_file() == skymodel_exists_after


@pytest.mark.parametrize("overwrite", [True, False])
def test_prepare_path_for_download_new_directory(tmp_path, overwrite):
    """Test the _prepare_path_for_download function creates new directory."""

    skymodel_path = tmp_path / "new_directory" / "fake.sky"
    _prepare_path_for_download(
        str(skymodel_path), skymodel_exists=False, overwrite=overwrite
    )
    assert skymodel_path.parent.is_dir()


def test_prepare_path_for_downloading_not_a_file(tmp_path):
    """Test the _prepare_path_for_download function raises ValueError if path is not a file."""

    skymodel_path = tmp_path / "not_a_file"
    skymodel_path.mkdir()
    with pytest.raises(ValueError):
        _prepare_path_for_download(
            str(skymodel_path), skymodel_exists=True, overwrite=False
        )
