import pytest

from lsmtool.filter_skymodel import (
    filter_skymodel_bdsf,
    filter_skymodel_sofia,
)
from .conftest import copy_test_data, assert_skymodels_are_equal


@pytest.mark.parametrize("output_diagnostics", [True, False])
def test_filter_skymodel_bdsf(
    output_diagnostics,
    test_ms,
    tmp_path,
    test_data_path,
):
    """Test skymodel filtering, with+without creating extra output files."""

    # PyBDSF writes its log to the directory containing the image, so copy it.
    copy_test_data("test_image.fits", tmp_path)

    image_path = tmp_path / "test_image.fits"
    apparent_sky_path = tmp_path / "out.apparent.txt"
    true_sky_path = tmp_path / "out.true.txt"

    catalog_path = tmp_path / "output-catalog.test.fits"
    true_rms_path = tmp_path / "output-true-sky-rms.test.fits"
    flat_noise_rms_path = tmp_path / "output-flat-noise-rms.test.fits"

    filter_skymodel_bdsf(
        image_path,
        image_path,
        test_data_path / "sector_1-sources-pb.txt",
        apparent_sky_path,
        true_sky_path,
        test_data_path / "expected_sector_1_vertices.pkl",
        beam_ms=test_ms,
        threshisl=4.0,
        threshpix=5.0,
        output_catalog=(catalog_path if output_diagnostics else ""),
        output_flat_noise_rms=(
            flat_noise_rms_path if output_diagnostics else ""
        ),
        output_true_rms=(true_rms_path if output_diagnostics else ""),
    )

    assert true_sky_path.exists()
    assert apparent_sky_path.exists()
    assert catalog_path.exists() is output_diagnostics
    assert true_rms_path.exists() is output_diagnostics
    assert flat_noise_rms_path.exists() is output_diagnostics

    assert_skymodels_are_equal(
        apparent_sky_path, test_data_path / "expected.apparent_sky.txt"
    )
    assert_skymodels_are_equal(
        true_sky_path, test_data_path / "expected.true_sky.txt"
    )


def test_filter_skymodel_bdsf_empty(test_ms, tmp_path, test_data_path):
    """Test skymodel filtering with too high thresholds"""

    # PyBDSF writes its log to the directory containing the image, so copy it.
    copy_test_data("test_image.fits", tmp_path)

    image_path = tmp_path / "test_image.fits"
    apparent_sky_path = tmp_path / "out.apparent.txt"
    true_sky_path = tmp_path / "out.true.txt"

    filter_skymodel_bdsf(
        image_path,
        image_path,
        "sector_1-sources-pb.txt",
        apparent_sky_path,
        true_sky_path,
        "expected_sector_1_vertices.pkl",
        beam_ms=test_ms,
        threshisl=400.0,
        threshpix=500.0,
    )

    assert apparent_sky_path.exists()
    assert true_sky_path.exists()

    assert_skymodels_are_equal(
        true_sky_path, test_data_path / "empty.skymodel"
    )
    assert_skymodels_are_equal(
        apparent_sky_path,
        test_data_path / "empty.skymodel",
    )


class TestSofia:
    """Test skymodel filtering with SoFiA-2."""

    @pytest.fixture()
    def image_paths(self, tmp_path):
        """Image paths for testing."""

        # Copy image to temp test folder
        copy_test_data("test_image.fits", tmp_path)

        image_name = tmp_path / "test_image.fits"
        skymodel_prefix = "output_sofia"
        true_sky_path = tmp_path / f"{skymodel_prefix}.true_sky.txt"
        apparent_sky_path = tmp_path / f"{skymodel_prefix}.apparent_sky.txt"

        return image_name, true_sky_path, apparent_sky_path

    def test_filter_skymodel_single_image(self, image_paths, test_data_path):
        """Test skymodel filtering with no true sky image provided."""

        image_name, true_sky_path, apparent_sky_path = image_paths

        # When supplying a single (flat noise) image, the generated apparent
        # and true skymodels should be equal.
        filter_skymodel_sofia(
            image_name,
            None,
            apparent_sky_path,
            true_sky_path,
        )

        # Assert
        assert apparent_sky_path.exists()
        assert true_sky_path.exists()

        assert_skymodels_are_equal(
            true_sky_path,
            apparent_sky_path,
            check_patch_names_sizes=False,
        )
        assert_skymodels_are_equal(
            apparent_sky_path,
            test_data_path / "expected_sofia.apparent_sky.txt",
            check_patch_names_sizes=False,
        )

    def test_filter_skymodel_true_sky_image(self, image_paths, test_data_path):
        """Test skymodel filtering with true sky image provided."""
        image_name, true_sky_path, apparent_sky_path = image_paths

        # Run with the true sky image
        filter_skymodel_sofia(
            image_name,
            image_name,
            apparent_sky_path,
            true_sky_path,
        )

        # Assert
        assert apparent_sky_path.exists()
        assert true_sky_path.exists()

        assert_skymodels_are_equal(
            apparent_sky_path,
            test_data_path / "expected_sofia.apparent_sky.txt",
            check_patch_names_sizes=False,
        )
        assert_skymodels_are_equal(
            true_sky_path,
            test_data_path / "expected_sofia.true_sky.txt",
            check_patch_names_sizes=False,
        )
