"""
Tests for filtering sources from the skymodel based on source detection.
"""

import contextlib as ctx

import pytest
from conftest import TEST_DATA_PATH, copy_test_data

from lsmtool.filter_skymodel import (
    KNOWN_SOURCE_FINDERS,
    bdsf,
    resolve_source_finder,
    sofia,
)
from lsmtool.testing import check_skymodels_equal


class TestResolveSourceFinder:
    null = ctx.nullcontext()

    message = (
        "'invalid' is not a valid value for 'source_finder'. Valid options are"
        f" {set(KNOWN_SOURCE_FINDERS.keys())}."
    )

    @pytest.mark.parametrize(
        "name, fallback, expected, context",
        [
            pytest.param("sofia", "", "sofia", null, id="sofia"),
            pytest.param("bdsf", "", "bdsf", null, id="bdsf"),
            pytest.param("SoFiA", "", "sofia", null, id="SoFiA"),
            pytest.param("BDSF", "", "bdsf", null, id="BDSF"),
            pytest.param("on", "bdsf", "bdsf", null, id="default"),
            pytest.param(True, "SoFiA", "sofia", null, id="fallback_sofia"),
            pytest.param(None, "bdsf", None, null, id="source_finder_off"),
            pytest.param(
                "invalid",
                "bdsf",
                "bdsf",
                pytest.raises(ValueError, match=message),
                id="invalid_raises",
            ),
        ],
    )
    def test_resolve_source_finder(self, name, fallback, expected, context):
        # Act
        with context:
            result = resolve_source_finder(name, fallback)

            # Assert
            assert result == expected


def get_image_paths(tmp_path, prefix):
    """Image paths for testing."""

    # Copy image to temp test folder
    copy_test_data("test_image.fits", tmp_path)

    image_name = tmp_path / "test_image.fits"
    true_sky_path = tmp_path / f"{prefix}.true_sky.txt"
    apparent_sky_path = tmp_path / f"{prefix}.apparent_sky.txt"

    return image_name, true_sky_path, apparent_sky_path


class TestBDSF:
    """Test skymodel filtering with pybdsf."""

    @pytest.fixture()
    def image_paths(self, tmp_path, midband_ms):
        """Image paths for testing."""
        return *get_image_paths(tmp_path, "output_bdsf"), midband_ms

    @pytest.fixture()
    def diagnostic_paths(self, tmp_path):
        """Image paths for testing."""
        return {
            "output_catalog": tmp_path / "output-catalog.test.fits",
            "output_true_rms": tmp_path / "output-true-sky-rms.test.fits",
            "output_flat_noise_rms": tmp_path
            / "output-flat-noise-rms.test.fits",
        }

    def test_filter_skymodel(self, image_paths, **kws):
        """
        Test skymodel filtering, with and without creating extra output files.
        """

        image_path, true_sky_path, apparent_sky_path, beam_ms = image_paths

        bdsf.filter_skymodel(
            flat_noise_image=image_path,
            true_sky_image=image_path,
            input_true_skymodel=TEST_DATA_PATH / "sector_1-sources-pb.txt",
            input_apparent_skymodel=None,
            output_apparent_sky=apparent_sky_path,
            output_true_sky=true_sky_path,
            vertices_file=TEST_DATA_PATH / "expected_sector_1_vertices.pkl",
            beam_ms=beam_ms,
            thresh_isl=4.0,
            thresh_pix=5.0,
            **kws,
        )

        assert true_sky_path.exists()
        assert apparent_sky_path.exists()

        assert check_skymodels_equal(
            apparent_sky_path, TEST_DATA_PATH / "expected.apparent_sky.txt"
        )
        assert check_skymodels_equal(
            true_sky_path, TEST_DATA_PATH / "expected.true_sky.txt"
        )

    def test_filter_skymodel_diagnostics(self, image_paths, diagnostic_paths):
        # run the base test
        self.test_filter_skymodel(image_paths, **diagnostic_paths)
        # check that the additional files were created
        assert diagnostic_paths["output_catalog"].exists()
        assert diagnostic_paths["output_true_rms"].exists()
        assert diagnostic_paths["output_flat_noise_rms"].exists()

    def test_filter_skymodel_empty(self, image_paths):
        """
        Test skymodel filtering with too high thresholds.
        filter_skymodel() should fallback to using create_dummy_skymodel().
        This test also tests create_dummy_skymodel(), including its output.
        """

        image_path, true_sky_path, apparent_sky_path, beam_ms = image_paths

        bdsf.filter_skymodel(
            flat_noise_image=image_path,
            true_sky_image=image_path,
            input_true_skymodel=TEST_DATA_PATH / "sector_1-sources-pb.txt",
            input_apparent_skymodel=None,
            output_apparent_sky=apparent_sky_path,
            output_true_sky=true_sky_path,
            vertices_file=TEST_DATA_PATH / "expected_sector_1_vertices.pkl",
            beam_ms=beam_ms,
            thresh_isl=400.0,
            thresh_pix=500.0,
        )

        assert apparent_sky_path.exists()
        assert true_sky_path.exists()

        assert check_skymodels_equal(
            true_sky_path, TEST_DATA_PATH / "single_point.sky"
        )
        assert check_skymodels_equal(
            apparent_sky_path,
            TEST_DATA_PATH / "single_point.sky",
        )


class TestSofia:
    """Test skymodel filtering with SoFiA-2."""

    @pytest.fixture()
    def image_paths(self, tmp_path):
        """Image paths for testing."""
        return get_image_paths(tmp_path, "output_sofia")

    def test_filter_skymodel_single_image(self, image_paths):
        """Test skymodel filtering with no true sky image provided."""

        image_name, true_sky_path, apparent_sky_path = image_paths

        # When supplying a single (flat noise) image, the generated apparent
        # and true skymodels should be equal.
        sofia.filter_skymodel(
            image_name,
            None,
            apparent_sky_path,
            true_sky_path,
        )

        # Assert
        assert apparent_sky_path.exists()
        assert true_sky_path.exists()

        assert check_skymodels_equal(
            true_sky_path,
            apparent_sky_path,
            check_patch_names_sizes=False,
        )
        assert check_skymodels_equal(
            apparent_sky_path,
            TEST_DATA_PATH / "expected_sofia.true_sky.txt",
            check_patch_names_sizes=False,
        )

    def test_filter_skymodel_true_sky_image(self, image_paths):
        """Test skymodel filtering with true sky image provided."""
        image_name, true_sky_path, apparent_sky_path = image_paths

        # Run with the true sky image
        sofia.filter_skymodel(
            image_name,
            image_name,
            apparent_sky_path,
            true_sky_path,
        )

        # Assert
        assert apparent_sky_path.exists()
        assert true_sky_path.exists()

        assert check_skymodels_equal(
            apparent_sky_path,
            TEST_DATA_PATH / "expected_sofia.true_sky.txt",
            check_patch_names_sizes=False,
        )
        assert check_skymodels_equal(
            true_sky_path,
            TEST_DATA_PATH / "expected_sofia.true_sky.txt",
            check_patch_names_sizes=False,
        )
