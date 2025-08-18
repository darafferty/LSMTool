"""
Tests for filtering sources from the skymodel based on source detection.
"""

import contextlib as ctx

import pytest
from conftest import TEST_DATA_PATH, copy_test_data

from lsmtool.filter_skymodel import bdsf, resolve_source_finder

try:
    from lsmtool.filter_skymodel import sofia

    have_sofia = True
except ImportError:
    have_sofia = False

from lsmtool.testing import check_skymodels_equal


class TestResolveSourceFinder:
    null_context = ctx.nullcontext()
    raises = pytest.raises(ValueError)

    @pytest.mark.parametrize(
        "name, expected, context",
        [
            pytest.param(
                "sofia",
                "sofia",
                null_context,
                id="sofia",
                marks=pytest.mark.skipif(not have_sofia, reason="SoFiA not available"),
            ),
            pytest.param("bdsf", "bdsf", null_context, id="bdsf"),
            pytest.param(
                "SoFiA",
                "sofia",
                null_context,
                id="SoFiA",
                marks=pytest.mark.skipif(not have_sofia, reason="SoFiA not available"),
            ),
            pytest.param("BDSF", "bdsf", null_context, id="BDSF"),
            pytest.param(None, None, raises, id="nonetype_raises"),
            pytest.param(True, None, raises, id="true_raises"),
            pytest.param("none", None, raises, id="invalid_string_raises"),
        ],
    )
    def test_resolve_source_finder(self, name, expected, context):
        # Act
        with context:
            # Assert
            assert resolve_source_finder(name) == expected


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
    def image_paths(self, tmp_path, midbands_ms):
        """Image paths for testing."""
        return *get_image_paths(tmp_path, "output_bdsf"), midbands_ms

    @pytest.fixture()
    def diagnostic_paths(self, tmp_path):
        """Image paths for testing."""
        return {
            "output_catalog": tmp_path / "output-catalog.test.fits",
            "output_true_rms": tmp_path / "output-true-sky-rms.test.fits",
            "output_flat_noise_rms": tmp_path / "output-flat-noise-rms.test.fits",
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

        assert check_skymodels_equal(true_sky_path, TEST_DATA_PATH / "single_point.sky")
        assert check_skymodels_equal(
            apparent_sky_path,
            TEST_DATA_PATH / "single_point.sky",
        )


@pytest.mark.skipif(not have_sofia, reason="SoFiA not available")
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
