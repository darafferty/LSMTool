"""
Tests for filtering sources from the skymodel based on source detection.
"""

import contextlib as ctx

import pytest
from conftest import copy_test_data

from lsmtool.filter_skymodel import bdsf, resolve_source_finder
from lsmtool.testing import check_skymodels_equal

sofia = None
with ctx.suppress(ImportError):
    from lsmtool.filter_skymodel import sofia


@pytest.mark.parametrize(
    "name, expected",
    [
        # -------------------------------------------------------------------- #
        # Nominal cases
        pytest.param(
            "sofia",
            "sofia",
            id="sofia",
            marks=pytest.mark.skipif(
                sofia is None, reason="SoFiA not available"
            ),
        ),
        pytest.param("bdsf", "bdsf", id="bdsf"),
        pytest.param(
            "SoFiA",
            "sofia",
            id="SoFiA",
            marks=pytest.mark.skipif(
                sofia is None, reason="SoFiA not available"
            ),
        ),
        pytest.param("BDSF", "bdsf", id="BDSF"),
        # -------------------------------------------------------------------- #
        # Error cases
        pytest.param(None, None, id="nonetype_raises"),
        pytest.param(True, None, id="true_raises"),
        pytest.param("none", None, id="invalid_string_raises"),
    ],
)
def test_resolve_source_finder(name, expected):

    context = (
        pytest.raises(ValueError) if expected is None else ctx.nullcontext()
    )
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
            "output_flat_noise_rms": tmp_path
            / "output-flat-noise-rms.test.fits",
        }

    @pytest.mark.parametrize("keep_mask", [True, False])
    def test_filter_skymodel(
        self, test_data_path, image_paths, keep_mask, **kws
    ):
        """
        Test skymodel filtering, with and without creating extra output files.
        """

        image_path, true_sky_path, apparent_sky_path, beam_ms = image_paths

        bdsf.filter_skymodel(
            flat_noise_image=image_path,
            true_sky_image=image_path,
            input_true_skymodel=test_data_path / "sector_1-sources-pb.txt",
            input_apparent_skymodel=None,
            output_apparent_sky=apparent_sky_path,
            output_true_sky=true_sky_path,
            vertices_file=test_data_path / "expected_sector_1_vertices.npy",
            beam_ms=beam_ms,
            thresh_isl=4.0,
            thresh_pix=5.0,
            keep_mask=keep_mask,
            **kws,
        )

        assert true_sky_path.exists()
        assert apparent_sky_path.exists()
        if keep_mask:
            assert next(true_sky_path.parent.glob("*.mask.fits"))

        assert check_skymodels_equal(
            apparent_sky_path, test_data_path / "expected.apparent_sky.txt"
        )
        assert check_skymodels_equal(
            true_sky_path, test_data_path / "expected.true_sky.txt"
        )

    @pytest.mark.parametrize("keep_mask", [True, False])
    def test_filter_skymodel_diagnostics(
        self, test_data_path, image_paths, diagnostic_paths, keep_mask
    ):
        # run the base test
        self.test_filter_skymodel(
            test_data_path, image_paths, keep_mask=keep_mask, **diagnostic_paths
        )
        # check that the additional files were created
        assert diagnostic_paths["output_catalog"].exists()
        assert diagnostic_paths["output_true_rms"].exists()
        assert diagnostic_paths["output_flat_noise_rms"].exists()

    def test_filter_skymodel_empty(self, test_data_path, image_paths):
        """
        Test skymodel filtering with too high thresholds.
        filter_skymodel() should fallback to using create_dummy_skymodel().
        This test also tests create_dummy_skymodel(), including its output.
        """

        image_path, true_sky_path, apparent_sky_path, beam_ms = image_paths

        bdsf.filter_skymodel(
            flat_noise_image=image_path,
            true_sky_image=image_path,
            input_true_skymodel=test_data_path / "sector_1-sources-pb.txt",
            input_apparent_skymodel=None,
            output_apparent_sky=apparent_sky_path,
            output_true_sky=true_sky_path,
            vertices_file=test_data_path / "expected_sector_1_vertices.npy",
            beam_ms=beam_ms,
            thresh_isl=400.0,
            thresh_pix=500.0,
        )

        assert apparent_sky_path.exists()
        assert true_sky_path.exists()

        assert check_skymodels_equal(
            true_sky_path, test_data_path / "single_point.sky"
        )
        assert check_skymodels_equal(
            apparent_sky_path,
            test_data_path / "single_point.sky",
        )


@pytest.mark.skipif(not sofia, reason="SoFiA not available")
class TestSofia:
    """Test skymodel filtering with SoFiA-2."""

    @pytest.fixture()
    def image_paths(self, tmp_path):
        """Image paths for testing."""
        return get_image_paths(tmp_path, "output_sofia")

    def test_filter_skymodel_single_image(self, test_data_path, image_paths):
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
            test_data_path / "expected_sofia.true_sky.txt",
            check_patch_names_sizes=False,
        )

    def test_filter_skymodel_true_sky_image(self, test_data_path, image_paths):
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
            test_data_path / "expected_sofia.true_sky.txt",
            check_patch_names_sizes=False,
        )
        assert check_skymodels_equal(
            true_sky_path,
            test_data_path / "expected_sofia.true_sky.txt",
            check_patch_names_sizes=False,
        )
