"""
Tests for skymodel assertion helpers.
"""

import numpy as np
import pytest
from scipy.stats import kstest
from scipy.stats.distributions import uniform

from lsmtool import load
from lsmtool.testing import (
    SkyModelGenerator,
    check_skymodels_equal,
    uniform_range,
)

# ---------------------------------------------------------------------------- #
# Fixtures


@pytest.fixture(scope="module")
def input_skymodel(test_data_path):
    return test_data_path / "expected.true_sky.txt"


@pytest.fixture
def modified_skymodel(request, tmp_path):
    # Use the input_skymodel fixture value
    input_skymodel = request.getfixturevalue("input_skymodel")
    return _modify_patches(input_skymodel, *request.param, tmp_path)


def _modify_patches(input_skymodel, update_patch_names, regroup, tmp_path):
    """
    Modify the patch definitions in the skymodel and save as a temporary file,
    returning the filename.
    """
    skymodel = load(input_skymodel)

    if update_patch_names:
        _update_patch_names(skymodel)

    if regroup:
        skymodel.group("every")

    suffix = f"{int(update_patch_names)}{int(regroup)}"
    filename = tmp_path / f"skymodel_testcase_{suffix}.sky"
    skymodel.write(filename, format="makesourcedb")
    return filename


def _update_patch_names(skymodel):
    last_patch_name = skymodel.getPatchNames()[-1]
    new_name = "Whatever"

    patch_pos = skymodel.getPatchPositions()
    patch_pos[new_name] = patch_pos.pop(last_patch_name)
    skymodel.setPatchPositions(patch_pos)

    patches = skymodel.table["Patch"]
    patches[patches == last_patch_name] = new_name

    patches = skymodel.table.groups.keys["Patch"]
    patches[patches == last_patch_name] = new_name


# ---------------------------------------------------------------------------- #
# Tests


@pytest.mark.parametrize(
    "modified_skymodel, check_patch_names_sizes, expected_equal",
    [
        pytest.param(
            # (update_patch_names, regroup),
            (False, True),
            False,  # check_names_sizes
            False,  # expected equal
            id="different_groups_check_names",
        ),
        pytest.param(
            (False, True),
            True,
            False,
            id="different_groups_ignore_names",
        ),
        pytest.param(
            (True, False),
            True,
            False,
            id="different_patch_names_check",
        ),
        pytest.param(
            (True, False),
            False,
            True,
            id="different_patch_names_ignore",
        ),
        pytest.param(
            (False, False),
            True,
            True,
            id="no_edit",
        ),
    ],
    indirect=["modified_skymodel"],
)
def test_check_skymodels_equal_patches(
    input_skymodel, modified_skymodel, check_patch_names_sizes, expected_equal
):
    """
    Test check_skymodels_equal in the case that the skymodels are identical,
    except for the patch sizes or names.
    """

    # Test that differences in patch names and sizes are checked when asked for
    assert (
        check_skymodels_equal(
            input_skymodel, modified_skymodel, check_patch_names_sizes
        )
        is expected_equal
    )


@pytest.mark.parametrize(
    "left_filename, right_filename, check_patch_names_sizes, expected_equal",
    [
        # Equal cases
        pytest.param(
            "single_point.sky",
            "single_point.sky",
            True,
            True,
            id="single_point_model",
        ),
        pytest.param(
            "single_spectralindx.sky",
            "single_spectralindx.sky",
            True,
            True,
            id="multi_point_single_si",
        ),
        pytest.param(
            "nans.sky",
            "nans.sky",
            True,
            True,
            id="patches_contains_nans",
        ),
        pytest.param(
            "transfer_patches_from.sky",
            "to_patched.sky",
            True,
            True,
            id="patch_positions_close_but_not_exact",
        ),
        pytest.param(
            "transfer_patches_from.sky",
            "transfer_patches_to.sky",
            False,
            True,
            id="ignore_patches",
        ),
        # Unequal cases
        pytest.param(
            "transfer_patches_from.sky",
            "transfer_patches_to.sky",
            True,
            False,
            id="check_patches",
        ),
        pytest.param(
            "single_point.sky",
            "single_spectralindx.sky",
            False,
            False,
            id="not_equal",
        ),
    ],
)
def test_check_skymodels_equal(
    test_data_path,
    left_filename,
    right_filename,
    check_patch_names_sizes,
    expected_equal,
):
    # Assert
    assert (
        check_skymodels_equal(
            test_data_path / left_filename,
            test_data_path / right_filename,
            check_patch_names_sizes,
        )
        is expected_equal
    )


class TestSkyModelGenerator:
    """
    Test the SkyModelGenerator class.
    """

    def test_minimal(self, tmp_path, rng):

        # create skymodel generator and sample 100 sources
        generator = SkyModelGenerator(
            q=None,
            u=None,
            v=None,
            reference_frequency=None,
            spectral_index=None,
            rotation_measure=None,
            major_axis=None,
            minor_axis=None,
            orientation=None,
        )
        # check that we can write and read the skymodel without errors
        path = tmp_path / "test_skymodel_generator.sky"
        generator.to_file(path, 10, rng)

        skymodel = load(path)
        assert skymodel.getColNames() == ["Name", "Type", "Ra", "Dec", "I"]
        assert len(skymodel) == 10

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({}, id="default"),
            pytest.param(
                {"ra": uniform_range(0, 45), "dec": uniform_range(-45, 45)},
                id="custom",
            ),
        ],
    )
    def test_skymodel_generator(self, config, rng):

        # create skymodel generator and draw a random sample of sources
        generator = SkyModelGenerator(**config)
        samples = generator.sample(n_sources=1_000, random_state=rng)

        # Check that the samples are within the expected ranges and have the
        # expected distribution.
        ra0 = generator.ra.kwds["loc"]
        ra1 = ra0 + generator.ra.kwds["scale"]
        dec0 = generator.dec.kwds["loc"]
        dec1 = dec0 + generator.dec.kwds["scale"]

        assert np.all((ra0 < samples["ra"]) & (samples["ra"] < ra1))
        assert np.all((dec0 < samples["dec"]) & (samples["dec"] < dec1))
        assert not any(
            map(len, np.nonzero([samples["q"], samples["u"], samples["v"]]))
        )
        assert np.all(samples["reference_frequency"] == 1.44e8)
        assert np.all(samples["minor_axis"] < samples["major_axis"])

        # test that samples are drawn from the correct distributions
        assert kstest(samples["ra"], uniform(ra0, ra1 - ra0).cdf).pvalue > 0.05
        assert (
            kstest(samples["dec"], uniform(dec0, dec1 - dec0).cdf).pvalue > 0.05
        )
