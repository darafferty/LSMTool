"""
Tests for assertion helpers used in tests
"""

import pytest

from lsmtool.testing import check_skymodels_equal
from lsmtool import load
from conftest import TEST_DATA_PATH


@pytest.fixture(scope="session")
def input_skymodel():
    return TEST_DATA_PATH / "expected.true_sky.txt"


@pytest.fixture
def modified_skymodel(request, tmp_path):
    # Use the input skymodel fixture value
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
    stem, nr = last_patch_name.rsplit("_", 1)
    new_name = f"{stem}_{int(nr) + 1}"

    patch_pos = skymodel.getPatchPositions()
    patch_pos[new_name] = patch_pos.pop(last_patch_name)
    skymodel.setPatchPositions(patch_pos)

    patches = skymodel.table["Patch"]
    patches[patches == last_patch_name] = new_name

    patches = skymodel.table.groups.keys["Patch"]
    patches[patches == last_patch_name] = new_name


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
        # pytest.param(
        #     "", skymodel_modified_patches, True, False, id="boop",
        # ),
        # pytest.param(
        #     "", "", True, False, id="different_default_values"
        # ),
        # pytest.param("", "", True, False, id="different_col_values"),
    ],
)
def test_check_skymodels_equal(
    left_filename, right_filename, check_patch_names_sizes, expected_equal
):

    # Assert
    assert (
        check_skymodels_equal(
            TEST_DATA_PATH / left_filename,
            TEST_DATA_PATH / right_filename,
            check_patch_names_sizes,
        )
        is expected_equal
    )
