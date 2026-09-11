"""
Tests for the `lsmtool.operations._threshold` module.
"""

import numpy as np
import pytest

from lsmtool.operations._threshold import getPatchNamesByThreshold


class MockSkyModel:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.ungrouped = False

    def ungroup(self):
        self.ungrouped = True

    def _getXY(self, crdelt):
        assert crdelt == 1.0 / 4.0 / 3600.0
        return self.x, self.y, 42, 42


@pytest.mark.parametrize("threshold", [0.1, 0.001])
def test_get_patch_names_by_threshold_single_source(threshold):
    """Test the grouping of a single source into a patch."""
    sky_model = MockSkyModel([0], [0])

    patch_names = getPatchNamesByThreshold(
        sky_model, fwhmArcsec=1.0, root="island", threshold=threshold
    )

    assert sky_model.ungrouped
    expected_name = "island_patch_1" if threshold == 0.001 else "patch_0"
    np.testing.assert_array_equal(patch_names, [expected_name])


def test_get_patch_names_by_threshold_groups_multiple_sources():
    """
    Test the grouping of multiple sources into patches.
    - The first two sources should be grouped together, since they're adjacent.
    - The third source should be in its own patch.
      It explicitly tests handling of negative coordinates. Because of those
      coordinates, it should be the first patch (with index 1).
    - The last two sources should be grouped together, since they're equal.
    """
    sky_model = MockSkyModel([0, 1, -5, 12, 12], [3, 3, -2, 6, 6])

    patch_names = getPatchNamesByThreshold(
        sky_model,
        fwhmArcsec=1.0,
        root="island",
        threshold=0.05,
    )

    np.testing.assert_array_equal(
        patch_names,
        [
            "island_patch_2",
            "island_patch_2",
            "island_patch_1",
            "island_patch_3",
            "island_patch_3",
        ],
    )


@pytest.mark.parametrize("pad_index", [True, False])
def test_get_patch_names_by_threshold_pads_patch_indices(pad_index):
    """Test different `pad_index` settings."""
    sky_model = MockSkyModel(np.arange(0, 110, 10), np.zeros(11))

    patch_names = getPatchNamesByThreshold(
        sky_model,
        fwhmArcsec=1.0,
        root="island",
        threshold=0.05,
        pad_index=pad_index,
    )

    if pad_index:
        expected_names = [f"island_patch_{index:02d}" for index in range(1, 12)]
    else:
        expected_names = [f"island_patch_{index}" for index in range(1, 12)]
    np.testing.assert_array_equal(patch_names, expected_names)
