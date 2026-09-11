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


@pytest.mark.parametrize("threshold", [0.1, 0.01])
def test_get_patch_names_by_threshold_single_source(threshold):
    """Test the grouping of a single source into a patch."""
    sky_model = MockSkyModel([0], [0])

    patch_names = getPatchNamesByThreshold(
        sky_model, fwhmArcsec=1.0, root="island", threshold=threshold
    )

    assert sky_model.ungrouped
    expected_name = "island_patch_1" if threshold == 0.01 else "patch_0"
    np.testing.assert_array_equal(patch_names, [expected_name])


def test_get_patch_names_by_threshold_groups_multiple_sources():
    """
    Test the grouping of multiple sources into patches.
    - The first two sources are adjacent, therefore reach the threshold and
      should be grouped together.
    - The third source is isolated and doesn't reach the threshold.
      It explicitly tests handling of negative coordinates.
    - The next two sources are isolated sources at the same position.
      Since an isolated source doesn't reach the threshold, they should
      be in their own patch, without the root prefix.
    - The last four sources form a 2x2 grid and thereby reach the threshold.
      Because of their low coordinates, they should have the first patch
      index with a root prefix.
    """
    sky_model = MockSkyModel(
        [4, 5, -5, 12, 12, 0, 1, 0, 1], [6, 6, -2, 6, 6, 0, 0, 1, 1]
    )

    patch_names = getPatchNamesByThreshold(
        sky_model,
        fwhmArcsec=1.0,
        root="island",
    )

    np.testing.assert_array_equal(
        patch_names,
        [
            "island_patch_2",
            "island_patch_2",
            "patch_0",
            "patch_1",
            "patch_2",
            "island_patch_1",
            "island_patch_1",
            "island_patch_1",
            "island_patch_1",
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
        threshold=0.01,
        pad_index=pad_index,
    )

    if pad_index:
        expected_names = [f"island_patch_{index:02d}" for index in range(1, 12)]
    else:
        expected_names = [f"island_patch_{index}" for index in range(1, 12)]
    np.testing.assert_array_equal(patch_names, expected_names)
