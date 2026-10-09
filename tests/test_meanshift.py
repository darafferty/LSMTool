"""Regression checks for the Python mean-shift implementation."""

import numpy as np
import pytest

from lsmtool.operations._meanshift import Grouper


class ReferenceGrouper(Grouper):
    """Original distance calculation for trajectory comparisons."""

    def euclid_distance(self, coord, coords):
        return np.sqrt(np.sum((coord - coords) ** 2, axis=1))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_distances_match_row_reduction(dtype):
    coords = np.array([[0, 0], [3, 4], [-3, -4], [12, -5]], dtype=dtype)
    grouper = Grouper(coords, np.ones(4), 1, 5, 5, 0.1)
    expected = np.sqrt(np.sum((coords[0] - coords) ** 2, axis=1))
    np.testing.assert_array_equal(
        grouper.euclid_distance(coords[0], coords), expected
    )
    # Neighbours exactly at the radius must still be excluded.
    np.testing.assert_array_equal(
        grouper.neighbourhood_points(coords[0], coords, 5)[0], [0]
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_meanshift_trajectory_and_clusters_unchanged(dtype):
    rng = np.random.default_rng(123)
    coords = rng.normal(size=(100, 2)).astype(dtype)
    fluxes = rng.uniform(0.1, 10, 100)
    actual = Grouper(coords, fluxes, 0.2, 10, 0.5, 0.05)
    expected = ReferenceGrouper(coords, fluxes, 0.2, 10, 0.5, 0.05)
    actual.run()
    expected.run()
    np.testing.assert_array_equal(actual.past_coords, expected.past_coords)
    for cluster, reference in zip(
        actual.grouping(), expected.grouping(), strict=True
    ):
        np.testing.assert_array_equal(cluster, reference)
