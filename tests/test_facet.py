from astropy.coordinates import SkyCoord
import contextlib

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from lsmtool.facet import (
    in_box,
    tessellate,
    voronoi,
    prepare_points_for_tessellate,
)

NULL_CONTEXT = contextlib.nullcontext()


@pytest.mark.parametrize(
    "cal_coords, bounding_box, expected, context",
    [
        # All points inside boundary
        pytest.param(
            np.array([[1, 1], [2, 2], [3, 3]]),
            np.array([0, 4, 0, 4]),
            np.array([True, True, True]),
            NULL_CONTEXT,
            id="all_inside",
        ),
        # Some points inside, some outside boundary
        pytest.param(
            np.array([[0, 0], [5, 5], [2, 2]]),
            np.array([1, 4, 1, 4]),
            np.array([False, False, True]),
            NULL_CONTEXT,
            id="some_inside_some_outside",
        ),
        # Points exactly on the boundary
        pytest.param(
            np.array([[1, 1], [4, 4], [1, 4], [4, 1]]),
            np.array([1, 4, 1, 4]),
            np.array([True, True, True, True]),
            NULL_CONTEXT,
            id="on_boundary",
        ),
        # Empty `cal_coords`
        pytest.param(
            np.empty((0, 2)),
            np.array([0, 1, 0, 1]),
            np.array([], dtype=bool),
            NULL_CONTEXT,
            id="empty_coords",
        ),
        # Bounding box with zero area (min == max)
        pytest.param(
            np.array([[1, 1], [1, 2], [2, 1]]),
            np.array([1, 1, 1, 1]),
            np.array([True, False, False]),
            NULL_CONTEXT,
            id="zero_area_box",
        ),
        # Negative coordinates
        pytest.param(
            np.array([[-2, -2], [-1, -1], [0, 0]]),
            np.array([-2, 0, -2, 0]),
            np.array([True, True, True]),
            NULL_CONTEXT,
            id="negative_coords",
        ),
        # Bounding box with min > max (inverted box)
        pytest.param(
            np.array([[0, 0], [1, 1]]),
            np.array([2, 0, 2, 0]),
            np.array([True, True]),
            NULL_CONTEXT,
            id="inverted_box",
        ),
        # -------------------------------------------------------------------- #
        # Error: `cal_coords` not 2D
        pytest.param(
            np.array([1, 2, 3]),
            np.array([0, 1, 0, 1]),
            None,
            pytest.raises(ValueError),
            id="cal_coords_not_2d",
        ),
        # Error: `cal_coords` with wrong number of columns
        pytest.param(
            np.array([[1, 1, 1], [2, 2, 2]]),
            np.array([0, 2, 0, 2]),
            None,
            pytest.raises(ValueError),
            id="cal_coords_wrong_columns",
        ),
        # Error: `bounding_box` wrong shape (too small)
        pytest.param(
            np.array([[1, 1]]),
            np.array([0, 1, 0]),
            None,
            pytest.raises(ValueError),
            id="bounding_box_too_short",
        ),
        # Error: `bounding_box` wrong shape (too big)
        pytest.param(
            np.array([[1, 1]]),
            np.array([0, 1, 0, 1, 2]),
            None,
            pytest.raises(ValueError),
            id="bounding_box_too_long",
        ),
    ],
)
def test_in_box(cal_coords, bounding_box, expected, context):
    # Act
    with context:
        result = in_box(cal_coords, bounding_box)

        # Assert
        assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "directions, bbox_midpoint, bbox_size, "
    "expected_facet_points, expected_facet_polygons",
    [
        pytest.param(
            directions := SkyCoord(
                ra=[119.73, 138.08, 124.13, 115.74],
                dec=[89.92, 89.91, 89.89, 89.89],
                unit="deg",
            ),
            bbox_midpoint := SkyCoord(ra=126.52, dec=90.0, unit="deg"),
            bbox_size := (0.3, 0.3),
            # expected_facet_points
            np.transpose([directions.ra.deg, directions.dec.deg]),
            #
            [
                [
                    [127.3, 89.9],
                    [278.9, 89.8],
                    [351.5, 89.8],
                    [51.9, 89.8],
                    [119.9, 89.9],
                    [127.3, 89.9],
                ],
                [
                    [127.3, 89.9],
                    [278.9, 89.8],
                    [261.5, 89.8],
                    [171.5, 89.8],
                    [146.9, 89.8],
                    [127.3, 89.9],
                ],
                [
                    [127.3, 89.9],
                    [146.9, 89.8],
                    [119.9, 89.8],
                    [119.9, 89.9],
                    [127.3, 89.9],
                ],
                [
                    [119.9, 89.9],
                    [51.9, 89.8],
                    [81.5, 89.8],
                    [119.9, 89.8],
                    [119.9, 89.9],
                ],
            ],
        )
    ],
)
def test_tessellate(
    directions,
    bbox_midpoint,
    bbox_size,
    expected_facet_points,
    expected_facet_polygons,
):
    """
    Test the tessellate function, using a region that encompasses the North
    Celestial Pole (NCP).
    """

    # Tessellate a region that encompasses the NCP.
    facet_points, facet_polys = tessellate(
        directions,
        bbox_midpoint,
        bbox_size,
    )

    # Check the facet points
    np.testing.assert_allclose(facet_points, expected_facet_points)

    # Check the facet polygons. Since the tessellate function is not
    # guaranteed to return the same order of vertices for identical
    # polygons, check only that the set of expected and actual vertices
    # are identical.
    facet_polys = [np.round(a, 1).tolist() for a in facet_polys]
    facet_polys_flat = _flatten(facet_polys)

    facet_polys_expected = _flatten(expected_facet_polygons)
    difference = facet_polys_flat.symmetric_difference(facet_polys_expected)
    assert not difference


def _flatten(iterable):
    return {tuple(point) for item in iterable for point in item}


@pytest.mark.parametrize(
    "cal_coords, bounding_box, expected_in_box, expected_vertices, expected_regions, context",
    [
        # Regular input cases
        # -------------------------------------------------------------------- #
        # 4 points in a square
        pytest.param(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            [0, 1, 0, 1],
            np.array([True, True, True, True]),
            np.array(
                [
                    [-0.5, -0.5],
                    [1.5, -0.5],
                    [0.5, -0.5],
                    [1.5, 1.5],
                    [1.5, 0.5],
                    [-0.5, 1.5],
                    [-0.5, 0.5],
                    [0.5, 0.5],
                    [0.5, 1.5],
                ]
            ),
            [],
            NULL_CONTEXT,
            id="square",
        ),
        # Edge cases
        # -------------------------------------------------------------------- #
        # Only one point inside bounding box
        pytest.param(
            np.array([[0.5, 0.5], [2, 2]]),
            [0, 1, 0, 1],
            np.array([True, False]),
            np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            [[3, 1, 0, 2]],
            NULL_CONTEXT,
            id="edge_case_one_inside",
        ),
        # All points on the boundary
        pytest.param(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            [0, 1, 0, 1],
            np.array([True, True, True, True]),
            np.array(
                [
                    [-0.5, -0.5],
                    [1.5, -0.5],
                    [0.5, -0.5],
                    [1.5, 1.5],
                    [1.5, 0.5],
                    [-0.5, 1.5],
                    [-0.5, 0.5],
                    [0.5, 0.5],
                    [0.5, 1.5],
                ]
            ),
            [],
            NULL_CONTEXT,
            id="all_on_boundary",
        ),
        # Degenerate: all points colinear
        pytest.param(
            np.array([[0, 0], [0.5, 0.5], [1, 1]]),
            [0, 1, 0, 1],
            np.array([True, True, True]),
            np.array(
                [
                    [-0.25, 1.25],
                    [1.25, -0.25],
                    [0.0, 1.0],
                    [0.0, 0.5],
                    [0.5, 0.0],
                    [1.0, 0.0],
                    [0.5, 1.0],
                    [1.0, 0.5],
                ]
            ),
            [[7, 5, 4, 3, 2, 6]],
            NULL_CONTEXT,
            id="colinear_points",
        ),
        # Duplicate points
        pytest.param(
            np.array([[0, 0], [0, 0], [1, 1], [1, 1]]),
            [0, 1, 0, 1],
            np.array([True, True, True, True]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            [],
            NULL_CONTEXT,
            id="duplicate_points",
        ),
        # Error cases
        # -------------------------------------------------------------------- #
        # All points outside bounding box
        pytest.param(
            np.array([[2, 2], [3, 3]]),
            [0, 1, 0, 1],
            ...,
            [],
            [],
            pytest.raises(ValueError),
            id="edge_case_all_outside",
        ),
    ],
)
def test_voronoi(
    cal_coords,
    bounding_box,
    expected_in_box,
    expected_vertices,
    expected_regions,
    context,
):
    # Arrange
    with context:
        # Act
        points, vertices, regions = voronoi(cal_coords, bounding_box)

        # Assert
        # filtered_points should match the points inside the bounding box
        expected_points = cal_coords[expected_in_box]
        assert_array_equal(points, expected_points)

        # check regions are as expected
        assert regions == expected_regions

        # check vertices are as expected
        assert_array_equal(vertices, expected_vertices)


@pytest.mark.parametrize(
    "cal_coords, bounding_box, expected_centre, context",
    [
        # Nominal cases
        # -------------------------------------------------------------------- #
        # All points inside the bounding box
        pytest.param(
            np.array([[0, 0], [1, 1], [0.5, 0.5]]),
            [0, 1, 0, 1],
            np.array([[0, 0], [1, 1], [0.5, 0.5]]),
            NULL_CONTEXT,
            id="all_inside",
        ),
        # Some points inside, some outside
        pytest.param(
            np.array([[0, 0], [2, 2], [1, 1]]),
            [0, 1, 0, 1],
            np.array([[0, 0], [1, 1]]),
            NULL_CONTEXT,
            id="some_inside_some_outside",
        ),
        # All points outside
        pytest.param(
            np.array([[2, 2], [3, 3]]),
            [0, 1, 0, 1],
            np.empty((0, 2)),
            NULL_CONTEXT,
            id="all_outside",
        ),
        # Points on the boundary
        pytest.param(
            np.array([[0, 0], [1, 1], [0, 1], [1, 0]]),
            [0, 1, 0, 1],
            np.array([[0, 0], [1, 1], [0, 1], [1, 0]]),
            NULL_CONTEXT,
            id="on_boundary",
        ),
        # Empty input
        pytest.param(
            np.empty((0, 2)),
            [0, 1, 0, 1],
            np.empty((0, 2)),
            NULL_CONTEXT,
            id="empty_input",
        ),
        # Negative coordinates
        pytest.param(
            np.array([[-1, -1], [0, 0], [1, 1]]),
            [-1, 1, -1, 1],
            np.array([[-1, -1], [0, 0], [1, 1]]),
            NULL_CONTEXT,
            id="negative_coords",
        ),
        # Inverted bounding box (min > max)
        pytest.param(
            np.array([[0, 0], [1, 1]]),
            [1, 0, 1, 0],
            np.array([[0, 0], [1, 1]]),
            NULL_CONTEXT,
            id="inverted_box",
        ),
        # Duplicate points
        pytest.param(
            np.array([[0, 0], [0, 0], [1, 1], [1, 1]]),
            [0, 1, 0, 1],
            np.array([[0, 0], [0, 0], [1, 1], [1, 1]]),
            NULL_CONTEXT,
            id="duplicate_points",
        ),
        # Single point inside
        pytest.param(
            np.array([[0.5, 0.5]]),
            [0, 1, 0, 1],
            np.array([[0.5, 0.5]]),
            NULL_CONTEXT,
            id="single_point_inside",
        ),
        # Single point outside
        pytest.param(
            np.array([[2, 2]]),
            [0, 1, 0, 1],
            np.empty((0, 2)),
            NULL_CONTEXT,
            id="single_point_outside",
        ),
        # Error cases
        # -------------------------------------------------------------------- #
        # cal_coords not 2D
        pytest.param(
            np.array([1, 2, 3]),
            [0, 1, 0, 1],
            None,
            pytest.raises(ValueError),
            id="cal_coords_not_2d",
        ),
        # cal_coords wrong number of columns
        pytest.param(
            np.array([[1, 2, 3], [4, 5, 6]]),
            [0, 1, 0, 1],
            None,
            pytest.raises(ValueError),
            id="cal_coords_wrong_columns",
        ),
        # bounding_box wrong shape (too short)
        pytest.param(
            np.array([[1, 1]]),
            [0, 1, 0],
            None,
            pytest.raises(ValueError),
            id="bounding_box_too_short",
        ),
        # bounding_box wrong shape (too long)
        pytest.param(
            np.array([[1, 1]]),
            [0, 1, 0, 1, 2],
            None,
            pytest.raises(ValueError),
            id="bounding_box_too_long",
        ),
    ],
)
def test_prepare_points_for_tessellate(
    cal_coords, bounding_box, expected_centre, context
):
    # Act
    with context:
        points_centre, points = prepare_points_for_tessellate(
            cal_coords, bounding_box
        )

        # If there are N points_centre, there should be N*5 points in total
        # (original + 4 mirrored)
        expected_points = [*points_centre]
        for i, interval in enumerate(np.reshape(bounding_box, (2, 2))):
            for edge in interval:
                mirrored_points = points_centre.copy()
                mirrored_points[:, i] = 2 * edge - points_centre[:, i]
                expected_points.extend(mirrored_points)

        # Assert
        assert set(map(tuple, expected_points)) == set(map(tuple, points))

        np.testing.assert_array_equal(points_centre, expected_centre)
