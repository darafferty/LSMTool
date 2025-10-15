"""
Benchmarking different implementations of functions used in `tessellate`.
"""

import json
import itertools as itt
import functools as ftl
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import scipy
import pytest
import numpy as np
import matplotlib.pyplot as plt
from lsmtool.facet import in_box

from lsmtool.facet import in_box


INDEX_OUTSIDE_DIAGRAM = -1
BBOX_SHAPE_FOR_XY_RANGES = (2, 2)

# ---------------------------------------------------------------------------- #


def in_box_v0(cal_coords, bounding_box):
    return np.logical_and(
        np.logical_and(
            bounding_box[0] <= cal_coords[:, 0],
            cal_coords[:, 0] <= bounding_box[1],
        ),
        np.logical_and(
            bounding_box[2] <= cal_coords[:, 1],
            cal_coords[:, 1] <= bounding_box[3],
        ),
    )


def in_box_v1(cal_coords, bounding_box):
    minx, maxx, miny, maxy = bounding_box
    minx, maxx = sorted([minx, maxx])
    miny, maxy = sorted([miny, maxy])
    x, y = np.transpose(cal_coords)
    return (minx <= x) & (x <= maxx) & (miny <= y) & (y <= maxy)


def in_box_v2(cal_coords, bounding_box):
    minx, maxx, miny, maxy = bounding_box
    minx, maxx = sorted([minx, maxx])
    miny, maxy = sorted([miny, maxy])
    x, y = cal_coords[..., 0], cal_coords[..., 1]
    return (minx <= x) & (x <= maxx) & (miny <= y) & (y <= maxy)


@pytest.mark.parametrize("version", range(3))
@pytest.mark.parametrize(
    "size", (10 ** np.linspace(1, 6, 11)).astype(int).tolist()
)
def test_in_box_benchmark(version, size, benchmark):
    # Act
    np.random.seed(0)
    bounding_box = [0, 1, 0, 1]
    data = np.random.rand(size, 2)
    in_box = eval(f"in_box_v{version}")
    benchmark(in_box, data, bounding_box)


# ---------------------------------------------------------------------------- #


def prepare_points_for_tessellate_v0(cal_coords, bounding_box):
    # Select calibrators inside the bounding box
    points_inside = in_box(cal_coords, bounding_box)

    # Mirror points
    points_center = cal_coords[points_inside, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (
        bounding_box[1] - points_right[:, 0]
    )
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(
        points_center,
        np.append(
            np.append(points_left, points_right, axis=0),
            np.append(points_down, points_up, axis=0),
            axis=0,
        ),
        axis=0,
    )
    return points_center, points


def prepare_points_for_tessellate_v1(cal_coords, bounding_box):
    # Select calibrators inside the bounding box
    points_centre = cal_coords[in_box(cal_coords, bounding_box)]

    # Mirror points
    points_mirror = np.tile(points_centre, (2, 2, 1, 1))
    intervals = np.reshape(bounding_box, (2, 1, 2))
    xy = 2 * intervals - points_centre.T[..., None]
    points_mirror[0, ..., 0] = xy[0].T
    points_mirror[1, ..., 1] = xy[1].T

    points = np.vstack([points_centre, points_mirror.reshape(-1, 2)])
    return points_centre, points


def prepare_points_for_tessellate_v2(cal_coords, bounding_box):
    # Select calibrators inside the bounding box
    points_centre = cal_coords[in_box(cal_coords, bounding_box)]

    # Mirror points
    points_mirror = np.tile(points_centre, (2, 2, 1, 1))
    intervals = np.reshape(bounding_box, (2, 1, 2))
    xy = 2 * intervals - points_centre.T[..., None]
    indices = [0, 1]
    points_mirror[indices, ..., indices] = xy[indices].swapaxes(-1, 1)
    points = np.vstack([points_centre, points_mirror.reshape(-1, 2)])
    return points_centre, points


def prepare_points_for_tessellate_v3(cal_coords, bounding_box):
    # Select calibrators inside the bounding box
    points_centre = cal_coords[in_box(cal_coords, bounding_box)]

    if len(points_centre) == 0:
        return points_centre, points_centre

    # Extract bounding box coordinates
    minx, maxx, miny, maxy = bounding_box

    # Create mirrored points more efficiently
    x_coords, y_coords = points_centre.T

    # Mirror across each boundary
    mirror_x_min = np.column_stack((2 * minx - x_coords, y_coords))
    mirror_x_max = np.column_stack((2 * maxx - x_coords, y_coords))
    mirror_y_min = np.column_stack((x_coords, 2 * miny - y_coords))
    mirror_y_max = np.column_stack((x_coords, 2 * maxy - y_coords))

    # Combine all points
    points = np.vstack(
        [points_centre, mirror_x_min, mirror_x_max, mirror_y_min, mirror_y_max]
    )

    return points_centre, points


def prepare_points_for_tessellate_v4(cal_coords, bounding_box):
    # Select calibrators inside the bounding box
    points_centre = cal_coords[in_box(cal_coords, bounding_box)]

    if len(points_centre) == 0:
        return points_centre, points_centre

    # Extract bounding box coordinates
    minx, maxx, miny, maxy = bounding_box

    # Create mirrored points more efficiently
    x_coords, y_coords = points_centre[..., 0], points_centre[..., 1]

    # Mirror across each boundary
    mirror_x_min = np.column_stack((2 * minx - x_coords, y_coords))
    mirror_x_max = np.column_stack((2 * maxx - x_coords, y_coords))
    mirror_y_min = np.column_stack((x_coords, 2 * miny - y_coords))
    mirror_y_max = np.column_stack((x_coords, 2 * maxy - y_coords))

    # Combine all points
    points = np.vstack(
        [points_centre, mirror_x_min, mirror_x_max, mirror_y_min, mirror_y_max]
    )

    return points_centre, points


def prepare_points_for_tessellate_v5(cal_coords, bounding_box):
    # Select calibrators inside the bounding box
    points_centre = cal_coords[in_box(cal_coords, bounding_box)]

    if len(points_centre) == 0:
        return points_centre, points_centre

    # Extract bounding box coordinates
    # from IPython import embed
    # embed(header="Embedded interpreter at 'tests/test_facet.py':621")
    minx, maxx, miny, maxy = bounding_box

    # Create mirrored points more efficiently
    x_coords, y_coords = points_centre[..., 0], points_centre[..., 1]

    # Mirror across each boundary
    points = np.column_stack(
        [
            (x_coords, y_coords),
            (2 * minx - x_coords, y_coords),  # mirror_x_min
            (2 * maxx - x_coords, y_coords),  # mirror_x_max
            (x_coords, 2 * miny - y_coords),  # mirror_y_min
            (x_coords, 2 * maxy - y_coords),  # mirror_y_max
        ]
    )

    return points_centre, points.T


def prepare_points_for_tessellate_v6(cal_coords, bounding_box):
    # Select calibrators inside the bounding box
    points_centre = cal_coords[in_box(cal_coords, bounding_box)]

    # Mirror points
    intervals = np.reshape(bounding_box, (1, 2, 2))
    xy = 2 * intervals - points_centre[..., np.newaxis]
    points_mirror = np.tile(points_centre, (3, 2, 1, 1))
    points_mirror[[0, 1], ..., [0, 1]] = np.moveaxis(xy, 0, -1)
    points_mirror = points_mirror.reshape(-1, 2)
    return points_centre, points_mirror


# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("version", np.arange(0, 7))
@pytest.mark.parametrize(
    "data", (np.random.rand(int(10**size), 2) for size in [4])
)
def test_prepare_benchmark(version, data, benchmark):
    # Act
    np.random.seed(0)
    bounding_box = [0, 1, 0, 1]

    prepare = eval(f"prepare_points_for_tessellate_v{version}")
    benchmark(prepare, data, bounding_box)


def in_box_v0(cal_coords, bounding_box):
    return np.logical_and(
        np.logical_and(
            bounding_box[0] <= cal_coords[:, 0],
            cal_coords[:, 0] <= bounding_box[1],
        ),
        np.logical_and(
            bounding_box[2] <= cal_coords[:, 1],
            cal_coords[:, 1] <= bounding_box[3],
        ),
    )


def in_box_v1(cal_coords, bounding_box):
    minx, maxx, miny, maxy = bounding_box
    minx, maxx = sorted([minx, maxx])
    miny, maxy = sorted([miny, maxy])
    x, y = np.transpose(cal_coords)
    return (minx <= x) & (x <= maxx) & (miny <= y) & (y <= maxy)


def in_box_v2(cal_coords, bounding_box):
    minx, maxx, miny, maxy = bounding_box
    minx, maxx = sorted([minx, maxx])
    miny, maxy = sorted([miny, maxy])
    x, y = cal_coords[..., 0], cal_coords[..., 1]
    return (minx <= x) & (x <= maxx) & (miny <= y) & (y <= maxy)


@pytest.mark.parametrize("version", np.arange(0, 3))
@pytest.mark.parametrize(
    "data", (np.random.rand(int(10**size), 2) for size in [4])
)
def test_in_box_benchmark(version, data, benchmark):
    # Act
    np.random.seed(0)
    bounding_box = [0, 1, 0, 1]

    in_box = eval(f"in_box_v{version}")
    benchmark(in_box, data, bounding_box)


# ---------------------------------------------------------------------------- #

def prepare_points_for_tessellate_v0(cal_coords, bounding_box):
    # Select calibrators inside the bounding box
    points_inside = in_box(cal_coords, bounding_box)

    # Mirror points
    points_center = cal_coords[points_inside, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (
        bounding_box[1] - points_right[:, 0]
    )
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(
        points_center,
        np.append(
            np.append(points_left, points_right, axis=0),
            np.append(points_down, points_up, axis=0),
            axis=0,
        ),
        axis=0,
    )
    return points_center, points


def prepare_points_for_tessellate_v1(cal_coords, bounding_box):
    # Select calibrators inside the bounding box
    points_centre = cal_coords[in_box(cal_coords, bounding_box)]

    # Mirror points
    points_mirror = np.tile(points_centre, (2, 2, 1, 1))
    intervals = np.reshape(bounding_box, (2, 1, 2))
    xy = 2 * intervals - points_centre.T[..., None]
    points_mirror[0, ..., 0] = xy[0].T
    points_mirror[1, ..., 1] = xy[1].T

    points = np.vstack([points_centre, points_mirror.reshape(-1, 2)])
    return points_centre, points


def prepare_points_for_tessellate_v2(cal_coords, bounding_box):
    # Select calibrators inside the bounding box
    points_centre = cal_coords[in_box(cal_coords, bounding_box)]

    # Mirror points
    points_mirror = np.tile(points_centre, (2, 2, 1, 1))
    intervals = np.reshape(bounding_box, (2, 1, 2))
    xy = 2 * intervals - points_centre.T[..., None]
    indices = [0, 1]
    points_mirror[indices, ..., indices] = xy[indices].swapaxes(-1, 1)
    points = np.vstack([points_centre, points_mirror.reshape(-1, 2)])
    return points_centre, points


def prepare_points_for_tessellate_v3(cal_coords, bounding_box):
    # Select calibrators inside the bounding box
    points_centre = cal_coords[in_box(cal_coords, bounding_box)]

    if len(points_centre) == 0:
        return points_centre, points_centre

    # Extract bounding box coordinates
    minx, maxx, miny, maxy = bounding_box

    # Create mirrored points more efficiently
    x_coords, y_coords = points_centre.T

    # Mirror across each boundary
    mirror_x_min = np.column_stack((2 * minx - x_coords, y_coords))
    mirror_x_max = np.column_stack((2 * maxx - x_coords, y_coords))
    mirror_y_min = np.column_stack((x_coords, 2 * miny - y_coords))
    mirror_y_max = np.column_stack((x_coords, 2 * maxy - y_coords))

    # Combine all points
    points = np.vstack(
        [points_centre, mirror_x_min, mirror_x_max, mirror_y_min, mirror_y_max]
    )

    return points_centre, points


def prepare_points_for_tessellate_v4(cal_coords, bounding_box):
    # Select calibrators inside the bounding box
    points_centre = cal_coords[in_box(cal_coords, bounding_box)]

    if len(points_centre) == 0:
        return points_centre, points_centre

    # Extract bounding box coordinates
    minx, maxx, miny, maxy = bounding_box

    # Create mirrored points more efficiently
    x_coords, y_coords = points_centre[..., 0], points_centre[..., 1]

    # Mirror across each boundary
    mirror_x_min = np.column_stack((2 * minx - x_coords, y_coords))
    mirror_x_max = np.column_stack((2 * maxx - x_coords, y_coords))
    mirror_y_min = np.column_stack((x_coords, 2 * miny - y_coords))
    mirror_y_max = np.column_stack((x_coords, 2 * maxy - y_coords))

    # Combine all points
    points = np.vstack(
        [points_centre, mirror_x_min, mirror_x_max, mirror_y_min, mirror_y_max]
    )

    return points_centre, points


def prepare_points_for_tessellate_v5(cal_coords, bounding_box):
    # Select calibrators inside the bounding box
    points_centre = cal_coords[in_box(cal_coords, bounding_box)]

    if len(points_centre) == 0:
        return points_centre, points_centre

    # Extract bounding box coordinates
    # from IPython import embed
    # embed(header="Embedded interpreter at 'tests/test_facet.py':621")
    minx, maxx, miny, maxy = bounding_box

    # Create mirrored points more efficiently
    x_coords, y_coords = points_centre[..., 0], points_centre[..., 1]

    # Mirror across each boundary
    points = np.column_stack(
        [
            (x_coords, y_coords),
            (2 * minx - x_coords, y_coords),  # mirror_x_min
            (2 * maxx - x_coords, y_coords),  # mirror_x_max
            (x_coords, 2 * miny - y_coords),  # mirror_y_min
            (x_coords, 2 * maxy - y_coords),  # mirror_y_max
        ]
    )

    return points_centre, points.T


def prepare_points_for_tessellate_v6(cal_coords, bounding_box):

    # Select calibrators inside the bounding box
    points_centre = cal_coords[in_box(cal_coords, bounding_box)]

    # Mirror points
    intervals = np.reshape(bounding_box, (1, 2, 2))
    xy = 2 * intervals - points_centre[..., np.newaxis]
    points_mirror = np.tile(points_centre, (3, 2, 1, 1))
    points_mirror[[0, 1], ..., [0, 1]] = np.moveaxis(xy, 0, -1)
    points_mirror = points_mirror.reshape(-1, 2)
    return points_centre, points_mirror


@pytest.mark.parametrize("version", range(7))
@pytest.mark.parametrize(
    "size", (10 ** np.linspace(1, 5.5, 10)).astype(int).tolist()
)
def test_prepare_benchmark(version, size, benchmark):
    # Act
    np.random.seed(0)
    bounding_box = [0, 1, 0, 1]
    data = np.random.rand(size, 2)

    prepare = eval(f"prepare_points_for_tessellate_v{version}")
    benchmark(prepare, data, bounding_box)


# ---------------------------------------------------------------------------- #

def filter_points_v0(voronoi_data):
    eps = 1e-6

    n, vor, points_centre, bounding_box = voronoi_data

    sorted_regions = np.array(vor.regions, dtype=object)[
        np.array(vor.point_region)
    ]

    # Filter regions
    filtered_regions = []
    for region in sorted_regions.tolist():
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not (
                    bounding_box[0] - eps <= x
                    and x <= bounding_box[1] + eps
                    and bounding_box[2] - eps <= y
                    and y <= bounding_box[3] + eps
                ):
                    flag = False
                    break
        if region and flag:
            filtered_regions.append(region)

    return points_centre, vor.vertices, filtered_regions


def filter_points_v1(voronoi_data):
    eps = 1e-6

    n, vor, points_centre, bounding_box = voronoi_data

    # Compute Voronoi, sorting the output regions to match the order of the
    # input coordinates
    sorted_regions = np.array(vor.regions, dtype=object)[
        np.array(vor.point_region)
    ]

    # Filter regions
    filtered_regions = []
    for region in sorted_regions.tolist():
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not (
                    bounding_box[0] - eps <= x
                    and x <= bounding_box[1] + eps
                    and bounding_box[2] - eps <= y
                    and y <= bounding_box[3] + eps
                ):
                    flag = False
                    break
        if region and flag:
            filtered_regions.append(region)

    return points_centre, vor.vertices, filtered_regions


def filter_points_v2(voronoi_data, eps=1e-6):
    n, vor, points_centre, bounding_box = voronoi_data

    # Compute Voronoi, sorting the output regions to match the order of the
    # input coordinates
    sorted_regions = np.array(vor.regions, dtype=object)[
        np.array(vor.point_region)
    ]

    # Add
    bounding_box = (
        np.reshape(bounding_box, BBOX_SHAPE_FOR_XY_RANGES) + (-eps, eps)
    ).ravel()

    # Filter regions
    filtered_regions = [
        region
        for region in sorted_regions
        if region
        and (INDEX_OUTSIDE_DIAGRAM not in region)
        and in_box(vor.vertices[region], bounding_box).all()
    ]
    return points_centre, vor.vertices, filtered_regions


def filter_points_v3(voronoi_data, eps=1e-6):
    n, vor, points_centre, bounding_box = voronoi_data

    # Add
    bounding_box = (
        np.reshape(bounding_box, BBOX_SHAPE_FOR_XY_RANGES) + (-eps, eps)
    ).ravel()

    # Filter regions
    filtered_regions = list(filter_regions(vor, bounding_box))

    return points_centre, vor.vertices, filtered_regions


def filter_regions(vor, bounding_box):
    sorted_regions = np.array(vor.regions, dtype=object)[vor.point_region]

    for region in sorted_regions:
        if not region:
            continue

        if INDEX_OUTSIDE_DIAGRAM in region:
            continue

        if in_box(vor.vertices[region], bounding_box).all():
            yield region


def filter_points_v4(voronoi_data, eps=1e-6):
    eps = 1e-6

    n, vor, points_centre, bounding_box = voronoi_data

    sorted_regions = np.array(vor.regions, dtype=object)[
        np.array(vor.point_region)
    ]

    minx, maxx, miny, maxy = bounding_box
    minx = minx - eps
    miny = miny - eps
    maxx = maxx + eps
    maxy = maxy + eps

    # Filter regions
    filtered_regions = []
    for region in sorted_regions.tolist():
        keep = True
        for index in region:
            if index == -1:
                keep = False
                break
            else:
                y = vor.vertices[index, 1]
                x = vor.vertices[index, 0]
                if (minx > x) or (x > maxx) or (miny > y) or (y > maxy):
                    keep = False
                    break
        if region and keep:
            filtered_regions.append(region)

    return points_centre, vor.vertices, filtered_regions


def filter_points_v5(voronoi_data, eps=1e-6):
    eps = 1e-6

    n, vor, points_centre, bounding_box = voronoi_data

    sorted_regions = np.array(vor.regions, dtype=object)[
        np.array(vor.point_region)
    ]

    minx, maxx, miny, maxy = bounding_box
    minx = minx - eps
    miny = miny - eps
    maxx = maxx + eps
    maxy = maxy + eps

    # Filter regions
    filtered_regions = []
    for region in sorted_regions:
        keep = True
        for index in region:
            if index == -1:
                keep = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if (minx > x) or (x > maxx) or (miny > y) or (y > maxy):
                    keep = False
                    break

        if region and keep:
            filtered_regions.append(region)

    return points_centre, vor.vertices, filtered_regions


def filter_points_v6(voronoi_data, eps=1e-6):
    eps = 1e-6

    n, vor, points_centre, bounding_box = voronoi_data

    sorted_regions = np.array(vor.regions, dtype=object)[
        np.array(vor.point_region)
    ]

    minx, maxx, miny, maxy = bounding_box
    minx -= eps
    miny -= eps
    maxx += eps
    maxy += eps

    # Filter regions
    # f = ftl.partial(filter_region, vor=vor, bbox=bounding_box)
    # filtered_regions = filter(f, sorted_regions)
    vertices = vor.vertices
    filtered_regions = [
        region
        for region in sorted_regions
        if keep_region(region, vertices, (minx, maxx, miny, maxy))
    ]

    return points_centre, vertices, list(filtered_regions)


def keep_region(region, vertices, bbox):
    minx, maxx, miny, maxy = bbox
    for index in region:
        if index == -1:
            return False

        x = vertices[index, 0]
        y = vertices[index, 1]
        if (minx > x) or (x > maxx) or (miny > y) or (y > maxy):
            # not ((minx < x < maxx) and (miny < y < maxy))
            return False

    return bool(region)


def _filter_points_v7(sorted_regions, vertices, bounding_box, eps=1e-6):
    minx, maxx, miny, maxy = bounding_box
    minx -= eps
    miny -= eps
    maxx += eps
    maxy += eps

    # Filter regions
    # f = ftl.partial(filter_region, vor=vor, bbox=bounding_box)
    # filtered_regions = filter(f, sorted_regions)
    for region in sorted_regions:
        if keep_region(region, vertices, (minx, maxx, miny, maxy)):
            yield region


def filter_points_v7(voronoi_data):
    n, vor, points_centre, bounding_box = voronoi_data

    sorted_regions = np.array(vor.regions, dtype=object)[
        np.array(vor.point_region)
    ]

    filtered_regions = _filter_points_v7(
        sorted_regions, vor.vertices, bounding_box
    )
    return points_centre, vor.vertices, list(filtered_regions)


def filter_points_v8(voronoi_data, eps=1e-6):
    n, vor, points_centre, bounding_box = voronoi_data

    sorted_regions = np.array(vor.regions, dtype=object)[
        np.array(vor.point_region)
    ]

    minx, maxx, miny, maxy = bounding_box
    bounding_box = (minx - eps, maxx + eps, miny - eps, maxy + eps)

    # Filter regions
    vertices = vor.vertices
    filtered_regions = [
        region
        for region in sorted_regions
        if keep_region8(region, vertices, bounding_box)
    ]

    return points_centre, vertices, filtered_regions


def keep_region8(region, vertices, bbox):
    minx, maxx, miny, maxy = bbox

    for index in region:
        if index == -1:
            return False

        if not (
            (minx < vertices[index, 0] < maxx)
            and (miny < vertices[index, 1] < maxy)
        ):
            return False

    return bool(region)


def filter_points_v9(voronoi_data, eps=1e-6):
    n, vor, points_centre, bounding_box = voronoi_data

    minx, maxx, miny, maxy = bounding_box
    bounding_box = (minx - eps, maxx + eps, miny - eps, maxy + eps)

    # Filter regions
    vertices = vor.vertices
    regions = vor.regions
    filtered_regions = [
        region
        for index in vor.point_region
        if keep_region8((region := regions[index]), vertices, bounding_box)
    ]

    return points_centre, vertices, filtered_regions


def keep_region10(position_index, region_index, regions, vertices, bbox):

    for region in regions[region_index]:
        if not keep_index(region, vertices, bbox):
            return position_index, False

    return position_index, bool(region)


def keep_index(index, vertices, bbox):
    minx, maxx, miny, maxy = bbox
    if index == -1:
        return False

    if not (
        (minx < vertices[index, 0] < maxx) and
        (miny < vertices[index, 1] < maxy)
    ):
        return False

    return True


def filter_points_v10(voronoi_data, eps=1e-6):
    n, vor, points_centre, bounding_box = voronoi_data

    minx, maxx, miny, maxy = bounding_box
    bounding_box = (minx - eps, maxx + eps, miny - eps, maxy + eps)

    # Filter regions
    vertices = vor.vertices
    regions = vor.regions
    keep_region = ftl.partial(keep_region10,
                              regions=regions,
                              vertices=vertices,
                              bbox=bounding_box)

    with ThreadPoolExecutor() as executor:
        results = executor.map(
            keep_region, range(len(vor.point_region)), vor.point_region
        )
    filtered_regions = [regions[vor.point_region[i]]
                        for i, keep in results if keep]

    # filtered_regions = asyncio.run(filtered_regions_gen(vor, bounding_box))
    return points_centre, vertices, filtered_regions


def filter_points_v11(voronoi_data, eps=1e-6):
    n, vor, points_centre, bounding_box = voronoi_data

    minx, maxx, miny, maxy = bounding_box
    bounding_box = (minx - eps, maxx + eps, miny - eps, maxy + eps)

    # Filter regions
    vertices = vor.vertices
    regions = vor.regions
    filtered_regions = [
        region
        for index in vor.point_region
        if keep_region11((region := regions[index]), vertices, bounding_box)
    ]

    return points_centre, vertices, filtered_regions


def keep_region11(region, vertices, bbox):

    for index in region:
        if index == -1:
            return False

        if not in_box(vertices[index], bbox):
            return False

    return bool(region)


@pytest.fixture(params=np.linspace(1, 5.5, 10))
def voronoi_data(request):
    size = request.param
    cal_coords = np.random.rand(int(10**size), 2)
    bounding_box = [0, 1, 0, 1]

    # Prepare points for tessellation
    points_centre, points = prepare_points_for_tessellate_v4(
        cal_coords, bounding_box
    )
    # Compute Voronoi, sorting the output regions to match the order of the
    # input coordinates
    vor = scipy.spatial.Voronoi(points)

    return len(cal_coords), vor, points_centre, bounding_box


@pytest.mark.parametrize(
    "version", np.arange(0, 12).astype(str)
)
def test_filter_points_benchmark(version, voronoi_data, benchmark):
    # Act
    filter_points_test = eval(f"filter_points_v{version}")
    benchmark(filter_points_test, voronoi_data)


# ---------------------------------------------------------------------------- #


def collect_results(filename):
    path = Path(filename)
    data = json.loads(path.read_text())

    db = defaultdict(lambda: defaultdict(list))
    for info in data["benchmarks"]:
        params = info["params"]
        stats = db[params["version"]]
        stats["n"].append(params["size"])
        for key, val in info["stats"].items():
            stats[key].append(val)

    return db


def plot_benchmark_results(ax, db, **kws):
    markers = itt.cycle("os^DvP*X")
    for version, stats in db.items():
        ax.errorbar(
            stats["n"],
            stats["median"],
            stats["iqr"],
            marker=next(markers),
            ms=5,
            capsize=0,
            lw=1,
            ls="-" if version == 0 else "",
            label=f"v={version}",
        )
    ax.legend()
    ax.grid()
    ax.set(xscale="log", yscale="log")

# ---------------------------------------------------------------------------- #


def main():
    test_folder = Path(__file__).parent
    path = test_folder / "benchmark/runtime-results/"
    files = sorted(path.rglob("*.json"))

    db = {}
    for json_file in files[-1:]:
        fig, axes = plt.subplots(figsize=(8, 5))
        name = json_file.stem[5:]

        # Collect results from benchmark
        stats = db[name] = collect_results(json_file)
        ax = axes  # [i]
        plot_benchmark_results(ax, stats)
        ax.set(title=name, xlabel="Number of facets", ylabel="Time [s]")

        versions = np.array(list(stats.keys()))
        fastest = np.argmin([s["median"] for s in stats.values()], 0)
        fastest_version_by_size, score = np.unique(
            versions[fastest], return_counts=True
        )
        print(name)
        print(
            "\n".join(
                map(
                    "{0:<9}|{1:<9}".format,
                    *zip(
                        *[
                            ("version", "score"),
                            *(zip(fastest_version_by_size, score)),
                        ]
                    ),
                )
            )
        )
        print()
        fig.savefig(test_folder / f"benchmark/{name}.pdf")


# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    np.random.seed(90909)
    main()
