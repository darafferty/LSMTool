#!/usr/bin/env python3

import filecmp
import pathlib
import tarfile
import tempfile
import requests
import unittest
import contextlib

import pytest
import numpy as np
from numpy.testing import assert_array_equal

import lsmtool
from lsmtool.operations_lib import (
    apply_beam,
    make_wcs,
    normalize_ra_dec,
    in_box,
    tessellate,
    voronoi
)


class TestOperationsLib(unittest.TestCase):
    """
    Test class for the module `operations_lib`

    For testing a Measurement Set will be downloaded, if not present in the
    `tests/resources` directory. The skymodel was generated, using the pointing
    information in this MS, using the following commands:
    ```
    s = lsmtool.load('TGSS', VOPosition=["08:13:36.067800", "+48.13.02.58100"], VORadius=1.0)
    s.write('tests/resources/LOFAR_HBA_MOCK.sky')
    ```
    """

    @classmethod
    def setUpClass(cls):
        """
        Download and unpack Measurement Set used for testing
        """
        # Path to directory containing test resources
        cls.resource_path = pathlib.Path(__file__).parent / "resources"

        # Path to the input Measurement Set (will be downloaded if non-existent)
        cls.ms_path = cls.resource_path / "LOFAR_HBA_MOCK.ms"

        # URL to download location of input Measurement Set
        cls.ms_url = "https://support.astron.nl/software/ci_data/EveryBeam/L258627-one-timestep.tar.bz2"

        # Path to the input Sky Model (will be created if non-existent)
        cls.skymodel_path = cls.resource_path / "LOFAR_HBA_MOCK.sky"

        # Create a temporary test directory
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = pathlib.Path(cls.temp_dir.name)

        # Download MS if it's not available in resource directory
        if not cls.ms_path.exists():
            cls.download_ms()

    @classmethod
    def download_ms(cls):
        """
        Download and unpack MS in our temporary test directory
        """
        # Filter to strip leading component from file names (like `tar --strip-components=1`)
        filter = lambda member, path: member.replace(
            name=pathlib.Path(*pathlib.Path(member.path).parts[1:])
        )
        with requests.get(cls.ms_url, stream=True) as req:
            with tarfile.open(fileobj=req.raw, mode="r|bz2") as tarobj:
                tarobj.extractall(path=cls.ms_path, filter=filter)

    def setUp(self):
        self.skymodel = lsmtool.load(str(self.skymodel_path))
        self.ra = self.skymodel.getColValues("RA")
        self.dec = self.skymodel.getColValues("Dec")
        self.flux = self.skymodel.getColValues("I")

    def test_apply_beam(self):
        """
        Test `apply_beam` function
        """
        outfile = str(self.temp_path / "test_apply_beam.out")
        reffile = str(self.resource_path / "test_apply_beam.out")
        result = apply_beam(
            str(self.ms_path),
            self.flux,
            self.ra,
            self.dec,
        )
        np.set_printoptions(precision=6)
        with open(outfile, "w") as f:
            f.write(str(result))
        assert filecmp.cmp(reffile, outfile, shallow=False)

    def test_apply_beam_invert(self):
        """
        Test `apply_beam` function with inverted beam
        """
        outfile = str(self.temp_path / "test_apply_beam_invert.out")
        reffile = str(self.resource_path / "test_apply_beam_invert.out")
        result = apply_beam(
            str(self.ms_path), self.flux, self.ra, self.dec, invert=True
        )
        np.set_printoptions(precision=6)
        with open(outfile, "w") as f:
            f.write(str(result))
        assert filecmp.cmp(reffile, outfile, shallow=False)

    def test_normalize_ra_dec(self):
        """
        Test `normalize_ra_dec` function
        """
        ra = 450.0
        dec = 95.0
        result = normalize_ra_dec(ra, dec)
        assert result.ra == 270.0 and result.dec == 85.0


def test_make_wcs_default():
    """Test `make_wcs` with default parameters."""
    ref_ra = 10
    ref_dec = -42
    crdelt = 0.066667
    w = make_wcs(ref_ra, ref_dec)
    assert w is not None
    assert w.naxis == 2
    assert_array_equal(w.wcs.crpix, [1000, 1000])
    assert_array_equal(w.wcs.cdelt, [-crdelt, crdelt])
    assert_array_equal(w.wcs.crval, [ref_ra, ref_dec])
    assert_array_equal(w.wcs.ctype, ["RA---TAN", "DEC--TAN"])


def test_make_wcs_custom():
    """Test `make_wcs` with a custom crdelt parameter."""
    ref_ra = -10
    ref_dec = 42
    crdelt = 0.42
    w = make_wcs(ref_ra, ref_dec, crdelt)
    assert w is not None
    assert w.naxis == 2
    assert_array_equal(w.wcs.crpix, [1000, 1000])
    assert_array_equal(w.wcs.cdelt, [-crdelt, crdelt])
    assert_array_equal(w.wcs.crval, [ref_ra, ref_dec])
    assert_array_equal(w.wcs.ctype, ["RA---TAN", "DEC--TAN"])


@pytest.mark.parametrize(
    "cal_coords, bounding_box, expected, context",
    [
        # All points inside boundary
        pytest.param(
            np.array([[1, 1], [2, 2], [3, 3]]),
            np.array([0, 4, 0, 4]),
            np.array([True, True, True]),
            null_context := contextlib.nullcontext(),
            id="all_inside",
        ),
        # Some points inside, some outside boundary
        pytest.param(
            np.array([[0, 0], [5, 5], [2, 2]]),
            np.array([1, 4, 1, 4]),
            np.array([False, False, True]),
            null_context,
            id="some_inside_some_outside",
        ),
        # Points exactly on the boundary
        pytest.param(
            np.array([[1, 1], [4, 4], [1, 4], [4, 1]]),
            np.array([1, 4, 1, 4]),
            np.array([True, True, True, True]),
            null_context,
            id="on_boundary",
        ),
        # Empty `cal_coords`
        pytest.param(
            np.empty((0, 2)),
            np.array([0, 1, 0, 1]),
            np.array([], dtype=bool),
            null_context,
            id="empty_coords",
        ),
        # Bounding box with zero area (min == max)
        pytest.param(
            np.array([[1, 1], [1, 2], [2, 1]]),
            np.array([1, 1, 1, 1]),
            np.array([True, False, False]),
            null_context,
            id="zero_area_box",
        ),
        # Negative coordinates
        pytest.param(
            np.array([[-2, -2], [-1, -1], [0, 0]]),
            np.array([-2, 0, -2, 0]),
            np.array([True, True, True]),
            null_context,
            id="negative_coords",
        ),
        # Bounding box with min > max (inverted box)
        pytest.param(
            np.array([[0, 0], [1, 1]]),
            np.array([2, 0, 2, 0]),
            np.array([True, True]),
            null_context,
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
    "ra_cal, ra_dec, ra_mid, dec_mid, width_ra, width_dec, "
    "expected_facet_points, expected_facet_polygons",
    [
        pytest.param(
            RA_CAL := [119.73, 138.08, 124.13, 115.74],
            DEC_CAL := [89.92, 89.91, 89.89, 89.89],
            126.52,
            90.0,
            0.3,
            0.3,
            # expected_facet_points
            np.transpose([RA_CAL, DEC_CAL]),
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
    ra_cal,
    ra_dec,
    ra_mid,
    dec_mid,
    width_ra,
    width_dec,
    expected_facet_points,
    expected_facet_polygons,
):
    """
    Test the tessellate function, using a region that encompasses the North
    Celestial Pole (NCP).
    """

    # Tessellate a region that encompasses the NCP.
    facet_points, facet_polys = tessellate(
        ra_cal,
        ra_dec,
        ra_mid,
        dec_mid,
        width_ra,
        width_dec,
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
    "cal_coords, bounding_box, in_box_expected, filtered_regions, context",
    [
        # 4 points in a square
        pytest.param(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            [0, 1, 0, 1],
            ...,
            [],
            null_context,
            id="square",
        ),
        # Edge case: only one point inside bounding box
        pytest.param(
            np.array([[0.5, 0.5], [2, 2]]),
            [0, 1, 0, 1],
            np.array([True, False]),
            [[3, 1, 0, 2]],
            null_context,
            id="edge_case_one_inside",
        ),
        # -------------------------------------------------------------------- #
        # Edge case: all points outside bounding box
        pytest.param(
            np.array([[2, 2], [3, 3]]),
            [0, 1, 0, 1],
            ...,
            [],
            pytest.raises(ValueError),
            id="edge_case_all_outside",
        ),
    ],
)
def test_voronoi(
    cal_coords,
    bounding_box,
    in_box_expected,
    filtered_regions,
    context
):
    # Arrange
    with context:
        # Act
        vor = voronoi(cal_coords, bounding_box)

        # Assert
        # filtered_points should match the points inside the bounding box
        expected_points = cal_coords[in_box_expected]
        assert_array_equal(vor.filtered_points, expected_points)

        # check filtered_regions matches expected
        assert vor.filtered_regions == filtered_regions


if __name__ == "__main__":
    unittest.main(verbosity=2)
