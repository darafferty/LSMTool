"""
Tests for the lsmtool.facet module.
"""

import contextlib
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from conftest import TEST_DATA_PATH
from numpy.testing import assert_array_equal

from lsmtool.facet import (
    Facet,
    SquareFacet,
    filter_skymodel,
    in_box,
    make_ds9_region_file,
    prepare_points_for_tessellate,
    read_ds9_region_file,
    read_skymodel,
    tessellate,
    voronoi,
)
from lsmtool.io import load

# ---------------------------------------------------------------------------- #
# Module constants

TEST_DATA_PATH = TEST_DATA_PATH / Path(__file__).stem

# ---------------------------------------------------------------------------- #
# Helper functions


def get_context(expected, **kws):
    """
    Get the appropriate runtime context for executing test code based on
    whether the expected result is an exception or not.

    Parameters
    ----------
    expected : Exception or object
        The expected result of the test. If this object is an ex

    Examples
    --------
    For tests that are expected to succeed:
    >>> @pytest.mark.parametrize("expected", [1])
    ... def test_success(expected):
    ...     with get_context(expected):
    ...         assert expected == 1

    For tests that are expected to fail:
    The following example will raise an IndexError, which will get caught by
    the `pytest.raises` context manager, leading to a successful test
    >>> @pytest.mark.parametrize("expected", [LookupError])
    ... def test_expected_failure(expected):
    ...     with get_context(expected):
    ...         [][1]

    Returns
    -------
    contextlib.AbstractContextManager
    """
    if isinstance(expected, type):
        if isinstance(expected, contextlib.AbstractContextManager):
            return expected

        if issubclass(expected, BaseException):
            return pytest.raises(expected, **kws)

    return contextlib.nullcontext(expected)


# ---------------------------------------------------------------------------- #
# Tests


class TestFacet:
    """Unit tests for the `lsmtool.facet.Facet` class."""

    reference_coords = 266.41681662, -29.00782497

    _reference_namespace = {
        "ra": reference_coords[0],
        "dec": reference_coords[1],
        "vertices": (
            _vertices := np.array(
                [
                    [266.9848842, -29.50105015],
                    [266.97941377, -28.50112422],
                    [265.84157768, -28.50107158],
                    [265.83598434, -29.50099529],
                ]
            )
        ),
    }
    _constructor_kws = {
        Facet: {"vertices": _reference_namespace["vertices"]},
        SquareFacet: {"width": 1},
    }
    _expected_namespace = {
        Facet: {
            **_reference_namespace,
            "size": 0.5,
            "ra_center": 266.41046451225884,
            "dec_center": -29.002269268446632,
            "ra_centroid": 266.4104645130159,
            "dec_centroid": -29.00226926732463,
            "x_center": 1000.0,
            "y_center": 1000.0,
        },
        SquareFacet: {
            **_reference_namespace,
            "vertices": (
                _vertices := np.array(
                    [
                        [266.9848842, -29.50105015],
                        [266.97941377, -28.50112422],
                        [265.84157768, -28.50107158],
                        [265.83598434, -29.50099529],
                    ]
                )
            ),
            "size": 0.5,
            "ra_center": 266.41046451354356,
            "dec_center": -29.00226926514979,
            "ra_centroid": 266.41046451354356,
            "dec_centroid": -29.00226926514978,
            "x_center": 1000.0,
            "y_center": 1000.0,
        },
    }

    @pytest.fixture(params=[Facet, SquareFacet])
    def facet_class(self, request):
        return request.param

    @pytest.fixture()
    def constructor_kws(self, facet_class):
        return self._constructor_kws[facet_class]

    @pytest.fixture()
    def expected_namespace(self, facet_class):
        return self._expected_namespace[facet_class]

    @pytest.mark.parametrize(
        "ra, dec",
        [
            pytest.param(
                *reference_coords,
                id="numeric RA and Dec",
            ),
            pytest.param(
                "17h45m40.03599s",
                "-29d00m28.1699s",
                id="string RA and Dec",
            ),
        ],
    )
    def test_init(
        self,
        facet_class,
        ra,
        dec,
        constructor_kws,
        expected_namespace,
    ):
        """
        Test the Facet class.
        """
        facet = facet_class(
            "test_facet",
            ra,
            dec,
            **constructor_kws,
        )
        for attr, val in expected_namespace.items():
            assert np.allclose(getattr(facet, attr), val), (
                f"Facet attribute {attr!r} does not match expected value."
            )

    # ------------------------------------------------------------------------ #
    @pytest.fixture()
    def facet(self, mocker):
        """
        Fixture to create a facet for testing.
        """
        facet = Facet(
            name="Square Facet",
            ra=1.0,
            dec=1,
            vertices=[(0, 2), (2, 2), (2, 0), (0, 0)],
        )

        # Attach mock skymodel
        facet.skymodel = mocker.MagicMock()

        return facet

    @pytest.fixture()
    def mock_download_panstarrs(self, mocker, request, facet):
        """
        Fixture to mock the download_panstarrs method of the
        `lsmtool.facet.Facet` class.
        """
        return mocker.patch.object(
            facet, "download_panstarrs", return_value=request.param
        )

    @pytest.mark.parametrize(
        "mock_download_panstarrs",
        [None],
        indirect=True,
    )
    def test_find_astrometry_offsets_with_comparison_skymodel_does_not_download(
        self, facet, mock_download_panstarrs
    ):
        """
        Test that find_astrometry_offsets does not attempt to download data
        from PanSTARRS if a comparison skymodel is provided.
        """
        mock_comparison_skymodel = "mock_LSMTool_skymodel"
        facet.find_astrometry_offsets(mock_comparison_skymodel, min_number=1)
        mock_download_panstarrs.assert_not_called()

    @pytest.mark.parametrize(
        "mock_download_panstarrs",
        (
            [],  # No sources found in PanSTARRS
            [0, 1, 2, 3, 4],  # Default minimum number of sources is 5
            [0, 1, 2, 3],  # Below the minimum number of sources
        ),
        indirect=True,
    )
    def test_find_astrometry_offsets_without_comparison_skymodel_downloads(
        self, facet, mock_download_panstarrs
    ):
        """
        Test that find_astrometry_offsets attempts to download data from
        PanSTARRS if no comparison skymodel is provided.
        """
        min_number = 5  # Default minimum number of sources is 5
        facet.find_astrometry_offsets(
            comparison_skymodel=None, min_number=min_number
        )
        mock_download_panstarrs.assert_called_once()
        if len(mock_download_panstarrs.return_value) < min_number:
            facet.skymodel.compare.assert_not_called()
        else:
            facet.skymodel.compare.assert_called_once()

    @pytest.mark.disable_socket
    def test_find_astrometry_offsets_with_comparison_skymodel_does_not_access_internet(
        self, facet
    ):
        """
        Test that find_astrometry_offsets does not access the internet if a
        comparison skymodel is provided.
        """
        mock_comparsion_skymodel = [
            1,
            1,
            1,
            1,
            1,
        ]  # Mock comparison skymodel with enough sources
        facet.find_astrometry_offsets(
            comparison_skymodel=mock_comparsion_skymodel, min_number=5
        )
        facet.skymodel.compare.assert_called_once_with(
            mock_comparsion_skymodel,
            radius="5 arcsec",
            excludeMultiple=True,
            make_plots=False,
        )

    @pytest.mark.filterwarnings(
        "ignore:The Pan-STARRS catalog could not be successfully downloaded"
        # since we are mocking the download_skymodel function, the returned
        # "skymodel" will not load correctly, which will trigger a warning.
        # Since we are only testing that the download_skymodel function was
        # called, we can safely ignore this
    )
    def test_download_panstarrs(self, facet, mocker):
        """
        Test that download_skymodel is called.
        """
        mock_download_skymodel = mocker.patch(
            "lsmtool.facet.download_skymodel",
            return_value="mock_skymodel",
        )
        _ = facet.download_panstarrs()
        assert mock_download_skymodel.called

    @pytest.mark.parametrize("use_wcs", [False, True])
    def test_get_matplotlib_patch(self, facet, use_wcs):
        """
        Test that get_matplotlib_patch returns a matplotlib patch object with
        extents matching the facet's polygon vertices.
        """
        patch = facet.get_matplotlib_patch(wcs=facet.wcs if use_wcs else None)
        extents = patch.get_extents()

        assert isinstance(patch, mpl.patches.Patch)
        assert np.allclose(extents.max, np.max(facet.polygon.exterior.xy, 1))
        assert np.allclose(extents.min, np.min(facet.polygon.exterior.xy, 1))

    def test_set_skymodel(self, mocker, facet):
        """
        Test that `set_skymodel` method runs the `filter_skymodel` function.
        """
        # Arrange
        mock_filter_skymodel = mocker.patch("lsmtool.facet.filter_skymodel")
        # Act
        facet.set_skymodel("mock_skymodel")
        # Assert
        mock_filter_skymodel.assert_called_once_with(
            facet.polygon, "mock_skymodel", facet.wcs
        )


class TestDS9RegionFile:
    """
    Tests for the `lsmtool.facet.read_ds9_region_file` and
    `lsmtool.facet.make_ds9_region_file` functions.
    """

    @pytest.fixture(params=["test.reg", "invalid.reg"])
    def ds9_region_file(self, request):
        """
        Fixture to create a DS9 region file for testing.
        """
        return TEST_DATA_PATH / request.param

    @pytest.fixture(params=["test.reg", "invalid.reg"])
    def expected_facet_attributes(self, ds9_region_file):
        if ds9_region_file.name == "test.reg":
            return [
                {
                    "name": "Patch_1",
                    "ra": 318.2026666666666,
                    "dec": 62.25055927777777,
                },
                {
                    "name": "Patch_10_with_spaces",
                    "ra": 312.59492416666666,
                    "dec": 60.46176975,
                },
                {
                    "name": "Patch_11",
                    "ra": 315.00718541666663,
                    "dec": 59.5211111388889,
                },
                {
                    "name": "Patch_12",
                    "ra": 310.31232124999997,
                    "dec": 59.54736030555557,
                },
            ]

        return ValueError

    def test_read_ds9_region_file(
        self, ds9_region_file, expected_facet_attributes
    ):
        """
        Test reading a DS9 region file.
        """
        with get_context(expected_facet_attributes):
            # Act
            facets = read_ds9_region_file(ds9_region_file)

            # Assert
            for i, (facet, expected) in enumerate(
                zip(facets, expected_facet_attributes, strict=False)
            ):
                for attr, value in expected.items():
                    assert getattr(facet, attr) == value, (
                        f"Facet {i} attribute {attr!r} does not match expected "
                        "value."
                    )

    @pytest.mark.parametrize(
        "ds9_region_file, expected_facet_attributes",
        [("test.reg", "test.reg")],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "enclose_names, context",
        [
            pytest.param(True, contextlib.nullcontext(), id="enclose_names"),
            pytest.param(
                False,
                pytest.raises(
                    ValueError, match='"text" property could not be parsed'
                ),
                id="no_enclose_names",
            ),
        ],
    )
    def test_write_ds9_region_file(
        self,
        tmp_path,
        ds9_region_file,
        expected_facet_attributes,
        enclose_names,
        context,
    ):
        """
        Test writing a DS9 region file.
        """
        # Arrange
        reg_out = tmp_path / "test_region_write.reg"
        facets = read_ds9_region_file(ds9_region_file)

        # Act
        make_ds9_region_file(facets, reg_out, enclose_names=enclose_names)

        # Assert
        with context:
            self.test_read_ds9_region_file(reg_out, expected_facet_attributes)


class TestReadSkymodel:
    """Tests for the `lsmtool.facet.read_skymodel` function."""

    @pytest.fixture(autouse=True)
    def mock_skymodel(self, mocker, request):

        patch_positions = (
            {"getPatchPositions.return_value": request.param}
            if (has_patches := request.param is not None)
            else {}
        )

        mock_skymodel = mocker.Mock(hasPatches=has_patches, **patch_positions)
        mocker.patch("lsmtool.facet.load", return_value=mock_skymodel)
        return mock_skymodel

    @pytest.mark.parametrize("mock_skymodel", [None], indirect=True)
    def test_read_skymodel_no_patches(self):
        """
        Test that read_skymodel raises ValueError if sky model has no patches.
        """
        with pytest.raises(ValueError, match="must be grouped into patches"):
            read_skymodel("fake.sky", 180.0, 45.0, 2.0, 2.0)

    @pytest.mark.parametrize(
        "mock_skymodel",
        [
            {
                "PatchA": (180.0 * u.degree, 45.0 * u.degree),
                "PatchB": (180.5 * u.degree, 45.5 * u.degree),
                "PatchC": (179.5 * u.degree, 44.5 * u.degree),
            }
        ],
        indirect=True,
    )
    def test_read_skymodel_returns_facets(self):
        """
        Test that read_skymodel returns correct facets from a patched sky model.
        """

        # Act
        facets = read_skymodel("fake.sky", 180.0, 45.0, 2.0, 2.0)

        # Assert
        assert all(isinstance(facet, Facet) for facet in facets)
        # Each facet name should be one of the patch names
        assert [(facet.name, facet.ra, facet.dec) for facet in facets] == [
            ("PatchA", 180.0, 45.0),
            ("PatchB", 180.5, 45.5),
            ("PatchC", 179.5, 44.5),
        ]


# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "coords, bounding_box, expected",
    [
        # All points inside boundary
        pytest.param(
            np.array([[1, 1], [2, 2], [3, 3]]),  # coords
            np.array([0, 4, 0, 4]),  # bounding box
            np.array([True, True, True]),  # expected
            id="all_inside",
        ),
        # Some points inside, some outside boundary
        pytest.param(
            np.array([[0, 0], [5, 5], [2, 2]]),  # coords
            np.array([1, 4, 1, 4]),  # bounding box
            np.array([False, False, True]),  # expected
            id="some_inside_some_outside",
        ),
        # Points exactly on the boundary
        pytest.param(
            np.array([[1, 1], [4, 4], [1, 4], [4, 1]]),  # coords
            np.array([1, 4, 1, 4]),  # bounding box
            np.array([True, True, True, True]),  # expected
            id="on_boundary",
        ),
        # Empty `coords`
        pytest.param(
            np.empty((0, 2)),  # coords
            np.array([0, 1, 0, 1]),  # bounding box
            np.array([], dtype=bool),  # expected
            id="empty_coords",
        ),
        # Bounding box with zero area (min == max)
        pytest.param(
            np.array([[1, 1], [1, 2], [2, 1]]),  # coords
            np.array([1, 1, 1, 1]),  # bounding box
            np.array([True, False, False]),  # expected
            id="zero_area_box",
        ),
        # Negative coordinates
        pytest.param(
            np.array([[-2, -2], [-1, -1], [0, 0]]),  # coords
            np.array([-2, 0, -2, 0]),  # bounding box
            np.array([True, True, True]),  # expected
            id="negative_coords",
        ),
        # Bounding box with min > max (inverted box)
        pytest.param(
            np.array([[0, 0], [1, 1]]),  # coords
            np.array([2, 0, 2, 0]),  # bounding box
            np.array([True, True]),  # expected
            id="inverted_box",
        ),
        # -------------------------------------------------------------------- #
        # Error: `coords` not 2D
        pytest.param(
            np.array([1, 2, 3]),  # coords
            np.array([0, 1, 0, 1]),  # bounding box
            ValueError,  # expected
            id="coords_not_2d",
        ),
        # Error: `coords` with wrong number of columns
        pytest.param(
            np.array([[1, 1, 1], [2, 2, 2]]),  # coords
            np.array([0, 2, 0, 2]),  # bounding box
            ValueError,  # expected
            id="coords_wrong_columns",
        ),
        # Error: `bounding_box` wrong shape (too small)
        pytest.param(
            np.array([[1, 1]]),  # coords
            np.array([0, 1, 0]),  # bounding box
            ValueError,  # expected
            id="bounding_box_too_short",
        ),
        # Error: `bounding_box` wrong shape (too big)
        pytest.param(
            np.array([[1, 1]]),  # coords
            np.array([0, 1, 0, 1, 2]),  # bounding box
            ValueError,  # expected
            id="bounding_box_too_long",
        ),
    ],
)
def test_in_box(coords, bounding_box, expected):
    # Act

    with get_context(expected):
        result = in_box(coords, bounding_box)

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
            # bbox_midpoint
            SkyCoord(ra=126.52, dec=90.0, unit="deg"),
            # bbox_size
            (0.3, 0.3),
            # expected_facet_points
            np.transpose([directions.ra.deg, directions.dec.deg]),
            # expected_facet_polygons
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
        ),
        pytest.param(
            # directions
            SkyCoord(
                ra=[119.73, 138.08, 124.13, 115.74],
                dec=[89.92, 89.91, 89.89, 89.89],
                unit="deg",
            ),
            # bbox_midpoint
            SkyCoord(ra=126.52, dec=90.0, unit="deg"),
            # bbox_size
            (-1, 0.3),
            # expected_facet_points
            ValueError,
            # expected_facet_polygons
            ...,
        ),
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
    with get_context(expected_facet_points):
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
    "coords, bounding_box,expected_in_box, expected_vertices, expected_regions",
    [
        # Regular input cases
        # -------------------------------------------------------------------- #
        # 4 points in a square
        pytest.param(
            # coords
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            # bounding_box
            [0, 1, 0, 1],
            # expected_in_box
            np.array([True, True, True, True]),
            # expected_vertices
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
            # expected_regions
            [],
            id="square",
        ),
        # Edge cases
        # -------------------------------------------------------------------- #
        # Only one point inside bounding box
        pytest.param(
            # coords
            np.array([[0.5, 0.5], [2, 2]]),
            # bounding_box
            [0, 1, 0, 1],
            # expected_in_box
            np.array([True, False]),
            # expected_vertices
            np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            # expected_regions
            [[3, 1, 0, 2]],
            id="edge_case_one_inside",
        ),
        # All points on the boundary
        pytest.param(
            # coords
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            # bounding_box
            [0, 1, 0, 1],
            # expected_in_box
            np.array([True, True, True, True]),
            # expected_vertices
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
            # expected_regions
            [],
            id="all_on_boundary",
        ),
        # Degenerate: all points colinear
        pytest.param(
            # coords
            np.array([[0, 0], [0.5, 0.5], [1, 1]]),
            # bounding_box
            [0, 1, 0, 1],
            # expected_in_box
            np.array([True, True, True]),
            # expected_vertices
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
            # expected_regions
            [[7, 5, 4, 3, 2, 6]],
            id="colinear_points",
        ),
        # Duplicate points
        pytest.param(
            # coords
            np.array([[0, 0], [0, 0], [1, 1], [1, 1]]),
            # bounding_box
            [0, 1, 0, 1],
            # expected_in_box
            np.array([True, True, True, True]),
            # expected_vertices
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            # expected_regions
            [],
            id="duplicate_points",
        ),
        # Error cases
        # -------------------------------------------------------------------- #
        # All points outside bounding box
        pytest.param(
            # coords
            np.array([[2, 2], [3, 3]]),
            # bounding_box
            [0, 1, 0, 1],
            # expected_in_box
            ...,
            # expected_vertices
            ...,
            # expected_regions
            ValueError,
            id="edge_case_all_outside",
        ),
    ],
)
def test_voronoi(
    coords,
    bounding_box,
    expected_in_box,
    expected_vertices,
    expected_regions,
):
    # Arrange
    with get_context(expected_regions):
        # Act
        points, vertices, regions = voronoi(coords, bounding_box)

        # Assert
        # filtered_points should match the points inside the bounding box
        expected_points = coords[expected_in_box]
        assert_array_equal(points, expected_points)

        # check regions are as expected
        assert regions == expected_regions

        # check vertices are as expected
        assert_array_equal(vertices, expected_vertices)


@pytest.mark.parametrize(
    "coords, bounding_box, expected_centre",
    [
        # Nominal cases
        # -------------------------------------------------------------------- #
        # All points inside the bounding box
        pytest.param(
            np.array([[0, 0], [1, 1], [0.5, 0.5]]),
            [0, 1, 0, 1],
            np.array([[0, 0], [1, 1], [0.5, 0.5]]),
            id="all_inside",
        ),
        # Some points inside, some outside
        pytest.param(
            np.array([[0, 0], [2, 2], [1, 1]]),
            [0, 1, 0, 1],
            np.array([[0, 0], [1, 1]]),
            id="some_inside_some_outside",
        ),
        # All points outside
        pytest.param(
            np.array([[2, 2], [3, 3]]),
            [0, 1, 0, 1],
            np.empty((0, 2)),
            id="all_outside",
        ),
        # Points on the boundary
        pytest.param(
            np.array([[0, 0], [1, 1], [0, 1], [1, 0]]),
            [0, 1, 0, 1],
            np.array([[0, 0], [1, 1], [0, 1], [1, 0]]),
            id="on_boundary",
        ),
        # Empty input
        pytest.param(
            np.empty((0, 2)),
            [0, 1, 0, 1],
            np.empty((0, 2)),
            id="empty_input",
        ),
        # Negative coordinates
        pytest.param(
            np.array([[-1, -1], [0, 0], [1, 1]]),
            [-1, 1, -1, 1],
            np.array([[-1, -1], [0, 0], [1, 1]]),
            id="negative_coords",
        ),
        # Inverted bounding box (min > max)
        pytest.param(
            np.array([[0, 0], [1, 1]]),
            [1, 0, 1, 0],
            np.array([[0, 0], [1, 1]]),
            id="inverted_box",
        ),
        # Duplicate points
        pytest.param(
            np.array([[0, 0], [0, 0], [1, 1], [1, 1]]),
            [0, 1, 0, 1],
            np.array([[0, 0], [0, 0], [1, 1], [1, 1]]),
            id="duplicate_points",
        ),
        # Single point inside
        pytest.param(
            np.array([[0.5, 0.5]]),
            [0, 1, 0, 1],
            np.array([[0.5, 0.5]]),
            id="single_point_inside",
        ),
        # Single point outside
        pytest.param(
            np.array([[2, 2]]),
            [0, 1, 0, 1],
            np.empty((0, 2)),
            id="single_point_outside",
        ),
        # Error cases
        # -------------------------------------------------------------------- #
        # coords not 2D
        pytest.param(
            np.array([1, 2, 3]),
            [0, 1, 0, 1],
            ValueError,
            id="coords_not_2d",
        ),
        # coords wrong number of columns
        pytest.param(
            np.array([[1, 2, 3], [4, 5, 6]]),
            [0, 1, 0, 1],
            ValueError,
            id="coords_wrong_columns",
        ),
        # bounding_box wrong shape (too short)
        pytest.param(
            np.array([[1, 1]]),
            [0, 1, 0],
            ValueError,
            id="bounding_box_too_short",
        ),
        # bounding_box wrong shape (too long)
        pytest.param(
            np.array([[1, 1]]),
            [0, 1, 0, 1, 2],
            ValueError,
            id="bounding_box_too_long",
        ),
    ],
)
def test_prepare_points_for_tessellate(coords, bounding_box, expected_centre):
    # Act
    with get_context(expected_centre):
        points_centre, points = prepare_points_for_tessellate(
            coords, bounding_box
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


@pytest.mark.parametrize(
    "facet, extent",
    [
        (
            SquareFacet(
                name="test_filter_skymodel",
                ra=255,
                dec=55,
                width=5,
            ),
            [250, 260, 50, 60],
        ),
        (
            Facet(
                name="test_filter_skymodel",
                ra=238.795,
                dec=50.98242,
                vertices=[(250, 60), (260, 60), (260, 50), (250, 50)],
            ),
            [250, 260, 50, 60],
        ),
    ],
)
def test_filter_skymodel(request, facet, extent):
    """
    Test that `facet.filter_skymodel` selects only sources that lie inside the
    input facet.
    """

    # Arrange
    skymodel = load(request.config.resource_dir / "no_patches.sky")

    # Act
    result = filter_skymodel(facet.polygon, skymodel, facet.wcs)

    # Assert
    assert "FILTER (with array of indices/bools)" in result.history[-1]

    ra = result.table["Ra"]
    dec = result.table["Dec"]

    ra0, ra1, dec0, dec1 = extent
    assert np.all((ra0 < ra) & (ra < ra1))
    assert np.all((dec0 < dec) & (dec < dec1))
