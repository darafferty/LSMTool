import astropy
import numpy as np
import pytest
from astropy.table import Column, MaskedColumn

from lsmtool.skymodel import SkyModel


def test_getxy_empty():
    """Test _get_xy on an empty SkyModel / empty RA/Dec lists."""
    # The SkyModel constructor does not support creating an empty SkyModel.
    # -> Create a model with a single source, and remove that source.
    sky = SkyModel(
        {"Name": "source1", "Type": "point", "Ra": 10.0, "Dec": 20.0, "I": 1.0}
    )
    assert len(sky) == 1
    assert sky._get_xy([], []) == ([0], [0], 0, 0)

    sky.remove("I>0")
    assert len(sky) == 0
    assert sky._get_xy() == ([0], [0], 0, 0)


@pytest.mark.parametrize(
    "ra, dec", [(10.0, 20.0), (350.0, -42.0), (-42.0, 90.0), (4242.0, -90.0)]
)
@pytest.mark.parametrize("ra_dec_args", [False, True])
def test_getxy_single_source(ra, dec, ra_dec_args):
    """Test _get_xy with a single source."""
    if ra_dec_args:
        sky = SkyModel(
            {"Name": "source1", "Type": "point", "Ra": 42, "Dec": 42, "I": 1.0}
        )
        x, y, mid_ra, mid_dec = sky._get_xy([ra], [dec])
    else:
        sky = SkyModel(
            {"Name": "source1", "Type": "point", "Ra": ra, "Dec": dec, "I": 1.0}
        )
        x, y, mid_ra, mid_dec = sky._get_xy()

    # make_wcs sets the reference pixel coordinates (crpix) to 1000, 1000.
    # Since numpy uses 0-based indexing, the x and y coordinates are 999.
    np.testing.assert_allclose(x, [999])
    np.testing.assert_allclose(y, [999])
    # The returned ra, dec coordinates should equal the input values.
    assert mid_ra == ra % 360.0  # The returned RA value should be normalized.
    assert mid_dec == dec


@pytest.mark.parametrize("ra, dec", [(0.0, -91.0), (0.0, 91.0)])
def test_getxy_unnormalised_dec(ra, dec):
    """Test _get_xy with non-normalised RA and Dec values."""
    sky = SkyModel(
        {"Name": "source1", "Type": "point", "Ra": 42, "Dec": 42, "I": 1.0}
    )
    with pytest.raises(astropy.wcs._wcs.InvalidTransformError):
        sky._get_xy([ra], [dec])


def test_getxy_identical_sources():
    """Test _get_xy on identical sources."""
    sky = SkyModel(
        {"Name": "source0", "Type": "point", "Ra": 42.0, "Dec": 42.0, "I": 1.0}
    )
    x, y, mid_ra, mid_dec = sky._get_xy([10, 10, 10, 10], [20, 20, 20, 20])
    np.testing.assert_allclose(x, [999, 999, 999, 999])
    np.testing.assert_allclose(y, [999, 999, 999, 999])
    assert mid_ra == 10.0
    assert mid_dec == 20.0


@pytest.mark.parametrize(
    "crdelt, expected_x, expected_y",
    [
        (
            None,
            [1167.2, 670.0, 999.0, 1337.9, 509.0, 833.3],
            [639.4, 1181.4, 819.0, 460.5, 1364.6, 999.6],
        ),
        (
            0.042,
            [1021.2, 955.5, 999.0, 1043.8, 934.2, 977.1],
            [951.4, 1023.1, 975.2, 927.8, 1047.4, 999.1],
        ),
    ],
)
@pytest.mark.parametrize("ra_dec_args", [False, True])
def test_getxy_multiple_sources(crdelt, expected_x, expected_y, ra_dec_args):
    """Test _get_xy on with multiple sources and varying crdelt."""

    if ra_dec_args:
        sky = SkyModel(
            {"Name": "source1", "Type": "point", "Ra": 42, "Dec": 42, "I": 1}
        )
        # Use out-of-order values, since _get_xy should correctly sort them.
        # Using numpy arrays provides test coverage for that argument type.
        ra_list = np.array([11, 14, 12, 10, 15, 13])
        dec_list = np.array([21, 24, 22, 20, 25, 23])
        x, y, mid_ra, mid_dec = sky._get_xy(ra_list, dec_list, crdelt=crdelt)
    else:
        # Create a SkyModel with the same RA and Dec values as above.
        sky = SkyModel(
            {"Name": "source1", "Type": "point", "Ra": 11, "Dec": 21, "I": 1}
        )
        for i in [4, 2, 0, 5, 3]:
            sky.add(
                {
                    "Name": f"source{i}",
                    "Type": "point",
                    "Ra": 10 + i,
                    "Dec": 20 + i,
                    "I": 1,
                }
            )
        x, y, mid_ra, mid_dec = sky._get_xy(crdelt=crdelt)

    # The midpoint RA and Dec values are 12.5 and 22.5, respectively.
    # Since the x value decreases as RA increases, the function returns the
    # first RA value smaller than the midpoint.
    assert mid_ra == 12.0
    # For Dec, the y value increases as Dec increases, so the function returns
    # the first Dec value larger than the midpoint.
    assert mid_dec == 23.0

    # Regression test for the x and y values.
    np.testing.assert_allclose(x, expected_x, atol=0.1)
    np.testing.assert_allclose(y, expected_y, atol=0.1)


@pytest.mark.parametrize("grouped", [False, True])
def test_row_index_sources(grouped, sky_no_patches, monkeypatch):
    """
    Lookups use exact names and never copy columns via the public helpers.
    """
    sky = sky_no_patches
    # Include duplicate names and literal wildcard characters.
    sky.table["Name"][0] = "literal*"
    sky.table["Name"][1] = "literalX"
    sky.table["Name"][2] = "literal*"
    if grouped:
        sky.group("single", root="all_sources")

    def unexpected_copy(*_, **__):
        pytest.fail("Row lookup must not use copying name/column helpers")

    monkeypatch.setattr(sky, "getColValues", unexpected_copy)
    monkeypatch.setattr(sky, "_getNameIndx", unexpected_copy)
    indices = sky.getRowIndex("literal*")
    np.testing.assert_array_equal(indices, [0, 2])
    with pytest.raises(ValueError, match="not recognized"):
        sky.getRowIndex("missing")


def test_row_index_patch_views(sky_patches, monkeypatch):
    """
    Patch selectors cover exactly the group and preserve shared storage.
    """
    sky = sky_patches
    names = sky.getPatchNames()

    def unexpected_copy(*_, **__):
        pytest.fail("Patch lookup must not copy full columns")

    monkeypatch.setattr(sky, "getColValues", unexpected_copy)
    monkeypatch.setattr(sky, "getPatchNames", unexpected_copy)
    for name in names:
        selector = sky.getRowIndex(name)
        assert isinstance(selector, slice)
        expected = np.flatnonzero(sky.table["Patch"] == name)
        np.testing.assert_array_equal(np.arange(len(sky))[selector], expected)
        assert np.shares_memory(sky.table["Ra"][selector], sky.table["Ra"])

    # A patch name wins even when a source outside that patch has the same name.
    sky.table["Name"][-1] = names[0]
    assert sky.getRowIndex(names[0]) == slice(0, sky.table.groups.indices[1])


def test_tessellate_by_patch(sky_grouped):
    """
    Regroup existing patches using slice selectors.
    """
    original_length = len(sky_grouped)
    sky_grouped.group("tessellate", targetFlux="100.0 Jy", byPatch=True)
    assert len(sky_grouped) == original_length
    assert sky_grouped.hasPatches


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("units", [None, "mJy"])
def test_column_values_independent(sky_no_patches, masked, units):
    """
    Fill and convert columns without mutating or aliasing the model.
    """
    sky = sky_no_patches
    values = np.arange(len(sky), dtype=float)
    column = Column(values, name="I", unit="Jy")
    if masked:
        column = MaskedColumn(column, mask=False, fill_value=-7.0)
        column.mask[1] = True
    sky.table.replace_column("I", column)
    expected = values.copy()
    if masked:
        expected[1] = -7.0
    if units:
        expected *= 1000
    result = sky.getColValues("I", units=units)
    np.testing.assert_allclose(result, expected)
    assert not np.shares_memory(result, sky.table["I"])
    result[:] = -99
    np.testing.assert_array_equal(sky.table["I"].data, column.data)
    assert sky.table["I"].unit == "Jy"
    if masked:
        assert sky.table["I"].mask[1]


def test_column_values_beam_independent(sky_no_patches, monkeypatch):
    """
    In-place beam attenuation must operate on an owned column.
    """
    sky = sky_no_patches
    original = sky.table["I"].copy()

    def attenuate(column):
        column[:] *= 0.5
        return column

    monkeypatch.setattr(sky, "_applyBeamToCol", attenuate)
    result = sky.getColValues("I", applyBeam=True)
    np.testing.assert_allclose(result, original.data * 0.5)
    np.testing.assert_array_equal(sky.table["I"], original)


@pytest.mark.parametrize("aggregate", ["sum", "mean", "wmean", "min", "max"])
def test_column_values_aggregate_independent(sky_patches, aggregate):
    """
    Converting and editing aggregated values preserves the model.
    """
    sky = sky_patches
    original = sky.table["I"].copy()
    expected = sky.getColValues("I", aggregate=aggregate)
    result = sky.getColValues("I", aggregate=aggregate, units="mJy")
    np.testing.assert_allclose(result, expected * 1000)
    result[:] = -99
    np.testing.assert_array_equal(sky.table["I"], original)
    assert sky.table["I"].unit == "Jy"
