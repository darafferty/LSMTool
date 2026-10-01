import numpy as np
import pytest
from astropy.table import Column, MaskedColumn

from lsmtool.skymodel import SkyModel


def test_getxy_empty():
    """Test _get_xy on an empty SkyModel."""
    # The SkyModel constructor does not support creating an empty SkyModel.
    # -> Create a model with a single source, and remove that source.
    sky = SkyModel(
        {"Name": "source1", "Type": "point", "Ra": 10.0, "Dec": 20.0, "I": 1.0}
    )
    sky.remove("I>0")
    assert len(sky) == 0
    assert sky._get_xy() == ([0], [0], 0, 0)


@pytest.mark.parametrize(
    "ra, dec", [(10.0, 20.0), (730.0, -340.0), (-710, 380.0)]
)
def test_getxy_single_source(ra, dec):
    """Test _get_xy on a SkyModel with a single source."""
    sky = SkyModel(
        {"Name": "source1", "Type": "point", "Ra": ra, "Dec": dec, "I": 1.0}
    )
    x, y, ra, dec = sky._get_xy()
    # make_wcs sets the reference pixel coordinates (crpix) to 1000, 1000.
    # Since numpy uses 0-based indexing, the x and y coordinates are 999.
    np.testing.assert_allclose(x, [999])
    np.testing.assert_allclose(y, [999])
    # The returned ra, dec coordinates should be normalized to 10, 20.
    assert ra == 10.0
    assert dec == 20.0


def test_getxy_identical_sources():
    """Test _get_xy on a SkyModel with sources with identical coordinates."""
    sky = SkyModel(
        {"Name": "source0", "Type": "point", "Ra": 10.0, "Dec": 20.0, "I": 1.0}
    )
    for i in range(1, 4):
        sky.add(
            {
                "Name": f"source{i}",
                "Type": "point",
                "Ra": 10.0,
                "Dec": 20.0,
                "I": 1.0,
            }
        )
    x, y, ra, dec = sky._get_xy()
    np.testing.assert_allclose(x, [999, 999, 999, 999])
    np.testing.assert_allclose(y, [999, 999, 999, 999])
    assert ra == 10.0
    assert dec == 20.0


def test_getxy_multiple_sources():
    """Test _get_xy on a SkyModel with multiple sources."""
    # Add sources out-of-order, since _get_xy should correctly sort them.
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
    x, y, ra, dec = sky._get_xy()
    # The midpoint RA and Dec values are 12.5 and 22.5, respectively.
    # Since the x value decreases as RA increases, the function returns the
    # first RA value smaller than the midpoint.
    assert ra == 12.0
    # For Dec, the y value increases as Dec increases, so the function returns
    # the first Dec value larger than the midpoint.
    assert dec == 23.0

    # Regression test for the x and y values.
    expected_x = [1167.2, 670.0, 999.0, 1337.9, 509.0, 833.3]
    expected_y = [639.4, 1181.4, 819.0, 460.5, 1364.6, 999.6]
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
    x, y, _, _ = sky._get_xy(patchName=names[0])
    assert len(x) == len(y) == sky.table.groups.indices[1]


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
