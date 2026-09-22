import pytest

@pytest.mark.parametrize('grouped', [False, True])
def test_row_index_sources(grouped, sky_no_patches, monkeypatch):
    """Lookups use exact names and never copy columns via the public helpers."""
    import numpy as np

    sky = sky_no_patches
    # Include duplicate names and literal wildcard characters.
    sky.table['Name'][0] = 'literal*'
    sky.table['Name'][1] = 'literalX'
    sky.table['Name'][2] = 'literal*'
    if grouped:
        sky.group('single', root='all_sources')

    def unexpected_copy(*args, **kwargs):
        pytest.fail('Row lookup must not use copying name/column helpers')

    monkeypatch.setattr(sky, 'getColValues', unexpected_copy)
    monkeypatch.setattr(sky, '_getNameIndx', unexpected_copy)
    indices = sky.getRowIndex('literal*')
    np.testing.assert_array_equal(indices, [0, 2])
    with pytest.raises(ValueError, match='not recognized'):
        sky.getRowIndex('missing')


def test_row_index_patch_views(sky_patches, monkeypatch):
    """Patch selectors cover exactly the group and preserve shared storage."""
    import numpy as np

    sky = sky_patches
    names = sky.getPatchNames()

    def unexpected_copy(*args, **kwargs):
        pytest.fail('Patch lookup must not copy full columns')

    monkeypatch.setattr(sky, 'getColValues', unexpected_copy)
    monkeypatch.setattr(sky, 'getPatchNames', unexpected_copy)
    for name in names:
        selector = sky.getRowIndex(name)
        assert isinstance(selector, slice)
        expected = np.flatnonzero(sky.table['Patch'] == name)
        np.testing.assert_array_equal(np.arange(len(sky))[selector], expected)
        assert np.shares_memory(sky.table['Ra'][selector], sky.table['Ra'])

    # A patch name wins even when a source outside that patch has the same name.
    sky.table['Name'][-1] = names[0]
    assert sky.getRowIndex(names[0]) == slice(0, sky.table.groups.indices[1])
    x, y, _, _ = sky._getXY(patchName=names[0])
    assert len(x) == len(y) == sky.table.groups.indices[1]


def test_tessellate_by_patch(sky_grouped):
    """Regroup existing patches using slice selectors."""
    original_length = len(sky_grouped)
    sky_grouped.group('tessellate', targetFlux='100.0 Jy', byPatch=True)
    assert len(sky_grouped) == original_length
    assert sky_grouped.hasPatches


@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('units', [None, 'mJy'])
def test_column_values_independent(sky_no_patches, masked, units):
    """Fill and convert columns without mutating or aliasing the model."""
    import numpy as np
    from astropy.table import Column, MaskedColumn

    sky = sky_no_patches
    values = np.arange(len(sky), dtype=float)
    column = Column(values, name='I', unit='Jy')
    if masked:
        column = MaskedColumn(column, mask=False, fill_value=-7.0)
        column.mask[1] = True
    sky.table.replace_column('I', column)
    expected = values.copy()
    if masked:
        expected[1] = -7.0
    if units:
        expected *= 1000
    result = sky.getColValues('I', units=units)
    np.testing.assert_allclose(result, expected)
    assert not np.shares_memory(result, sky.table['I'])
    result[:] = -99
    np.testing.assert_array_equal(sky.table['I'].data, column.data)
    assert sky.table['I'].unit == 'Jy'
    if masked:
        assert sky.table['I'].mask[1]


def test_column_values_beam_independent(sky_no_patches, monkeypatch):
    """In-place beam attenuation must operate on an owned column."""
    import numpy as np

    sky = sky_no_patches
    original = sky.table['I'].copy()

    def attenuate(column):
        column[:] *= 0.5
        return column

    monkeypatch.setattr(sky, '_applyBeamToCol', attenuate)
    result = sky.getColValues('I', applyBeam=True)
    np.testing.assert_allclose(result, original.data * 0.5)
    np.testing.assert_array_equal(sky.table['I'], original)


@pytest.mark.parametrize('aggregate', ['sum', 'mean', 'wmean', 'min', 'max'])
def test_column_values_aggregate_independent(sky_patches, aggregate):
    """Converting and editing aggregated values preserves the model."""
    import numpy as np

    sky = sky_patches
    original = sky.table['I'].copy()
    expected = sky.getColValues('I', aggregate=aggregate)
    result = sky.getColValues('I', aggregate=aggregate, units='mJy')
    np.testing.assert_allclose(result, expected * 1000)
    result[:] = -99
    np.testing.assert_array_equal(sky.table['I'], original)
    assert sky.table['I'].unit == 'Jy'
