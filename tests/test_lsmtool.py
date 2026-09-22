"""Runs an example of each operation"""

import lsmtool
import filecmp
import pytest


@pytest.fixture()
def sky_no_patches():
    return lsmtool.load('tests/resources/no_patches.sky')


@pytest.fixture()
def sky_patches():
    return lsmtool.load('tests/resources/patches.sky')


def test_select(sky_no_patches):
    """Select individual sources with Stokes I fluxes above 1 Jy."""
    assert len(sky_no_patches) == 1210
    sky_no_patches.select('I > 1.0 Jy')
    assert len(sky_no_patches) == 965


def test_transfer(sky_no_patches, sky_patches):
    """Transfer patches from patches.sky."""
    assert not sky_no_patches.hasPatches
    sky_no_patches.transfer(sky_patches)
    assert sky_no_patches.hasPatches
    expected = dict(zip(sky_patches.table['Name'], sky_patches.table['Patch']))
    for name, patch in zip(sky_no_patches.table['Name'], sky_no_patches.table['Patch']):
        if name in expected:
            assert patch == expected[name]


def test_remove(sky_no_patches):
    """Remove sources with total fluxes below 2 Jy."""
    assert len(sky_no_patches) == 1210
    sky_no_patches.remove('I < 2.0 Jy', aggregate='sum')
    assert len(sky_no_patches) == 389


def test_ungroup(sky_patches):
    """Ungroup a skymodel with patches."""
    assert sky_patches.hasPatches
    sky_patches.ungroup()
    assert not sky_patches.hasPatches


def test_concatenate(sky_no_patches):
    """Concatenate with concat.sky."""
    assert len(sky_no_patches) == 1210
    sky_no_patches.concatenate('tests/resources/concat.sky', matchBy='position',
                               radius='30 arcsec', keep='from2')
    assert len(sky_no_patches) == 2898


def test_concatenate_differing_spectral_index(sky_no_patches):
    """Concatenate with single_spectralindx.sky."""
    original_length = len(sky_no_patches)
    sky_no_patches.concatenate('tests/resources/single_spectralindx.sky',
                               matchBy='position', radius='30 arcsec', keep='from2')
    assert len(sky_no_patches) == original_length


def test_compare(sky_no_patches, tmp_path):
    """Compare to concat.sky."""
    flux_ratio_path = tmp_path / "flux_ratio_vs_distance.pdf"
    sky_concat = lsmtool.load('tests/resources/concat.sky')
    sky_concat.ungroup()
    sky_concat.select('I > 5.0 Jy')
    sky_no_patches.compare(sky_concat, outDir=str(tmp_path))
    assert flux_ratio_path.is_file()


def test_add(sky_no_patches):
    """Add a source."""
    original_length = len(sky_no_patches)
    sky_no_patches.add({'Name': 'src1', 'Type': 'POINT', 'Ra': 277.4232, 'Dec': 48.3689,
                        'I': 0.69})
    assert len(sky_no_patches) == original_length + 1


@pytest.fixture
def sky_grouped(sky_no_patches):
    """Group sky_no_patches using tessellation to a target flux of 50 Jy."""
    sky_no_patches.group('tessellate', targetFlux='50.0 Jy')
    return sky_no_patches


def test_group(sky_grouped):
    """Basic check for the grouped skymodel."""
    assert len(sky_grouped.getPatchNames()) == 79


def test_move(sky_grouped):
    """Move patch Patch_1 to 16:04:16.2288, 58.03.06.912."""
    sky_grouped.move('Patch_1', position=['16:04:16.2288', '58.03.06.912'])
    assert round(sky_grouped.getPatchPositions()['Patch_1'][0].value, 4) == 241.0676


def test_merge(sky_grouped):
    """Merge patches Patch_0 and Patch_2."""
    patch_count = len(sky_grouped.getPatchNames())
    sky_grouped.merge(['Patch_0', 'Patch_2'], name='merged_patch')
    assert len(sky_grouped.getPatchNames()) == patch_count - 1


def test_setPatchPositions(sky_grouped):
    """Set patch positions to midpoint of patch."""
    sky_grouped.merge(['Patch_0', 'Patch_2'], name='merged_patch')
    sky_grouped.setPatchPositions(method='mid')
    assert round(sky_grouped.getPatchPositions()['merged_patch'][0].value, 4) == 274.1166


def test_facet_write(sky_no_patches, tmp_path):
    """Write ds9 facet file."""
    # Note: differences in the libraries used can cause slight differences in the
    # resulting facet file, so it is not possible to compare with a reference
    # file. Instead, we just check that the file exists
    facet_path = tmp_path / "facet.reg"
    sky_no_patches.group("single")
    sky_no_patches.write(str(facet_path), format='facet', clobber=True)
    assert facet_path.is_file()


@pytest.fixture
def final_model(sky_no_patches, sky_patches):
    """Create a skymodel resembling the steps in 'validation.parset'."""
    sky_no_patches.select('I > 1.0 Jy')
    sky_no_patches.transfer(sky_patches)
    sky_no_patches.remove('I < 2.0 Jy', aggregate='sum')
    sky_no_patches.concatenate('tests/resources/concat.sky', matchBy='position',
                                       radius='30 arcsec', keep='from2')
    sky_no_patches.add({'Name': 'src1', 'Type': 'POINT', 'Ra': 277.4232,
                        'Dec': 48.3689, 'I': 0.69})
    sky_no_patches.group('tessellate', targetFlux='50.0 Jy')
    sky_no_patches.move('Patch_1', position=['16:04:16.2288', '58.03.06.912'])
    sky_no_patches.merge(['Patch_0', 'Patch_2'], name='merged_patch')
    sky_no_patches.setPatchPositions(method='mid')
    return sky_no_patches


def test_write(final_model, tmp_path):
    """Write final model to file."""
    final_path = tmp_path / "final.sky"
    final_model.write(str(final_path), clobber=True, addHistory=False)
    assert filecmp.cmp(final_path, 'tests/resources/final.sky')


def test_plot(final_model, tmp_path):
    """Plot the model."""
    plot_path = tmp_path / "plot.pdf"
    final_model.plot(str(plot_path))
    assert plot_path.is_file()


def test_meanshift():
    """Group the model with the meanshift algorithm."""
    sky_apparent = lsmtool.load('tests/resources/apparent.sky')
    sky_apparent.group('meanshift', byPatch=True, lookDistance=0.075, groupingDistance=0.01)
    assert len(sky_apparent.getPatchPositions()) == 67


def test_meanshift_with_nans():
    """Load a model that contains NaNs and group it with the meanshift algorithm."""
    sky_nans  = lsmtool.load('tests/resources/nans.sky')
    sky_nans.group('meanshift', byPatch=True, lookDistance=0.075, groupingDistance=0.01)
    assert len(sky_nans.getPatchPositions()) == 7


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
