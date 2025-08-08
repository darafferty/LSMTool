"""Runs an example of each operation"""

import lsmtool
import filecmp
import pytest


@pytest.fixture()
def sky_no_patches():
    return lsmtool.load('tests/resources/no_patches.sky')


def test_select(sky_no_patches):
    """Select individual sources with Stokes I fluxes above 1 Jy."""
    assert len(sky_no_patches) == 1210
    sky_no_patches.select('I > 1.0 Jy')
    assert len(sky_no_patches) == 965


def test_transfer(sky_no_patches):
    """Transfer patches from patches.sky."""
    assert not sky_no_patches.hasPatches
    sky_no_patches.transfer('tests/resources/patches.sky')
    assert sky_no_patches.hasPatches


def test_remove(sky_no_patches):
    """Remove patches with total fluxes below 2 Jy."""
    assert len(sky_no_patches) == 1210
    sky_no_patches.remove('I < 2.0 Jy', aggregate='sum')
    assert len(sky_no_patches) == 389


def test_ungroup(sky_no_patches):
    """Ungroup the skymodel."""
    sky_no_patches.group("every")
    assert sky_no_patches.hasPatches

    sky_no_patches.ungroup()
    assert not sky_no_patches.hasPatches


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


def test_group(sky_no_patches):
    """Group using tessellation to a target flux of 50 Jy."""
    sky_no_patches.group('tessellate', targetFlux='50.0 Jy')
    assert len(sky_no_patches.getPatchNames()) == 79


def test_move(sky_no_patches):
    """Move patch Patch_1 to 16:04:16.2288, 58.03.06.912."""
    test_group(sky_no_patches)
    sky_no_patches.move('Patch_1', position=['16:04:16.2288', '58.03.06.912'])
    assert round(sky_no_patches.getPatchPositions()['Patch_1'][0].value, 4) == 241.0676


def test_merge(sky_no_patches):
    """Merge patches Patch_0 and Patch_2."""
    test_group(sky_no_patches)
    patch_count = len(sky_no_patches.getPatchNames())
    sky_no_patches.merge(['Patch_0', 'Patch_2'], name='merged_patch')
    assert len(sky_no_patches.getPatchNames()) == patch_count - 1


def test_setPatchPositions(sky_no_patches):
    """Set patch positions to midpoint of patch."""
    test_merge(sky_no_patches)
    sky_no_patches.setPatchPositions(method='mid')
    assert round(sky_no_patches.getPatchPositions()['merged_patch'][0].value, 4) == 274.1166


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
def final_model(sky_no_patches):
    """Create a skymodel resembling the steps in 'validation.parset'."""
    sky_no_patches.select('I > 1.0 Jy')
    sky_no_patches.transfer('tests/resources/patches.sky')
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
