#! /usr/bin/env python
# Runs an example of each operation
import lsmtool
import os
import filecmp


sky_no_patches = lsmtool.load('tests/resources/no_patches.sky')
sky_apparent = lsmtool.load('tests/resources/apparent.sky')


def test_select():
    print('Select individual sources with Stokes I fluxes above 1 Jy')
    sky_no_patches.select('I > 1.0 Jy')
    assert len(sky_no_patches) == 965


def test_transfer():
    print('Transfer patches from patches.sky')
    sky_no_patches.transfer('tests/resources/patches.sky')
    assert sky_no_patches.hasPatches


def test_remove():
    print('Remove patches with total fluxes below 2 Jy')
    sky_no_patches.remove('I < 2.0 Jy', aggregate='sum')
    assert len(sky_no_patches) == 433


def test_upgroup():
    print('Ungroup the skymodel')
    sky_no_patches.ungroup()
    assert ~sky_no_patches.hasPatches


def test_concatenate():
    print('Concatenate with concat.sky')
    sky_no_patches.concatenate('tests/resources/concat.sky', matchBy='position',
                               radius='30 arcsec', keep='from2')
    assert len(sky_no_patches) == 2165


def test_concatenate_differing_spectral_index():
    print('Concatenate with single_spectralindx.sky')
    sky_no_patches_orig = lsmtool.load('tests/resources/no_patches.sky')
    sky_no_patches_orig.concatenate('tests/resources/single_spectralindx.sky',
                                    matchBy='position', radius='30 arcsec', keep='from2')
    assert len(sky_no_patches_orig) == 1210


def test_compare():
    print('Compare to concat.sky')
    if os.path.exists('tests/flux_ratio_vs_distance.pdf'):
        os.remove('tests/flux_ratio_vs_distance.pdf')
    sky_concat = lsmtool.load('tests/resources/concat.sky')
    sky_concat.ungroup()
    sky_concat.select('I > 5.0 Jy')
    sky_no_patches.ungroup()
    sky_no_patches.compare(sky_concat, outDir='tests/')
    assert os.path.exists('tests/flux_ratio_vs_distance.pdf')


def test_add():
    print('Add a source')
    sky_no_patches.add({'Name': 'src1', 'Type': 'POINT', 'Ra': 277.4232, 'Dec': 48.3689,
                        'I': 0.69})
    assert len(sky_no_patches) == 2166


def test_group():
    print('Group using tessellation to a target flux of 50 Jy')
    sky_no_patches.group('tessellate', targetFlux='50.0 Jy')
    assert sky_no_patches.hasPatches


def test_move():
    print('Move patch Patch_1 to 16:04:16.2288, 58.03.06.912')
    sky_no_patches.move('Patch_1', position=['16:04:16.2288', '58.03.06.912'])
    assert round(sky_no_patches.getPatchPositions()['Patch_1'][0].value, 4) == 241.0676


def test_merge():
    print('Merge patches Patch_0 and Patch_2')
    sky_no_patches.merge(['Patch_0', 'Patch_2'], name='merged_patch')
    assert len(sky_no_patches.getPatchNames()) == 139


def test_setPatchPositions():
    print('Set patch positions to midpoint of patch')
    sky_no_patches.setPatchPositions(method='mid')
    assert round(sky_no_patches.getPatchPositions()['merged_patch'][0].value, 4) == 274.2124


def test_facet_write():
    print('Write ds9 facet file')
    if os.path.exists('tests/facet.reg'):
        os.remove('tests/facet.reg')
    sky_no_patches.write('tests/facet.reg', format='facet', clobber=True)

    # Note: Python 3.10+ produces a slightly different file (with insignificant
    # differences), so we allow a match to either one
    assert filecmp.cmp('tests/facet.reg', 'tests/resources/facet.reg') or filecmp.cmp('tests/facet.reg', 'tests/resources/facet2.reg')


def test_write():
    print('Write final model to file')
    if os.path.exists('tests/final.sky'):
        os.remove('tests/final.sky')
    sky_no_patches.write('tests/final.sky', clobber=True, addHistory=False)
    assert filecmp.cmp('tests/final.sky', 'tests/resources/final.sky')


def test_plot():
    print('Plot the model')
    if os.path.exists('tests/plot.pdf'):
        os.remove('tests/plot.pdf')
    sky_no_patches.plot('tests/plot.pdf')
    assert os.path.exists('tests/plot.pdf')


def test_meanshift():
    print('Group the model with the meanshift algorithm')
    sky_apparent.group('meanshift', byPatch=True, lookDistance=0.075, groupingDistance=0.01)
    assert len(sky_apparent.getPatchPositions()) == 67


def test_meanshift_with_nans():
    print('Load a model that contains NaNs and group it with the meanshift algorithm')
    sky_nans  = lsmtool.load('tests/resources/nans.sky')
    sky_nans.group('meanshift', byPatch=True, lookDistance=0.075, groupingDistance=0.01)
    assert len(sky_nans.getPatchPositions()) == 7
