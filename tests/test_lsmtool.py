#! /usr/bin/env python
# Runs an example of each operation
import lsmtool
import os


s = lsmtool.load('tests/no_patches.sky')
s2 = lsmtool.load('tests/apparent.sky')


def test_select():
    print('Select individual sources with Stokes I fluxes above 1 Jy')
    s.select('I > 1.0 Jy')
    assert len(s) == 965


def test_transfer():
    print('Transfer patches from patches.sky')
    s.transfer('tests/patches.sky')
    assert s.hasPatches


def test_remove():
    print('Remove patches with total fluxes below 2 Jy')
    s.remove('I < 2.0 Jy', aggregate='sum')
    assert len(s) == 433


def test_upgroup():
    print('Ungroup the skymodel')
    s.ungroup()
    assert ~s.hasPatches


def test_concatenate():
    print('Concatenate with concat.sky')
    s.concatenate('tests/concat.sky', matchBy = 'position', radius = '30 arcsec', keep = 'from2')
    assert len(s) == 2165


def test_compare():
    print('Compare to concat.sky')
    if os.path.exists('tests/flux_ratio_vs_distance.sky'):
        os.remove('tests/flux_ratio_vs_distance.sky')
    c = lsmtool.load('tests/concat.sky')
    c.ungroup()
    c.select('I > 5.0 Jy')
    s.ungroup()
    s.compare(c, outDir='tests/')
    assert os.path.exists('tests/flux_ratio_vs_distance.pdf')


def test_add():
    print('Add a source')
    s.add({'Name': 'src1', 'Type': 'POINT', 'Ra': 277.4232, 'Dec': 48.3689, 'I': 0.69})
    assert len(s) == 2166


def test_group():
    print('Group using tessellation to a target flux of 50 Jy')
    s.group('tessellate', targetFlux = '50.0 Jy')
    assert s.hasPatches


def test_move():
    print('Move patch Patch_1 to 16:04:16.2288, 58.03.06.912')
    s.move('Patch_1', position=['16:04:16.2288', '58.03.06.912'])
    assert round(s.getPatchPositions()['Patch_1'][0].value, 4) == 241.0676


def test_merge():
    print('Merge patches Patch_0 and Patch_2')
    s.merge(['Patch_0', 'Patch_2'], name = 'merged_patch')
    assert len(s.getPatchNames()) == 139


def test_setPatchPositions():
    print('Set patch positions to midpoint of patch')
    s.setPatchPositions(method='mid')
    assert round(s.getPatchPositions()['merged_patch'][0].value, 4) == 274.2124


def test_write():
    print('Write final model to file')
    if os.path.exists('tests/final.sky'):
        os.remove('tests/final.sky')
    s.write('tests/final.sky', clobber=True)
    assert os.path.exists('tests/final.sky')


def test_plot():
    print('Plot the model')
    if os.path.exists('tests/plot.pdf'):
        os.remove('tests/plot.pdf')
    s.plot('tests/plot.pdf')
    assert os.path.exists('tests/plot.pdf')


def test_meanshift():
    print('Group the model with the meanshift algorithm')
    s2.group('meanshift', byPatch=True, lookDistance=0.075, groupingDistance=0.01)
    assert len(s2.getPatchPositions()) == 67
