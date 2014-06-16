# Runs steps for validation

import lsmtool


s = lsmtool.load('tests/no_patches.sky')

print('Select individual sources with Stokes I fluxes above 1 Jy')
s.select('I > 1.0 Jy')

print('Transfer patches from test2.sky')
s.transfer('tests/patches.sky')

print('Remove patches with total fluxes below 2 Jy')
s.remove('I < 2.0 Jy', aggregate='sum')

print('Remove all patches')
s.ungroup()

print('Group using tessellation to a target flux of 50 Jy')
s.group('tessellate', targetFlux = '50.0 Jy', method = 'mid')

print('Move patch bin1 by 0.001 degrees in RA')
s.move('bin1', shift = [0.001, 0.0])

print('Merge patches bin0 and bin2')
s.merge(['bin0', 'bin2'], name = 'merged_patch')

print('Concatenate with concat_sky_model.sky')
s.concatenate('tests/patches.sky', matchBy = 'position', radius = '30 arcsec', keep = 'from2')

print('Add a source and write final sky model')
s.add({'Name': 'src1', 'Type': 'POINT', 'Patch': 'src1_patch', 'Ra': 277.4232, 'Dec': 48.3689, 'I': 0.69})
s.write('tests/test_final.sky', clobber=True)

print('Plot the sky model')
s.plot('tests/test_plot.pdf')
