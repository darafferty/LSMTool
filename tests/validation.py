# Runs steps for validation

import lsmtool


s = lsmtool.load('tests/no_patches.sky')

print('Select individual sources with Stokes I fluxes above 1 Jy')
s.select('I > 1.0 Jy')

print('Transfer patches from patches.sky')
s.transfer('tests/patches.sky')

print('Remove patches with total fluxes below 2 Jy')
s.remove('I < 2.0 Jy', aggregate='sum')

print('Ungroup the skymodel')
s.ungroup()

print('Concatenate with concat.sky')
s.concatenate('tests/concat.sky', matchBy = 'position', radius = '30 arcsec', keep = 'from2')

print('Compare to concat.sky')
c = lsmtool.load('tests/concat.sky')
c.ungroup()
c.select('I > 5.0 Jy')
s.ungroup()
s.compare(c, outDir='tests/')

print('Add a source')
s.add({'Name': 'src1', 'Type': 'POINT', 'Ra': 277.4232, 'Dec': 48.3689, 'I': 0.69})

print('Group using tessellation to a target flux of 50 Jy')
s.group('tessellate', targetFlux = '50.0 Jy')

print('Move patch Patch_1 to 16:04:16.2288, 58.03.06.912')
s.move('Patch_1', position =  ['16:04:16.2288', '58.03.06.912'])

print('Merge patches Patch_0 and Patch_2')
s.merge(['Patch_0', 'Patch_2'], name = 'merged_patch')

print('Reset patch positions to midpoint of patch and write final model to file')
s.setPatchPositions(method='mid')
s.write('tests/final.sky', clobber=True)

print('Plot the sky model')
s.plot('tests/plot.pdf', labelBy='patch')
