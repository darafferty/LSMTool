#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This operation implements merging of patches
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

import logging

log = logging.getLogger('LSMTool.MERGE')
log.debug('Loading MERGE module.')


def run(step, parset, LSM):

    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )
    patches = parset.getStringVector('.'.join(["LSMTool.Steps", step, "Patches"]), [] )
    name = parset.getString('.'.join(["LSMTool.Steps", step, "Name"]), '' )
    if name == '':
        name = None

    try:
        result = merge(LSM, patches, name)
        result = 0
    except Exception as e:
        log.error(e.message)
        result = 1

    # Write to outFile
    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result


def merge(LSM, patches, name=None):
    """
    Merge two or more patches together

    Parameters
    ----------
    LSM : SkyModel object
        Input sky model
    patches : list of str
        List of patches to merge
    name : str, optional
        Name of resulting merged patch. If None, the merged patch uses the name
        of the first patch in the input patches list

    Examples
    --------
    Merge three patches into one patch named 'binmerged'::

        >>> LSM = lsmtool.load('sky.model')
        >>> merge(LSM, ['bin0', 'bin1', 'bin2'], 'binmerged')

    """
    if len(LSM) == 0:
        log.error('Sky model is empty.')
        return

    if name is None:
        name = patches[0]

    indices = []
    for patchName in patches:
        indices.append(LSM.getRowIndex(patchName))
    for patchName, ind in zip(patches, indices):
        if ind is None:
            raise ValueError("Could not find patch '{0}'.".format(patchName))
        else:
            LSM.table['Patch'][ind] = name
    LSM._updateGroups()
    LSM._addHistory("MERGE ('{0}' into '{1}')".format(patches, name))
    LSM._info()
