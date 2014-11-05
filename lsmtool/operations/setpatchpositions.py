#
# This operation implements setting of patch positions
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

log = logging.getLogger('LSMTool.SETPATCHPOSITIONS')
log.debug('Loading SETPATCHPOSITIONS module.')


def run(step, parset, LSM):

    method = parset.getString('.'.join(["LSMTool.Steps", step, "Method"]), 'mid' )
    applyBeam = parset.getBool('.'.join(["LSMTool.Steps", step, "ApplyBeam"]), False )
    outFile = parset.getString('.'.join(["LSMTool.Steps", step, "OutFile"]), '' )

    try:
        LSM.setPatchPositions(method=method, applyBeam=applyBeam)
        result = 0
    except Exception as e:
        log.error(e.message)
        result = 1

    # Write to outFile
    if outFile != '' and result == 0:
        LSM.write(outFile, clobber=True)

    return result
