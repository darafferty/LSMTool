#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This is the command-line script that performs the operations defined in a
# LSMTool parset.
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

# Authors:
# David Raffery
_author = "David Rafferty (drafferty@hs.uni-hamburg.de)"

import sys
import os
import time
import logging
from lsmtool import _version, _logging, skymodel, operations
from lofar_parameterset.parameterset import parameterset


def main():
    # Options
    import optparse
    opt = optparse.OptionParser(usage='%prog <skymodel> <parset> [<beam MS>] \n'
            +_author, version='%prog ' + _version.__version__)
    opt.add_option('-q', help='Quiet', action='store_true', default=False)
    opt.add_option('-v', help='Verbose', action='store_true', default=False)
    (options, args) = opt.parse_args()

    if options.q:
        _logging.setLevel('warning')
    if options.v:
        _logging.setLevel('debug')

    # Check options
    if len(args) not in [2, 3]:
        opt.print_help()
        sys.exit()

    try:
        skyModelFile = args[0]
    except:
        logging.critical('Missing sky model file.')
        sys.exit(1)
    try:
        parsetFile = args[1]
    except:
        logging.critical('Missing parset file.')
        sys.exit(1)
    try:
        beamMS = args[2]
    except:
        beamMS = None

    if not os.path.isfile(skyModelFile):
        logging.critical("Missing skymodel file.")
        sys.exit(1)
    if not os.path.isfile(parsetFile):
        logging.critical("Missing parset file.")
        sys.exit(1)

    # Load the skymodel
    try:
        LSM = skymodel.SkyModel(skyModelFile, beamMS=beamMS)
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    # from ~vdtol/Expion-2011-05-03/src
    parset = parameterset.fromFile(parsetFile)
    steps = parset.getStringVector("LSMTool.Steps", [])

    # Possible operations, linked to relevant function
    availableOperations = {"REMOVE": operations.remove,
                           "SELECT": operations.select,
                           "GROUP": operations.group,
                           "UNGROUP": operations.ungroup,
                           "CONCATENATE": operations.concatenate,
                           "ADD": operations.add,
                           "MERGE": operations.merge,
                           "MOVE": operations.move,
                           "PLOT": operations.plot,
                           "SETPATCHPOSITIONS": operations.setpatchpositions,
                           "TRANSFER": operations.transfer,
                           "COMPARE": operations.compare
                           }

    for step in steps:
        operation = parset.getString('.'.join(["LSMTool.Steps", step, "Operation"]))
        logging.info("--> Starting \'" + step + "\' step (operation: " + operation + ").")
        start = time.perf_counter()
        returncode = availableOperations[operation].run(step, parset, LSM)
        if returncode != 0:
           logging.error("Step \'" + step + "\' incomplete. Trying to continue anyway...")
        else:
           logging.info("Step \'" + step + "\' completed successfully.")
        elapsed = (time.perf_counter() - start)
        logging.debug("Time for this step: "+str(elapsed)+" s.")

    logging.info("Done.")

