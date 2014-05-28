#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors:
# David Raffery
_author = "David Rafferty (drafferty@hs.uni-hamburg.de)"

import sys
import os
import time
import logging
import _version
import _logging
import lofar.parameterset
import skymodel


if __name__=='__main__':
    # Options
    import optparse
    opt = optparse.OptionParser(usage='%prog <skymodel> <parset> [<beam MS>] \n'
            +_author, version='%prog '+_version.__version__)
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
        logging.critical('Missing skymodel file.')
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
    LSM = skymodel.SkyModel(skyModelFile, beamMS=beamMS)

    # from ~vdtol/Expion-2011-05-03/src
    parset = lofar.parameterset.parameterset( parsetFile )
    steps = parset.getStringVector( "LSMTool.Steps", [] )

    # Possible operations, linked to relevant function
    import operations
    operations = {"REMOVE": operations.remove,
                  "SELECT": operations.select,
                  "GROUP": operations.group,
                  "UNGROUP": operations.ungroup,
                  "CONCATENATE": operations.concatenate,
                  "ADD": operations.add,
                  "MERGE": operations.merge,
                  "MOVE": operations.move,
                  "PLOT": operations.plot,
                  "TRANSFER": operations.transfer
                  }

    for step in steps:
       operation = parset.getString( '.'.join( [ "LSMTool.Steps", step, "Operation" ] ) )
       logging.info("--> Starting \'" + step + "\' step (operation: " + operation + ").")
       start = time.clock()
       returncode = operations[ operation ].run( step, parset, LSM )
       if returncode != 0:
          logging.error("Step \'" + step + "\' incomplete. Try to continue anyway.")
       else:
          logging.info("Step \'" + step + "\' completed successfully.")
       elapsed = (time.clock() - start)
       logging.debug("Time for this step: "+str(elapsed)+" s.")

    logging.info("Done.")

