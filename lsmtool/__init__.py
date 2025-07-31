# -*- coding: utf-8 -*-

"""
This initializes the LSMTool module by importing the necessary submodules
containing utility (:py:mod:`lsmtool.utils`) and input-output
(:py:mod:`lsmtool.io`) functions. Table input-output operations using
makesourcedb format are registered with astropy upon import.

.. moduleauthor:: David Rafferty <drafferty@hs.uni-hamburg.de>
"""

# NOTE: tableio import below registers makesourcedb reader/writer in
# astropy.table

from . import _logging as logger
from . import io, tableio, utils
from .io import load

__all__ = ["io", "utils", "load", "tableio"]

logger.setLevel("info")
