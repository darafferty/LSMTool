# -*- coding: utf-8 -*-
#
# Defines the custom logger

import logging

# set the logging format and default level (info)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def setLevel(level):
    """
    Change the verbosity of the logger

    Parameters
    ----------
    level : str
        Desired level (from less to more verbosity): 'warning', 'info', 'debug'
    """
    level = level.lower()
    if level not in ['warning', 'info', 'debug']:
        raise ValueError("Invalid level name specified. Level name must be one "
            "of: 'warning', 'info', 'debug'")
    if level == 'warning':
        logging.root.setLevel(logging.WARNING)
    elif level == 'info':
        logging.root.setLevel(logging.INFO)
    elif level == 'debug':
        logging.root.setLevel(logging.DEBUG)

