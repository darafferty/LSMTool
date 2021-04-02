# -*- coding: utf-8 -*-
#
# This module stores the version and changelog

# Version number
__version__ = '1.4.2.post1'

# Change log
def changelog():
    log = """
    LSMTool Changelog.
    -----------------------------------------------

    2021/04/02 - Version 1.4.2.post1

        Publish on PyPI

    2019/10/01 - Version 1.4.2

        Fix to incorrect header on write

        Add meanshift grouping algorithm

    2019/04/12 - Version 1.4.1

        Fix installation on Python 3 systems

        Update GSM url

    2019/03/21 - Version 1.4.0

        Allow 2D images as masks in group operation

        Add TGSS and GSM VO queries

        Add option to apply beam attenuation during write

    2018/05/04 - Version 1.3.1

        Improve packaging support

    2018/05/03 - Version 1.3.0

        Add option to tessellate using patches

    2017/06/15 - Version 1.2.0

    2017/05/09

        Add a "pad_index" option to lsm.group() to allow the patch index to be
        padded with leading zeros

    2017/05/05

        Improve the "tessellate" grouping algorithm when negative fluxes are
        present

    2017/04/21

        Add support for LogarithmicSI column

        Add a "voronoi" option to lsm.group() to allow a previously grouped sky
        model to be regrouped using Voronoi tessellation

    2016/01/28 - Version 1.1.0

        Add support for FACTOR-formatted output

    2014/06/25 - Version 1.0.0 (initial release)
    """

    print(log)
