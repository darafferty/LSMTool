# -*- coding: utf-8 -*-
#
# This module stores the version and changelog

# Version number
__version__ = '1.4.10'

# Change log
def changelog():
    log = """
    LSMTool Changelog.
    -----------------------------------------------

    2023/01/09 - Version 1.4.10

        Fix issues with concatenating two sky models

        Fix beam attenuation when using EveryBeam

        Fix handling of LogarithmicSI and OrientationIsAbsolute columns

        Fix skymodel compare operation

        Replace deprecated `numpy.float` with plain `float`

    2022/08/29 - Version 1.4.9

        Documentation moved to https://lsmtool.readthedocs.io.

        Repository moved to https://git.astron.nl/RD/LSMTool.git,
        https://github.com/darafferty/LSMTool.git is now a mirror.

        Added a new script correct_gaussian_orientation.py, which computes
        the absolute Gaussian position angle for a skymodel file.

        Fixed numpy deprecation warnings.

        A few minor bug fixes.

    2022/06/09 - Version 1.4.8

        Fix bug in concatenation of sky models when number of spectral index
        terms differs

    2022/02/23 - Version 1.4.7

        Fix reading of patchless tables for astropy versions < 4.1

    2022/02/08 - Version 1.4.6

        Fix table column converters for astropy versions >= 4.1

    2022/02/04 - Version 1.4.5

        Fix regression in handling of blank values introduced in v1.4.4

    2022/01/27 - Version 1.4.4

        Fix improper handling of blank values

        Add support for EveryBeam

    2021/04/07 - Version 1.4.3

        Publish on PyPI

        Add faster version of meanshift algorithm

        Fix to incorrect filtering with mask images

        Fix to Astropy registry check

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
