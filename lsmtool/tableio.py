# -*- coding: utf-8 -*-
#
# Defines astropy.table reader and writer functions for the following formats
#   - makesourcedb/BBS (reader and writer)
#   - ds9 (writer only)
#   - kvis (writer only)
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
import astropy
from astropy.table import Table, Column, MaskedColumn
from astropy.coordinates import Angle
from astropy.io import registry
import astropy.io.ascii as ascii
from packaging.version import Version
import numpy as np
import numpy.ma as ma
import re
import logging
import os
from copy import deepcopy
from .operations_lib import normalize_ra_dec, tessellate

# Python 3 compatibility
try:
    dict.iteritems
except AttributeError:
    # Python 3
    def itervalues(d):
        return iter(d.values())

    def iteritems(d):
        return iter(d.items())
    numpy_type = "U"
else:
    # Python 2
    def itervalues(d):
        return d.itervalues()

    def iteritems(d):
        return d.iteritems()
    numpy_type = "S"
try:
    unicode = unicode
except NameError:
    # Python 3
    basestring = (str, bytes)
else:
    # Python 2
    basestring = basestring
import io
try:
    # Python 2
    file_types = (file, io.IOBase)
except NameError:
    # Python 3
    file_types = (io.IOBase,)

# Define the valid columns here as dictionaries. The entry key is the lower-case
# name of the column, the entry value is the key used in the astropy table of the
# SkyModel object. For details, see:
# https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb
allowedColumnNames = {'name': 'Name', 'type': 'Type', 'patch': 'Patch',
                      'ra': 'Ra', 'dec': 'Dec', 'i': 'I', 'q': 'Q', 'u': 'U', 'v': 'V',
                      'majoraxis': 'MajorAxis', 'minoraxis': 'MinorAxis',
                      'orientation': 'Orientation', 'orientationisabsolute': 'OrientationIsAbsolute',
                      'ishapelet': 'IShapelet', 'qshapelet': 'QShapelet', 'ushapelet': 'UShapelet',
                      'vshapelet': 'VShapelet', 'category': 'Category', 'logarithmicsi': 'LogarithmicSI',
                      'rotationmeasure': 'RotationMeasure', 'polarizationangle': 'PolarizationAngle',
                      'polarizedfraction': 'PolarizedFraction', 'referencewavelength': 'ReferenceWavelength',
                      'referencefrequency': 'ReferenceFrequency', 'spectralindex': 'SpectralIndex'}

allowedColumnUnits = {'name': None, 'type': None, 'patch': None, 'ra': 'degree',
                      'dec': 'degree', 'i': 'Jy', 'i-apparent': 'Jy', 'q': 'Jy', 'u': 'Jy', 'v': 'Jy',
                      'majoraxis': 'arcsec', 'minoraxis': 'arcsec', 'orientation': 'degree',
                      'orientationisabsolute': None,
                      'ishapelet': None, 'qshapelet': None, 'ushapelet': None,
                      'vshapelet': None, 'category': None, 'logarithmicsi': None,
                      'rotationmeasure': 'rad/m^2', 'polarizationangle': 'rad',
                      'polarizedfraction': 'PolarizedFraction',
                      'referencewavelength': 'ReferenceWavelength',
                      'referencefrequency': 'Hz', 'spectralindex': None}

allowedColumnDefaults = {'name': 'N/A', 'type': 'N/A', 'patch': 'N/A', 'ra': 0.0,
                         'dec': 0.0, 'i': 0.0, 'q': 0.0, 'u': 0.0, 'v': 0.0, 'majoraxis': 0.0,
                         'minoraxis': 0.0, 'orientation': 0.0, 'orientationisabsolute': 'false',
                         'ishapelet': 'N/A', 'qshapelet': 'N/A', 'ushapelet': 'N/A',
                         'vshapelet': 'N/A', 'category': 2, 'logarithmicsi': 'true',
                         'rotationmeasure': 0.0, 'polarizationangle': 0.0,
                         'polarizedfraction': 0.0, 'referencewavelength': 'N/A',
                         'referencefrequency': 0.0, 'spectralindex': [0.0]}

requiredColumnNames = ['Name', 'Type', 'Ra', 'Dec', 'I']

allowedVOServices = {
    'nvss': 'http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=VIII/65&amp;',
    'wenss': 'http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=VIII/62A&amp;',
    'vlssr': 'http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=VIII/97&amp;'
}

# Define the various non-VO URLs used for downloading sky models
TGSS_URL = 'http://tgssadr.strw.leidenuniv.nl/cgi-bin/gsmv5.cgi'
GSM_URL = 'https://lcs165.lofar.eu/cgi-bin/gsmv1.cgi'
LOTSS_URL = 'https://vo.astron.nl/lotss_dr2/q/gaus_cone/form'


def raformat(val):
    """
    Column formatter for RA values.

    Parameters
    ----------
    val : float
        Input RA value in deg

    Returns
    -------
    valstr : str
        Formatted string as 'hh:mm:ss.s'

    """
    return Angle(val, unit='degree').to_string(unit='hourangle', sep=':')


def decformat(val):
    """
    Column formatter for Dec values.

    Parameters
    ----------
    val : float
        Input Dec value in deg

    Returns
    -------
    valstr : str
        Formatted string as 'dd.mm.ss.s'

    """
    return Angle(val, unit='degree').to_string(unit='degree', sep='.')


def fluxformat(val):
    """
    Column formatter for flux density values.

    Parameters
    ----------
    val : float
        Input flux density value in Jy

    Returns
    -------
    valstr : str
        Formatted string to 3 digits

    """
    if type(val) is ma.core.MaskedConstant:
        return '{}'.format(val)
    else:
        return '{0:0.3f}'.format(val)


def skyModelReader(fileName, header_start=0):
    """
    Reads a makesourcedb sky model file into an astropy table.

    See https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb
    for details. Note that source names, types, and patch names are limited to
    a length of 100 characters.

    Parameters
    ----------
    fileName : str
        Input ASCII file from which the sky model is read. Must
        respect the makesourcedb format
    header_start : int, optional
        Line number at which header starts

    Returns
    -------
    table : astropy.table.Table object

    """
    log = logging.getLogger('LSMTool.Load')

    # Open the input file
    with open(fileName) as modelFile:
        log.debug('Reading {0}'.format(fileName))

        # Read format line
        formatString = None
        for line in modelFile.readlines()[header_start:]:
            if 'format' in line.lower():
                formatString = line
                break
    if formatString is None:
        raise IOError("No valid format line found in file '{0}'.".format(fileName))

    # Process the header
    colNames, hasPatches, colDefaults, metaDict = processFormatString(formatString)

    # Read model into astropy table object
    outlines = []
    log.debug('Reading file...')
    with open(fileName) as f:
        for line in f:
            outline, metaDict = processLine(line, metaDict, colNames)
            if outline is not None:
                outlines.append(outline)
    outlines.append('\n')  # needed in case of single-line sky models

    # Create table
    table = createTable(outlines, metaDict, colNames, colDefaults)

    return table


def createTable(outlines, metaDict, colNames, colDefaults):
    """
    Creates an astropy table from inputs.

    Parameters
    ----------
    outlines : list of str
        Input lines
    metaDict : dict
        Input meta data
    colNames : list of str
        Input column names
    colDefaults : list
        Input column default values

    Returns
    -------
    table : astropy.table.Table object

    """
    log = logging.getLogger('LSMTool.Load')

    # Before loading table into an astropy Table object, set lengths of Name,
    # Patch, and Type columns to 100 characters to ensure long names are not
    # truncated. The LogarithmicSI and OrientationIsAbsolute columns are set
    # to 5 characters to allow true/false values to be stored as strings without
    # truncation. Due to a change in the astropy table API with v4.1, we have to
    # check the version and use the appropriate column names
    if Version(astropy.__version__) < Version('4.1'):
        # Use the input column names for the converters
        nameCol = 'col{0}'.format(colNames.index('Name')+1)
        typeCol = 'col{0}'.format(colNames.index('Type')+1)
        if 'Patch' in colNames:
            patchCol = 'col{0}'.format(colNames.index('Patch')+1)
        if 'LogarithmicSI' in colNames:
            logSICol = 'col{0}'.format(colNames.index('LogarithmicSI')+1)
        if 'OrientationIsAbsolute' in colNames:
            orienCol = 'col{0}'.format(colNames.index('OrientationIsAbsolute')+1)
    else:
        # Use the output column names for the converters
        nameCol = 'Name'
        typeCol = 'Type'
        patchCol = 'Patch'
        logSICol = 'LogarithmicSI'
        orienCol = 'OrientationIsAbsolute'
    converters = {}
    converters[nameCol] = [ascii.convert_numpy('{}100'.format(numpy_type))]
    converters[typeCol] = [ascii.convert_numpy('{}100'.format(numpy_type))]
    if 'Patch' in colNames:
        converters[patchCol] = [ascii.convert_numpy('{}100'.format(numpy_type))]
    if 'LogarithmicSI' in colNames:
        converters[logSICol] = [ascii.convert_numpy('{}5'.format(numpy_type))]
    if 'OrientationIsAbsolute' in colNames:
        converters[orienCol] = [ascii.convert_numpy('{}5'.format(numpy_type))]

    log.debug('Creating table...')
    table = Table.read('\n'.join(outlines), guess=False, format='ascii.no_header', delimiter=',',
                       names=colNames, comment='#', data_start=0, converters=converters)

    # Convert spectral index values from strings to arrays.
    if 'SpectralIndex' in table.keys():
        log.debug('Converting spectral indices...')
        specOld = table['SpectralIndex'].data.tolist()
        specVec = []
        maskVec = []
        maxLen = 0
        for entry in specOld:
            try:
                if type(entry) is float or type(entry) is int:
                    maxLen = 1
                else:
                    specEntry = [float(f) for f in entry.split(';')]
                    if len(specEntry) > maxLen:
                        maxLen = len(specEntry)
            except:
                pass
        defSpeclen = len(colDefaults[colNames.index('SpectralIndex')])
        if defSpeclen > maxLen:
            maxLen = defSpeclen
        log.debug('Maximum number of spectral-index terms in model: {0}'.format(maxLen))
        for entry in specOld:
            try:
                # Take existing entry and fix type
                if type(entry) is float or type(entry) is int:
                    specEntry = [float(entry)]
                    specMask = [False]
                else:
                    specEntry = [float(f) for f in entry.split(';')]
                    specMask = [False] * len(specEntry)
            except:
                # No entry in table, so use default value
                specEntry = colDefaults[colNames.index('SpectralIndex')]
                specMask = [False] * len(specEntry)
            while len(specEntry) < maxLen:
                # Add masked values to any entries that are too short
                specEntry.append(0.0)
                specMask.append(True)
            specVec.append(specEntry)
            maskVec.append(specMask)
        specCol = MaskedColumn(name='SpectralIndex', data=np.array(specVec, dtype=float))
        specCol.mask = maskVec
        specIndx = table.keys().index('SpectralIndex')
        table.remove_column('SpectralIndex')
        table.add_column(specCol, index=specIndx)

    # Convert RA and Dec to Angle objects
    log.debug('Converting RA and Dec...')
    RARaw = table['Ra'].data.tolist()
    DecRaw = table['Dec'].data.tolist()
    RANorm, DecNorm = RADec2Angle(RARaw, DecRaw)

    RACol = Column(name='Ra', data=RANorm)
    RACol.format = raformat
    RAIndx = table.keys().index('Ra')
    table.remove_column('Ra')
    table.add_column(RACol, index=RAIndx)

    DecCol = Column(name='Dec', data=DecNorm)
    DecCol.format = decformat
    DecIndx = table.keys().index('Dec')
    table.remove_column('Dec')
    table.add_column(DecCol, index=DecIndx)

    table.columns['I'].format = fluxformat

    # Set column units and default values
    for i, colName in enumerate(colNames):
        log.debug("Setting units for column '{0}' to {1}".format(
            colName, allowedColumnUnits[colName.lower()]))
        table.columns[colName].unit = allowedColumnUnits[colName.lower()]

        if hasattr(table.columns[colName], 'filled') and colDefaults[i] is not None:
            fillVal = colDefaults[i]
            log.debug("Setting default value for column '{0}' to {1}".
                      format(colName, fillVal))
            if colName == 'SpectralIndex':
                # We cannot set the fill value to a list/array, so just use a float
                fillVal = 0.0
            table.columns[colName].set_fill_value(fillVal)
    table.meta = metaDict

    return table


def processFormatString(formatString):
    """
    Proccesses the header string.

    Parameters
    ----------
    formatString : str
        Header line

    Returns
    -------
    colNames : list of str
        Output column names
    hasPatches : bool
        Flag for patches
    colDefaults : dict
        Default values
    metaDict : dict
        Output meta data

    """
    formatString = formatString.strip()
    formatString = formatString.strip('# ')
    if formatString.lower().endswith('format'):
        parts = formatString.split('=')[:-1]
        formatString = 'FORMAT = ' + '='.join(parts).strip('# ()')
    elif formatString.lower().startswith('format'):
        parts = formatString.split('=')[1:]
        formatString = 'FORMAT = ' + '='.join(parts).strip('# ()')
    else:
        raise IOError("Format line not understood.")

    # Check whether sky model has patches
    if 'Patch' in formatString:
        hasPatches = True
    else:
        hasPatches = False

    # Get column names and default values. Non-string columns have default
    # values of 0.0 unless a different value is given in the header.
    if ',' not in formatString:
        raise IOError("Sky model must use ',' as a field separator.")
    colNames = formatString.split(',')

    # Check if a default value in the format string is a list. If it is, make
    # sure the list is complete
    cnStart = None
    cnEnd = None
    for cn in colNames:
        if '[' in cn and ']' not in cn:
            cnStart = cn
        if ']' in cn and '[' not in cn:
            cnEnd = cn
    if cnStart is not None:
        indx1 = colNames.index(cnStart)
        indx2 = colNames.index(cnEnd)
        colNamesFixed = []
        toJoin = []
        for i, cn in enumerate(colNames):
            if i < indx1:
                colNamesFixed.append(cn)
            elif i >= indx1 and i <= indx2:
                toJoin.append(cn)
                if i == len(colNames)-1:
                    colNamesFixed.append(','.join(toJoin))
            elif i > indx2:
                if i == indx2 + 1:
                    colNamesFixed.append(','.join(toJoin))
                    colNamesFixed.append(cn)
                else:
                    colNamesFixed.append(cn)
        colNames = colNamesFixed

    # Now get the defaults
    colDefaults = [None] * len(colNames)
    metaDict = {}
    colNames[0] = colNames[0].split('=')[1]
    for i in range(len(colNames)):
        parts = colNames[i].split('=')
        colName = parts[0].strip().lower()
        if len(parts) == 2:
            try:
                if '[' in parts[1]:
                    # Default is a list
                    defParts = parts[1].strip("'[]").split(',')
                    defaultVal = []
                    for p in defParts:
                        defaultVal.append(float(p.strip()))
                elif 'true' in parts[1].lower():
                    defaultVal = True
                elif 'false' in parts[1].lower():
                    defaultVal = False
                else:
                    defaultVal = float(parts[1].strip("'"))
            except ValueError:
                defaultVal = None
        else:
            defaultVal = None

        if colName == '':
            raise IOError('Skipping of columns is not yet supported.')
        if colName not in allowedColumnNames:
            raise IOError("Column '{0}' is not currently allowed".format(colName))
        else:
            colNames[i] = allowedColumnNames[colName]
            if defaultVal is not None:
                colDefaults[i] = defaultVal
                metaDict[colNames[i]] = defaultVal
            elif allowedColumnDefaults[colName] is not None:
                # Note: we used deepcopy() here to ensure that the original
                # is not altered by later changes to colDefaults
                colDefaults[i] = deepcopy(allowedColumnDefaults)[colName]

    # Check for required columns
    for reqCol in requiredColumnNames:
        if reqCol not in colNames:
            raise IOError("Sky model must have a '{0}' column.".format(reqCol))

    return colNames, hasPatches, colDefaults, metaDict


def processLine(line, metaDict, colNames):
    """
    Processes a makesourcedb line.

    Parameters
    ----------
    line : str
        Data line
    metaDict : dict
        Input meta data
    colNames : list of str
        Input column names

    Returns
    -------
    line : str
        Processed line
    metaDict : dict
        Output meta data

    """
    log = logging.getLogger('LSMTool.Load')

    # Ignore format line and comments
    if line.lower().startswith("format") or line.startswith("#"):
        return None, metaDict

    # Check for SpectralIndex entries, which are unreadable as they use
    # the same separator for multiple orders as used for the columns
    line = line.strip('\n')
    a = re.search(r'\[.*\]', line)
    if a is not None:
        b = line[a.start(): a.end()]
        c = b.strip('[]')
        if ',' in c:
            c = c.replace(',', ';')
        line = line.replace(b, c)
    colLines = line.split(',')

    # Skip empty lines
    if all([col.strip() == '' for col in colLines]):
        return None, metaDict

    # Check for 'nan' in any column except string columns (indicated by 'N/A' in
    # the defaults dict)
    checkColNames = [name for name in colNames if allowedColumnDefaults[name.lower()] != 'N/A']
    for name in checkColNames:
        try:
            if 'nan' in colLines[colNames.index(name)]:
                log.warning('One or more NaNs found in the sky model. Sources and '
                            'patches with NaNs will be ignored.')
                return None, metaDict
        except IndexError:
            # Line does not contain this column
            pass

    # Check for patch lines as any line with an empty Name entry. If found,
    # store patch positions in the table meta data.
    nameIndx = colNames.index('Name')
    if colLines[nameIndx].strip() == '':
        if len(colLines) > 4:
            patchIndx = colNames.index('Patch')
            patchName = colLines[patchIndx].strip()
            RAIndx = colNames.index('Ra')
            DecIndx = colNames.index('Dec')
            if colLines[RAIndx].strip() == '' or colLines[DecIndx].strip() == '':
                patchRA = [Angle(0.0, unit='degree')]
                patchDec = [Angle(0.0, unit='degree')]
            else:
                patchRA, patchDec = RADec2Angle(colLines[RAIndx].strip(),
                                                colLines[DecIndx].strip())
            metaDict[patchName] = [patchRA[0], patchDec[0]]
        return None, metaDict

    while len(colLines) < len(colNames):
        colLines.append(' ')

    return ','.join(colLines), metaDict


def RADec2Angle(RA, Dec):
    """
    Returns normalized Angle objects for input RA, Dec values.

    Parameters
    ----------
    RA : str, float or list of str, float
        Values of RA to convert. Can be strings in makesourcedb format or floats
        in degrees (astropy.coordinates.Angle are also supported)
    Dec : str, float or list of str, float
        Values of Dec to convert. Can be strings in makesourcedb format or floats
        in degrees (astropy.coordinates.Angle are also supported)

    Returns
    -------
    RAAngle : list of astropy.coordinates.Angle objects
        The RA, normalized to [0, 360)
    DecAngle : list of astropy.coordinates.Angle objects
        The Dec, normalized to [-90, 90].
    """
    import astropy.units as u

    if type(RA) is not list:
        RA = [RA]
    if type(Dec) is not list:
        Dec = [Dec]

    if type(RA[0]) is str:
        try:
            RAAngle = Angle(Angle(RA, unit=u.hourangle), unit=u.deg)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            raise ValueError('RA not understood (must be string in '
                             'makesourcedb format or float in degrees): {0}'.format(e))
    else:
        RAAngle = Angle(RA, unit=u.deg)

    if type(Dec[0]) is str:
        try:
            DecAngle = Angle(Dec, unit=u.deg)
        except KeyboardInterrupt:
            raise
        except ValueError:
            try:
                DecSex = [decstr.replace('.', ':', 2) for decstr in Dec]
                DecAngle = Angle(DecSex, unit=u.deg)
            except Exception as e:
                raise ValueError('Dec not understood (must be string in '
                                 'makesourcedb format or float in degrees): {0}'.format(e))
        except Exception as e:
            raise ValueError('Dec not understood (must be string in '
                             'makesourcedb format or float in degrees): {0}'.format(e))
    else:
        DecAngle = Angle(Dec, unit=u.deg)

    RANorm = []
    DecNorm = []
    for RA, Dec in zip(RAAngle, DecAngle):
        RADec = normalize_ra_dec(RA, Dec)
        RANorm.append(RADec.ra)
        DecNorm.append(RADec.dec)

    return Angle(RANorm, unit=u.deg), Angle(DecNorm, unit=u.deg)


def skyModelIdentify(origin, *args, **kwargs):
    """
    Identifies valid makesourcedb sky model files.
    """
    # Search for a format line. If found, assume file is valid
    try:
        if isinstance(args[0], basestring):
            f = open(args[0])
        elif isinstance(args[0], file_types):
            f = args[0]
        else:
            return False
        for line in f:
            if line.startswith("FORMAT") or line.startswith("format"):
                return True
        return False
    except UnicodeDecodeError:
        return False


def skyModelWriter(table, fileName):
    """
    Writes table to a makesourcedb sky model file.

    Parameters
    ----------
    fileName : str
        Output ASCII file to which the sky model is written

    """
    log = logging.getLogger('LSMTool.Write')

    modelFile = open(fileName, 'w')
    log.debug('Writing model to {0}'.format(fileName))

    # Make sure all columns have the correct makesourcedb units
    for colName in table.columns:
        units = allowedColumnUnits[colName.lower()]
        if units is not None:
            table[colName].convert_unit_to(units)

    # Add format line
    outLines = []
    formatString = []
    for colKey in table.keys():
        if colKey.lower() not in allowedColumnNames:
            continue
        colName = allowedColumnNames[colKey.lower()]

        if colName in table.meta and colName != 'Patch':
            colHeader = "{0}='{1}'".format(colName, table.meta[colName])
        elif colName == 'SpectralIndex':
            colHeader = "{0}='[]'".format(colName)
        else:
            colHeader = colName
        formatString.append(colHeader)
    outLines.append('FORMAT = {0}'.format(', '.join(formatString)))
    if 'History' in table.meta:
        outLines.append('\n\n# LSMTool history:\n# ')
        outLines.append('\n# '.join(table.meta['History']))
    outLines.append('\n')
    outLines.append('\n')

    # Add source lines
    if 'Patch' in table.keys():
        table = table.group_by('Patch')
        patchNames = table.groups.keys['Patch']
        for i, patchName in enumerate(patchNames):
            if patchName in table.meta:
                try:
                    gRA, gDec = table.meta[patchName]
                except ValueError:
                    raise ValueError('Multiple positions per patch. Please set'
                                     'the patch positions.')
            else:
                gRA = 0.0
                gDec = 0.0
            gRADec = normalize_ra_dec(gRA, gDec)
            gRAStr = Angle(gRADec.ra, unit='degree').to_string(unit='hourangle', sep=':', precision=4)
            gDecStr = Angle(gRADec.dec, unit='degree').to_string(unit='degree', sep='.', precision=4)

            outLines.append(' , , {0}, {1}, {2}\n'.format(patchName, gRAStr,
                                                          gDecStr))
        for row in table.filled(fill_value=-9999):
            line = rowStr(row, table.meta)
            outLines.append(', '.join(line))
            outLines.append('\n')
    else:
        for row in table.filled(fill_value=-9999):
            line = rowStr(row, table.meta)
            outLines.append(', '.join(line))
            outLines.append('\n')

    modelFile.writelines(outLines)
    modelFile.close()


def rowStr(row, metaDict):
    """
    Returns makesourcedb representation of a row.

    Parameters
    ----------
    row : astropy.table.Row object
        Row to process
    metaDict : dict
        Table meta dictionary

    Returns
    -------
    line : str
        Sting representing a row in a makesourcedb sky model file

    """
    line = []
    for colKey in row.columns:
        try:
            colName = allowedColumnNames[colKey.lower()]
        except KeyError:
            continue

        # Determine whether the header (metaDict) defined a fill value and,
        # if so, use that for blank entries. If not, use the default value
        defaultVal = allowedColumnDefaults[colName.lower()]
        if colName in metaDict:
            fillVal = metaDict[colName]
            hasfillVal = True
        else:
            fillVal = defaultVal
            hasfillVal = False

        d = row[colKey]
        if str(d).startswith('-9999'):
            if hasfillVal:
                dstr = ' '
            else:
                dstr = str(fillVal)
        else:
            if type(d) is np.ndarray:
                if np.all(d == -9999):
                    if hasfillVal:
                        dstr = ' '
                    else:
                        dstr = str(fillVal)
                else:
                    dlist = d.tolist()
                    # Blank the value if it's equal to fill values
                    if hasfillVal and dlist == fillVal:
                        dlist = []
                    # Remove blanked values
                    if len(dlist) > 0:
                        while dlist[-1] == -9999:
                            dlist.pop()
                            if len(dlist) == 0:
                                break
                    dstr = str(dlist)
            else:
                if colKey == 'Ra':
                    dstr = Angle(d, unit='degree').to_string(unit='hourangle', sep=':')
                elif colKey == 'Dec':
                    dstr = Angle(d, unit='degree').to_string(unit='degree', sep='.')
                else:
                    dstr = str(d)
        line.append('{0}'.format(dstr))

    while line[-1] == ' ':
        line.pop()
    return line


def ds9RegionWriter(table, fileName):
    """
    Writes table to a ds9 region file.

    Parameters
    ----------
    table : astropy.table.Table object
        Input sky model table
    fileName : str
        Output file to which the sky model is written

    """
    log = logging.getLogger('LSMTool.Write')

    regionFile = open(fileName, 'w')
    log.debug('Writing ds9 region file to {0}'.format(fileName))

    outLines = []
    outLines.append('# Region file format: DS9 version 4.0\nglobal color=green '
                    'font="helvetica 10 normal" select=1 highlite=1 edit=1 '
                    'move=1 delete=1 include=1 fixed=0 source\nfk5\n')

    # Make sure all columns have the correct units
    for colName in table.columns:
        units = allowedColumnUnits[colName.lower()]
        if units is not None:
            table[colName].convert_unit_to(units)

    for row in table:
        ra = row['Ra']
        dec = row['Dec']
        name = row['Name']
        if row['Type'].lower() == 'gaussian':
            a = row['MajorAxis'] / 3600.0  # deg
            b = row['MinorAxis'] / 3600.0  # deg
            pa = row['Orientation']  # deg

            # ds9 can't handle 1-D Gaussians, so make sure they are 2-D
            if a < 1.0 / 3600.0:
                a = 1.0 / 3600.0  # deg
            if b < 1.0 / 3600.0:
                b = 1.0 / 3600.0  # deg
            region = 'ellipse({0}, {1}, {2}, {3}, {4}) # text={{{5}}}\n'.format(ra, dec, a, b,
                                                                                pa+90.0, name)
        else:
            region = 'point({0}, {1}) # point=cross width=2 text={{{2}}}\n'.format(ra, dec, name)
        outLines.append(region)

    regionFile.writelines(outLines)
    regionFile.close()


def kvisAnnWriter(table, fileName):
    """
    Writes table to a kvis annotation file.

    Parameters
    ----------
    table : astropy.table.Table object
        Input sky model table
    fileName : str
        Output file to which the sky model is written

    """
    log = logging.getLogger('LSMTool.Write')

    kvisFile = open(fileName, 'w')
    log.debug('Writing kvis annotation file to {0}'.format(fileName))

    # Make sure all columns have the correct units
    for colName in table.columns:
        units = allowedColumnUnits[colName.lower()]
        if units is not None:
            table[colName].convert_unit_to(units)

    outLines = []
    for row in table:
        ra = row['Ra']
        dec = row['Dec']
        name = row['Name']

        if row['Type'].lower() == 'gaussian':
            a = row['MajorAxis'] / 3600.0  # degree
            b = row['MinorAxis'] / 3600.0  # degree
            pa = row['Orientation']  # degree
            outLines.append('ELLIPSE W {0} {1} {2} {3} {4}\n'.format(ra, dec, a, b, pa))
        else:
            outLines.append('CIRCLE W {0} {1} 0.02\n'.format(ra, dec))
        outLines.append('TEXT W {0} {1} {2}\n'.format(ra - 0.07, dec, name))

    kvisFile.writelines(outLines)
    kvisFile.close()


def casaRegionWriter(table, fileName):
    """
    Writes model to a casa region file.

    Parameters
    ----------
    table : astropy.table.Table object
        Input sky model table
    fileName : str
        Output file to which the sky model is written

    """
    log = logging.getLogger('LSMTool.Write')

    casaFile = open(fileName, 'w')
    log.debug('Writing CASA box file to {0}'.format(fileName))

    outLines = []
    outLines.append('#CRTFv0\n')
    outLines.append('global coord=J2000\n\n')

    # Make sure all columns have the correct units
    for colName in table.columns:
        units = allowedColumnUnits[colName.lower()]
        if units is not None:
            table[colName].convert_unit_to(units)

    minSize = 10.0 / 3600.0  # min size in degrees
    for row in table:
        ra = row['Ra']
        dec = row['Dec']

        if row['Type'].lower() == 'gaussian':
            a = row['MajorAxis'] / 3600.0  # degree
            if a < minSize:
                a = minSize
            b = row['MinorAxis'] / 3600.0  # degree
            if b < minSize:
                b = minSize

            pa = row['Orientation']  # degree
            outLines.append('ellipse[[{0}deg, {1}deg], [{2}deg, {3}deg], '
                            '{4}deg]\n'.format(ra, dec, a, b, pa))
        else:
            outLines.append('ellipse[[{0}deg, {1}deg], [{2}deg, {3}deg], '
                            '{4}deg]\n'.format(ra, dec, minSize, minSize, 0.0))

    casaFile.writelines(outLines)
    casaFile.close()


def factorDirectionsWriter(table, fileName):
    """
    Writes patches to a Factor directions file.

    Note that Factor respects the order of patches and they are sorted here by
    apparent flux from brightest to faintest.

    Parameters
    ----------
    table : astropy.table.Table object
        Input sky model table; must have patches defined
    fileName : str
        Output file to which the sky model is written

    """
    log = logging.getLogger('LSMTool.Write')

    regionFile = open(fileName, 'w')
    log.debug('Writing Factor directions file to {0}'.format(fileName))

    outLines = []
    outLines.append('# name position atrous_do mscale_field_do cal_imsize '
                    'solint_ph solint_amp dynamic_range region_selfcal '
                    'region_facet peel_skymodel outlier_source cal_size_deg cal_flux_mJy\n')
    if 'History' in table.meta:
        outLines.append('\n# LSMTool history:\n# ')
        outLines.append('\n# '.join(table.meta['History']))
    outLines.append('\n')
    outLines.append('\n')

    # Make sure all columns have the correct units
    for colName in table.columns:
        units = allowedColumnUnits[colName.lower()]
        if units is not None:
            table[colName].convert_unit_to(units)

    table = table.group_by('Patch')
    patchNames = table.groups.keys['Patch']
    if 'patch_order' in table.meta:
        indx = table.meta['patch_order']
    else:
        indx = range(len(table.groups))
    if 'patch_size' in table.meta:
        sizes = table.meta['patch_size']
    else:
        sizes = [''] * len(table.groups)
    if 'patch_flux' in table.meta:
        fluxes = table.meta['patch_flux']
    else:
        fluxes = [''] * len(table.groups)
    for patchName, size, flux in zip(patchNames[indx], sizes[indx], fluxes[indx]):
        if patchName in table.meta:
            gRA, gDec = table.meta[patchName]
        else:
            gRA = Angle(0.0, unit='degree')
            gDec = Angle(0.0, unit='degree')
        outLines.append('{0} {1},{2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} '
                        '{13} {14}\n'.format(patchName, gRA.to_string(unit='hourangle', sep='hms'),
                                             gDec.to_string(sep='dms'), 'empty', 'empty', 0, 0, 0,
                                             'LD', 'empty', 'empty', 'empty', False, size, flux))

    regionFile.writelines(outLines)
    regionFile.close()


def facetRegionWriter(table, fileName):
    """
    Writes the model patches to a ds9 facet region file.

    The resulting file is suitable for use with WSClean in faceting mode.

    Parameters
    ----------
    table : astropy.table.Table object
        Input sky model table; must have patches defined
    fileName : str
        Output file to which the sky model is written

    """
    log = logging.getLogger('LSMTool.Write')

    # Get the positions of the calibration patches
    table = table.group_by('Patch')
    patchNames = table.groups.keys['Patch']
    patchRA = []
    patchDec = []
    for patchName in patchNames:
        gRA, gDec = table.meta[patchName]
        gRADec = normalize_ra_dec(gRA, gDec)
        patchRA.append(gRADec.ra)
        patchDec.append(gRADec.dec)

    # Do the tessellation
    facet_points, facet_polys = tessellate(patchRA, patchDec, table.meta['refRA'],
                                           table.meta['refDec'], table.meta['width'])

    # For each facet, match the correct name (some patches in the sky model may have
    # been filtered out if they lie outside the bounding box)
    facet_names = []
    for ra, dec, name in zip(patchRA, patchDec, patchNames):
        for facet_point in facet_points:
            if np.isclose(ra, facet_point[0]) and np.isclose(dec, facet_point[1]):
                facet_names.append(name)
                break

    # Make the ds9 region file
    lines = []
    lines.append('# Region file format: DS9 version 4.0\nglobal color=green '
                 'font="helvetica 10 normal" select=1 highlite=1 edit=1 '
                 'move=1 delete=1 include=1 fixed=0 source=1\nfk5\n')
    for name, center_coord, vertices in zip(facet_names, facet_points, facet_polys):
        radec_list = []
        RAs = vertices.T[0]
        Decs = vertices.T[1]
        for ra, dec in zip(RAs, Decs):
            radec_list.append('{0}, {1}'.format(ra, dec))
        lines.append('polygon({0})\n'.format(', '.join(radec_list)))
        if name is None:
            lines.append('point({0}, {1})\n'.format(center_coord[0], center_coord[1]))
        else:
            lines.append('point({0}, {1}) # text={{{2}}}\n'.format(center_coord[0], center_coord[1], name))

    log.debug('Writing facet region file to {0}'.format(fileName))
    with open(fileName, 'w') as f:
        f.writelines(lines)


def broadcastTable(fileName):
    """
    Sends a table via SAMP.

    Parameters
    ----------
    fileName : str
        Name of sky model file to broadcast

    """
    from astropy.vo.samp import SAMPIntegratedClient
    import urllib.parse

    client = SAMPIntegratedClient()
    client.connect()

    params = {}
    params["url"] = urllib.parse.urljoin('file:', os.path.abspath(fileName))
    params["name"] = "LSMTool sky model"
    message = {}
    message["samp.mtype"] = "table.load.votable"
    message["samp.params"] = params

    # Send message
    client.call_all('lsmtool', message)

    # Disconnect from the SAMP hub
    client.disconnect()


def coneSearch(VOService, position, radius):
    """
    Returns table from a VO cone search.

    Parameters
    ----------
    VOService : str
        Name of VO service to query (must be one of allowedVOServices)
    position : list of floats
        A list specifying a position as [RA, Dec] in either makesourcedb
        format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
        [123.2312, 23.3422])
    radius : float or str, optional
        Radius in degrees (if float) or 'value unit' (if str; e.g.,
        '30 arcsec') for cone search region

    """
    import pyvo as vo

    log = logging.getLogger('LSMTool.Load')

    # Define the mapping to go from the VO catalog column names to the
    # makesourcedb column names
    columnMapping = {
        'nvss': {'NVSS': 'name', 'RAJ2000': 'ra', 'DEJ2000': 'dec', 'S1.4': 'i',
                 'MajAxis': 'majoraxis', 'MinAxis': 'minoraxis'},
        'wenss': {'Name': 'name', 'RAJ2000': 'ra', 'DEJ2000': 'dec', 'Sint': 'i',
                  'MajAxis': 'majoraxis', 'MinAxis': 'minoraxis', 'PA': 'orientation'},
        'vlssr': {'Name': 'name', 'RAJ2000': 'ra', 'DEJ2000': 'dec', 'Sp': 'i',
                  'MajAx': 'majoraxis', 'MinAx': 'minoraxis', 'PA': 'orientation'}
        }

    # Define various properties of the VO catalog:
    #   fluxtype - type of flux density: "int" for total integrated flux,
    #              "peak" for peak flux density
    #   fluxunits - units of flux density
    #   deconvolved - whether the semimajor and semiminor axes are deconvolved or not
    #   psf - the point spread function in degrees
    #   referencefrequency - the reference frequency in Hz
    catalogProperties = {
        'nvss': {'fluxtype': 'int', 'fluxunits': 'mJy', 'deconvolved': True, 'psf': 0.0125,
                 'referencefrequency': 1.4e9},
        'wenss': {'fluxtype': 'int', 'fluxunits': 'mJy', 'deconvolved': True, 'psf': 0.015,
                  'referencefrequency': 325e6},
        'vlssr': {'fluxtype': 'peak', 'fluxunits': 'Jy', 'deconvolved': False, 'psf': 0.0208,
                  'referencefrequency': 74e6}
        }

    if VOService.lower() in allowedVOServices:
        url = allowedVOServices[VOService.lower()]
    else:
        raise ValueError('VO query service not known. Allowed services are: '
                         '{0}'.format(allowedVOServices.keys()))

    # Get raw VO catalog
    log.debug('Querying VO service...')
    try:
        RANorm, DecNorm = RADec2Angle(position[0], position[1])
        position = [RANorm[0].value, DecNorm[0].value]
    except TypeError:
        raise ValueError('VO query positon not understood.')
    try:
        radius = Angle(radius, unit='degree').value
    except TypeError:
        raise ValueError('VO query radius not understood.')
    try:
        VOcatalog = vo.conesearch(url, position, radius=radius)
    except (vo.dal.exceptions.DALQueryError, vo.dal.DALServiceError) as e:
        raise ConnectionError('Problem communicating with the VO service: {0}'.format(e))

    log.debug('Creating table...')
    try:
        table = Table.read(VOcatalog.votable)
    except IndexError:
        # Empty query result
        log.error('No sources found. Sky model is empty.')
        table = makeEmptyTable()
        return table
    table = convertExternalTable(table, columnMapping[VOService.lower()],
                                 catalogProperties[VOService.lower()])

    return table


def convertExternalTable(table, columnMapping, catalogProperties):
    """
    Converts an external table to a makesourcedb compatible one.

    Parameters
    ----------
    table : Table
        External table to convert
    columnMapping : dict
        Dict that defines the column name mapping from external table to
        makesourcedb columns
    catalogProperties : dict
        Dict that defines the catalog properties. Currently, these consist
        of 'fluxtype', 'deconvolved', 'psf', 'referencefrequency', and
        'fluxunits'
    """
    log = logging.getLogger('LSMTool.Load')

    # Add required columns
    for colName in requiredColumnNames:
        for k, v in columnMapping.items():
            if v == colName.lower():
                tableColname = k
                break
        if tableColname not in table.colnames:
            if colName.lower() == 'name':
                # If the "name" column is missing, generate simple source names
                col = Column(name=tableColname, data=[f'source_{indx}' for indx in range(len(table))])
                table.add_column(col)
            elif colName.lower() != 'type':
                # If any other column is missing (except "type", which is set
                # later), raise an error
                raise ValueError(f'VO table lacks the expected column "{tableColname}". '
                                 'Please check the VO service for problems.')

    # Remove unneeded columns
    colsToRemove = []
    for colName in table.colnames:
        if colName not in columnMapping:
            colsToRemove.append(colName)
        elif columnMapping[colName] not in allowedColumnNames:
            colsToRemove.append(colName)
    for colName in colsToRemove:
        table.remove_column(colName)

    # Rename columns to match makesourcedb conventions
    for colName in table.colnames:
        if colName != allowedColumnNames[columnMapping[colName]]:
            table.rename_column(colName, allowedColumnNames[columnMapping[colName]])

    # Convert RA and Dec to Angle objects
    log.debug('Converting RA and Dec...')
    RARaw = table['Ra'].data.tolist()
    DecRaw = table['Dec'].data.tolist()
    RANorm, DecNorm = RADec2Angle(RARaw, DecRaw)

    RACol = Column(name='Ra', data=RANorm)
    RACol.format = raformat
    RAIndx = table.keys().index('Ra')
    table.remove_column('Ra')
    table.add_column(RACol, index=RAIndx)

    DecCol = Column(name='Dec', data=DecNorm)
    DecCol.format = decformat
    DecIndx = table.keys().index('Dec')
    table.remove_column('Dec')
    table.add_column(DecCol, index=DecIndx)

    # Make sure Name is a str column
    NameRaw = table['Name'].data.tolist()
    NameCol = Column(name='Name', data=NameRaw, dtype='{}100'.format(numpy_type))
    table.remove_column('Name')
    table.add_column(NameCol, index=0)

    # Convert flux and axis values to floats
    for name in ['I', 'MajorAxis', 'MinorAxis', 'Orientation']:
        if name in table.colnames:
            indx = table.index_column(name)
            intRaw = table[name].data.tolist()
            floatCol = Column(name=name, data=intRaw, dtype='float')
            table.remove_column(name)
            table.add_column(floatCol, index=indx)

    # Add source-type column and convert fluxes to integrated values if needed
    types = ['POINT'] * len(table)
    if 'minoraxis' in columnMapping.values() and 'majoraxis' in columnMapping.values():
        for i, (minor, major) in enumerate(zip(table[allowedColumnNames['minoraxis']],
                                               table[allowedColumnNames['majoraxis']])):
            if (
                (catalogProperties['deconvolved'] and minor > 0.0) or
                (not catalogProperties['deconvolved'] and minor > catalogProperties['psf'])
            ):
                types[i] = 'GAUSSIAN'
                if catalogProperties['fluxtype'] == 'peak':
                    # For extended sources in catalogs with peak flux, we need
                    # to correct from peak to total flux using the source size
                    table.columns[allowedColumnNames['i']][i] *= minor * major / catalogProperties['psf']**2
            else:
                # Make sure semimajor and semiminor axes and orientation are 0 for POINT type
                table[allowedColumnNames['minoraxis']][i] = 0.0
                table[allowedColumnNames['majoraxis']][i] = 0.0
                if 'orientation' in columnMapping.values():
                    table[allowedColumnNames['orientation']][i] = 0.0
    col = Column(name='Type', data=types, dtype='{}100'.format(numpy_type))
    table.add_column(col, index=1)

    # Add reference-frequency column
    refFreq = catalogProperties['referencefrequency']
    col = Column(name='ReferenceFrequency', data=np.array([refFreq]*len(table), dtype=float))
    table.add_column(col)

    # Set column units and default values
    for i, colName in enumerate(table.colnames):
        log.debug("Setting units for column '{0}' to {1}".format(
            colName, allowedColumnUnits[colName.lower()]))
        if colName == 'I':
            table.columns[colName].unit = catalogProperties['fluxunits']
            table.columns[colName].convert_unit_to('Jy')
            table.columns[colName].format = fluxformat
        else:
            table.columns[colName].unit = allowedColumnUnits[colName.lower()]

        if hasattr(table.columns[colName], 'filled') and allowedColumnDefaults[colName.lower()] is not None:
            # Note: we used deepcopy() here to ensure that the original
            # is not altered by later changes to fillVal
            fillVal = deepcopy(allowedColumnDefaults)[colName.lower()]
            log.debug("Setting default value for column '{0}' to {1}".
                      format(colName, fillVal))
            if colName == 'SpectralIndex':
                # We cannot set the fill value to a list/array, so just use a float
                fillVal = 0.0
            table.columns[colName].set_fill_value(fillVal)

    return table


def getQueryInputs(position, radius):
    """
    Returns the inputs for a non-VO-compatible catalog search.

    Parameters
    ----------
    position : list of floats
        A list specifying a position as [RA, Dec] in either makesourcedb
        format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
        [123.2312, 23.3422])
    radius : float or str
        Radius in degrees (if float) or 'value unit' (if str; e.g.,
        '30 arcsec') for cone search region

    Returns
    -------
    RA : float
        RA in degrees of query position
    Dec : float
        Dec in degrees of query position
    radius : float
        Radius in degrees of query cone

    Raises
    ------
    ValueError
        Raised when the input radius cannot be converted to degrees,
        usually due to improperly specified units
    """
    RANorm, DecNorm = RADec2Angle(position[0], position[1])
    try:
        radius = Angle(radius, unit='degree').value
    except TypeError:
        raise ValueError('Query radius "{}" not understood.'.format(radius))

    return (RANorm[0].value, DecNorm[0].value, radius)


def queryNonVOService(url, format='makesourcedb'):
    """
    Returns the table from a non-VO service.

    Parameters
    ----------
    url : str
        URL of catalog
    format : str, optional
        Format to use when reading the catalog. Any format accepted by
        Table.read() is valid

    Raises
    ------
    ConnectionError
        Raised when the wget call returns a nonzero return code, indicating
        a problem with the connection to the service

    """
    import tempfile
    import subprocess

    # Use a temp file in the current working directory, as typical temp
    # directories like /tmp may be too small
    with tempfile.NamedTemporaryFile(dir=os.getcwd()) as outFile:
        cmd = ['wget', '-nv', '-O', outFile.name, url]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise ConnectionError(cp.stderr)

        table = Table.read(outFile.name, format=format, header_start=0)

    return table


def getTGSS(position, radius):
    """
    Returns the table from a TGSS search.

    Parameters
    ----------
    position : list of floats
        A list specifying a position as [RA, Dec] in either makesourcedb
        format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
        [123.2312, 23.3422])
    radius : float or str
        Radius in degrees (if float) or 'value unit' (if str; e.g.,
        '30 arcsec') for cone search region

    """
    log = logging.getLogger('LSMTool.Load')

    log.debug('Querying TGSS...')
    RA, Dec, radius = getQueryInputs(position, radius)
    url = TGSS_URL + '?coord={0},{1}&radius={2}&unit=deg&deconv=y'.format(RA, Dec, radius)
    table = queryNonVOService(url, format='makesourcedb')

    return table


def getGSM(position, radius):
    """
    Returns the table from a GSM search.

    Parameters
    ----------
    position : list of floats
        A list specifying a position as [RA, Dec] in either makesourcedb
        format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
        [123.2312, 23.3422])
    radius : float or str
        Radius in degrees (if float) or 'value unit' (if str; e.g.,
        '30 arcsec') for cone search region

    """
    log = logging.getLogger('LSMTool.Load')

    log.debug('Querying GSM...')
    RA, Dec, radius = getQueryInputs(position, radius)
    url = GSM_URL + '?coord={0},{1}&radius={2}&unit=deg&deconv=y'.format(RA, Dec, radius)
    table = queryNonVOService(url, format='makesourcedb')

    return table


def getLoTSS(position, radius):
    """
    Returns table from a LoTSS search.

    Parameters
    ----------
    position : list of floats
        A list specifying a position as [RA, Dec] in either makesourcedb
        format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
        [123.2312, 23.3422])
    radius : float or str
        Radius in degrees (if float) or 'value unit' (if str; e.g.,
        '30 arcsec') for cone search region

    """
    log = logging.getLogger('LSMTool.Load')

    columnMapping = {'Source_Name': 'name', 'RA': 'ra', 'DEC': 'dec', 'Total_flux': 'i',
                     'DC_Maj': 'majoraxis', 'DC_Min': 'minoraxis', 'PA': 'orientation'}
    catalogProperties = {'fluxtype': 'int', 'fluxunits': 'mJy', 'deconvolved': True,
                         'psf': 0.00167, 'referencefrequency': 1.4e8}

    log.debug('Querying LoTSS...')
    RA, Dec, radius = getQueryInputs(position, radius)
    radius *= 60  # LoTSS query requires arcmin, not degrees
    url = (LOTSS_URL + '?__nevow_form__=genForm&'
           'hscs_pos={0}%2C%20{1}&hscs_sr={2}&_DBOPTIONS_ORDER=&'
           '_DBOPTIONS_DIR=ASC&MAXREC=100000&_FORMAT=CSV&submit=Go'.format(RA, Dec, radius))
    table = queryNonVOService(url, format='ascii.csv')
    table = convertExternalTable(table, columnMapping, catalogProperties)

    return table


def makeEmptyTable():
    """
    Returns an empty sky model table.
    """
    outlines = ['Z, Z, 0.0, 0.0, 0.0\n']
    colNames = ['Name', 'Type', 'Ra', 'Dec', 'I']
    converters = {}
    nameCol = 'col{0}'.format(colNames.index('Name')+1)
    converters[nameCol] = [ascii.convert_numpy('{}100'.format(numpy_type))]
    typeCol = 'col{0}'.format(colNames.index('Type')+1)
    converters[typeCol] = [ascii.convert_numpy('{}100'.format(numpy_type))]
    table = Table.read(outlines, guess=False, format='ascii.no_header', delimiter=',',
                       names=colNames, comment='#', data_start=0, converters=converters)
    table.remove_rows(0)
    return table


# Register the file reader, identifier, and writer functions with astropy.io
registry.register_reader('makesourcedb', Table, skyModelReader)
registry.register_identifier('makesourcedb', Table, skyModelIdentify)
registry.register_writer('makesourcedb', Table, skyModelWriter)
registry.register_writer('ds9', Table, ds9RegionWriter)
registry.register_writer('kvis', Table, kvisAnnWriter)
registry.register_writer('casa', Table, casaRegionWriter)
registry.register_writer('factor', Table, factorDirectionsWriter)
registry.register_writer('facet', Table, facetRegionWriter)
