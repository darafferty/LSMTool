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

from astropy.table import Table, Column, MaskedColumn
from astropy.coordinates import Angle
from astropy.io import registry
import astropy.io.ascii as ascii
import numpy as np
import numpy.ma as ma
import re
import logging
import os

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
    basestring = (str,bytes)
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
# http://www.lofar.org/operations/doku.php?id=engineering:software:tools:makesourcedb#format_string
allowedColumnNames = {'name':'Name', 'type':'Type', 'patch':'Patch',
    'ra':'Ra', 'dec':'Dec', 'i':'I', 'q':'Q', 'u':'U', 'v':'V',
    'majoraxis':'MajorAxis', 'minoraxis':'MinorAxis', 'orientation':'Orientation',
    'ishapelet':'IShapelet', 'qshapelet':'QShapelet', 'ushapelet':'UShapelet',
    'vshapelet':'VShapelet', 'category':'Category', 'logarithmicsi':'LogarithmicSI',
    'rotationmeasure':'RotationMeasure', 'polarizationangle':'PolarizationAngle',
    'polarizedfraction':'PolarizedFraction', 'referencewavelength':'ReferenceWavelength',
    'referencefrequency':'ReferenceFrequency', 'spectralindex':'SpectralIndex'}

allowedColumnUnits = {'name':None, 'type':None, 'patch':None, 'ra':'degree',
    'dec':'degree', 'i':'Jy', 'i-apparent':'Jy', 'q':'Jy', 'u':'Jy', 'v':'Jy',
    'majoraxis':'arcsec', 'minoraxis':'arcsec', 'orientation':'degree',
    'ishapelet':None, 'qshapelet':None, 'ushapelet':None,
    'vshapelet':None, 'category':None, 'logarithmicsi':None,
    'rotationmeasure':'rad/m^2', 'polarizationangle':'rad',
    'polarizedfraction':'PolarizedFraction', 'referencewavelength':'ReferenceWavelength',
    'referencefrequency':'Hz', 'spectralindex':None}

allowedColumnDefaults = {'name':'N/A', 'type':'N/A', 'patch':'N/A', 'ra':0.0,
    'dec':0.0, 'i':0.0, 'q':0.0, 'u':0.0, 'v':0.0, 'majoraxis':0.0,
    'minoraxis':0.0, 'orientation':0.0,
    'ishapelet':'N/A', 'qshapelet':'N/A', 'ushapelet':'N/A',
    'vshapelet':'N/A', 'category':2, 'logarithmicsi': True,
    'rotationmeasure':0.0, 'polarizationangle':0.0,
    'polarizedfraction':0.0, 'referencewavelength':'N/A',
    'referencefrequency':0.0, 'spectralindex':[0.0]}

requiredColumnNames = ['Name', 'Type', 'Ra', 'Dec', 'I']

allowedVOServices = {
    'nvss':'http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=VIII/65&amp;',
    'wenss':'http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=VIII/62A&amp;'}


def skyModelReader(fileName):
    """
    Reads a makesourcedb sky model file into an astropy table.

    See http://www.lofar.org/operations/doku.php?id=engineering:software:tools:makesourcedb#format_string
    for details. Note that source names, types, and patch names are limited to
    a length of 100 characters.

    Parameters
    ----------
    fileName : str
        Input ASCII file from which the sky model is read. Must
        respect the makesourcedb format

    Returns
    -------
    table : astropy.table.Table object

    """
    log = logging.getLogger('LSMTool.Load')

    # Open the input file
    try:
        modelFile = open(fileName)
        log.debug('Reading {0}'.format(fileName))
    except IOError as e:
        raise IOError('Could not open {0}: {1}'.format(fileName, e.strerror))

    # Read format line
    formatString = None
    for l, line in enumerate(modelFile):
        if 'format' in line.lower():
            formatString = line
            break
    modelFile.close()
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
    outlines.append('\n') # needed in case of single-line sky models

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
    # Before loading table into an astropy Table object, set lengths of Name,
    # Patch, and Type columns to 100 characters
    log = logging.getLogger('LSMTool.Load')

    converters = {}
    nameCol = 'col{0}'.format(colNames.index('Name')+1)
    converters[nameCol] = [ascii.convert_numpy('{}100'.format(numpy_type))]
    typeCol = 'col{0}'.format(colNames.index('Type')+1)
    converters[typeCol] = [ascii.convert_numpy('{}100'.format(numpy_type))]
    if 'Patch' in colNames:
        patchCol = 'col{0}'.format(colNames.index('Patch')+1)
        converters[patchCol] = [ascii.convert_numpy('{}100'.format(numpy_type))]

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
        for l in specOld:
            try:
                if type(l) is float or type(l) is int:
                    maxLen = 1
                else:
                    specEntry = [float(f) for f in l.split(';')]
                    if len(specEntry) > maxLen:
                        maxLen = len(specEntry)
            except:
                pass
        log.debug('Maximum number of spectral-index terms in model: {0}'.format(maxLen))
        for l in specOld:
            try:
                if type(l) is float or type(l) is int:
                    specEntry = [float(l)]
                    specMask = [False]
                else:
                    specEntry = [float(f) for f in l.split(';')]
                    specMask = [False] * len(specEntry)
                while len(specEntry) < maxLen:
                    specEntry.append(0.0)
                    specMask.append(True)
                specVec.append(specEntry)
                maskVec.append(specMask)
            except:
                specVec.append([0.0]*maxLen)
                maskVec.append([True]*maxLen)
        specCol = MaskedColumn(name='SpectralIndex', data=np.array(specVec, dtype=np.float))
        specCol.mask = maskVec
        specIndx = table.keys().index('SpectralIndex')
        table.remove_column('SpectralIndex')
        table.add_column(specCol, index=specIndx)

    # Convert RA and Dec to Angle objects
    log.debug('Converting RA...')
    RARaw = table['Ra'].data.tolist()
    RACol = Column(name='Ra', data=RA2Angle(RARaw))
    def raformat(val):
        return Angle(val, unit='degree').to_string(unit='hourangle', sep=':')
    RACol.format = raformat
    RAIndx = table.keys().index('Ra')
    table.remove_column('Ra')
    table.add_column(RACol, index=RAIndx)

    log.debug('Converting Dec...')
    DecRaw = table['Dec'].data.tolist()
    DecCol = Column(name='Dec', data=Dec2Angle(DecRaw))
    def decformat(val):
        return Angle(val, unit='degree').to_string(unit='degree', sep='.')
    DecCol.format = decformat
    DecIndx = table.keys().index('Dec')
    table.remove_column('Dec')
    table.add_column(DecCol, index=DecIndx)

    def fluxformat(val):
        if type(val) is ma.core.MaskedConstant:
            return '{}'.format(val)
        else:
            return '{0:0.3f}'.format(val)
    table.columns['I'].format = fluxformat

    # Set column units and default values
    for i, colName in enumerate(colNames):
        log.debug("Setting units for column '{0}' to {1}".format(
            colName, allowedColumnUnits[colName.lower()]))
        table.columns[colName].unit = allowedColumnUnits[colName.lower()]

        if hasattr(table.columns[colName], 'filled') and colDefaults[i] is not None:
            fillVal = colDefaults[i]
            if colName == 'SpectralIndex':
                while len(fillVal) < maxLen:
                    fillVal.append(0.0)
            log.debug("Setting default value for column '{0}' to {1}".
                format(colName, fillVal))
            table.columns[colName].fill_value = fillVal
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
                colDefaults[i] = allowedColumnDefaults[colName]

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
    if line.lower().startswith("format") or line.startswith("#"):
        return None, metaDict

    # Check for SpectralIndex entries, which are unreadable as they use
    # the same separator for multiple orders as used for the columns
    line = line.strip('\n')
    a = re.search('\[.*\]', line)
    if a is not None:
        b = line[a.start(): a.end()]
        c = b.strip('[]')
        if ',' in c:
            c = c.replace(',', ';')
        line = line.replace(b, c)
    colLines = line.split(',')

    # Check for patch lines as any line with an empty Name entry. If found,
    # store patch positions in the table meta data.
    nameIndx = colNames.index('Name')
    if colLines[nameIndx].strip() == '':
        if len(colLines) > 4:
            patchIndx = colNames.index('Patch')
            patchName = colLines[patchIndx].strip()
            RAIndx = colNames.index('Ra')
            if colLines[RAIndx].strip() == '':
                patchRA = [0.0]
            else:
                patchRA = RA2Angle(colLines[RAIndx].strip())
            DecIndx = colNames.index('Dec')
            if colLines[DecIndx].strip() == '':
                patchDec = [0.0]
            else:
                patchDec = Dec2Angle(colLines[DecIndx].strip())
            metaDict[patchName] = [patchRA[0], patchDec[0]]
        return None, metaDict

    while len(colLines) < len(colNames):
        colLines.append(' ')

    return ','.join(colLines), metaDict


def RA2Angle(RA):
    """
    Returns Angle objects for input RA values.

    Parameters
    ----------
    RA : str, float or list of str, float
        Values of RA to convert. Can be strings in makesourcedb format or floats
        in degrees.

    Returns
    -------
    RAAngle : astropy.coordinates.Angle object

    """
    import astropy.units as u

    if type(RA) is not list:
        RA = [RA]

    if type(RA[0]) is str:
        try:
            RAAngle = Angle(Angle(RA, unit=u.hourangle), unit=u.deg)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            raise ValueError('RA not understood (must be string in '
                'makesourcedb format or float in degrees): {0}'.format(e.message))
    else:
        RAAngle = Angle(RA, unit=u.deg)

    return RAAngle


def Dec2Angle(Dec):
    """
    Returns Angle objects for input Dec values.

    Parameters
    ----------
    Dec : str, float or list of str, float
        Values of Dec to convert. Can be strings in makesourcedb format or floats
        in degrees

    Returns
    -------
    DecAngle : astropy.coordinates.Angle object

    """
    import astropy.units as u

    if type(Dec) is not list:
        Dec = [Dec]

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
                    'makesourcedb format or float in degrees): {0}'.format(e.message))
        except Exception as e:
            raise ValueError('Dec not understood (must be string in '
                'makesourcedb format or float in degrees): {0}'.format(e.message))
    else:
        DecAngle = Angle(Dec, unit=u.deg)

    return DecAngle


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
            gRAStr = Angle(gRA, unit='degree').to_string(unit='hourangle', sep=':')
            gDecStr = Angle(gDec, unit='degree').to_string(unit='degree', sep='.')

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
        d = row[colKey]
        if np.all(d == -9999):
            dstr = ' '
        else:
            defaultVal = allowedColumnDefaults[colName.lower()]
            if colName in metaDict:
                fillVal = metaDict[colName]
                hasfillVal = True
            else:
                fillVal = defaultVal
                hasfillVal = False
            if type(d) is np.ndarray:
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
    outLines.append('# Region file format: DS9 version 4.0\nglobal color=green '\
                           'font="helvetica 10 normal" select=1 highlite=1 edit=1 '\
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
            a = row['MajorAxis'] / 3600.0 # deg
            b = row['MinorAxis'] / 3600.0 # deg
            pa = row['Orientation'] # degree

            # ds9 can't handle 1-D Gaussians, so make sure they are 2-D
            if a < 1.0 / 3600.0:
                a = 1.0 / 3600.0 # deg
            if b < 1.0 / 3600.0:
                b = 1.0 / 3600.0 # deg
            stype = 'GAUSSIAN'
            region = 'ellipse({0}, {1}, {2}, {3}, {4}) # text={{{5}}}\n'.format(ra,
                dec, a, b, pa+90.0, name)
        else:
            stype = 'POINT'
            region = 'point({0}, {1}) # point=cross width=2 text={{{2}}}\n'.format(ra,
                dec, name)
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
            a = row['MajorAxis'] / 3600.0 # degree
            b = row['MinorAxis'] / 3600.0 # degree
            pa = row['Orientation'] # degree
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

    minSize = 10.0 / 3600.0 # min size in degrees
    for row in table:
        ra = row['Ra']
        dec = row['Dec']
        name = row['Name']

        if row['Type'].lower() == 'gaussian':
            a = row['MajorAxis'] / 3600.0 # degree
            if a < minSize:
                a = minSize
            b = row['MinorAxis'] / 3600.0 # degree
            if b < minSize:
                b = minSize

            pa = row['Orientation'] # degree
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
            gRA = Angle(0.0)
            gDec = Angle(0.0)
        outLines.append('{0} {1},{2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} '
            '{13} {14}\n'.format(patchName,
            gRA.to_string(unit='hourangle', sep='hms'), gDec.to_string(sep='dms'),
            'empty', 'empty', 0, 0, 0, 'LD', 'empty', 'empty', 'empty', False,
            size, flux))

    regionFile.writelines(outLines)
    regionFile.close()


def broadcastTable(fileName):
    """
    Sends a table via SAMP.

    Parameters
    ----------
    fileName : str
        Name of sky model file to broadcast

    """
    from astropy.vo.samp import SAMPHubServer, SAMPIntegratedClient, SAMPHubError
    import urlparse

    client = SAMPIntegratedClient()
    client.connect()

    params = {}
    params["url"] = urlparse.urljoin('file:', os.path.abspath(fileName))
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
        Name of VO service to query (must be one of 'WENSS' or 'NVSS')
    position : list of floats
        A list specifying a new position as [RA, Dec] in either makesourcedb
        format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
        [123.2312, 23.3422])
    radius : float or str, optional
        Radius in degrees (if float) or 'value unit' (if str; e.g.,
        '30 arcsec') for cone search region in degrees
    """
    import pyvo as vo

    log = logging.getLogger('LSMTool.Load')

    # Define allowed cone-search databases. These are the ones we know how to
    # convert to makesourcedb-formated sky models.
    columnMapping = {
        'nvss':{'NVSS':'name', 'RAJ2000':'ra', 'DEJ2000':'dec', 'S1.4':'i',
            'MajAxis':'majoraxis', 'MinAxis':'minoraxis', 'referencefrequency':1.4e9},
        'wenss':{'Name':'name', 'RAJ2000':'ra', 'DEJ2000':'dec', 'Sint':'i',
            'MajAxis':'majoraxis', 'MinAxis':'minoraxis', 'PA':'orientation',
            'referencefrequency':325e6}
        }

    if VOService.lower() in allowedVOServices:
        url = allowedVOServices[VOService.lower()]
    else:
        raise ValueError('VO query service not known. Allowed services are: '
            '{0}'.format(allowedVOServices.keys()))

    # Get raw VO catalog
    log.debug('Querying VO service...')
    try:
        position = [RA2Angle(position[0])[0].value, Dec2Angle(position[1])[0].value]
    except TypeError:
        raise ValueError('VO query positon not understood.')
    try:
        radius = Angle(radius, unit='degree').value
    except TypeError:
        raise ValueError('VO query radius not understood.')
    VOcatalog = vo.conesearch(url, position, radius=radius)

    log.debug('Creating table...')
    try:
        table = Table.read(VOcatalog.votable)
    except IndexError:
        # Empty query result
        log.error('No sources found. Sky model is empty.')
        table = makeEmptyTable()
        return table

    # Remove unneeded columns
    colsToRemove = []
    for colName in table.colnames:
        if colName not in columnMapping[VOService.lower()]:
            colsToRemove.append(colName)
        elif columnMapping[VOService.lower()][colName] not in allowedColumnNames:
            colsToRemove.append(colName)
    for colName in colsToRemove:
        table.remove_column(colName)

    # Rename columns to match makesourcedb conventions
    for colName in table.colnames:
        if colName != allowedColumnNames[columnMapping[VOService.lower()][colName]]:
            table.rename_column(colName, allowedColumnNames[columnMapping[
                VOService.lower()][colName]])

    # Convert RA and Dec to Angle objects
    log.debug('Converting RA...')
    RARaw = table['Ra'].data.tolist()
    RACol = Column(name='Ra', data=RA2Angle(RARaw))
    def raformat(val):
        return Angle(val, unit='degree').to_string(unit='hourangle', sep=':')
    RACol.format = raformat
    RAIndx = table.keys().index('Ra')
    table.remove_column('Ra')
    table.add_column(RACol, index=RAIndx)

    log.debug('Converting Dec...')
    DecRaw = table['Dec'].data.tolist()
    DecCol = Column(name='Dec', data=Dec2Angle(DecRaw))
    def decformat(val):
        return Angle(val, unit='degree').to_string(unit='degree', sep='.')
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


    # Add source-type column
    types = ['POINT'] * len(table)
    if 'majoraxis' in columnMapping[VOService.lower()].values():
        for i, maj in enumerate(table[allowedColumnNames['majoraxis']]):
            if maj > 0.0:
                types[i] = 'GAUSSIAN'
    col = Column(name='Type', data=types, dtype='{}100'.format(numpy_type))
    table.add_column(col, index=1)

    # Add reference-frequency column
    refFreq = columnMapping[VOService.lower()]['referencefrequency']
    col = Column(name='ReferenceFrequency', data=np.array([refFreq]*len(table), dtype=np.float))
    table.add_column(col)

    # Set column units and default values
    def fluxformat(val):
        return '{0:0.3f}'.format(val)
    for i, colName in enumerate(table.colnames):
        log.debug("Setting units for column '{0}' to {1}".format(
            colName, allowedColumnUnits[colName.lower()]))
        if colName == 'I':
            table.columns[colName].unit = 'mJy'
            table.columns[colName].convert_unit_to('Jy')
            table.columns[colName].format = fluxformat
        else:
            table.columns[colName].unit = allowedColumnUnits[colName.lower()]

        if hasattr(table.columns[colName], 'filled') and allowedColumnDefaults[colName.lower()] is not None:
            fillVal = allowedColumnDefaults[colName.lower()]
            if colName == 'SpectralIndex':
                while len(fillVal) < 1:
                    fillVal.append(0.0)
            log.debug("Setting default value for column '{0}' to {1}".
                format(colName, fillVal))
            table.columns[colName].fill_value = fillVal

    return table


def getTGSS(position, radius):
    """
    Returns the file name from a TGSS search.

    Parameters
    ----------
    position : list of floats
        A list specifying a new position as [RA, Dec] in either makesourcedb
        format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
        [123.2312, 23.3422])
    radius : float or str, optional
        Radius in degrees (if float) or 'value unit' (if str; e.g.,
        '30 arcsec') for cone search region in degrees

    """
    import tempfile
    import subprocess

    log = logging.getLogger('LSMTool.Load')

    outFile = tempfile.NamedTemporaryFile()
    RA = RA2Angle(position[0])[0].value
    Dec = Dec2Angle(position[1])[0].value
    try:
        radius = Angle(radius, unit='degree').value
    except TypeError:
        raise ValueError('TGSS query radius not understood.')

    url = 'http://tgssadr.strw.leidenuniv.nl/cgi-bin/gsmv3.cgi?coord={0},{1}&radius={2}&unit=deg&deconv=y'.format(
          RA, Dec, radius)
    cmd = ['wget', '-O', outFile.name, url]
    subprocess.call(cmd)

    return outFile


def getGSM(position, radius):
    """
    Returns the file name from a GSM search.

    Parameters
    ----------
    position : list of floats
        A list specifying a new position as [RA, Dec] in either makesourcedb
        format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
        [123.2312, 23.3422])
    radius : float or str, optional
        Radius in degrees (if float) or 'value unit' (if str; e.g.,
        '30 arcsec') for cone search region in degrees

    """
    import tempfile
    import subprocess

    log = logging.getLogger('LSMTool.Load')

    outFile = tempfile.NamedTemporaryFile()
    RA = RA2Angle(position[0])[0].value
    Dec = Dec2Angle(position[1])[0].value
    try:
        radius = Angle(radius, unit='degree').value
    except TypeError:
        raise ValueError('GSM query radius not understood.')

    url = 'https://lcs165.lofar.eu/cgi-bin/gsmv1.cgi?coord={0},{1}&radius={2}&unit=deg&deconv=y'.format(
          RA, Dec, radius)
    cmd = ['wget', '-O', outFile.name, url]
    subprocess.call(cmd)

    return outFile


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

