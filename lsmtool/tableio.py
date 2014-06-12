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

from astropy.table import Table, Column
from astropy.coordinates import Angle
from astropy.io import registry
import numpy as np
import re
import logging
import os


allowedColumnNames = {'name':'Name', 'type':'Type', 'patch':'Patch',
    'ra':'Ra', 'dec':'Dec', 'i':'I', 'q':'Q', 'u':'U', 'v':'V',
    'majoraxis':'MajorAxis', 'minoraxis':'MinorAxis', 'orientation':'Orientation',
    'referencefrequency':'ReferenceFrequency', 'spectralindex':'SpectralIndex'}

allowedColumnUnits = {'name':None, 'type':None, 'patch':None, 'ra':'degree',
    'dec':'degree', 'i':'Jy', 'i-apparent':'Jy', 'q':'Jy', 'u':'Jy', 'v':'Jy',
    'majoraxis':'arcsec', 'minoraxis':'arcsec', 'orientation':'degree',
    'referencefrequency':'Hz', 'spectralindex':None}

allowedColumnDefaults = {'name':'N/A', 'type':'N/A', 'patch':'N/A', 'ra':'N/A',
    'dec': 'N/A', 'i':0.0, 'q':0.0, 'u':0.0, 'v':0.0, 'majoraxis':0.0,
    'minoraxis':0.0, 'orientation':0.0, 'referencefrequency':0.0,
    'spectralindex':0.0}


def skyModelReader(fileName):
    """
    Reads a makesourcedb sky model file into an astropy table

    Parameters
    ----------
    fileName : str
        Input ASCII file from which the sky model is read. Must
        respect the makesourcedb format

    Returns
    -------
    table : astropy.table.Table object
    """
    # Open the input file
    try:
        modelFile = open(fileName)
        logging.debug('Reading {0}'.format(fileName))
    except IOError:
        raise Exception('Could not open {0}'.format(fileName))

    # Read format line
    formatString = None
    for l, line in enumerate(modelFile):
        if 'format' in line.lower()
            formatString = line
            break
    modelFile.close()
    if formatString is None:
        raise Exception("File '{0}' does not appear to be a valid makesourcedb "
            'sky model (no valid format line found).'.format(fileName))
    formatString = formatString.strip('# ')
    if formatString.lower().endswith('format'):
        parts = formatString.split('=')[:-1]
        formatString = 'FORMAT = ' + '='.join(parts).strip('# ()')
    elif formatString.lower().startswith('format'):
        parts = formatString.split('=')[1:]
        formatString = 'FORMAT = ' + '='.join(parts).strip('# ()')
    else:
        raise Exception("Format line in file '{0}' not understood.".format(fileName))

    # Check whether sky model has patches
    if 'Patch' in formatString:
        hasPatches = True
    else:
        hasPatches = False

    # Get column names and default values. Non-string columns have default
    # values of 0.0 unless a different value is given in the header.
    colNames = formatString.split(',')
    colDefaults = [None] * len(colNames)
    metaDict = {}
    colNames[0] = colNames[0].split('=')[1]
    for i in range(len(colNames)):
        parts = colNames[i].split('=')
        colName = parts[0].strip().lower()
        if len(parts) == 2:
            try:
                defaultVal = float(parts[1].strip("'[]"))
            except ValueError:
                defaultVal = None
        else:
            defaultVal = None

        if colName not in allowedColumnNames:
            raise Exception('Column name "{0}" found in file {1} is not a valid column'.format(colName, fileName))
        else:
            colNames[i] = allowedColumnNames[colName]
            if defaultVal is not None:
                colDefaults[i] = defaultVal
                metaDict[colNames[i]] = defaultVal
            elif allowedColumnDefaults[colName] is not None:
                colDefaults[i] = allowedColumnDefaults[colName]

    # Read model into astropy table object
    modelFile = open(fileName)
    lines = modelFile.readlines()
    outlines = []
    for line in lines:
        if line.startswith("FORMAT") or line.startswith("format") or line.startswith("#"):
            continue

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
        if colLines[0].strip() == '':
            if len(colLines) > 4:
                patchName = colLines[2].strip()
                patchRA = RA2Angle(colLines[3].strip())
                patchDec = Dec2Angle(colLines[4].strip())
                metaDict[patchName] = [patchRA[0], patchDec[0]]
            continue

        while len(colLines) < len(colNames):
            colLines.append(' ')
        outlines.append(','.join(colLines))
    modelFile.close()

    table = Table.read('\n'.join(outlines), guess=False, format='ascii.no_header', delimiter=',',
        names=colNames, comment='#', data_start=0)

    # Convert spectral index values from strings to arrays.
    if 'SpectralIndex' in table.keys():
        specOld = table['SpectralIndex'].data.tolist()
        specVec = []
        maskVec = []
        for l in specOld:
            try:
                if type(l) is float:
                    specEntry = [l]
                else:
                    specEntry = [float(f) for f in l.split(';')][0:2]
                specMask = [False]*len(specEntry)
                if len(specEntry) < 2:
                    specEntry.append(0.0)
                    specMask.append(True)
                specVec.append(specEntry)
                maskVec.append(specMask)
            except:
                specVec.append([0.0, 0.0])
                maskVec.append([True, True])
        specCol = Column(name='SpectralIndex', data=np.ma.array(specVec, mask=maskVec, dtype=np.float))
        specIndx = table.keys().index('SpectralIndex')
        table.remove_column('SpectralIndex')
        table.add_column(specCol, index=specIndx)

    # Convert RA and Dec to Angle objects
    RARaw = table['Ra'].data.tolist()
    RACol = Column(name='Ra', data=RA2Angle(RARaw))
    def raformat(val):
        return Angle(val, unit='degree').to_string(unit='hourangle', sep=':')
    RACol.format = raformat
    RAIndx = table.keys().index('Ra')
    table.remove_column('Ra')
    table.add_column(RACol, index=RAIndx)

    DecRaw = table['Dec'].data.tolist()
    DecCol = Column(name='Dec', data=Dec2Angle(DecRaw))
    def decformat(val):
        return Angle(val, unit='degree').to_string(unit='degree', sep='.')
    DecCol.format = decformat
    DecIndx = table.keys().index('Dec')
    table.remove_column('Dec')
    table.add_column(DecCol, index=DecIndx)

    # Set column units and default values
    for i, colName in enumerate(colNames):
        logging.debug("Setting units for column '{0}' to {1}".format(
            colName, allowedColumnUnits[colName.lower()]))
        table.columns[colName].unit = allowedColumnUnits[colName.lower()]

        if hasattr(table.columns[colName], 'filled') and colDefaults[i] is not None:
            logging.debug("Setting default value for column '{0}' to {1}".
                format(colName, colDefaults[i]))
            table.columns[colName].fill_value = colDefaults[i]
    table.meta = metaDict

    return table


def RA2Angle(RA):
    """
    Returns Angle objects for input RA values.
    """
    if type(RA) is str or type(RA) is float:
        RA = [RA]

    if type(RA[0]) is str:
        RAStr = []
        for ras in RA:
            RAStr.append(ras+' hours')
        RAAngle = Angle(RAStr, unit='degree')
    else:
        RAAngle = Angle(RA, unit='degree')

    return RAAngle


def Dec2Angle(Dec):
    """
    Returns Angle objects for input Dec values.
    """
    if type(Dec) is str or type(Dec) is float:
        Dec = [Dec]

    if type(Dec[0]) is str:
        DecStr = []
        for decs in Dec:
            DecStr.append(decs.replace('.', ':', 2))
        DecAngle = Angle(DecStr, unit='degree')
    else:
        DecAngle = Angle(Dec, unit='degree')

    return DecAngle


def skyModelIdentify(origin, *args, **kwargs):
    """
    Identifies valid makesourcedb sky model files.
    """
    # Search for a format line. If found, assume file is valid
    if isinstance(args[0], basestring):
        f = open(args[0])
    elif isinstance(args[0], file):
        f = args[0]
    else:
        return False
    for line in f:
        if line.startswith("FORMAT") or line.startswith("format"):
            return True
    return False


def skyModelWriter(table, fileName):
    """
    Writes table to a makesourcedb sky model file.

    Parameters
    ----------
    fileName : str
        Output ASCII file to which the sky model is written.
    """
    modelFile = open(fileName, 'w')
    logging.debug('Writing model to {0}'.format(fileName))

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

        if colName in table.meta:
            if colName == 'SpectralIndex':
                colHeader = "{0}='[{1}]'".format(colName, table.meta[colName])
            else:
                colHeader = "{0}='{1}'".format(colName, table.meta[colName])
        elif colName == 'SpectralIndex':
            colHeader = "{0}='[]'".format(colName)
        else:
            colHeader = colName
        formatString.append(colHeader)
    outLines.append('FORMAT = {0}'.format(', '.join(formatString)))
    outLines.append('\n')
    outLines.append('\n')

    # Add source lines
    if 'Patch' in table.keys():
        table = table.group_by('Patch')
        patchNames = table.groups.keys['Patch']
        for i, patchName in enumerate(patchNames):
            if patchName in table.meta:
                gRA, gDec = table.meta[patchName]
            else:
                gRA = 0.0
                gDec = 0.0
            gRAStr = Angle(gRA, unit='degree').to_string(unit='hourangle', sep=':')
            gDecStr = Angle(gDec, unit='degree').to_string(unit='degree', sep='.')

            outLines.append(' , , {0}, {1}, {2}\n'.format(patchName, gRAStr,
                gDecStr))
        for row in table.filled(fill_value=-9999):
            line = rowStr(row)
            outLines.append(', '.join(line))
            outLines.append('\n')
    else:
        for row in table.filled(fill_value=-9999):
            line = rowStr(row)
            outLines.append(', '.join(line))
            outLines.append('\n')

    modelFile.writelines(outLines)
    modelFile.close()


def rowStr(row):
    """
    Returns makesourcedb representation of a row

    Parameters
    ----------
    row : astropy.table.Row object
    """
    line = []
    for colKey in row.columns:
        try:
            colName = allowedColumnNames[colKey.lower()]
        except KeyError:
            continue
        d = row[colKey]
        if np.any(d == -9999):
            dstr = ' '
        else:
            if type(d) is np.ndarray:
                dstr = str(d.tolist())
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
        Output ASCII file to which the sky model is written.
    """
    from operations_lib import convertRAdeg, convertDecdeg

    regionFile = open(fileName, 'w')
    logging.debug('Writing ds9 region file to {0}'.format(fileName))

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
            a = row['MajorAxis'] # arcsec
            b = row['MinorAxis'] # arcsec
            pa = row['Orientation'] # degree

            # ds9 can't handle 1-D Gaussians, so make sure they are 2-D
            if a < 1.0/3600.0: a = 1.0 # arcsec
            if b < 1.0/3600.0: b = 1.0 # arcsec
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
        Output ASCII file to which the sky model is written.
    """
    from operations_lib import convertRAdeg, convertDecdeg

    kvisFile = open(fileName, 'w')
    logging.debug('Writing kvis annotation file to {0}'.format(fileName))

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
            a = row['MajorAxis']/3600.0 # degree
            b = row['MinorAxis']/3600.0 # degree
            pa = row['Orientation'] # degree
            outLines.append('ELLIPSE W {0} {1} {2} {3} {4}\n'.format(ra, dec, a, b, pa))
        else:
            outLines.append('CIRCLE W {0} {1} 0.02\n'.format(ra, dec))
        outLines.append('TEXT W {0} {1} {2}\n'.format(ra - 0.07, dec, name))

    kvisFile.writelines(outLines)
    kvisFile.close()


# Register the file reader, identifier, and writer functions with astropy.io
registry.register_reader('makesourcedb', Table, skyModelReader)
registry.register_identifier('makesourcedb', Table, skyModelIdentify)
registry.register_writer('makesourcedb', Table, skyModelWriter)
registry.register_writer('ds9', Table, ds9RegionWriter)
registry.register_writer('kvis', Table, kvisAnnWriter)

