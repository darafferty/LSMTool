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
from astropy.io import registry
import numpy as np
import re
import logging
import os


inputColumnNames = {'name':'Name', 'type':'Type', 'patch':'Patch',
    'ra':'RA-HMS', 'dec':'Dec-DMS', 'i':'I', 'q':'Q', 'u':'U', 'v':'V',
    'majoraxis':'MajorAxis', 'minoraxis':'MinorAxis',
    'orientation':'Orientation', 'referencefrequency':'ReferenceFrequency',
    'spectralindex':'SpectralIndex'}

outputColumnNames = {'name':'Name', 'type':'Type', 'patch':'Patch',
    'ra-hms':'Ra', 'dec-dms':'Dec', 'i':'I', 'q':'Q', 'u':'U', 'v':'V',
    'majoraxis':'MajorAxis', 'minoraxis':'MinorAxis',
    'orientation':'Orientation', 'referencefrequency':'ReferenceFrequency',
    'spectralindex':'SpectralIndex'}

allowedColumnNames = {'name':'Name', 'type':'Type', 'patch':'Patch',
    'ra':'RA', 'dec':'Dec', 'ra-hms':'RA-HMS', 'dec-dms':'Dec-DMS', 'i':'I',
    'i-apparent':'I-Apparent', 'q':'Q', 'u':'U', 'v':'V', 'majoraxis':'MajorAxis',
    'minoraxis':'MinorAxis', 'orientation':'Orientation',
    'referencefrequency':'ReferenceFrequency', 'spectralindex':'SpectralIndex'}

allowedColumnUnits = {'name':None, 'type':None, 'patch':None, 'ra':'degree',
    'dec':'degree', 'ra-hms':None, 'dec-dms':None, 'i':'Jy', 'i-apparent':'Jy',
    'q':'Jy', 'u':'Jy', 'v':'Jy', 'majoraxis':'arcsec', 'minoraxis':'arcsec',
    'orientation':'degree', 'referencefrequency':'Hz', 'spectralindex':None}

allowedColumnDefaults = {'name':'N/A', 'type':'N/A', 'patch':'N/A', 'ra':0.0,
    'dec':0.0, 'ra-hms':'N/A', 'dec-dms': 'N/A', 'i':0.0, 'q':0.0, 'u':0.0,
    'v':0.0, 'majoraxis':0.0, 'minoraxis':0.0, 'orientation':0.0,
    'referencefrequency':0.0, 'spectralindex':0.0}


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
        if line.startswith("FORMAT") or line.startswith("format"):
            formatString = line
    modelFile.close()
    if formatString is None:
        # Look for old version of format
        modelFile = open(fileName)
        for l, line in enumerate(modelFile):
            line.strip()
            if line.startswith("#") and "format" in line and 'Name' in line:
                formatString = line
        modelFile.close()
        if formatString is not None:
            # Change old version to new one
            formatString = '='.join(formatString.split('=')[:-1])
            formatString = formatString.strip('#() ')
            formatString.replace('SpectralIndex:0', 'SpectralIndex')
            parts = formatString.split(',')
            for i, part in enumerate(parts[:]):
                if 'SpectralIndexDegree' in part:
                    parts.pop(i)
            formatString = 'FORMAT = ' + ','.join(parts)
        else:
            raise Exception("File '{0}' does not appear to be a valid makesourcedb "
                'sky model (no format line found).'.format(fileName))

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

        if colName not in inputColumnNames:
            raise Exception('Column name "{0}" found in file {1} is not a valid column'.format(colName, fileName))
        else:
            colNames[i] = inputColumnNames[colName]
            if defaultVal is not None:
                colDefaults[i] = defaultVal
                metaDict[colNames[i]] = defaultVal
            elif allowedColumnDefaults[colName] is not None:
                colDefaults[i] = allowedColumnDefaults[colName]

    # Read model into astropy table object
    modelFile = open(fileName)
    lines = modelFile.readlines()
    outlines = []
    lenSI = 0
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
                if len(c.split(',')) > lenSI:
                    lenSI = len(c.split(','))
                c = c.replace(',', ';')
            line = line.replace(b, c)
        colLines = line.split(',')

        # Check for patch lines. If found, store patch positions in the table
        # meta data
        if colLines[0] == '':
            if len(colLines) > 4:
                patchName = colLines[2].strip()
                patchRA = convertRAdeg(colLines[3].strip())
                patchDec = convertDecdeg(colLines[4].strip())
                metaDict[patchName] = [patchRA[0], patchDec[0]]
            continue

        while len(colLines) < len(colNames):
            colLines.append(' ')
        outlines.append(','.join(colLines))
    modelFile.close()

    table = Table.read('\n'.join(outlines), guess=False, format='ascii.no_header', delimiter=',',
        names=colNames, comment='#', data_start=0)

    # Convert RA and Dec columns to degrees
    RADeg = convertRAdeg(table['RA-HMS'].tolist())
    RACol = Column(name='RA', data=RADeg, unit='degree')
    DecDeg = convertDecdeg(table['Dec-DMS'].tolist())
    DecCol = Column(name='Dec', data=DecDeg, unit='degree')
    RAIndx = table.index_column('RA-HMS')
    table.add_column(RACol, index=RAIndx+2)
    table.add_column(DecCol, index=RAIndx+3)

    # Convert spectral index values from strings to arrays.
    if 'SpectralIndex' in table.keys():
        specOld = table['SpectralIndex'].data.tolist()
        specNew = []
        for spec in specOld:
            while len(spec.split(';')) < lenSI:
                spec += '; '
            specNew.append(spec)
        SItable = Table.read('\n'.join(specNew), guess=False, format='ascii.no_header', delimiter=';', comment='#', data_start=0)

        toStack = []
        for k in SItable.keys():
            toStack.append(SItable[k].data)
        specVec = np.ma.dstack(toStack)

#         specVec = []
#         maskVec = []
#         for l in specOld:
#             try:
#                 specEntry = [float(f) for f in l.split(';')]  # np.fromstring(str(l), dtype=float, sep=';')
#                 specVec.append(specEntry)
#                 maskVec.append([False]*len(specEntry))
#             except:
#                 specVec.append([0])
#                 maskVec.append([True])
#         specCol = Column(name='SpectralIndex', data=np.ma.array(specVec, mask=maskVec))

        specCol = Column(name='SpectralIndex', data=specVec[0])
        specIndx = table.keys().index('SpectralIndex')
        table.remove_column('SpectralIndex')
        table.add_column(specCol, index=specIndx)

    # Set column units and default values
    for i, colName in enumerate(colNames):
        if colName == 'RA-HMS' or colName == 'Dec-DMS':
            continue
        logging.debug("Setting units for column '{0}' to {1}".format(
            colName, allowedColumnUnits[colName.lower()]))
        table.columns[colName].unit = allowedColumnUnits[colName.lower()]

        if hasattr(table.columns[colName], 'filled') and colDefaults[i] is not None:
            logging.debug("Setting default value for column '{0}' to {1}".
                format(colName, colDefaults[i]))
            table.columns[colName].fill_value = colDefaults[i]
    table.meta = metaDict

    # Group by patch name
    if hasPatches:
        table = table.group_by('Patch')

    return table


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


def skyModelWriter(table, fileName, groupByPatch=False):
    """
    Writes table to a makesourcedb sky model file.

    Parameters
    ----------
    fileName : str
        Output ASCII file to which the sky model is written.
    groupByPatch: bool, optional
        If True, group lines by patch (can be slow). This option does not
        affect the sources or patches in the sky model, only the order in which
        they are written.
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
        if colKey.lower() not in outputColumnNames:
            continue
        colName = outputColumnNames[colKey.lower()]

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
            gRAStr = convertRAHHMMSS(gRA)
            gDecStr = convertDecDDMMSS(gDec)
            outLines.append(' , , {0}, {1}, {2}\n'.format(patchName, gRAStr,
                gDecStr))
            if groupByPatch:
                g = table.groups[i]
                for row in g.filled(fill_value=-9999):
                    line = rowStr(row)
                    outLines.append(', '.join(line))
                    outLines.append('\n')
        if not groupByPatch:
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
            colName = outputColumnNames[colKey.lower()]
        except KeyError:
            continue
        d = row[colKey]
        if np.any(d == -9999):
            dstr = ' '
        else:
            if type(d) is np.ndarray:
                dstr = str(d.tolist())
            else:
                dstr = str(d)
        line.append('{0}'.format(dstr))

    while line[-1] == ' ':
        line.pop()
    return line


def convertRAdeg(raStr):
    """
    Takes makesourcedb string of RA and returns RA in degrees.

    Parameters
    ----------
    raStr : str
        RA string such as '12:23:31.232'
    """
    if type(raStr) is float:
        return raStr
    if type(raStr) is str:
        raStr = [raStr]
    has_non_str = False
    if type(raStr) is list:
        for ra in raStr:
            if type(ra) is not str:
                has_non_str = True
    if type(raStr) is not list or has_non_str:
        logging.error('Input must be a string or a list of strings.')
        return

    raDeg = []
    for ra in raStr:
        raSrc = ra.split(':')
        raDeg.append(float(raSrc[0])*15.0 + (float(raSrc[1])/60.0)*15.0 + (float(raSrc[2])
            /3600.0)*15.0)
    return np.array(raDeg)


def convertDecdeg(decStr):
    """
    Takes makesourcedb string of Dec and returns Dec in degrees.

    Parameters
    ----------
    decStr : str
        Dec string such as '12.23.31.232'
    """
    if type(decStr) is float:
        return decStr
    if type(decStr) is str:
        decStr = [decStr]
    has_non_str = False
    if type(decStr) is list:
        for dec in decStr:
            if type(dec) is not str:
                has_non_str = True
    if type(decStr) is not list or has_non_str:
        logging.error('Input must be a string or a list of strings.')
        return

    decDeg = []
    for dec in decStr:
        decSrc = dec.split('.')
        if len(decSrc) == 3:
            decDeg.append(float(decSrc[0]) + (float(decSrc[1])/60.0) + (float(decSrc[2])
                /3600.0))
        else:
            decDeg.append(float(decSrc[0]) + (float(decSrc[1])/60.0) + (float(decSrc[2]
                + '.' + decSrc[3])/3600.0))
    return np.array(decDeg)


def convertRAHHMMSS(deg):
    """
    Convert RA coordinate (in degrees) to a makesourcedb string.

    Parameters
    ----------
    deg : float
        RA in degrees
    """
    from math import modf

    if type(deg) is str:
        return deg

    if deg < 0:
        deg += 360.0
    x, hh = modf(deg/15.)
    x, mm = modf(x*60)
    ss = x*60

    return str(int(hh)).zfill(2)+':'+str(int(mm)).zfill(2)+':'+str("%.3f" % (ss)).zfill(6)


def convertDecDDMMSS(deg):
    """
    Convert Dec coordinate (in degrees) to makesourcedb string.

    Parameters
    ----------
    deg : float
        Dec in degrees
    """
    from math import modf

    if type(deg) is str:
        return deg

    sign = (-1 if deg < 0 else 1)
    x, dd = modf(abs(deg))
    x, ma = modf(x*60)
    sa = x*60
    decsign = ('-' if sign < 0 else '+')
    return decsign+str(int(dd)).zfill(2)+'.'+str(int(ma)).zfill(2)+'.'+str("%.3f" % (sa)).zfill(6)


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
        ra = row['RA']
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
    kvisFile = open(fileName, 'w')
    logging.debug('Writing kvis annotation file to {0}'.format(fileName))

    # Make sure all columns have the correct units
    for colName in table.columns:
        units = allowedColumnUnits[colName.lower()]
        if units is not None:
            table[colName].convert_unit_to(units)

    outLines = []
    for row in table:
        ra = row['RA']
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

