# -*- coding: utf-8 -*-
#
# This module defines the SkyModel object used by LSMTool for all operations.
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

import logging
from . import _logging
from . import tableio
from . import operations

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


class SkyModel(object):
    """
    Object that stores the sky model and provides methods for accessing it.
    """
    def __init__(self, fileName, beamMS=None, checkDup=False, VOPosition=None,
                 VORadius=None):
        """
        Initializes SkyModel object.

        Parameters
        ----------
        fileName : str
            Input ASCII file from which the sky model is read (must respect the
            makesourcedb format), name of VO service to query (must be one of
            'WENSS', 'NVSS', 'TGSS', or 'GSM'), or dict (single source only)
        beamMS : str, optional
            Measurement set from which the primary beam will be estimated. A
            column of attenuated Stokes I fluxes will be added to the table
        checkDup: bool, optional
            If True, the sky model is checked for duplicate sources (with the
            same name)
        VOPosition : list of floats
            A list specifying a new position as [RA, Dec] in either makesourcedb
            format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
            [123.2312, 23.3422]) for a cone search
        VORadius : float or str, optional
            Radius in degrees (if float) or 'value unit' (if str; e.g.,
            '30 arcsec') for cone search region in degrees

        Examples
        --------
        Create a SkyModel object::

            >>> s = SkyModel('sky.model')

        Create a SkyModel object with a beam MS so that apparent fluxes will
        be available::

            >>> s = SkyModel('sky.model', beamMS='SB100.MS')

        Load a WENSS catalog into a SkyModel object::

            >>> s = SkyModel('WENSS', VOPosition=[212.8352792, 52.202644],
                VOradius=5.0)

        """
        from astropy.table import Table
        from .tableio import processFormatString, processLine, createTable

        self.log = logging.getLogger('LSMTool')
        self.history = []
        if type(fileName) is str:
            if fileName.lower() in tableio.allowedVOServices:
                self.log.debug("Attempting to load model from VO service '{0}'...".format(fileName))
                self.table = tableio.coneSearch(fileName, VOPosition, VORadius)
                self.log.debug("Successfully loaded model from VO service '{0}'".format(fileName))
                self._fileName = None
                self._addHistory("LOAD (from {0} at position {1})".format(fileName, VOPosition))
            elif fileName.lower() == 'tgss':
                self.log.debug("Attempting to load model from TGSS...")
                fileObj = tableio.getTGSS(VOPosition, VORadius)
                self.table = Table.read(fileObj.name, format='makesourcedb')
                fileObj.close()
                self.log.debug("Successfully loaded model from TGSS")
                self._fileName = None
                self._addHistory("LOAD (from TGSS at position {0})".format(VOPosition))
            elif fileName.lower() == 'gsm':
                self.log.debug("Attempting to load model from GSM...")
                fileObj = tableio.getGSM(VOPosition, VORadius)
                self.table = Table.read(fileObj.name, format='makesourcedb')
                fileObj.close()
                self.log.debug("Successfully loaded model from GSM")
                self._fileName = None
                self._addHistory("LOAD (from GSM at position {0})".format(VOPosition))
            else:
                self.log.debug("Attempting to load model from file '{0}'...".format(fileName))
                self.table = Table.read(fileName, format='makesourcedb')
                self.log.debug("Successfully loaded model from file '{0}'".format(fileName))
                self._fileName = fileName
                self._addHistory("LOAD (from file '{0}')".format(fileName))
        elif type(fileName) is dict:
            self.log.debug("Attempting to create model from input dict...")
            # Create header
            formatString = '#FORMAT = ' + ', '.join(fileName.keys())

            # Process the header
            colNames, hasPatches, colDefaults, metaDict = processFormatString(formatString)

            # Process the model
            outlines = []
            stringValues = ['{0}'.format(v) for v in fileName.values()]
            line = ', '.join(stringValues)
            outline, metaDict = processLine(line, metaDict, colNames)
            if outline is not None:
                outlines.append(outline)
            outlines.append('\n')  # needed in case of single-line sky models
            table = createTable(outlines, metaDict, colNames, colDefaults)
            self.table = table
            self.log.debug("Successfully created model from input dict")
            self._fileName = None
            self._addHistory("LOAD (from input dict)")
        else:
            raise ValueError("Filename not understood. Exiting...")

        if beamMS is not None:
            self.beamMS = beamMS
            self._hasBeam = True
            self.beamTime = 0.5
        else:
            self.beamMS = None
            self._hasBeam = False
            self.beamTime = None

        if checkDup:
            self.log.debug('Checking model for duplicate lines...')
            self._clean()

        self.log.debug('Processing patches (if any)...')
        self._patchMethod = None
        self._updateGroups()

        self.log.debug("Successfully loaded sky model")

    def __len__(self):
        """
        Returns the table len() value (number of rows).
        """
        return self.table.__len__()

    def __str__(self):
        """
        Returns string with info about sky model contents.
        """
        return self.table.__str__()

    def _updateGroups(self):
        """
        Updates the grouping of the table by patch name.
        """
        if 'Patch' in self.table.keys():
            self.table = self.table.group_by('Patch')
            self.hasPatches = True
        else:
            self.hasPatches = False

    def _addHistory(self, entry=""):
        """
        Adds entry to the history with current date and time

        Parameters
        ----------
        entry : str, optional
            String to add to history

        """
        import datetime
        current_time = str(datetime.datetime.now()).split('.')[0]
        self.history.append(current_time + ": " + str(entry))

    def _info(self, useLogInfo=False):
        """
        Prints information about the sky model.
        """
        import numpy as np

        if self.hasPatches:
            nPatches = len(set(self.getPatchNames()))
        else:
            nPatches = 0

        nPoint = len(np.where(self.getColValues('Type') == 'POINT')[0])
        nGaus = len(np.where(self.getColValues('Type') == 'GAUSSIAN')[0])

        if nPatches == 1:
            plur = ''
        else:
            plur = 'es'
        if useLogInfo:
            logCall = self.log.info
        else:
            logCall = self.log.debug

        x, y, refRA, refDec = self._getXY()
        totFlux = np.sum(self.getColValues('I', units='Jy'))

        info = 'Model contains {0} sources in {1} patch{2} of which:\n'\
               '      {3} are type POINT\n'\
               '      {4} are type GAUSSIAN\n'\
               '      Associated beam MS: {5}\n'\
               '      Approximate RA, Dec of center: {6}, {7}\n'\
               '      Total flux: {8} Jy\n\n'\
               '      History:\n'\
               '      {9}'.format(len(self.table), nPatches, plur,
                                  nPoint, nGaus, self.beamMS, refRA, refDec, totFlux,
                                  '\n      '.join(self.history))
        logCall(info)
        return info

    def info(self):
        """
        Prints information about the sky model.
        """
        infoStr = self._info(useLogInfo=True)

    def copy(self):
        """
        Returns a copy of the sky model.
        """
        import copy

        # The logger's stream handlers are not copyable with deepcopy, so copy
        # them by hand:
        self.log = None
        LSMCopy = copy.deepcopy(self)
        LSMCopy.log = logging.getLogger('LSMTool')
        LSMCopy._addHistory('COPY')
        self.log = logging.getLogger('LSMTool')

        return LSMCopy

    def more(self, colName=None, patchName=None, sourceName=None, sortBy=None,
             lowToHigh=False):
        """
        Prints the sky model table to the screen with more-like commands.

        Parameters
        ----------
        colName : str, list of str, optional
            Name of column or columns to print. If None, all columns are printed
        patchName : str, list of str, optional
            If given, returns column values for specified patch or patches only
        sourceName : str, list of str, optional
            If given, returns column value for specified source or sources only
        sortBy : str or list of str, optional
            Name of columns to sort on. If None, no sorting is done. If
            a list is given, sorting is done on the columns in the order given
        lowToHigh : bool, optional
            If True, sort values from low to high instead of high to low

        Examples
        --------
        Print the entire model::

            >>> s.more()

        Print only the 'Name' and 'I' columns for the 'bin0' patch::

            >>> s.more(['Name', 'I'], 'bin0', sortBy=['I'])

        """
        if patchName is not None and sourceName is not None:
            raise ValueError('patchName and sourceName cannot both be specified.')

        table = self.table

        # Get columns
        colName = self._verifyColName(colName)
        if colName is not None:
            if type(colName) is str:
                colName = [colName]  # needed in order to get a table instead of a column
            table = table[colName]

        # Get patches
        if patchName is not None:
            pindx = self._getNameIndx(patchName, patch=True)
            if pindx is not None:
                table = table.groups[pindx]

        # Get sources
        if sourceName is not None:
            sindx = self._getNameIndx(sourceName)
            if sindx is not None:
                table = table[sindx]

        # Sort if desired
        if sortBy is not None:
            colName = self._verifyColName(sortBy)
            indx = table.argsort(colName)
            if not lowToHigh:
                indx = indx[::-1]
            table = table[indx]

        table.more(show_unit=True)

    def _verifyColName(self, colName, onlyExisting=True, applyBeam=False,
                       quiet=False):
        """
        Verifies that column(s) exist and returns correctly formatted string or
        list of strings suitable for accessing the data table.

        Parameters
        ----------
        colName : str, list of str
            Name of column or columns
        onlyExisting : bool, optional
            If True, only columns that exist in the table are allowed. If False,
            columns that are valid but do not exist in the table are returned
            without error.
        applyBeam : bool, optional
            If True and colName = 'I', the attenuated Stokes I column will be
            returned
        quiet : bool, optional
            If True, errors will be suppressed

        Returns
        -------
        colName : str, None
            Properly formatted name of column or None if colName not found

        """
        if type(colName) is str:
            colNameLower = colName.lower()
            if colNameLower not in tableio.allowedColumnNames:
                if not quiet:
                    raise ValueError('Column name "{0}" is not a valid makesourcedb '
                                     'column.'.format(colName))
                return None
            else:
                colNameKey = tableio.allowedColumnNames[colNameLower]
            if colNameKey not in self.table.keys() and onlyExisting:
                if not quiet:
                    raise ValueError('Column name "{0}" not found in sky model.'.
                                     format(colName))
                return None

        elif type(colName) is list:
            colNameLower = [c.lower() for c in colName]
            for name in colNameLower[:]:
                badNames = []
                if name not in tableio.allowedColumnNames:
                    badNames.append(name)
                    colNameLower.remove(name)
                else:
                    colNameKey = tableio.allowedColumnNames[name]
                    if colNameKey not in self.table.keys():
                        badNames.append(name)
                        colNameLower.remove(name)

            if len(badNames) > 0:
                if len(badNames) == 1:
                    plur = ''
                else:
                    plur = 's'
                if not quiet:
                    self.log.warn("Column name{0} '{1}' not recognized. Ignoring.".
                                  format(plur, ','.join(badNames)))
            if len(colNameLower) == 0:
                return None
            else:
                colNameKey = [tableio.allowedColumnNames[n] for n in colNameLower]
        else:
            colNameKey = None

        return colNameKey

    def getPatchPositions(self, patchName=None, asArray=False, method=None,
                          applyBeam=False, perPatchProjection=True):
        """
        Returns arrays or a dict of patch positions (as {'patchName':(RA, Dec)}).

        Parameters
        ----------
        patchName : str or list, optional
            List of patch names for which the positions are desired
        asArray : bool, optional
            If True, returns arrays of RA, Dec instead of a dict
        method : None or str, optional
            This parameter specifies the method used to calculate the patch
            positions. If None, the current patch positions stored in the sky
            model, if any, will be returned.
            - 'mid' => calculate the midpoint of the patch
            - 'mean' => calculate the mean RA and Dec of the patch
            - 'wmean' => calculate the flux-weighted mean RA and Dec of the patch
            - None => current patch positions are returned
            Note that the mid, mean, and wmean positions are calculated from TAN-
            projected values.
        applyBeam : bool, optional
            If True, fluxes used as weights will be attenuated by the beam.
        perPatchProjection : bool, optional
            If True, a different projection center is used per patch. If False,
            a single projection center is used for all patches.

        Returns
        -------
        positions : numpy array or dict
            (RA, Dec) arrays (if asArray is False) of patch positions or a
            dictionary of {'patchName':(RA, Dec)}.

        Examples
        --------
        Get the current patch positions::

            >>> s.getPatchPositions()
            {'bin0': [<Angle 91.77565208333331 deg>, <Angle 41.57834805555555 deg>],
             'bin1': [<Angle 91.59991874999997 deg>, <Angle 41.90387583333333 deg>],
             'bin2': [<Angle 90.83773333333332 deg>, <Angle 42.189861944444445 deg>],

        Get them as RA and Dec arrays in degrees::

            >>> s.getPatchPositions(asArray=True)
            (array([ 91.77565208,  91.59991875,  90.83773333]),
             array([ 41.57834806,  41.90387583,  42.18986194]))

        Calculate the flux-weighted mean positions of each patch::

            >>> s.getPatchPositions(method='wmean', asArray=True)

        """
        import numpy as np
        from .operations_lib import radec2xy, xy2radec
        from astropy.table import Column
        from .tableio import RA2Angle, Dec2Angle

        if self.hasPatches:
            if patchName is None:
                patchName = self.getPatchNames()
            if type(patchName) is str or type(patchName) is np.string_ or type(patchName) is np.unicode_:
                patchName = [patchName]
            if method is None:
                patchDict = {}
                for patch in patchName:
                    if patch in self.table.meta:
                        patchDict[patch] = self.table.meta[patch]
                    else:
                        patchDict[patch] = [RA2Angle(0.0)[0], Dec2Angle(0.0)[0]]
            else:
                patchDict = {}

                # Add projected x and y columns.
                if perPatchProjection:
                    # Each patch has a different projection center
                    xAll = []  # has length = num of sources
                    yAll = []
                    midRAAll = []  # has length = num of patches
                    midDecAll = []
                    for name in patchName:
                        x, y, midRA, midDec = self._getXY(patchName=name)
                        xAll.extend(x)
                        yAll.extend(y)
                        midRAAll.append(midRA)
                        midDecAll.append(midDec)
                else:
                    xAll, yAll, midRA, midDec = self._getXY()
                    midRAAll = []  # has length = num of patches
                    midDecAll = []
                    for name in patchName:
                        midRAAll.append(midRA)
                        midDecAll.append(midDec)

                xCol = Column(name='X', data=xAll)
                yCol = Column(name='Y', data=yAll)
                self.table.add_column(xCol)
                self.table.add_column(yCol)

                if method == 'mid':
                    minX = self._getMinColumn('X')
                    maxX = self._getMaxColumn('X')
                    minY = self._getMinColumn('Y')
                    maxY = self._getMaxColumn('Y')
                    midX = minX + (maxX - minX) / 2.0
                    midY = minY + (maxY - minY) / 2.0
                    for i, name in enumerate(patchName):
                        gRA = RA2Angle(xy2radec([midX[i]], [midY[i]], midRAAll[i],
                                       midDecAll[i])[0])[0]
                        gDec = Dec2Angle(xy2radec([midX[i]], [midY[i]], midRAAll[i],
                                         midDecAll[i])[1])[0]
                        patchDict[name] = [gRA, gDec]
                elif method == 'mean' or method == 'wmean':
                    if method == 'mean':
                        weight = False
                    else:
                        weight = True
                    meanX = self._getAveragedColumn('X', applyBeam=applyBeam,
                                                    weight=weight)
                    meanY = self._getAveragedColumn('Y', applyBeam=applyBeam,
                                                    weight=weight)
                    for i, name in enumerate(patchName):
                        gRA = RA2Angle(xy2radec([meanX[i]], [meanY[i]], midRAAll[i],
                                       midDecAll[i])[0])[0]
                        gDec = Dec2Angle(xy2radec([meanX[i]], [meanY[i]], midRAAll[i],
                                         midDecAll[i])[1])[0]
                        patchDict[name] = [gRA, gDec]
                self.table.remove_column('X')
                self.table.remove_column('Y')

            if asArray:
                RA = []
                Dec = []
                for patch in patchName:
                    RA.append(patchDict[patch][0].value)
                    Dec.append(patchDict[patch][1].value)
                return np.array(RA), np.array(Dec)
            else:
                return patchDict

        else:
            return None

    def setPatchPositions(self, patchDict=None, method='mid', applyBeam=False,
                          perPatchProjection=True):
        """
        Sets the patch positions.

        Parameters
        ----------
        patchDict : dict, optional
            Dict specifying patch names and positions as {'patchName':[RA, Dec]}
            where both RA and Dec are degrees J2000 or in makesourcedb format.
            If None, positions are set for all patches using the method given
            by the 'method' parameter.
        method : None or str, optional
            If no patchDict is given, this parameter specifies the method used
            to set the patch positions:
            - 'mid' => the position is set to the midpoint of the patch
            - 'mean' => the position is set to the mean RA and Dec of the patch
            - 'wmean' => the position is set to the flux-weighted mean RA and
            Dec of the patch
            - 'zero' => set all positions to [0.0, 0.0]

            Note that the mid, mean, and wmean positions are calculated from TAN-
            projected values.
        applyBeam : bool, optional
            If True, fluxes used as weights will be attenuated by the beam.
        perPatchProjection : bool, optional
            If True, a different projection center is used per patch. If False,
            a single projection center is used for all patches.

        Examples
        --------
        Set all patch positions to their (projected) midpoints::

            >>> s.setPatchPositions()

        Set all patch positions to their (projected) flux-weighted mean
        positions::

             >>> s.setPatchPositions(method='wmean')

        Set new position for the 'bin0' patch only::

            >>> s.setPatchPositions({'bin0': [123.231, 23.4321]})

        """
        from .tableio import RA2Angle, Dec2Angle

        if self.hasPatches:
            if method not in ['mid', 'mean', 'wmean', 'zero']:
                raise ValueError('Invalid method parameter')

            if patchDict is None:
                # Delete any previous patch positions
                patchNames = self.getPatchNames()
                for patchName in patchNames:
                    if patchName in self.table.meta:
                        self.table.meta.pop(patchName)
                if method == 'zero':
                    patchDict = {}
                    for n in patchNames:
                        patchDict[n] = [RA2Angle(0.0), Dec2Angle(0.0)]
                else:
                    patchDict = self.getPatchPositions(method=method, applyBeam=applyBeam,
                                                       perPatchProjection=perPatchProjection)

            for patch, pos in iteritems(patchDict):
                if type(pos[0]) is str or type(pos[0]) is float:
                    pos[0] = RA2Angle(pos[0])
                if type(pos[1]) is str or type(pos[1]) is float:
                    pos[1] = Dec2Angle(pos[1])
                self.table.meta[patch] = pos
            self._addHistory("SETPATCHPOSITIONS (method = '{0}')".format(method))
        else:
            raise RuntimeError('Sky model does not have patches.')

    def _getXY(self, patchName=None, crdelt=None, byPatch=False):
        """
        Returns lists of projected x and y values for all sources.

        Parameters
        ----------
        patchName : str, optional
            If given, return x and y for specified patch only
        crdelt: float, optional
            Delta in degrees for sky grid
        byPatch : bool, optional
            Use patches instead of by sources

        Returns
        -------
        x, y, midRA, midDec : numpy array, numpy array, float, float
            arrays of x and y values and the midpoint RA and
            Dec values

        """
        from .operations_lib import radec2xy, xy2radec
        import numpy as np

        if len(self.table) == 0:
            return [0], [0], 0, 0

        if byPatch:
            if 'Patch' not in self.table.keys():
                raise ValueError('Sky model must be grouped before "byPatch" can be used.')
            RA = self.getColValues('Ra', aggregate='wmean')
            Dec = self.getColValues('Dec', aggregate='wmean')
        else:
            RA = self.getColValues('Ra')
            Dec = self.getColValues('Dec')
            if patchName is not None:
                ind = self.getRowIndex(patchName)
                RA = RA[ind]
                Dec = Dec[ind]
        x, y = radec2xy(RA, Dec, crdelt=crdelt)

        # Refine x and y using midpoint
        if len(x) > 1:
            xmid = min(x) + (max(x) - min(x)) / 2.0
            ymid = min(y) + (max(y) - min(y)) / 2.0
            xind = np.argsort(x)
            yind = np.argsort(y)
            try:
                midxind = np.where(np.array(x)[xind] > xmid)[0][0]
                midyind = np.where(np.array(y)[yind] > ymid)[0][0]
                midRA = RA[xind[midxind]]
                midDec = Dec[yind[midyind]]
                x, y = radec2xy(RA, Dec, midRA, midDec, crdelt=crdelt)
            except IndexError:
                midRA = RA[0]
                midDec = Dec[0]
        else:
            midRA = RA[0]
            midDec = Dec[0]

        return np.array(x), np.array(y), midRA, midDec

    def getDefaultValues(self):
        """
        Returns dict of {colName:default} values for all columns with defaults.

        Returns
        -------
        defaultDict : dict
            Dict of {colName:default} values

        """
        colNames = self.getColNames()
        defaultDict = {}
        for colName in colNames:
            if colName in self.table.meta:
                defaultDict[colName] = self.table.meta[colName]
        return defaultDict

    def setDefaultValues(self, colDict):
        """
        Sets default column values.

        Parameters
        ----------
        colDict : dict
            Dict specifying column names and default values as
            {'colName':value} where the value is in the units accepted by
            makesourcedb (e.g., Hz for 'ReferenceFrequency').

        Examples
        --------
        Set new default value for ReferenceFrequency::

            >>> s.setDefaultValues({'ReferenceFrequency': 140e6})

        """
        for colName, default in iteritems(colDict):
            self.table.meta[colName] = default

    def ungroup(self):
        """
        Removes all patches from the sky model.

        Examples
        --------
        Remove all patches::

            >>> s.ungroup()

        """
        if self.hasPatches:
            for patchName in self.getPatchNames():
                if patchName in self.table.meta:
                    self.table.meta.pop(patchName)
            self.table.remove_column('Patch')
            self._updateGroups()
            self._addHistory('UNGROUP')
            self._info()

    def getColNames(self):
        """
        Returns a list of all available column names.

        Returns
        -------
        colNames : list
            List of all column names

        Examples
        --------
        Get column names::

            >>> s.getColNames()

        """
        return self.table.keys()

    def getColValues(self, colName, units=None, aggregate=None,
                     applyBeam=False):
        """
        Returns a numpy array of column values.

        Parameters
        ----------
        colName : str
            Name of column
        units : str, optional
            Output units (the values are converted as needed). By default, the
            units are those used by makesourcedb, with the exception of RA and
            Dec which have default output units of degrees.
        aggregate : {'sum', 'mean', 'wmean', 'min', max'}, optional
            If set, the array returned will be of values aggregated
            over the patch members. The following aggregation functions are
            available:
                - 'sum': sum of patch values
                - 'mean': mean of patch values
                - 'wmean': Stokes-I-weighted mean of patch values
                - 'min': minimum of patch values
                - 'max': maximum of patch values
            Note that, in some cases, certain aggregation functions will not
            produce meaningful results. For example, asking for the sum of
            the MajorAxis values per patch will not give a good indication of
            the size of the patch (to get the sizes, use the getPatchSizes()
            method). Additionally, applying the 'mean' or 'wmean' functions to
            the RA or Dec columns may give strange results near the poles or
            near RA = 0h. For aggregated RA and Dec values, use the
            getPatchPositions() method instead which projects the sources onto
            the image plane before aggregation.
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam. This attenuation
            also applies to fluxes used in aggregation functions.

        Returns
        -------
        colValues : numpy array
            Array of column values. None is returned if column is not found.

        Examples
        --------
        Get Stokes I fluxes in Jy::

            >>> s.getColValues('I')
            array([ 60.4892,   1.2413,   1.216 , ...,   1.12  ,   1.25  ,   1.16  ])

        Get Stokes I fluxes in mJy::

            >>> s.getColValues('I', units='mJy')
            array([ 60489.2,   1241.3,   1216. , ...,   1120. ,   1250. ,   1160. ])

        Get total Stokes I flux for the patches::

            >>> s.getColValues('I', aggregate='sum')
            array([ 61.7305,   1.216 ,   3.9793, ...,   1.12  ,   1.25  ,   1.16  ])

        Get flux-weighted average RA and Dec for the patches. As noted above, the
        getColValues() method is not appropriate for use with RA or Dec, so
        we must use getPatchPositions() instead::

            >>> RA, Dec = s.getPatchPositions(method='wmean', asArray=True)

        """
        colName = self._verifyColName(colName)
        if colName is None:
            return None
        if type(colName) is list:
            if len(colName) > 1:
                raise ValueError('Only one column can be specified.')
            else:
                colName = colName[0]

        allowedFcns = ['sum', 'mean', 'wmean', 'min', 'max']
        if aggregate not in allowedFcns and aggregate is not None:
            raise ValueError("Value of parameter 'aggregate' not understood.")
        if aggregate in allowedFcns and self.hasPatches:
            col = self._getAggregatedColumn(colName, aggregate, applyBeam=applyBeam)
        else:
            col = self._getColumn(colName, applyBeam=applyBeam)

        if col is None:
            return None

        if hasattr(col, 'filled'):
            outcol = col.filled().copy()
        else:
            outcol = col.copy()

        if units is not None:
            outcol.convert_unit_to(units)

        return outcol.data

    def setColValues(self, colName, values, mask=None, index=None):
        """
        Sets column values.

        Parameters
        ----------
        colName : str
            Name of column. If not already present in the table, a new column
            will be created.
        values : list, numpy array, or dict
            Array of values or dict of {sourceName:value} pairs. If list or
            array, the length must match the number of rows in the table. If
            dict, missing values will be masked unless already present. Values
            are assumed to be in units required by makesourcedb.
        mask : list or array of bools, optional
            If values is a list or array, a mask can be specified (True means
            the value is masked).
        index : int, optional
            Index that specifies the column position in the table, if column is
            not already present in the table.

        Examples
        --------
        Set Stokes I fluxes::

            >>> s.setColValues('I', [1.0, 1.1, 1.2, 0.0, 1.3], mask=[False,
                    False, False, True, False])

        """
        from astropy.table import Column
        from .tableio import RA2Angle, Dec2Angle
        import numpy as np

        colName = self._verifyColName(colName, onlyExisting=False)
        if colName is None:
            return None
        if type(colName) is list:
            if len(colName) > 1:
                raise ValueError('Only one column can be specified.')
            else:
                colName = colName[0]

        if isinstance(values, dict):
            if colName in self.table.keys():
                data = self.table[colName].data
                mask = self.table[colName].mask
            else:
                data = [0] * len(self.table)
                mask = [True] * len(self.table)
            for sourceName, value in iteritems(values):
                indx = self._getNameIndx(sourceName)
                if colName == 'Ra':
                    val = RA2Angle(value)[0]
                elif colName == 'Dec':
                    val = Dec2Angle(value)[0]
                else:
                    val = value
                data[indx] = val
                mask[indx] = False
        else:
            if len(values) != len(self.table):
                raise ValueError('Length of input values must match length of table.')
            else:
                if colName == 'Ra':
                    vals = RA2Angle(values)
                elif colName == 'Dec':
                    vals = Dec2Angle(values)
                else:
                    vals = values
                data = vals

        if mask is not None:
            data = np.ma.masked_array(data, mask)
        else:
            data = np.array(data)
        if colName in self.table.keys():
            units = self.table.columns[colName].unit
            self.table[colName] = data
            self.table.columns[colName].unit = units
        else:
            if colName == 'Patch':
                # Specify length of 50 characters
                newCol = Column(name=colName, data=data, dtype='{}50'.format(numpy_type))
            else:
                newCol = Column(name=colName, data=data)
            self.table.add_column(newCol, index=index)
        if colName == 'Patch':
            self._updateGroups()

    def getRowValues(self, rowName):
        """
        Returns an astropy table or table row for specified source or patch.

        Parameters
        ----------
        rowName : str
            Name of the source or patch

        Returns
        -------
        rowValues : astropy table or row
            Table (if more than one source) or row (if one source). None is
            returned if source is not found.

        Examples
        --------
        Get row values for the source 'src1'::

            >>> rows = s.getRowValues('src1')

        Sum over the fluxes of sources in the 'bin1' patch::

            >>> tot = 0.0
            >>> for row in s.getRowValues('bin1'): tot += row['I']

        """
        sourceNames = self.getColValues('Name')
        patchNames = self.getPatchNames()
        if patchNames is None:
            patchNames = []
        if rowName in sourceNames:
            indx = self._getNameIndx(rowName)
            return self.table.filled()[indx]
        elif rowName in patchNames:
            pindx = self._getNameIndx(rowName, patch=True)
            table = self.table.groups[pindx]
            table = table.group_by('Patch')  # ensure that grouping is preserved
            return table
        else:
            raise ValueError("Row name '{0}' not recognized.".format(rowName))

    def getRowIndex(self, rowName):
        """
        Returns index or indices for specified source or patch as a list.

        Parameters
        ----------
        rowName : str
            Name of the source or patch

        Returns
        -------
        indices : list
            List of indices. ValueError is raised if the source is not found.

        Examples
        --------
        Get row index for the source 'src1'::

            >>> s.getRowIndex('src1')
            [0]

        Get row indices for the patch 'bin1' and verify the patch name::

            >>> ind = s.getRowIndex('bin1')
            >>> print(s.getColValues('patch')[ind])
            ['bin1', 'bin1', 'bin1']

        """
        import numpy as np

        sourceNames = self.getColValues('Name')
        if self.hasPatches:
            patchNames = self.getColValues('Patch')
        else:
            patchNames = []

        if rowName in sourceNames:
            return self._getNameIndx(rowName)
        elif rowName in patchNames:
            return np.where(patchNames == rowName)[0].tolist()
        else:
            raise ValueError("Row name '{0}' not recognized.".format(rowName))

    def setRowValues(self, values, mask=None, returnVerified=False):
        """
        Sets values for a single row.

        If a row with the given name already exists, its values are
        updated. If not, a new row is made and appended to the table.

        Parameters
        ----------
        values : list, numpy array, or dict
            Array of values or dict of {colName:value} pairs. If list or
            array, the length must match the number and order of the columns in
            the table. If dict, missing values will be masked unless already
            present.
        mask : list or array of bools, optional
            If values is a list or array, a mask can be specified (True means
            the value is masked).
        returnVerified : bool, optional
            If True, the values are verified and returned, allowing them to be
            passed to table.add_row().

        Examples
        --------
        Set row values for the source 'src1' (which can be a new source or an
        existing source)::

            >>> s.setRowValues({'Name':'src1', 'Ra':213.123, 'Dec':23.1232,
                'I':23.2, 'Type':'POINT'})

        The RA and Dec values can be in degrees (as above) or in makesourcedb
        format. E.g.::

            >>> s.setRowValues({'Name':'src1', 'Ra':'12:22:21.1',
                'Dec':'+14.46.31.5', 'I':23.2, 'Type':'POINT'})

        """
        # Read model into astropy table object
        tempLSM = SkyModel(values)

        # Concatenate tables
        self.concatenate(tempLSM, matchBy='name', keep='from2', inheritPatches=False)

    def getPatchSizes(self, units=None, weight=False, applyBeam=False):
        """
        Returns array of patch sizes.

        Parameters
        ----------
        units : str, optional
            Units for returned sizes (e.g., 'arcsec', 'degree')
        weight : bool, optional
            If True, weight the source positions inside the patch by flux
        applyBeam : bool, optional
            If True and weight is True, attenuate the fluxes used for weighting
            by the beam

        Returns
        -------
        data : numpy array
            Array of patch sizes. None is returned if the sky model
            does not have patches

        """
        if self.hasPatches:
            col = self._getSizeColumn(weight=weight, applyBeam=applyBeam)
            if units is not None:
                col.convert_unit_to(units)
            return col.data
        else:
            return None

    def getPatchNames(self):
        """
        Returns array of all patch names in the sky model, with duplicates removed.

        Note: use getColValues('Patch') if you want the patch names for each source
        in the sky model.

        Returns
        -------
        names : numpy array
            Array of patch names. None is returned if the sky model
            does not have patches

        """
        if self.hasPatches:
            col = self.table.groups.keys['Patch']
            if hasattr(col, 'filled'):
                outcol = col.filled().copy()
            else:
                outcol = col.copy()
            return outcol.data
        else:
            return None

    def _getNameIndx(self, name, patch=False):
        """
        Returns a list of indices corresponding to the given names.

        Parameters
        ----------
        name : str, list of str
            source or patch name or list of names (UNIX-style wildcards are
            allowed)
        patch : bool
            if True, return the index of the group corresponding to the given
            name; otherwise return the index of the source

        Returns
        -------
        indices : list
            List of indices

        """
        import numpy as np
        import fnmatch

        if patch:
            if self.hasPatches:
                names = self.getPatchNames().tolist()
            else:
                return None
        else:
            names = self.getColValues('Name').tolist()

        if type(name) is str or type(name) is np.string_:
            indx = [i for i, item in enumerate(names) if fnmatch.fnmatch(item, name)]
            if len(indx) == 0:
                return None
            return indx
        elif type(name) is list:
            indx = []
            for n in name:
                badNames = []
                nindx = [i for i, item in enumerate(names) if fnmatch.fnmatch(item, n)]
                if len(nindx) == 0:
                    badNames.append(n)
                else:
                    indx += nindx
            if len(badNames) > 0:
                if len(badNames) == 1:
                    plur = ''
                else:
                    plur = 's'
                self.log.warn("Name{0} '{1}' not recognized. Ignoring.".
                              format(plur, ','.join(badNames)))
            if len(indx) == 0:
                raise ValueError("None of the specified names were found.")
            return indx
        else:
            return None

    def _getColumn(self, colName, applyBeam=False):
        """
        Returns the appropriate column (nonaggregated).

        Parameters
        ----------
        colName : str
            Name of column to get. If not found, None is returned
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Nonaggregated Column object

         """
        colName = self._verifyColName(colName)
        if colName is None:
            return None

        col = self.table[colName].copy()

        if applyBeam and colName in ['I', 'Q', 'U', 'V']:
            col = self._applyBeamToCol(col)

        return col

    def _getAggregatedColumn(self, colName, aggregate='sum', applyBeam=False):
        """
        Returns the appropriate column aggregated by group.

        Parameters
        ----------
        colName : str
            Name of column to get. If not found, None is returned
        aggregate : str, optional
            If set, the array returned will be of values aggregated
            over the patch members. The following aggregation functions are
            available:
                - 'sum': sum of patch values
                - 'mean': mean of patch values
                - 'wmean': Stokes I weighted mean of patch values
                - 'min': minimum of patch values
                - 'max': maximum of patch values
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Column object with aggregated values

         """
        colName = self._verifyColName(colName)
        if colName is None:
            return None

        if aggregate == 'mean':
            col = self._getAveragedColumn(colName, weight=False,
                                          applyBeam=applyBeam)
        elif aggregate == 'wmean':
            col = self._getAveragedColumn(colName, weight=True,
                                          applyBeam=applyBeam)
        elif aggregate == 'sum':
            col = self._getSummedColumn(colName, applyBeam=applyBeam)
        elif aggregate == 'min':
            col = self._getMinColumn(colName, applyBeam=applyBeam)
        elif aggregate == 'max':
            col = self._getMaxColumn(colName, applyBeam=applyBeam)
        else:
            raise ValueError('Aggregation function not understood.'.format(colName))
        return col

    def _applyBeamToCol(self, col, patch=False):
        """
        Applies beam attenuation to the column values.

        Parameters
        ----------
        col : astropy Column
            Column of flux values to attenuate
        patch : bool, optional
            If True, col is assumed to be aggregated over patches

        Returns
        -------
        col : astropy Column
            Column object with flux values attenuated by the beam

        """
        from .operations_lib import attenuate

        if not self._hasBeam:
            self.log.warn('No beam MS has been specified. No beam attenuation applied.')
            return col

        if patch:
            if self._patchMethod is not None:
                # Try to get patch positions from the meta data
                RADeg, DecDeg = self.getPatchPositions(asArray=True)
            else:
                # If patch positions are not set, use weighted mean positions
                RADeg = self.getColValues('Ra', applyBeam=True, aggregate='wmean')
                DecDeg = self.getColValues('Dec', applyBeam=True, aggregate='wmean')
        else:
            RADeg = self.getColValues('Ra')
            DecDeg = self.getColValues('Dec')

        flux = col.data
        try:
            vals = attenuate(self.beamMS, flux, RADeg, DecDeg, timeIndx=self.beamTime)
        except Exception as e:
            self.log.warn('{0}. No beam attenuation applied.'.format(e.message))
            return col

        col[:] = vals
        return col

    def _getSummedColumn(self, colName, applyBeam=False):
        """
        Returns column summed by group.

        Parameters
        ----------
        colName : str
            Column name
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Column object with aggregated sum of values

        """
        import numpy as np

        def npsum(array):
            return np.sum(array, axis=0)

        if hasattr(self.table[colName], 'filled'):
            col = self.table[colName].filled()
            gcol = col.group_by(self.table['Patch'])
            gcol = gcol.groups.aggregate(npsum)
        else:
            gcol = self.table[colName].groups.aggregate(npsum)
        if applyBeam and colName in ['I', 'Q', 'U', 'V']:
            gcol = self._applyBeamToCol(gcol, patch=True)

        return gcol

    def _getMinColumn(self, colName, applyBeam=False):
        """
        Returns column minimum value by group.

        Parameters
        ----------
        colName : str
            Column name
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Column object with aggregated min values

        """
        import numpy as np

        def npmin(array):
            return np.min(array, axis=0)

        if hasattr(self.table[colName], 'filled'):
            col = self.table[colName].filled()
            gcol = col.group_by(self.table['Patch'])
            gcol = gcol.groups.aggregate(npmin)
        else:
            gcol = self.table[colName].groups.aggregate(npmin)
        if applyBeam and colName in ['I', 'Q', 'U', 'V']:
            gcol = self._applyBeamToCol(gcol, patch=True)

        return gcol

    def _getMaxColumn(self, colName, applyBeam=False):
        """
        Returns column maximum value by group.

        Parameters
        ----------
        colName : str
            Column name
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Column object with aggregated max values

        """
        import numpy as np

        def npmax(array):
            return np.max(array, axis=0)

        if hasattr(self.table[colName], 'filled'):
            col = self.table[colName].filled()
            gcol = col.group_by(self.table['Patch'])
            gcol = gcol.groups.aggregate(npmax)
        else:
            gcol = self.table[colName].groups.aggregate(npmax)
        if applyBeam and colName in ['I', 'Q', 'U', 'V']:
            gcol = self._applyBeamToCol(gcol, patch=True)

        return gcol

    def _getAveragedColumn(self, colName, weight=True, applyBeam=False):
        """
        Returns column averaged by group.

        Parameters
        ----------
        colName : str
            Column name
        weight : bool, optional
            If True, return average weighted by flux
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Column object with aggregated mean values

        """
        from astropy.table import Column
        import numpy as np

        if weight:
            def npsum(array):
                return np.sum(array, axis=0)

            if hasattr(self.table[colName], 'filled'):
                vals = self.table[colName].filled().data
            else:
                vals = self.table[colName].data
            if weight:
                weights = self.getColValues('I', applyBeam=applyBeam)
                if weights.shape != vals.shape:
                    weights = np.resize(weights, vals.shape)
                weightCol = Column(name='Weight', data=weights)
                valWeightCol = Column(name='ValWeight', data=vals*weights)
                self.table.add_column(valWeightCol)
                self.table.add_column(weightCol)
                numer = self.table['ValWeight'].groups.aggregate(npsum).data
                denom = self.table['Weight'].groups.aggregate(npsum).data
                self.table.remove_column('ValWeight')
                self.table.remove_column('Weight')
            else:
                valCol = Column(name='Val', data=vals)
                self.table.add_column(valCol)
                numer = self.table['Val'].groups.aggregate(npsum).data
                self.table.remove_column('Val')

            return Column(name=colName, data=np.array(numer/denom),
                          unit=self.table[colName].unit)
        else:
            def npavg(c):
                return np.average(c, axis=0)

            return self.table[colName].groups.aggregate(npavg)

    def _getSizeColumn(self, weight=True, applyBeam=False):
        """
        Returns column of source largest angular sizes.

        Parameters
        ----------
        weight : bool, optional
            If True, return size weighted by flux
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Column object with sizes from MajorAxis or from aggregated values if
            the model has patches

        """
        from astropy.table import Column
        import numpy as np

        if weight:
            method = 'wmean'
        else:
            method = 'mean'

        if self.hasPatches:
            # Get patch positions
            RAAvg, DecAvg = self.getPatchPositions(method=method, asArray=True,
                                                   applyBeam=applyBeam)

            # Fill out the columns by repeating the average value over the
            # entire group
            RAAvgFull = np.zeros(len(self.table), dtype=np.float)
            DecAvgFull = np.zeros(len(self.table), dtype=np.float)
            for i, ind in enumerate(self.table.groups.indices[1:]):
                RAAvgFull[self.table.groups.indices[i]: ind] = RAAvg[i]
                DecAvgFull[self.table.groups.indices[i]: ind] = DecAvg[i]

            dist = self._calculateSeparation(self.table['Ra'],
                                             self.table['Dec'], RAAvgFull, DecAvgFull)
            if weight:
                if applyBeam and self._hasBeam:
                    appFluxes = self.getColValues('I', applyBeam=True)
                    weightCol = Column(name='Weight', data=appFluxes)
                    valWeightCol = Column(name='ValWeight', data=dist*appFluxes)
                else:
                    weightCol = Column(name='Weight', data=self.table['I'].data)
                    valWeightCol = Column(name='ValWeight', data=dist*self.table['I'].data)
                self.table.add_column(valWeightCol)
                self.table.add_column(weightCol)
                numer = self.table['ValWeight'].groups.aggregate(np.sum).data * 2.0
                denom = self.table['Weight'].groups.aggregate(np.sum).data
                self.table.remove_column('ValWeight')
                self.table.remove_column('Weight')
                col = Column(name='Size', data=numer/denom,
                             unit='degree')
            else:
                valCol = Column(name='Val', data=dist)
                self.table.add_column(valCol)
                size = self.table['Val'].groups.aggregate(np.max).data * 2.0
                self.table.remove_column('Val')
                col = Column(name='Size', data=size, unit='degree')
        else:
            if 'majoraxis' in self.table.colnames:
                col = self.table['MajorAxis']
            else:
                col = Column(name='Size', data=np.zeros(len(self.table)), unit='degree')

        if hasattr(col, 'filled'):
            outcol = col.filled(fill_value=0.0)
        else:
            outcol = col
        outcol.convert_unit_to('arcsec')

        return outcol

    def _calculateSeparation(self, ra1, dec1, ra2, dec2):
        """
        Returns angular separation between two coordinates (all in degrees)

        Parameters
        ----------
        ra1 : float or array
            RA of coordinate 1 in degrees
        dec1 : float or array
            Dec of coordinate 1 in degrees
        ra2 : float
            RA of coordinate 2 in degrees
        dec2 : float
            Dec of coordinate 2 in degrees

        Returns
        -------
        separation : Angle object or array
            Angular separation in degrees

        """
        from .operations_lib import calculateSeparation

        return calculateSeparation(ra1, dec1, ra2, dec2)

    def getDistance(self, RA, Dec, byPatch=False, units=None):
        """
        Returns angular distance for each source or patch to specified position

        Parameters
        ----------
        RA : float or str
            RA of position to which the distance is desired (in degrees or
            makesourcedb format)
        Dec : float or str
            Dec of position to which the distance is desired (in degrees or
            makesourcedb format)
        byPatch : bool, optional
            Calculate distance by patches instead of by sources
        units : str, optional
            Units for resulting distance. If None, units are degrees

        Returns
        -------
        dist : array
            Array of distances

        Examples
        --------
        Find distance in degrees to a position for all sources::

            >>> s.getDistance(94.0, 42.0)

        Find distance in arcmin::

            >>> s.getDistance(94.0, 42.0, units='arcmin')

        Find distance to patch centers:

            >>> s.setPatchPositions(method='mid')
            >>> s.getDistance(94.0, 42.0, byPatch=True)

        """
        from .tableio import RA2Angle, Dec2Angle

        if byPatch and self.hasPatches:
            # Get patch positions
            sRA, sDec = self.getPatchPositions(asArray=True)
        else:
            sRA = self.getColValues('RA')
            sDec = self.getColValues('Dec')

        RA = RA2Angle(RA)
        Dec = Dec2Angle(Dec)

        dist = self._calculateSeparation(sRA, sDec, RA, Dec)
        if units is not None:
            return dist.to(units).value
        else:
            return dist.value

    def write(self, fileName=None, format='makesourcedb', clobber=False,
              sortBy=None, lowToHigh=False, addHistory=True, applyBeam=False,
              invertBeam=False, adjustSI=False):
        """
        Writes the sky model to a file.

        Parameters
        ----------
        filename : str
            Name of output file.
        format: str, optional
            Format of the output file. Allowed formats are:
                - 'makesourcedb' (BBS format)
                - 'fits'
                - 'votable'
                - 'hdf5'
                - 'ds9'
                - 'kvis'
                - 'casa'
                - 'factor'
                - plus all other formats supported by the astropy.table package
        clobber : bool, optional
            If True, an existing file is overwritten.
        sortBy : str or list of str, optional
            Name of columns to sort on. If None, no sorting is done. If
            a list is given, sorting is done on the columns in the order given.
        lowToHigh : bool, optional
            If True, sort values from low to high instead of high to low.
        addHistory : bool, optional
            If True, the history of operations is written to the sky model
            header.
        applyBeam : bool, optional
            If True, fluxes will be adjusted for the beam before being written.
        invertBeam : bool, optional
            If True, the beam correction is inverted (i.e., from apparent sky to
            true sky).
        adjustSI : bool, optional
            If True, adjust the spectral index column for the beam (only works if
            the spectral index in non-logarithmic. I.e., the 'LogarithmicSI' column
            entries are all False)

        Examples
        --------
        Write the model to a makesourcedb sky model file suitable for use with
        BBS::

            >>> s.write('modsky.model')

        Write to a fits catalog::

            >>> s.write('sky.fits', format='fits')

        Write to a ds9 region file::

            >>> s.write('sky.reg', format='ds9')

        """
        import os
        import numpy as np
        from .operations_lib import attenuate

        if fileName is None:
            if self._fileName is None:
                raise IOError("No file name specified.")
            else:
                fileName = self._fileName

        if os.path.exists(fileName):
            if clobber:
                os.remove(fileName)
            else:
                raise IOError("The output file '{0}' exists and clobber = False.".
                              format(fileName))

        table = self.table.copy()

        # Apply beam attenuation
        if applyBeam:
            if 'LogarithmicSI' in self.getColNames():
                if np.any(self.getColValues('LogarithmicSI') == "true"):
                    adjustSI = False
            else:
                # Default is LogarithmicSI=true
                adjustSI = False
            I_orig = self.getColValues('I')
            RADeg = self.getColValues('Ra')
            DecDeg = self.getColValues('Dec')
            if adjustSI:
                spectralIndex = self.getColValues('SpectralIndex')
                referenceFrequency = self.getColValues('ReferenceFrequency')
                I_adj, SI_adj = attenuate(self.beamMS, I_orig, RADeg, DecDeg,
                                          timeIndx=self.beamTime, invert=invertBeam,
                                          spectralIndex=spectralIndex,
                                          referenceFrequency=referenceFrequency)
            else:
                I_adj = attenuate(self.beamMS, I_orig, RADeg, DecDeg,
                                  timeIndx=self.beamTime, invert=invertBeam)
            units = self.table.columns['I'].unit
            table['I'] = I_adj
            table.columns['I'].unit = units
            if adjustSI:
                table['SpectralIndex'] = SI_adj

        # Sort if desired. For 'factor' output, save the order of patches in the
        # table meta
        if sortBy is not None:
            colName = self._verifyColName(sortBy)
            if format.lower() == 'factor' and self.hasPatches:
                indx = np.argsort(self.getColValues('I', aggregate='sum'))
                if not lowToHigh:
                    indx = indx[::-1]
                table.meta['patch_order'] = indx
            else:
                indx = table.argsort(colName)
                if not lowToHigh:
                    indx = indx[::-1]
                table = table[indx]

        if addHistory:
            table.meta['History'] = self.history

        # Add patch sizes in degrees
        if format.lower() == 'factor' and self.hasPatches:
            table.meta['patch_size'] = self.getPatchSizes(units='deg')

        # Add patch fluxes in mJy
        if format.lower() == 'factor' and self.hasPatches:
            table.meta['patch_flux'] = self.getColValues('I', aggregate='sum',
                                                         units='mJy')

        if format.lower() != 'makesourcedb' and format.lower() != 'factor':
            table.meta = {}
        table.write(fileName, format=format.lower())

    def broadcast(self):
        """
        Sends the model to another application using SAMP.

        Both the SAMP hub and the receiving application must be running before
        the table is broadcasted. Examples of SMAP-aware applications are
        TOPCAT, Aladin, and ds9.

        Examples
        --------
        Send the model to TOPCAT. First, start TOPCAT, then run the command::

            >>> s.broadcast()

        TOPCAT should then load the table.

        """
        import tempfile

        tfile = tempfile.NamedTemporaryFile()
        self.table.write(tfile, format='votable')
        tableio.broadcastTable(tfile.name)
        tfile.close()

    def _clean(self):
        """
        Removes duplicate entries.
        """
        names = self.getColValues('Name')
        nameSet = set(names)
        if len(names) == len(nameSet):
            return

        filtNames = []
        filtIndices = []
        for i, name in enumerate(self.getColValues('Name')):
            if name in filtNames:
                filtIndices.append(i)
            else:
                filtNames.append(name)
        nRowsOrig = len(self.table)
        self.table = self.table[filtIndices]
        nRowsNew = len(self.table)
        if nRowsOrig-nRowsNew > 0:
            self.log.info('Removed {0} duplicate sources.'.format(nRowsOrig-nRowsNew))
        self._updateGroups()

    def select(self, filterExpression, aggregate=None, applyBeam=False,
               useRegEx=False, force=True):
        """
        Filters the sky model, keeping all sources that meet the given expression.

        After filtering, the sky model contains only those sources for which the
        given filter expression is true.

        Parameters
        ----------
        filterExpression : str, dict, list, or numpy array
            - If string:
                A string specifying the filter expression in the form:
                '<property> <operator> <value> [<units>]'
                (e.g., 'I <= 10.5 Jy').

            - If dict:
                The filter can also be given as a dictionary in the form:
                {'filterProp':property, 'filterOper':operator,
                    'filterVal':value, 'filterUnits':units}

            - If list:
                The filter can also be given as a list of:
                [property, operator, value] or
                [property, operator, value, units]

            - If numpy array:
                The indices to filter on can be specified directly as a numpy array
                of row or patch indices such as:
                array([ 0,  2, 19, 20, 31, 37])

                or as a numpy array of bools with the same length as the sky model.

                If a numpy array is given and the indices correspond to patches, then
                set aggregate=True.

            The property to filter on must be one of the following:
                - a valid column name
                - the filename of a mask image

            Supported operators are:
                - !=
                - <=
                - >=
                - >
                - <
                - = (or '==')
            Units are optional and must be specified as required by astropy.units.
        aggregate : str, optional
            If set, the selection will be done on values aggregated
            over the patch members. The following aggregation functions are
            available:
                - 'sum': sum of patch values
                - 'mean': mean of patch values
                - 'wmean': Stokes I weighted mean of patch values
                - 'min': minimum of patch values
                - 'max': maximum of patch values
                - True: only valid when the filter indices are specified directly
                as a numpy array. If True, filtering is done on patches instead
                of sources.
        applyBeam : bool, optional
            If True, apparent fluxes will be used.
        useRegEx : bool, optional
            If True, string matching will use regular expression matching. If
            False, string matching uses Unix filename matching.
        force : bool, optional
            If True, selections that result in empty sky models are allowed. If
            False, such selections are not applied and the sky model is unaffected.

        Examples
        --------
        Filter on column 'I' (Stokes I flux). This filter will select all sources
        with Stokes I flux greater than 1.5 Jy::

            >>> s.select('I > 1.5 Jy')
            INFO: Kept 1102 sources.

        If the sky model has patches and the filter is desired per patch, use
        ``aggregate = function``. For example, to select on the sum of the patch
        fluxes::

            >>> s.select('I > 1.5 Jy', aggregate='sum')

        Or, to filter on patches smaller than 5 arcmin in size::

            >>> sizes = s.getPatchSizes(units='arcmin')
            >>> s.select(sizes < 5.0, aggregate=True)

        Filter on source names, keeping those that match "src*_1?"::

            >>> s.select('Name == src*_1?')

        Use a CASA clean mask image named 'clean_mask.mask' to select sources
        that lie in masked regions::

            >>> s.select('clean_mask.mask == True')

        """
        operations.select.select(self, filterExpression, aggregate=aggregate,
                                 applyBeam=applyBeam, useRegEx=useRegEx, force=force)

    def remove(self, filterExpression, aggregate=None, applyBeam=None,
               useRegEx=False, force=True):
        """
        Filters the sky model, removing all sources that meet the given expression.

        After filtering, the sky model contains only those sources for which the
        given filter expression is false.

        Parameters
        ----------
        filterExpression : str, dict, list, or numpy array
            - If string:
                A string specifying the filter expression in the form:
                '<property> <operator> <value> [<units>]'
                (e.g., 'I <= 10.5 Jy').

            - If dict:
                The filter can also be given as a dictionary in the form:
                {'filterProp':property, 'filterOper':operator,
                    'filterVal':value, 'filterUnits':units}

            - If list:
                The filter can also be given as a list of:
                [property, operator, value] or
                [property, operator, value, units]

            - If numpy array:
                The indices to filter on can be specified directly as a numpy array
                of row or patch indices such as:
                array([ 0,  2, 19, 20, 31, 37])

                or as a numpy array of bools with the same length as the sky model.

                If a numpy array is given and the indices correspond to patches, then
                set aggregate=True.

            The property to filter on must be one of the following:
                - a valid column name
                - the filename of a mask image

            Supported operators are:
                - !=
                - <=
                - >=
                - >
                - <
                - = (or '==')
            Units are optional and must be specified as required by astropy.units.
        aggregate : str, optional
            If set, the selection will be done on values aggregated
            over the patch members. The following aggregation functions are
            available:
            - 'sum': sum of patch values
            - 'mean': mean of patch values
            - 'wmean': Stokes I weighted mean of patch values
            - 'min': minimum of patch values
            - 'max': maximum of patch values
            - True: only valid when the filter indices are specified directly as
            a numpy array. If True, filtering is done on patches instead of
            sources.
        applyBeam : bool, optional
            If True, apparent fluxes will be used.
        useRegEx : bool, optional
            If True, string matching will use regular expression matching. If
            False, string matching uses Unix filename matching.
        force : bool, optional
            If True, filters that result in empty sky models are allowed. If
            False, such filters are not applied and the sky model is unaffected.

        Examples
        --------
        Filter on column 'I' (Stokes I flux). This filter will remove all sources
        with Stokes I flux greater than 1.5 Jy::

            >>> s.remove('I > 1.5 Jy')
            INFO: Removed 1102 sources.

        If the sky model has patches and the filter is desired per patch, use
        ``aggregate = function``. For example, to filter on the sum of the patch
        fluxes::

            >>> s.remove('I > 1.5 Jy', aggregate='sum')

        Or, to filter on patches smaller than 5 arcmin in size::

            >>> sizes = s.getPatchSizes(units='arcmin')
            >>> s.remove(sizes < 5.0, aggregate=True)

        Filter on source names, removing those that match "src*_1?" (e.g.,
        'src2345_15', 'srcB2_1a', etc.)::

            >>> s.remove('Name == src*_1?')

        Use a CASA clean mask image named 'clean_mask.mask' to remove sources
        that lie in masked regions::

            >>> s.remove('clean_mask.mask == True')

        """
        operations.remove.remove(self, filterExpression, aggregate=aggregate,
                                 applyBeam=applyBeam, useRegEx=useRegEx, force=force)

    def group(self, algorithm, targetFlux=None, weightBySize=False, numClusters=100, FWHM=None,
              threshold=0.1, applyBeam=False, root='Patch', pad_index=False,
              method='mid', facet="", byPatch=False, kernelSize=0.1,
              nIterations=100, lookDistance=0.2, groupingDistance=0.01):
        """
        Groups sources into patches.

        Parameters
        ----------
        LSM : SkyModel object
            Input sky model.
        algorithm : str
            Algorithm to use for grouping:
            - 'single' => all sources are grouped into a single patch
            - 'every' => every source gets a separate patch named 'source_patch'
            - 'cluster' => SAGECAL clustering algorithm that groups sources into
                specified number of clusters (specified by the numClusters parameter)
            - 'tessellate' => group into tiles whose total flux approximates
                the target flux (specified by the targetFlux parameter)
            - 'threshold' => group by convolving the sky model with a Gaussian beam
                and then thresholding to find islands of emission (NOTE: all sources
                are currently considered to be point sources of flux unity)
            - 'facet' => group by facets using as an input a fits file. It requires
                the use of the additional parameter 'facet' to enter the name of the
                fits file.
            - 'voronoi' => given a previously grouped sky model, voronoi tesselate
                using the patch positions for patches above the target flux (specified
                by the targetFlux parameter)
            - 'meanshift' => use the meanshift clustering algorithm
            - the filename of a mask image => group by masked regions (where mask =
                True). Sources outside of masked regions are given patches of their
                own
        targetFlux : str or float, optional
            Target flux for tessellation (the total flux of each tile will be close
            to this value) and voronoi algorithms. The target flux can be specified
            as either a float in Jy or as a string with units (e.g., '25.0 mJy')
        weightBySize : bool, optional
            If True, fluxes are weighted by patch size (as median_size / size) when
            the targetFlux criterion is applied. Patches with sizes below the median
            (flux-weighted) size are upweighted and those above the mean are
            downweighted
        numClusters : int, optional
            Number of clusters for clustering. Sources are grouped around the
            numClusters brightest sources.
        FWHM : str or float, optional
            FWHM of convolving Gaussian used for thresholding. The FWHM can
            be specified as either a float in degrees or as a string with units
            (e.g., '25.0 arcsec')
        threshold : float, optional
            Value between 0 and 1 above which emission is considered for thresholding
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam.
        root : str, optional
            Root string from which patch names are constructed. For 'single', the
            patch name will be set to root; for the other grouping algorithms, the
            patch names will be 'root_INDX', where INDX is an integer ranging from
            (0:nPatches).
        pad_index : bool, optional
            If True, pad the INDX is used in the patch names. E.g., facet_patch_001
            instead of facet_patch_1
        method : None or str, optional
            This parameter specifies the method used to set the patch positions:
            - 'mid' => the position is set to the midpoint of the patch
            - 'mean' => the positions is set to the mean RA and Dec of the patch
            - 'wmean' => the position is set to the flux-weighted mean RA and
            Dec of the patch
            - 'zero' => set all positions to [0.0, 0.0]
        facet : str, optional
            Facet fits file used with the algorithm 'facet'
        byPatch : bool, optional
            For the 'tessellate' or 'meanshift' algorithms, use patches instead of
            sources
        kernelSize : float, optional
            Kernel size in degrees for 'meanshift' grouping
        nIterations : int, optional
            Number of iterations for 'meanshift' grouping
        lookDistance : float, optional
            Look distance in degrees for 'meanshift' grouping
        groupingDistance : float, optional
            Grouping distance in degrees for 'meanshift' grouping

        Examples
        --------
        Tesselate the sky model into patches with approximately 30 Jy total
        flux:

            >>> s.group('tessellate', targetFlux=30.0)

        """
        operations.group.group(self, algorithm, targetFlux=targetFlux, weightBySize=weightBySize,
                               numClusters=numClusters, FWHM=FWHM, threshold=threshold,
                               applyBeam=applyBeam, root=root, pad_index=pad_index,
                               method=method, facet=facet, byPatch=byPatch,
                               kernelSize=kernelSize, nIterations=nIterations,
                               lookDistance=lookDistance,
                               groupingDistance=groupingDistance)

    def transfer(self, patchSkyModel, matchBy='name', radius=0.1):
        """
        Transfer patches from the input sky model.

        Sources matching those in patchSkyModel will be grouped into
        the patches defined in patchSkyModel. Sources that do not appear in
        patchSkyModel will be placed into separate patches (one per source).
        Patch positions are not transferred (as they may no longer be appropriate
        after transfer).

        Parameters
        ----------
        patchSkyModel : str or SkyModel object
            Input sky model from which to transfer patches.
        matchBy : str, optional
            Determines how matching sources are determined:
            - 'name' => matches are identified by name
            - 'position' => matches are identified by radius. Sources within the
                radius specified by the radius parameter are considered matches
        radius : float or str, optional
            Radius in degrees (if float) or 'value unit' (if str; e.g., '30 arcsec')
            for matching when matchBy='position'

        Examples
        --------
        Transfer patches from one sky model to another and set their positions
        (matching sources are identified by name)::

            >>> s.transfer('master_sky.model')
            >>> s.setPatchPositions(method='mid')

        """
        operations.transfer.transfer(self, patchSkyModel, matchBy=matchBy,
                                     radius=radius)

    def move(self, name, position=None, shift=None):
        """
        Move or shift a source or sources.

        If both a position and a shift are specified, a source is moved to the
        new position and then shifted. Note that only a single source can be
        moved to a new position. However, multiple sources can be shifted.

        If an xyshift is specified, a FITS file must also be specified to define
        the WCS system. If a position, a shift, and an xyshift are all specified,
        a source is moved to the new position, shifted in RA and Dec, and then
        shifted in x and y.

        Parameters
        ----------
        LSM : SkyModel object
            Input sky model
        name : str or list
            Source name or list of names (can include wildcards)
        position : list, optional
            A list specifying a new position as [RA, Dec] in either makesourcedb
            format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
            [123.2312, 23.3422])
        shift : list, optional
            A list specifying the shift as [RAShift, DecShift] in degrees (e.g.,
            [0.02312, 0.00342])
        xyshift : list, optional
            A list specifying the shift as [xShift, yShift] in pixels. A FITS file
            must be specified with the fitsFILE argument
        fitsFile : str, optional
            A FITS file from which to take WCS information to transform the pixel
            coordinates to RA and Dec values. The xyshift argument must be specfied
            for this to be useful

        Examples
        --------
        Move source '1609.6+6556' to a new position::

            >>> s.move('1609.6+6556', position=['16:10:00', '+65.57.00'])

        Shift the source by 10 arcsec in Dec::

            >>> s.move('1609.6+6556', shift=[0.0, 10.0/3600.0])

        Shift all sources by 10 pixels in x::

            >>> s.move('*', xyshift=[10, 0], fitsFile='image.fits')

        """
        operations.move.move(self, name, position=position, shift=shift)

    def add(self, colNamesVals):
        """
        Add a source to the sky model.

        Parameters
        ----------
        colNamesVals : dict
            A dictionary that specifies the column values for the source to be
            added

        Examples:
        ---------
        Add a point source::

            >>> source = {'Name':'src1', 'Type':'POINT', 'Ra':'12:32:10.1',
                'Dec':'23.43.21.21', 'I':2.134}
            >>> s.add(source)

        """
        operations.add.add(self, colNamesVals)

    def merge(self, patches, name=None):
        """
        Merge two or more patches together.

        Parameters
        ----------
        patches : list of str
            List of patch names to merge
        name : str, optional
            Name of resulting merged patch. If None, the merged patch uses the
            name of the first patch in the input patches list

        Examples
        --------
        Merge three patches into one named 'binmerged'::

            >>> s.merge(['bin0', 'bin1', 'bin2'], 'binmerged')

         """
        operations.merge.merge(self, patches, name=name)

    def concatenate(self, LSM2, matchBy='name', radius=0.1, keep='all',
                    inheritPatches=False):
        """
        Concatenate two sky models.

        Parameters
        ----------
        LSM2 : str or SkyModel object
            Secondary sky model to concatenate with the parent sky model
        matchBy : str, optional
            Determines how duplicate sources are determined:
            - 'name' => duplicates are identified by name
            - 'position' => duplicates are identified by radius. Sources within the
                radius specified by the radius parameter are considered duplicates
        radius : float or str, optional
            Radius in degrees (if float) or 'value unit' (if str; e.g., '30 arcsec')
            for matching when matchBy='position'
        keep : str, optional
            Determines how duplicates are treated:
            - 'all' => all duplicates are kept; those with identical names are re-
                named
            - 'from1' => duplicates kept are those from sky model 1 (the parent)
            - 'from2' => duplicates kept are those from sky model 2 (the secondary)
        inheritPatches : bool, optional
            If True, duplicates inherit the patch name from the parent sky model. If
            False, duplicates keep their own patch names.

        Examples
        --------
        Concatenate two sky models, identifying duplicates by matching to the source
        names. When duplicates are found, keep the source from the parent sky model
        and discard the duplicate from secondary sky model (this might be useful when
        merging two gsm.py sky models that have some overlap)::

            >>> LSM2 = lsmtool.load('gsm_sky2.model')
            >>> s.concatenate(LSM2, matchBy='name', keep='from1')

        Concatenate two sky models, identifying duplicates by matching to the source
        positions within a radius of 10 arcsec. When duplicates are found, keep the
        source from the secondary sky model and discard the duplicate from the parent
        sky model (this might be useful when replacing parts of a low-resolution
        sky model with a high-resolution one)::

            >>> LSM2 = lsmtool.load('high_res_sky.model')
            >>> s.concatenate(LSM2, matchBy='position', radius=10.0/3600.0,
                keep='from2')

        """
        if type(LSM2) is str:
            LSM2 = SkyModel(LSM2)
        operations.concatenate.concatenate(self, LSM2, matchBy=matchBy,
                                           radius=radius, keep=keep,
                                           inheritPatches=inheritPatches)

    def compare(self, LSM2, radius='10 arcsec', outDir='.', labelBy=None,
                ignoreSpec=None, excludeMultiple=True, excludeByFlux=False, name1=None,
                name2=None, format='pdf'):
        """
        Compare two sky models.

        Comparison plots and a text file with statistics are written out to the
        an output directory. Plots are made for:
            - flux ratio vs. radius from sky model center
            - flux ratio vs. sky position
            - flux ratio vs flux
            - position offsets
        The following statistics are saved to 'stats.txt' in the output directory:
            - mean and standard deviation of flux ratio
            - mean and standard deviation of RA offsets
            - mean and standard deviation of Dec offsets
        These statistics are also returned as a dictionary.

        Parameters
        ----------
        LSM2 : SkyModel object
            Secondary sky model to compare to the parent sky model
        radius : float or str, optional
            Radius in degrees (if float) or 'value unit' (if str; e.g., '30 arcsec')
            for matching
        outDir : str, optional
            Plots are saved to this directory
        labelBy : str, optional
            One of 'source' or 'patch': label points using source names ('source') or
            patch names ('patch')
        ignoreSpec : float, optional
            Ignore sources with this spectral index
        excludeMultiple : bool, optional
            If True, sources with multiple matches are excluded. If False, the
            nearest of the multiple matches will be used for comparison
        excludeByFlux : bool, optional
            If True, matches whose predicted fluxes differ from the parent model
            fluxes by 25% are excluded from the positional offset plot.
        name1 : str, optional
            Name to use in the plots for the primary sky model. If None, 'Model 1' is used.
        name2 : str, optional
            Name to use in the plots for LSM2. If None, 'Model 2' is used.
        format : str, optional
            Format of plot files.

        Returns
        -------
        stats : dict
            Dict of statistics with the following keys (where the clipped values
            are after 3-sigma clipping):
                - 'meanRatio'
                - 'stdRatio'
                - 'meanRAOffsetDeg'
                - 'stdRAOffsetDeg'
                - 'meanDecOffsetDeg'
                - 'stdDecOffsetDeg'
                - 'meanClippedRatio'
                - 'stdClippedRatio'
                - 'meanClippedRAOffsetDeg'
                - 'stdClippedRAOffsetDeg'
                - 'meanClippedDecOffsetDeg'
                - 'stdClippedDecOffsetDeg'

        Examples
        --------
        Compare two sky models and save plots::

            >>> LSM2 = lsmtool.load('sky2.model')
            >>> s.compare(LSM2, outDir='comparison_results/')

        Compare a LOFAR sky model to a global sky model made from VLSS+TGSS+NVSS (where
        refRA and refDec are the approximate center of the LOFAR sky model coverage)::

            >>> LSM2 = lsmtool.load('GSM', VOPosition=[refRA, refDec], VORadius='5 deg')
            >>> s.compare(LSM2, radius='30 arcsec', excludeMultiple=True,
                outDir='comparison_results/', name1='LOFAR', name2='GSM', format='png')

        """
        if type(LSM2) is str:
            LSM2 = SkyModel(LSM2)
        stats = operations.compare.compare(self, LSM2, radius=radius, outDir=outDir,
                                           labelBy=labelBy, ignoreSpec=ignoreSpec,
                                           excludeMultiple=excludeMultiple,
                                           excludeByFlux=excludeByFlux,
                                           name1=name1, name2=name2, format=format)
        return stats

    def plot(self, fileName=None, labelBy=None):
        """
        Shows a simple plot of the sky model.

        The circles in the plot are scaled with flux. If the sky model is grouped
        into patches, sources are colored by patch and the patch positions are
        indicated with stars.

        Parameters
        ----------
        fileName : str, optional
            If given, the plot is saved to a file instead of displayed.
        labelBy : str, optional
            One of 'source' or 'patch': label points using source names ('source')
            or patch names ('patch')

        Examples:
        ---------
        Plot and display to the screen::

            >>> s.plot()

        Plot and save to a PDF file::

            >>> s.plot('sky_plot.pdf')

        """
        operations.plot.plot(self, fileName=fileName, labelBy=labelBy)

    def rasterize(self, cellsize, fileRoot=None, writeRegionFile=False, clobber=False):
        """
        Rasterize the sky model to FITS images (one image per spectral term).

        The resulting images can be used with DDECal in DPPP for prediction using IDG
        (if the sky model is grouped into contiguous patches, a ds9 region file defining
        the Voronio patches can also written).

        Note: currently, only sky models with LogarithmicSI = False are supported at
        this time.

        Parameters
        ----------
        cellsize : float
            The cellsize in degrees for the output image.
        fileRoot : str, optional
            Filename root for the output FITS images. The images will be named
            fileRoot_0.fits, fileRoot_1.fits, etc. (one for each spectral term in
            the sky model). If writeRegionFile is True, a ds9 region file is also
            written as fileRoot.reg.
        writeRegionFile : bool, optional
            If True and the sky model is grouped into contiguous patches, a ds9 region
            file defining the Voronio patches will be written (this file is required
            for DDECal)
        clobber : bool, optional
            If True, existing files are overwritten.
        """
        import numpy as np
        from astropy.io import fits as pyfits
        from astropy import wcs
        from .operations_lib import make_template_image, gaussian_fcn, tessellate, xy2radec, radec2xy

        # Make a blank image for each spectral term
        referenceFrequency = self.getColValues('ReferenceFrequency')
        refFreq = referenceFrequency[0]  # TODO: allow per-source ref freq
        fluxes = self.getColValues('I')
        types = self.getColValues('Type')
        nsources = len(fluxes)
        if 'SpectralIndex' in self.getColNames():
            spectral_indices = self.getColValues('SpectralIndex')
        else:
            spectral_indices = [[]] * nsources
        nterms = len(spectral_indices[0]) + 1

        # Check that LogarithmicSI = False for all entries
        if nterms > 1:
            logsi = self.getColValues('LogarithmicSI')
            if np.any(logsi == 'true'):
                raise RuntimeError('Sky model has one or more sources with '
                                   'LogarithmicSI = True. Only sky models with '
                                   'LogarithmicSI = False are supported at this time.')

        image_names = ['{0}_{1}.fits'.format(fileRoot, i) for i in range(nterms)]
        x, y, refRA, refDec = self._getXY(crdelt=cellsize)
        if 'GAUSSIAN' in types:
            fwhm = np.max(self.getColValues('MajorAxis', units='degree') * cellsize)
            max_source_size = int(np.ceil(fwhm * 1.5))
        else:
            max_source_size = 2
        xpadding = int(0.2 * (np.max(x) - np.min(x)))
        ypadding = int(0.2 * (np.max(y) - np.min(y)))
        xpadding += max_source_size
        if xpadding % 2:
            xpadding += 1
        ypadding += max_source_size
        if ypadding % 2:
            ypadding += 1
        xsize = int(np.max(x) - np.min(x)) + xpadding
        ysize = int(np.max(y) - np.min(y)) + ypadding

        # Now we have the size, refine the RA, Dec of the image center
        xcen = np.min(x) + (np.max(x) - np.min(x)) / 2.0
        ycen = np.min(y) + (np.max(y) - np.min(y)) / 2.0
        refRA, refDec = xy2radec([xcen], [ycen], refRA=refRA, refDec=refDec, crdelt=cellsize)
        RA = self.getColValues('Ra')
        Dec = self.getColValues('Dec')

        for image_name in image_names:
            make_template_image(image_name, refRA[0], refDec[0], refFreq,
                                ximsize=xsize, yimsize=ysize, cellsize_deg=cellsize)

        # Build each image, one at a time (to minimize memory usage)
        for t, image_name in enumerate(image_names):
            # Read in image data
            hdu = pyfits.open(image_name, memmap=False)
            imdata = hdu[0].data
            w = wcs.WCS(hdu[0].header)

            # Loop over sources, adding them to the images (note that the spectral terms
            # sum together when summing polynomials, just as the flux densities do)
            if t == 0:
                # Flux densities
                itervalues = fluxes
            else:
                # Spectral terms
                itervalues = spectral_indices
            for i, (ra_src, dec_src, val, type) in enumerate(zip(RA, Dec, itervalues, types)):
                if t > 0:
                    v = val[t-1]
                    const = True
                else:
                    v = val
                    const = False
                ra_dec = np.array([[ra_src, dec_src, 0.0, 0.0]])
                xs, ys = w.wcs_world2pix(ra_dec, 0)[0][0], w.wcs_world2pix(ra_dec, 0)[0][1]
                if type == 'POINT':
                    imdata[0, 0, int(np.round(ys)), int(np.round(xs))] += v
                elif type == 'GAUSSIAN':
                    S1 = self.getColValues('MajorAxis', units='degree')[i] / cellsize  # pixels
                    S2 = self.getColValues('MinorAxis', units='degree')[i] / cellsize  # pixels
                    Th = self.getColValues('Orientation')[i]  # degrees
                    C1, C2 = ys, xs
                    b = np.ceil(S1 * 2.5)
                    bbox = np.s_[max(0, int(C1-b)):min(xsize, int(C1+b+1)),
                                 max(0, int(C2-b)):min(ysize, int(C2+b+1))]
                    x_ax, y_ax = np.mgrid[bbox]
                    g = [v, C1, C2, S1, S2, Th]
                    ffimg = gaussian_fcn(g, x_ax, y_ax, const=const)
                    imdata[0, 0, :, :][bbox] += ffimg
            hdu[0].data = imdata
            hdu.writeto(image_name, overwrite=True)

        # Make region file if needed
        if writeRegionFile and self.hasPatches:
            RA, Dec = self.getPatchPositions(asArray=True)
            patch_names = self.getPatchNames()
            x_pix_list = []
            y_pix_list = []
            for ra_src, dec_src in zip(RA, Dec):
                ra_dec = np.array([[ra_src, dec_src, 0.0, 0.0]])
                y_pix, x_pix = w.wcs_world2pix(ra_dec, 0)[0][0], w.wcs_world2pix(ra_dec, 0)[0][1]
                x_pix_list.append(x_pix)
                y_pix_list.append(y_pix)
            dist_pix = np.sqrt(xsize**2 + ysize**2)
            vertices = tessellate(x_pix_list, y_pix_list, w, dist_pix)
            lines = []
            lines.append('# Region file format: DS9 version 4.0\nglobal color=green '
                         'font="helvetica 10 normal" select=1 highlite=1 edit=1 '
                         'move=1 delete=1 include=1 fixed=0 source=1\nfk5\n')
            for verts, pname in zip(vertices, patch_names):
                xylist = []
                varray = np.array(verts).T
                RAs = varray[0]
                Decs = varray[1]
                for x, y in zip(RAs, Decs):
                    xylist.append('{0}, {1}'.format(x, y))
                lines.append('polygon({0}) # text={{{1}}}\n'.format(', '.join(xylist), pname))

            outputfile = '{0}.reg'.format(fileRoot)
            with open(outputfile, 'w') as f:
                f.writelines(lines)


