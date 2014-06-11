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
import _logging
import tableio
import operations


class SkyModel(object):
    """
    Object that stores the sky model and provides methods for accessing it.
    """
    def __init__(self, fileName, beamMS=None):
        """
        Initializes SkyModel object.

        Parameters
        ----------
        fileName : str
            Input ASCII file from which the sky model is read. Must
            respect the makesourcedb format
        beamMS : str, optional
            Measurement set from which the primary beam will be estimated. A
            column of attenuated Stokes I fluxes will be added to the table.

        Examples
        --------
        Create a SkyModel object::

            >>> s = SkyModel('sky.model')

        Create a SkyModel object with a beam MS so that apparent fluxes will
        be available as well as intrinsic fluxes:::

            >>> s = SkyModel('sky.model', 'SB100.MS')

        """
        from astropy.table import Table, Column

        self.table = Table.read(fileName, format='makesourcedb')
        self._fileName = fileName

        if beamMS is not None:
            self._beamMS = beamMS
            self._hasBeam = True
        else:
            self._hasBeam = False

        self._clean()
        self._patchMethod = None
        self._updateGroups()

        logging.debug("Successfully read file '{0}'".format(fileName))


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


    def _updateGroups(self, method=None):
        """
        Updates the grouping of the table by patch name.
        """
        if 'Patch' in self.table.keys():
            self.table = self.table.group_by('Patch')
            self._hasPatches = True
            if method is None:
                method = self._patchMethod
            else:
                self._patchMethod = method
            self.setPatchPositions(method=method)
        else:
            self._hasPatches = False


    def _info(self, useLogInfo=False):
        """
        Prints information about the sky model.
        """
        import numpy as np

        if self._hasPatches:
            nPatches = len(set(self.getColValues('Patch')))
        else:
            nPatches = 0

        nPoint = len(np.where(self.getColValues('Type') == 'POINT')[0])
        nGaus = len(np.where(self.getColValues('Type') == 'GAUSSIAN')[0])

        if nPatches == 1:
            plur = ''
        else:
            plur = 'es'
        if useLogInfo:
            logCall = logging.info
        else:
            logCall = logging.debug
        logCall('Model contains {0} sources in {1} patch{2} of which:\n'
            '      {3} are type POINT\n'
            '      {4} are type GAUSSIAN'.format(len(self.table), nPatches, plur,
            nPoint, nGaus))


    def info(self):
        """
        Prints information about the sky model.
        """
        self._info(useLogInfo=True)


    def copy(self):
        """
        Returns a copy of the sky model
        """
        import copy

        return copy.deepcopy(self)


    def more(self, colName=None, patchName=None, sourceName=None, sortBy=None,
        lowToHigh=False):
        """
        Prints the sky model table to the screen with more-like commands

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
            logging.error('patchName and sourceName cannot both be specified.')
            return

        table = self.table

        # Get columns
        colName = self._verifyColName(colName)
        if colName is not None:
            if type(colName) is str:
                colName = [colName] # needed in order to get a table instead of a column
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
                    logging.error('Column name "{0}" is not a valid makesourcedb '
                        'column.'.format(colName))
                return None
            else:
                colNameKey = tableio.allowedColumnNames[colNameLower]
            if colNameKey not in self.table.keys() and onlyExisting:
                if not quiet:
                    logging.error('Column name "{0}" not found in sky model.'.
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
                    logging.error("Column name{0} '{1}' not recognized. Ignoring".
                        format(plur, ','.join(badNames)))
            if len(colNameLower) == 0:
                return None
            else:
                colNameKey = [tableio.allowedColumnNames[n] for n in colNameLower]
        else:
            colNameKey = None

        return colNameKey


    def getPatchPositions(self, patchName=None):
        """
        Returns a dict of patch positions as {'patchName':(RA, Dec)}.

        Parameters
        ----------
        patchName : str or list, optional
            List of patch names for which the positions are desired

        Examples
        --------
        Get all patch positions::

            >>> s.getPatchPositions()
            {'bin0': [0.0, 0.0], 'bin1': [0.0, 0.0], 'bin2': [0.0, 0.0],
            'bin3': [0.0, 0.0]}

        """
        if self._hasPatches:
            if patchName is None:
                patchName = self.getColValues('Patch', aggregate=True)
            if type(patchName) is str:
                patchName = [patchName]
            positionDict = {}
            for patch in patchName:
                positionDict[patch] = self.table.meta[patch]
            return positionDict
        else:
            return {}


    def setPatchPositions(self, patchDict=None, method='mid'):
        """
        Sets the patch positions from the input dict of patch positions.

        Parameters
        ----------
        patchDict : dict
            Dict specifying patch names and positions as {'patchName':[RA, Dec]}
            where both RA and Dec are degrees J2000.
        method : None or str, optional
            If no patchDict is given, this parameter specifies the method used
            to set the patch positions:
            - 'mid' => the position is set to the midpoint of the patch
            - 'mean' => the positions is set to the mean RA and Ded of the patch
            - 'wmean' => the position is set to the flux-weighted mean RA and
               Dec of the patch
            - 'zero' => set all positions to [0.0, 0.0]
            - None => no changes are made

        Examples
        --------
        Set all patch positions to their midpoints::

            >>> s.setPatchPositions()

        Set all patch positions to their flux-weighted mean postions::

             >>> s.setPatchPositions(method='wmean')

        Set new position for the 'bin0' patch only::

            >>> s.setPatchPositions({'bin0': [123.231, 23.4321]})

        """
        from tableio import RA2Angle, Dec2Angle

        if self._hasPatches:
            if method is None:
                return None

            # Delete any previous patch positions
            for patchName in self.getColValues('Patch', aggregate=True):
                if patchName in self.table.meta:
                    self.table.meta.pop(patchName)

            if patchDict is None:
                patchDict = {}
                patchNames = self.getColValues('Patch', aggregate=True)
                if method == 'mid':
                    minRA = self._getMinColumn('Ra')
                    maxRA = self._getMaxColumn('Ra')
                    minDec = self._getMinColumn('Dec')
                    maxDec = self._getMaxColumn('Dec')
                    gRA = RA2Angle(minRA + (maxRA - minRA) / 2.0)
                    gDec = Dec2Angle(minDec + (maxDec - minDec) / 2.0)
                    for i, patchName in enumerate(patchNames):
                        patchDict[patchName] = [gRA[i], gDec[i]]
                elif method == 'mean':
                    RA = RA2Angle(self.getColValues('Ra', aggregate=True))
                    Dec = Dec2Angle(self.getColValues('Dec', aggregate=True))
                    for n, r, d in zip(patchNames, RA, Dec):
                        patchDict[n] = [r, d]
                elif method == 'wmean':
                    RA = RA2Angle(self.getColValues('Ra', aggregate=True, weight=True))
                    Dec = Dec2Angle(self.getColValues('Dec', aggregate=True, weight=True))
                    for n, r, d in zip(patchNames, RA, Dec):
                        patchDict[n] = [r, d]
                elif method == 'zero':
                    for n in patchNames:
                        patchDict[n] = [RA2Angle(0.0), Dec2Angle(0.0)]

            for patch, pos in patchDict.iteritems():
                self.table.meta[patch] = pos
        else:
            logging.error('Sky model does not have patches.')
            return None


    def ungroup(self):
        """
        Removes all patches from the sky model.

        Examples
        --------
        Remove all patches::

            >>> s.ungroup()

        """
        if self._hasPatches:
            for patchName in self.getColValues('Patch', aggregate=True):
                if patchName in self.table.meta:
                    self.table.meta.pop(patchName)
            self.table.remove_column('Patch')
            self._updateGroups()
            self._info()


    def getColNames(self):
        """
        Returns a list of column names.

        Examples
        --------
        Get column names::

            >>> s.getColNames()

        """
        return self.table.keys()


    def getColValues(self, colName, units=None, rowName=None,
        aggregate=False, weight=False, applyBeam=False):
        """
        Returns a numpy array of column values.

        Parameters
        ----------
        colName : str
            Name of column
        units : str, optional
            Output units (the values are converted as needed)
        rowName : str, optional
            Source or patch name. If given, returns column values for specified
            source or patch only.
        aggregate : bool, optional
            If True, the column returned will be of values aggregated
            over the patch members as follows:
                - RA, Dec, referenceFrequency, spectralIndex, orientation =>
                  average with optional weighting by Stokes I flux
                - I, Q, U, V => sum
                - majoraxis, minoraxis => patch size with optional weighting by
                  Stokes I flux
                - other columns not supported
        weight : bool, optional
            If True, aggregated values will be weighted when appropriate by the
            Stokes I flux
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam.

        Examples
        --------
        Get Stokes I fluxes in Jy::

            >>> s.getColValues('I')
            array([ 60.4892,   1.2413,   1.216 , ...,   1.12  ,   1.25  ,   1.16  ])

        Get Stokes I fluxes in mJy::

            >>> s.getColValues('I', units='mJy')
            array([ 60489.2,   1241.3,   1216. , ...,   1120. ,   1250. ,   1160. ])

        Get total Stokes I flux for the patches::

            >>> s.getColValues('I', aggregate=True)
            array([ 61.7305,   1.216 ,   3.9793, ...,   1.12  ,   1.25  ,   1.16  ])

        Get flux-weighted average RA for the patches::

            >>> s.getColValues('Ra', aggregate=True, weight=True)
            array([ 242.41450289,  243.11192   ,  243.50561817, ...,  271.51929   ,
            271.63612   ,  272.05412   ])

        """
        colName = self._verifyColName(colName)
        if colName is None:
            return None
        if type(colName) is list:
            if len(colName) > 1:
                logging.error('Only one column can be specified.')
                return None
            else:
                colName = colName[0]

        if rowName is not None:
            if rowName in self.table['Name'].data.tolist():
                sourceName = rowName
                patchName = None
            elif rowName in self.table['Patch'].data.tolist():
                sourceName = None
                patchName = rowName
            else:
                logging.error("Input row name '{0}' not found in sources or "
                    "patches.".format(rowName))
                return None
        else:
            sourceName = None
            patchName = None

        if patchName is None:
            table = self.table
        else:
            pindx = self._getNameIndx(patchName, patch=True)
            if pindx is not None:
                table = self.table.groups[pindx]
                table = table.group_by('Patch') # ensure that grouping is preseved
            else:
                return None

        if sourceName is not None:
            sindx = self._getNameIndx(sourceName)
            table = self.table[sindx]

        if aggregate and self._hasPatches:
            col = self._getAggregatedColumn(colName, weight=weight,
                table=table, applyBeam=applyBeam)
        else:
            col = table[colName]

        if col is None:
            return None

        if hasattr(col, 'filled'):
            outcol = col.filled().copy()
        else:
            outcol = col.copy()

        if units is not None:
            outcol.convert_unit_to(units)
        vals = outcol.data

        if colName.lower() == 'i' and applyBeam and self._hasBeam:
            from operations_lib import attenuate
            RADeg = table['Ra']
            DecDeg = table['Dec']
            fluxCol = outcol
            if units is not None:
                fluxCol.convert_unit_to(units)
            flux = fluxCol.data
            vals = attenuate(self._beamMS, flux, RADeg, DecDeg)

        return vals


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
            not already present in the table

        Examples
        --------
        Set Stokes I fluxes::

            >>> s.setColValues('I', [1.0, 1.1, 1.2, 0.0, 1.3], mask=[False,
                    False, False, True, False])

        """
        from astropy.table import Column
        from tableio import RA2Angle, Dec2Angle
        import numpy as np

        colName = self._verifyColName(colName, onlyExisting=False)
        if colName is None:
            return None
        if type(colName) is list:
            if len(colName) > 1:
                logging.error('Only one column can be specified.')
                return
            else:
                colName = colName[0]

        if isinstance(values, dict):
            if colName in self.table.keys():
                data = self.table[colName].data
                mask = self.table[colName].mask
            else:
                data = [0] * len(self.table)
                mask = [True] * len(self.table)
            for sourceName, value in values.iteritems():
                indx = self._getNameIndx(sourceName)
                if colName == 'Ra':
                    val = RA2Angle(value)[0]
                elif colName == 'Dec':
                    val = Dec2Angle(value)[0]
                else:
                    val = value
                data[indx] = value
                mask[indx] = False
        else:
            if len(values) != len(self.table):
                logging.error('Length of input values must match length of table.')
                return
            else:
                if colName == 'Ra':
                    vals = RA2Angle(values)
                elif colName == 'Dec':
                    vals = Dec2Angle(values)
                else:
                    vals = values
                data = values

        if mask is not None:
            data = np.ma.masked_array(data, mask)
        else:
            data = np.array(data)
        if colName in self.table.keys():
            self.table[colName] = data
        else:
            newCol = Column(name=colName, data=data)
            self.table.add_column(newCol, index=index)


    def getRowValues(self, rowName, colName=None):
        """
        Returns an astropy table or table row for specified source or patch.

        Parameters
        ----------
        rowName : str
            Name of the source or patch
        colName : str, optional
            Column name. If given, returns row values for specified
            column only.

        Examples
        --------
        Get row values for the source 'src1'::

            >>> r = s.getRowValues('src1')

        Sum over the fluxes of sources in the 'bin1' patch::

            >>> rows = s.getRowValues('bin1')
            >>> for r in rows: tot += r['I']

        """
        if colName is not None:
            colName = self._verifyColName(colName)

        sourceNames = self.getColValues('Name')
        patchNames = self.getColValues('Patch', aggregate=True)
        if rowName in sourceNames:
            indx = self._getNameIndx(rowName)
            if colName is not None:
                return self.table[colName].filled()[indx]
            else:
                return self.table.filled()[indx]
        elif rowName in patchNames:
            pindx = self._getNameIndx(rowName, patch=True)
            table = self.table.groups[pindx]
            table = table.group_by('Patch') # ensure that grouping is preserved
            return table
        else:
            logging.error("Row name '{0}' not recognized.".format(rowName))
            return None


    def getRowIndex(self, rowName):
        """
        Returns index or indices for specified source or patch as a list.

        Parameters
        ----------
        rowName : str
            Name of the source or patch

        Examples
        --------
        Get row index for the source 'src1'::

            >>> s.getRowIndex('src1')
            [0]

        Get row indices for the patch 'bin1' and verify the patch name::

            >>> ind = s.getRowIndex('bin1')
            >>> print(s.getColValues('Patch')[ind])
            ['bin1' 'bin1']

        """
        import numpy as np

        sourceNames = self.getColValues('Name')
        patchNames = self.getColValues('Patch')
        if rowName in sourceNames:
            return self._getNameIndx(rowName)
        elif rowName in patchNames:
            return np.where(patchNames == rowName)[0].tolist()
        else:
            logging.error("Row name '{0}' not recognized.".format(rowName))
            return None


    def setRowValues(self, values, mask=None):
        """
        Sets values for a single row.

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

        Examples
        --------
        Set row values for the source 'src1'::

            >>> s.setRowValues({'Name':'src1', 'Ra':213.123, 'Dec':23.1232,
                'I':23.2, 'Type':'POINT'}

        The RA and Dec values can be in degrees (as above) or in makesourcedb
        format. E.g.::

            >>> s.setRowValues({'Name':'src1', 'Ra':'12:22:21.1',
                'Dec':'+14.46.31.5', 'I':23.2, 'Type':'POINT'}

        """
        from tableio import RA2Angle, Dec2Angle
        import numpy as np

        requiredValues = ['Name', 'Ra', 'Dec', 'I', 'Type']
        if self._hasPatches:
            requiredValues.append('Patch')

        if isinstance(values, dict):
            verifiedValues = {}
            for valReq in requiredValues:
                found = False
                for val in values:
                    if self._verifyColName(valReq) == self._verifyColName(val):
                        found = True
                        verifiedValues[self._verifyColName(val)] = values[val]
                if not found:
                    logging.error("A value must be specified for '{0}'.".format(valReq))
                    return 1

            RA = verifiedValues['Ra']
            Dec = verifiedValues['Dec']
            try:
                verifiedValues['Ra'] = RA2Angle(RA)[0].value
                verifiedValues['Dec'] = Dec2Angle(Dec)[0].value
            except:
                logging.error('RA and/or Dec not understood.')
                return 1

            rowName = str(values['Name'])
            indx = self._getNameIndx(rowName)
            if indx is None:
                self.table.add_row(verifiedValues)
            else:
                for colName, value in verifiedValues.iteritems():
                    self.table[colName][indx] = value
                    self.table[colName][indx].mask = False
        elif type(dict) is list:
            if len(values) != len(self.table.columns):
                logging.error('Length of input values must match number of tables.')
                return 1
            else:
                self.table.add_row(values, mask=mask)
        else:
            logging.error('Input row values not understood.')
            return 1

        if self._hasPatches:
            self._updateGroups(method='mid')
        return 0


    def _getNameIndx(self, name, patch=False):
        """
        Returns a list of indices corresponding to the given names.

        Parameters
        ----------
        name : str, list of str
            source or patch name or list of names
        patch : bool
            if True, return the index of the group corresponding to the given
            name; otherwise return the index of the source

        """
        import numpy as np

        if patch:
            if self._hasPatches:
                names = self.getColValues('Patch', aggregate=True).tolist()
            else:
                return None
        else:
            names = self.getColValues('Name').tolist()

        if type(name) is str or type(name) is np.string_:
            if name not in names:
                return None
            indx = names.index(name)
            return [indx]
        elif type(name) is list:
            indx = []
            for n in name:
                badNames = []
                if n not in names:
                    badNames.append(n)
                else:
                    indx.append(names.index(n))
            if len(badNames) > 0:
                if len(badNames) == 1:
                    plur = ''
                else:
                    plur = 's'
                logging.warn("Name{0} '{1}' not recognized. Ignoring.".
                    format(plur, ','.join(badNames)))
            if len(indx) == 0:
                logging.error("None of the specified names were found.")
                return None
            return indx
        else:
            return None


    def _getAggregatedColumn(self, colName, weight=False, table=None,
        applyBeam=False):
        """
        Returns the appropriate colum values aggregated by group.

        Parameters
        ----------
        colName : str
            Name of column. If not already present in the table, a new column
            will be created.
        weight : bool, optional
            If True, aggregated values will be weighted when appropriate by the
            Stokes I flux
        table : astropy Table, optional
            If given, use this table; otherwise use self.table
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam.

         """
        colsToAverage = ['Ra', 'Dec', 'ReferenceFrequency', 'Orientation',
            'SpectralIndex']
        colsToSum = ['I', 'Q', 'U', 'V']
        colName = self._verifyColName(colName)
        if colName is None:
            return None

        if colName in colsToAverage:
            col = self._getAveragedColumn(colName, weight=weight,
                table=table, applyBeam=applyBeam)
        elif colName in colsToSum:
            col = self._getSummedColumn(colName, table=table)
        elif colName == 'MajorAxis' or colName == 'MinorAxis':
            col = self._getSizeColumn(weight=weight, table=table,
                applyBeam=applyBeam)
        elif colName == 'Patch':
            col = self.table.groups.keys['Patch']
        else:
            logging.error('Column {0} cannot be aggregated.'.format(colName))
            col = None
        return col


    def _getSummedColumn(self, colName, table=None):
        """
        Returns column summed by group.

        Parameters
        ----------
        colName : str
            Column name
        table : astropy Table, optional
            If given, use this table; otherwise use self.table

        """
        import numpy as np

        if table is None:
            table = self.table
        return table[colName].groups.aggregate(np.sum)


    def _getMinColumn(self, colName, table=None):
        """
        Returns column minimum value by group.

        Parameters
        ----------
        colName : str
            Column name
        table : astropy Table, optional
            If given, use this table; otherwise use self.table

        """
        import numpy as np

        if table is None:
            table = self.table
        return table[colName].groups.aggregate(np.min)


    def _getMaxColumn(self, colName, table=None):
        """
        Returns column maximum value by group.

        Parameters
        ----------
        colName : str
            Column name
        table : astropy Table, optional
            If given, use this table; otherwise use self.table

        """
        import numpy as np

        if table is None:
            table = self.table
        return table[colName].groups.aggregate(np.max)


    def _getAveragedColumn(self, colName, weight=True, table=None,
        applyBeam=False):
        """
        Returns column averaged by group.

        Parameters
        ----------
        colName : str
            Column name
        weight : bool, optional
            If True, return average weighted by flux
        table : astropy Table, optional
            If given, use this table; otherwise use self.table
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam.

        """
        from astropy.table import Column
        import numpy as np

        if table is None:
            table = self.table

        if weight:
            vals = self.getColValues(colName)
            if weight:
                weights = self.getColValues('I', applyBeam=applyBeam)
                weightCol = Column(name='Weight', data=weights)
                valWeightCol = Column(name='ValWeight', data=vals*weights)
                table.add_column(valWeightCol)
                table.add_column(weightCol)
                numer = table['ValWeight'].groups.aggregate(np.sum).data
                denom = table['Weight'].groups.aggregate(np.sum).data
                table.remove_column('ValWeight')
                table.remove_column('Weight')
            else:
                valCol = Column(name='Val', data=vals)
                table.add_column(valCol)
                numer = table['Val'].groups.aggregate(np.sum).data
                demon = 1.0
                table.remove_column('Val')
            return Column(name=colName, data=np.array(numer/denom),
                units=self.table[colName].units.name)
        else:
            def avg(c):
                return np.average(c, axis=0)
            return table[colName].groups.aggregate(avg)


    def _getSizeColumn(self, weight=True, table=None, applyBeam=False):
        """
        Returns column of source largest angular sizes.

        Parameters
        ----------
        weight : bool, optional
            If True, return size weighted by flux
        table : astropy Table, optional
            If given, use this table; otherwise use self.table
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam.

        """
        from astropy.table import Column
        import numpy as np

        if table is None:
            table = self.table
        if self._hasPatches:
            # Get weighted average RAs and Decs
            RAAvg = self._getAveragedColumn('Ra', weight=weight, table=table)
            DecAvg = self._getAveragedColumn('Dec', weight=weight, table=table)

            # Fill out the columns by repeating the average value over the
            # entire group
            RAAvgFull = np.zeros(len(table), dtype=np.float)
            DecAvgFull = np.zeros(len(table), dtype=np.float)
            for i, ind in enumerate(table.groups.indices[1:]):
                RAAvgFull[table.groups.indices[i]: ind] = RAAvg[i]
                DecAvgFull[table.groups.indices[i]: ind] = DecAvg[i]

            dist = self._calculateSeparation(table['Ra'],
                table['Dec'], RAAvgFull, DecAvgFull)
            if weight:
                if applyBeam and self._hasBeam:
                    appFluxes = self.getColValues('I', applyBeam=True)
                    weightCol = Column(name='Weight', data=appFluxes)
                    valWeightCol = Column(name='ValWeight', data=dist*appFluxes)
                else:
                    weightCol = Column(name='Weight', data=table['I'].data)
                    valWeightCol = Column(name='ValWeight', data=dist*table['I'].data)
                table.add_column(valWeightCol)
                table.add_column(weightCol)
                numer = table['ValWeight'].groups.aggregate(np.sum).data * 2.0
                denom = table['Weight'].groups.aggregate(np.sum).data
                table.remove_column('ValWeight')
                table.remove_column('Weight')
                col = Column(name='Size', data=numer/denom,
                    units='degree')
            else:
                valCol = Column(name='Val', data=dist)
                table.add_column(valCol)
                size = table['Val'].groups.aggregate(np.max).data * 2.0
                table.remove_column('Val')
                col = Column(name='Size', data=size, units='degree')
        else:
            if 'majoraxis' in table.colnames:
                col = table['MajorAxis']
            else:
                col = Column(name='Size', data=np.zeros(len(table)), units='degree')

        if hasattr(col, 'filled'):
            outcol = col.filled(fill_value=0.0)
        else:
            outcol = col
        outcol.convert_unit_to('arcsec')

        return outcol


    def _calculateSeparation(self, ra1, dec1, ra2, dec2):
        """
        Returns angular separation between two coordinates (all in degrees).

        Parameters
        ----------
        ra1 : float
            RA of coordinate 1 in degrees
        dec1 : float
            Dec of coordinate 1 in degrees
        ra2 : float
            RA of coordinate 2 in degrees
        dec2 : float
            Dec of coordinate 2 in degrees

        """
        from astropy.coordinates import FK5
        import astropy.units as u

        coord1 = FK5(ra1, dec1, unit=(u.degree, u.degree))
        coord2 = FK5(ra2, dec2, unit=(u.degree, u.degree))

        return coord1.separation(coord2)


    def write(self, fileName=None, format='makesourcedb', clobber=False, sortBy=None,
        lowToHigh=False):
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
                - plus all other formats supported by the astropy.table package
        clobber : bool, optional
            If True, an existing file is overwritten.
        sortBy : str or list of str, optional
            Name of columns to sort on. If None, no sorting is done. If
            a list is given, sorting is done on the columns in the order given
        lowToHigh : bool, optional
            If True, sort values from low to high instead of high to low

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
        if fileName is None:
            fileName = self._fileName

        if os.path.exists(fileName):
            if clobber:
                os.remove(fileName)
            else:
                logging.error("The output file '{0}' exists and clobber = False.".
                    format(fileName))
                return

        table = self.table

        # Sort if desired
        if sortBy is not None:
            colName = self._verifyColName(sortBy)
            indx = table.argsort(colName)
            if not lowToHigh:
                indx = indx[::-1]
            table = table[indx]

        table.write(fileName, format=format)


    def _clean(self):
        """
        Removes duplicate entries.
        """
        filt = []
        for i, name in enumerate(self.getColValues('Name')):
            if name not in filt:
                filt.append(i)
        nRowsOrig = len(self.table)
        self.table = self.table[filt]
        nRowsNew = len(self.table)
        if nRowsOrig-nRowsNew > 0:
            logging.info('Removed {0} duplicate sources.'.format(nRowsOrig-nRowsNew))


    def select(self, *args, **kwargs):
        """
        Selects table rows on column values with the given expression.

        See operations.select.select() for details.
        """
        operations.select.select(self, *args, **kwargs)


    def remove(self, *args, **kwargs):
        """
        Removes table rows on column values with the given expression.

        See operations.remove.remove() for details.
        """
        operations.remove.remove(self, *args, **kwargs)


    def group(self, algorithm, targetFlux=None, numClusters=100, applyBeam=False,
        method='mid'):
        """
        Groups sources into patches

        Parameters
        ----------
        algorithm : str
            Algorithm to use for grouping:
            - 'single' => all sources are grouped into a single patch
            - 'every' => every source gets a separate patch
            - 'cluster' => SAGECAL clustering algorithm that groups sources into
                specified number of clusters (specified by the numClusters parameter).
            - 'tessellate' => group into tiles whose total flux approximates
                the target flux (specified by the targetFlux parameter).
        targetFlux : str or float, optional
            Target flux for tessellation (the total flux of each tile will be close
            to this value). The target flux can be specified as either a float in Jy
            or as a string with units (e.g., '25.0 mJy').
        numClusters : int, optional
            Number of clusters for clustering. Sources are grouped around the
            numClusters brightest sources.
        applyBeam : bool, optional
            If True, fluxes will be attenuated by the beam.
        method : str, optional
            Method by which patch positions will be calculated:
            - 'mid' => use the midpoint of the patch
            - 'mean' => use the mean position
            - 'wmean' => use the flux-weighted mean position

        Examples
        --------
        Tesselate the sky model into patches with approximately 30 Jy total
        flux:

            >>> s.group('tessellate', targetFlux=30.0)

        """
        operations.group.group(self, algorithm, targetFlux, numClusters, applyBeam,
        method)


    def transfer(self, *args, **kwargs):
        """
        Transfers the patch scheme from the input sky model.

        See operations.transfer.transfer() for details.
        """
        operations.transfer.transfer(self, *args, **kwargs)


    def move(self, *args, **kwargs):
        """
        Moves a source or patch.

        See operations.move.move() for details.
        """
        operations.move.move(self, *args, **kwargs)


    def add(self, colNamesVals):
        """
        Add a source to the sky model.

        Parameters
        ----------
        colNamesVals : dict
            A dictionary that specifies the row values for the source to be added.

        Examples:
        ---------
        Add a point source::

            >>> source = {'Name':'src1', 'Type':'POINT', 'Ra':'12:32:10.1',
                'Dec':'23.43.21.21', 'I':2.134}
            >>> s.add(source)

        """
        operations.add.add(self, colNamesVals)


    def merge(self, *args, **kwargs):
        """
        Merges two or more patches.

        See operations.merge.merge() for details.
        """
        operations.merge.merge(self, *args, **kwargs)


    def concatenate(self, LSM2, matchBy='name', radius=0.1, keep='all'):
        """
        Concatenate two sky models

        Parameters
        ----------
        LSM2 : SkyModel object
            Sky model to concatenate with the parent sky model
        matchBy : str, optional
            Determines how duplicate sources are determined:
            - 'name' => duplicates are identified by name
            - 'position' => duplicates are identified by radius. Sources within the
                radius specified by the radius parameter are considered duplicates
        radius : float, optional
            Radius in degrees for matching when matchBy='position'
        keep : str, optional
            Determines how duplicates are treated:
            - 'all' => all duplicates are kept; those with identical names are re-
                named
            - 'from1' => duplicates kept are those from sky model 1 (the parent)
            - 'from2' => duplicates kept are those from sky model 2 (LSM2)

        Examples
        --------
        Concatenate two sky models, identifying duplicates by matching to the source
        names. When duplicates are found, keep the source from the parent sky model
        and discard the duplicate from second sky model (this might be useful when
        merging two gsm.py sky models that have some overlap)::

            >>> LSM2 = lsmtool.load('gsm_sky2.model')
            >>> s.concatenate(LSM2, matchBy='name', keep='from1')

        Concatenate two sky models, identifying duplicates by matching to the source
        positions within a radius of 10 arcsec. When duplicates are found, keep the
        source from the second sky model and discard the duplicate from the parent
        sky model (this might be useful when replacing parts of a low-resolution
        sky model with a high-resolution one)::

            >>> LSM2 = lsmtool.load('high_res_sky.model')
            >>> s.concatenate(LSM2, matchBy='position', radius=10.0/3600.0,
                keep='from2')

        """
        operations.concatenate.concatenate(self, LSM2, matchBy, radius, keep)


    def plot(self, *args, **kwargs):
        """
        Plot the sky model.

        See operations.plot.plot() for details.
        """
        operations.plot.plot(self, *args, **kwargs)


# Forward the operation doc strings to the appropriate methods of the SkyModel
# object.
SkyModel.remove.__func__.__doc__ = operations.remove.remove.__doc__
SkyModel.select.__func__.__doc__ = operations.select.select.__doc__
SkyModel.transfer.__func__.__doc__ = operations.transfer.transfer.__doc__
SkyModel.move.__func__.__doc__ = operations.move.move.__doc__
SkyModel.plot.__func__.__doc__ = operations.plot.plot.__doc__
SkyModel.merge.__func__.__doc__ = operations.merge.merge.__doc__

