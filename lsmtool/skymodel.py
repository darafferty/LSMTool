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


def load(fileName):
    """
    Loads a sky model from a file and returns a SkyModel object.

    Parameters
    ----------
    fileName : str
        Input ASCII file from which the sky model is read. Must
        respect the makesourcedb format

    Examples
    --------
    Create a SkyModel object::

        >>> from lsmtool import skymodel
        >>> s = skymodel.load('sky.model')

    """
    return SkyModel(fileName)


class SkyModel(object):
    """
    Object that stores the sky model and provides methods for accessing it.
    """
    def __init__(self, fileName):
        """
        Initializes SkyModel object.

        Parameters
        ----------
        fileName : str
            Input ASCII file from which the sky model is read. Must
            respect the makesourcedb format

        Examples
        --------
        Create a SkyModel object::

            >>> from lsmtool.skymodel import SkyModel
            >>> s = SkyModel('sky.model')

        """
        from astropy.table import Table

        self.table = Table.read(fileName, format='makesourcedb')
        self._fileName = fileName

        if 'Patch' in self.table.keys():
            self._hasPatches = True
        else:
            self._hasPatches = False

        logging.info("Successfully read file '{0}'".format(fileName))
        self._clean()
#         self.info()


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
        self.table = self.table.group_by('Patch')
        self._hasPatches = True
        self.setPatchPositions(method=method)


    def info(self):
        """
        Prints information about the sky model.
        """
        if self._hasPatches:
            nPatches = len(self.table.groups)
        else:
            nPatches = 0

        g = self.table.group_by('Type')
        nPoint = 0
        nGaus = 0
        for grp in g.groups:
            if grp['Type'][0].lower() == 'point':
                nPoint = len(grp)
            if grp['Type'][0].lower() == 'gaussian':
                nGaus = len(grp)

        if nPatches == 1:
            plur = ''
        else:
            plur = 'es'
        logging.info('Model contains {0} sources in {1} patch{2} of which:\n'
            '      {3} are type POINT\n'
            '      {4} are type GAUSSIAN'.format(len(self.table), nPatches, plur,
            nPoint, nGaus))


    def show(self, colName=None, patchName=None, sourceName=None, more=False):
        """
        Prints the sky model table to the screen.

        Parameters
        ----------
        colName : str, list of str, optional
            Name of column or columns to print. If None, all columns are printed
        patchName : str, list of str, optional
            If given, returns column values for specified patch or patches only
        sourceName : str, list of str, optional
            If given, returns column value for specified source or sources only
        more : bool, optional
            If True, allows interactive paging

        Examples
        --------
        Print the entire model::

            >>> s = SkyModel('sky.model')
            >>> s.show()

        Page through the model using more-like commands::

            >>> s.show(more=True)

        Print only the 'Name' and 'I' columns for the 'bin0' patch::

            >>> s.show(['Name', 'I'], 'bin0')

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

        if more:
            table.more(show_unit=True)
        else:
            table.pprint(show_unit=True)


    def _verifyColName(self, colName, onlyExisting=True):
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

        Returns
        -------
        colName : str, None
            Properly formatted name of column or None if colName not found
        """
        if type(colName) is str:
            colNameLower = colName.lower()
            if colNameLower not in tableio.allowedColumnNames:
                logging.error('Column name "{0}" is not a valid makesourcedb '
                    'column.'.format(colName))
                return None
            else:
                colNameKey = tableio.allowedColumnNames[colNameLower]
            if colNameKey not in self.table.keys() and onlyExisting:
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
                logging.error("Column name{0} '{1}' not recognized. Ignoring".
                    format(plur, ','.join(badNames)))
            if len(colNameLower) == 0:
                return None
            else:
                colNameKey = [tableio.allowedColumnNames[n] for n in colNameLower]
        else:
            colNameKey = None

        return colNameKey


    def sort(self, colName=None, reverse=False):
        """
        Sorts the sky model table by column values (high to low).

        Parameters
        ----------
        colName : str or list of str, optional
            Name of columns to sort on. If None, the Stokes I flux is used. If
            a list is given, sorting is done on the columns in the order given.
        reverse : bool, optional
            If True, sort from low to high instead

        Examples
        --------
        Sort on Stokes I flux, with largest values first::

            >>> s = SkyModel('sky.model')
            >>> s.sort()

        Sort on RA, with smallest values first::

            >>> s.sort('RA', reverse=True)

        Sort on Patch name, and sort sources in each patch by Stokes I flux,
        with largest values first::

            >>> s.sort(['Patch', 'I'])

        """
        if colName is None:
            colName = 'I'
            logging.info('No column name specified. Sorting on Stokes I flux.')

        colName = self._verifyColName(colName)
        self.table.sort(colName)

        if not reverse:
            self.table.reverse()


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

            >>> s = SkyModel('sky.model')
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
            - None => set all positions to [0.0, 0.0]

        Examples
        --------
        Set all patch positions to their midpoints::

            >>> s = SkyModel('sky.model')
            >>> s.setPatchPositions()

        Set all patch positions to their flux-weighted mean postions::

             >>> s.setPatchPositions(method='wmean')

        Set new position for the 'bin0' patch only::

            >>> s.setPatchPositions({'bin0': [123.231, 23.4321]})

        """
        if self._hasPatches:
            # Delete any previous patch positions
            for patchName in self.getColValues('Patch', aggregate=True):
                if patchName in self.table.meta:
                    self.table.meta.pop(patchName)

            if patchDict is None:
                patchDict = {}
                patchNames = self.getColValues('Patch', aggregate=True)
                if method == 'mid':
                    minRA = self._getMinColumn('RA')
                    maxRA = self._getMaxColumn('RA')
                    minDec = self._getMinColumn('Dec')
                    maxDec = self._getMaxColumn('Dec')
                    gRA = minRA + (maxRA - minRA) / 2.0
                    gDec = minDec + (maxDec - minDec) / 2.0
                    for i, patchName in enumerate(patchNames):
                        patchDict[patchName] = [gRA[i], gDec[i]]
                elif method == 'mean':
                    RA = self.getColValues('RA', aggregate=True)
                    Dec = self.getColValues('Dec', aggregate=True)
                    for n, r, d in zip(patchNames, RA, Dec):
                        patchDict[n] = [r, d]
                elif method == 'wmean':
                    RA = self.getColValues('RA', aggregate=True, weight=True)
                    Dec = self.getColValues('Dec', aggregate=True, weight=True)
                    for n, r, d in zip(patchNames, RA, Dec):
                        patchDict[n] = [r, d]
                else:
                    for n in patchNames:
                        patchDict[n] = [0.0, 0.0]

            for patch, pos in patchDict.iteritems():
                self.table.meta[patch] = pos
        else:
            logging.error('Sky model does not have patches.')
            return None


    def ungroup(self):
        """
        Removes all patches from the sky model.
        """
        if self._hasPatches:
            for patchName in self.getColValues('Patch', aggregate=True):
                if patchName in self.table.meta:
                    self.table.meta.pop(patchName)
            self.table.remove_column('Patch')
            self._hasPatches = False


    def getColValues(self, colName, units=None, rowName=None,
        aggregate=False, weight=False, beamMS=None):
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
        beamMS : string, optional
            Measurement set from which the primary beam will be estimated. If
            beamMS is specified, fluxes will be attenuated by the beam.

        Examples
        --------
        Get Stokes I fluxes in Jy::

            >>> s = SkyModel('sky.model')
            >>> s.getColValues('I')
            array([ 60.4892,   1.2413,   1.216 , ...,   1.12  ,   1.25  ,   1.16  ])

        Get Stokes I fluxes in mJy::

            >>> s.getColValues('I', units='mJy')
            array([ 60489.2,   1241.3,   1216. , ...,   1120. ,   1250. ,   1160. ])

        Get total Stokes I flux for the patches::

            >>> s.getColValues('I', aggregate=True)
            array([ 61.7305,   1.216 ,   3.9793, ...,   1.12  ,   1.25  ,   1.16  ])

        Get flux-weighted average RA for the patches::

            >>> s.getColValues('RA', aggregate=True, weight=True)
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
                table=table)
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

        if beamMS is not None and colName == 'I':
            from operations_lib import applyBeam
            RADeg = table['RA']
            DecDeg = table['Dec']
            vals = applyBeam(beamMS, outcol.data, RADeg, DecDeg)
        else:
            vals = outcol.data

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
            dict, missing values will be masked unless already present.
        mask : list or array of bools, optional
            If values is a list or array, a mask can be specified (True means
            the value is masked).
        index : int, optional
            Index that specifies the column position in the table, if column is
            not already present in the table

        Examples
        --------
        Set Stokes I fluxes::

            >>> s = SkyModel('sky.model')
            >>> s.setColValues('I', [1.0, 1.1, 1.2, 0.0, 1.3], mask=[False,
                    False, False, True, False])

        """
        from astropy.table import Column
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
                data[indx] = value
                mask[indx] = False
        else:
            if len(values) != len(self.table):
                logging.error('Length of input values must match length of table.')
                return
            else:
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

            >>> s = SkyModel('sky.model')
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

            >>> s = SkyModel('sky.model')
            >>> s.getRowIndex('src1')
            [0]

        Get row indices for the patch 'bin1' and verify the patch name::

            >>> s = SkyModel('sky.model')
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
        Sets row values

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

            >>> s = SkyModel('sky.model')
            >>> s.setRowValues({'Name':'src1', 'RA':213.123, 'Dec':23.1232,
                'I':23.2, 'Type':'POINT'}

        """
        requiredValues = ['Name', 'RA', 'Dec', 'I', 'Type']
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
            rowName = str(values['Name'])
            indx = self._getNameIndx(rowName)
            if indx is None:
                self.table.add_row(verifiedValues)
            else:
                for colName, value in verifiedValues.iteritems():
                    self.table[colName][indx] = value
                    self.table[colName][indx].mask = False
        else:
            if len(values) != len(self.table.columns):
                logging.error('Length of input values must match number of tables.')
                return 1
            else:
                self.table.add_row(values, mask=mask)

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


    def _getAggregatedColumn(self, colName, weight=False, table=None):
        """
        Returns the appropriate colum values aggregated by group.
        """
        colsToAverage = ['RA', 'Dec', 'ReferenceFrequency',
            'SpectralIndex', 'Orientation']
        colsToSum = ['I', 'Q', 'U', 'V']
        colName = self._verifyColName(colName)
        if colName is None:
            return None

        if colName in colsToAverage:
            col = self._getAveragedColumn(colName, weight=weight,
                table=table)
        elif colName in colsToSum:
            col = self._getSummedColumn(colName, table=table)
        elif colName == 'MajorAxis' or colName == 'MinorAxis':
            col = self._getSizeColumn(weight=weight, table=table)
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


    def _getAveragedColumn(self, colName, weight=True, table=None):
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
        """
        from astropy.table import Column
        import numpy as np

        if table is None:
            table = self.table
        if weight:
            weightCol = Column(name='Weight', data=table['I'].filled().data)
            valWeightCol = Column(name='ValWeight', data=table[colName].filled().data*
                table['I'].filled().data)
            table.add_column(valWeightCol)
            table.add_column(weightCol)
            numer = table['ValWeight'].groups.aggregate(np.sum).data
            denom = table['Weight'].groups.aggregate(np.sum).data
            table.remove_column('ValWeight')
            table.remove_column('Weight')
            return Column(name=colName, data=np.array(numer/denom),
                units=self.table[colName].units.name)
        else:
            def avg(c):
                return np.average(c, axis=0)
            return table[colName].groups.aggregate(avg)


    def _getSizeColumn(self, weight=True, table=None):
        """
        Returns column of source largest angular sizes.

        Parameters
        ----------
        weight : bool, optional
            If True, return size weighted by flux
        table : astropy Table, optional
            If given, use this table; otherwise use self.table
        """
        from astropy.table import Column
        import numpy as np

        if table is None:
            table = self.table
        if self._hasPatches:
            # Get weighted average RAs and Decs
            RAAvg = self._getAveragedColumn('RA', weight=weight, table=table)
            DecAvg = self._getAveragedColumn('Dec', weight=weight, table=table)

            # Fill out the columns by repeating the average value over the
            # entire group
            RAAvgFull = np.zeros(len(table), dtype=np.float)
            DecAvgFull = np.zeros(len(table), dtype=np.float)
            for i, ind in enumerate(table.groups.indices[1:]):
                RAAvgFull[table.groups.indices[i]: ind] = RAAvg[i]
                DecAvgFull[table.groups.indices[i]: ind] = DecAvg[i]

            dist = self._calculateSeparation(table['RA'],
                table['Dec'], RAAvgFull, DecAvgFull)
            if weight:
                weightCol = Column(name='Weight', data=table['I'].filled().data)
                valWeightCol = Column(name='ValWeight', data=dist*table['I'].filled().data)
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


    def write(self, fileName=None, format='makesourcedb', clobber=False):
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

        Examples
        --------
        Write the model to a makesourcedb sky model file suitable for use with
        BBS::

            >>> s = SkyModel('sky.model')
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

        self.table.write(fileName, format=format)


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


    def group(self, *args, **kwargs):
        """
        Groups sources into patches.

        See operations.group.group() for details.
        """
        operations.group.group(self, *args, **kwargs)


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


    def add(self, *args, **kwargs):
        """
        Adds a source.

        See operations.add.add() for details.
        """
        operations.add.add(self, *args, **kwargs)


    def merge(self, *args, **kwargs):
        """
        Merges two or more patches.

        See operations.merge.merge() for details.
        """
        operations.merge.merge(self, *args, **kwargs)


    def concatenate(self, *args, **kwargs):
        """
        Concatenate two sky models.

        See operations.concatenate.concatenate() for details.
        """
        operations.concatenate.concatenate(self, *args, **kwargs)


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
SkyModel.group.__func__.__doc__ = operations.group.group.__doc__
SkyModel.transfer.__func__.__doc__ = operations.transfer.transfer.__doc__
SkyModel.move.__func__.__doc__ = operations.move.move.__doc__
SkyModel.add.__func__.__doc__ = operations.add.add.__doc__
SkyModel.plot.__func__.__doc__ = operations.plot.plot.__doc__
SkyModel.merge.__func__.__doc__ = operations.merge.merge.__doc__
SkyModel.concatenate.__func__.__doc__ = operations.concatenate.concatenate.__doc__

