"""
Defines the SkyModel object used by LSMTool for all operations.
"""

import contextlib
import copy
import datetime
import fnmatch
import logging
import os
import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits as pyfits
from astropy.io.ascii import InconsistentTableError
from astropy.table import Column, Table
from astropy.wcs import WCS

# relative
from . import operations, tableio
from .operations_lib import (
    apply_beam as apply_beam_operation,
)
from .operations_lib import (
    calculateSeparation,
    gaussian_fcn,
    make_template_image,
    make_wcs,
    normalize_ra_dec,
)
from .tableio import createTable, processFormatString, processLine


class SkyModel(object):
    """
    Object that stores the sky model and provides methods for accessing it.
    """

    def __init__(
        self,
        filename,
        beamMS=None,
        checkDup=False,
        VOPosition=None,
        VORadius=None,
    ):
        """
        Initializes SkyModel object.

        Parameters
        ----------
        filename : str
            Input ASCII file from which the sky model is read (must respect the
            makesourcedb format or the LSM/GSM format), name of VO service to
            query (must be one of 'GSM', 'LOTSS', 'NVSS', 'TGSS', 'VLSSR', or
            'WENSS'), or dict (single source only).
        beamMS : str, optional
            Measurement set from which the primary beam will be estimated. A
            column of attenuated Stokes I fluxes will be added to the table
        checkDup: bool, optional
            If True, the sky model is checked for duplicate sources (with the
            same name).
        VOPosition : list of floats
            A list specifying a new position as [RA, Dec] in either makesourcedb
            format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
            [123.2312, 23.3422]) for a cone search.
        VORadius : float or str, optional
            Radius in degrees (if float) or 'value unit' (if str; e.g.,
            '30 arcsec') for cone search region in degrees.

        Examples
        --------
        Create a SkyModel object::

        >>> s = SkyModel("sky.model")

        Create a SkyModel object with a beam MS so that apparent fluxes will
        be available::

        >>> s = SkyModel("sky.model", beamMS="SB100.MS")

        Load a WENSS catalog into a SkyModel object::

        >>> s = SkyModel(
        ...     "WENSS", VOPosition=[212.8352792, 52.202644], VORadius=5.0
        ... )
        """

        beam_ms = beamMS
        check_dup = checkDup
        vo_position = VOPosition
        vo_radius = VORadius

        self.log = logging.getLogger("LSMTool")
        self.history = []
        self._filename = None

        if isinstance(filename, (str, Path)):
            # First check if filename points to a VO query
            if vo_position is not None and vo_radius is not None:
                self.table = self.load_from_remote_service(
                    filename, vo_position, vo_radius
                )
            else:
                self.table = self.load_from_file(filename)

        elif isinstance(filename, dict):
            self.table = self.load_from_dict(filename)
        else:
            raise ValueError("Filename not understood. Exiting...")

        if beam_ms is not None:
            self.beamMS = beam_ms
            self._has_beam = True
            self.beamTime = 0.5
        else:
            self.beamMS = None
            self._has_beam = False
            self.beamTime = None

        if check_dup:
            self.log.debug("Checking model for duplicate lines...")
            self._clean()

        self.log.debug("Processing patches (if any)...")
        self._patch_method = None
        self._updateGroups()

        self.log.debug("Successfully loaded sky model")

    def __len__(self):
        """
        Returns the table len() value (number of rows).
        """
        return len(self.table)

    def __str__(self):
        """
        Returns string with info about sky model contents.
        """
        return str(self.table)

    def load_from_file(self, filename):
        """
        Load a SkyModel from a local file.

        This method attempts to load a sky model from the specified local file.
        It first checks if the file is in the LSM format and uses the
        appropriate loader. If the file is not in LSM format, it attempts to
        read it using the Astropy Table reader with the "makesourcedb" format.

        Parameters
        ----------
        filename : str or Path
            _description_

        Returns
        -------
        astropy.table.Table
            Table containing the source catalog returned by the remote service.

        """

        if tableio.validateLSMFormat(filename):
            with self._load_context(filename, f"LSM file {filename!r}"):
                return tableio.loadTableFromLSM(filename)

        else:
            # If filename does not point to a VO query, assume it points to
            # a local file
            if filename.lower() in (
                "wenss",
                "nvss",
                "tgss",
                "gsm",
                "lotss",
                "vlssr",
            ):
                self.log.warning(
                    "It appears from the filename that you may be trying to"
                    " query a VO service. If so, you must provide values "
                    "for both vo_position and vo_radius."
                )

            with self._load_context(filename, f"file {filename}"):
                return Table.read(filename, format="makesourcedb")

    def load_from_remote_service(self, service, vo_position, vo_radius):
        """
        Load a SkyModel from a remote service based on the given filename and
        VO parameters.

        Parameters
        ----------
        service : {"nvss", "wenss", "vlssr", "tgss", "gsm", "lotts"}
            Name of the remote service to query.
        vo_position : _type_
            Position for the VO query.
        vo_radius : _type_
            Cone search radius.

        Returns
        -------
        astropy.table.Table
            Table containing the source catalog returned by the remote service.

        Raises
        ------
        ValueError
            If the specified VO service is not recognized or the query fails.
        """
        key = service.lower()
        if key in tableio.ALLOWED_VO_SERVICES:
            with self._load_remote_context(
                service, f"VO service {service}", vo_position, vo_radius
            ):
                return tableio.coneSearch(key, vo_position, vo_radius)

        if key in tableio.ALLOWED_SURVEYS:
            with self._load_remote_context(
                service, f"{service} survey", vo_position, vo_radius
            ):
                return tableio.get_survey(key, vo_position, vo_radius)

        raise ValueError(
            f"VO service {service!r} not understood. Must be one of "
            f"{tableio.ALLOWED_REMOTE_SERVICES}. If you want instead to load a "
            f"model from a local file, do not set vo_position or vo_radius."
        )

    @contextlib.contextmanager
    def _load_remote_context(self, filename, service, vo_position, vo_radius):
        """Context manager for loading SkyModels from remote services."""

        with self._load_context(filename, service, vo_position):
            try:
                yield
            except (IndexError, InconsistentTableError):
                # Empty result due to no coverage in the catalog at the queried
                # position
                self.log.warning(
                    "No sources found for the given VO query parameters "
                    "(%s  with vo_position = %s and vo_radius = %s). "
                    "Returning an empty %s instance.",
                    service,
                    vo_position,
                    vo_radius,
                    self.__class__.__name__,
                )
                self._filename = None
                return tableio.makeEmptyTable()
            else:
                self._filename = f"{filename}_vo"

    @contextlib.contextmanager
    def _load_context(self, filename, service, vo_position=None):
        """Context manager for loading SkyModels."""

        self.log.debug(
            "Attempting to load %s from %s.", self.__class__.__name__, service
        )

        # Control flow goes back to the caller to perform the actual loading of
        # the model during the yield block
        try:
            yield
        except:
            raise
        else:
            self.log.debug(
                "Successfully loaded %s from %s.",
                self.__class__.__name__,
                service,
            )
            pos = f" at position {vo_position}" if vo_position else ""
            self._add_history(f"LOAD (from {service}{pos})")
            self._filename = filename

    def load_from_dict(self, columns):
        """
        Load SkyModel from dict of columns.

        Parameters
        ----------
        columns : dict
            Dictionary where keys are column names and values are column data.

        Returns
        -------
        astropy.table.Table
            Table containing the source catalog returned by the remote service.

        """
        with self._load_context(None, "input dict"):
            # Create header
            format_string = f"#FORMAT = {', '.join(columns)}"

            # Process the header
            col_names, has_patches, col_defaults, meta_dict = (
                processFormatString(format_string)
            )

            # Process the model
            line = ", ".join(map(str, columns.values()))
            outline, meta_dict = processLine(line, meta_dict, col_names)
            outlines = []
            if outline is not None:
                outlines.append(outline)
            outlines.append("\n")  # needed in case of single-line sky models
            return createTable(outlines, meta_dict, col_names, col_defaults)

    def _updateGroups(self):
        """
        Updates the grouping of the table by patch name.
        """
        if "Patch" in self.table.keys():
            self.table = self.table.group_by("Patch")
            self.hasPatches = True

            # Check if any patches have undefined positions
            patch_dict = {}
            for patch_name in self.getPatchNames():
                if patch_name not in self.table.meta:
                    patch_dict.update({patch_name: None})
            if patch_dict:
                self.setPatchPositions(patch_dict, method="mid")
        else:
            self.hasPatches = False

    def _add_history(self, entry=""):
        """
        Adds entry to the history with current date and time

        Parameters
        ----------
        entry : str, optional
            String to add to history

        """
        current_time = str(datetime.datetime.now()).split(".")[0]
        self.history.append(current_time + ": " + str(entry))

    def _info(self, use_log_info=False):
        """
        Prints information about the sky model.
        """
        if self.hasPatches:
            n_patches = len(set(self.getPatchNames()))
        else:
            n_patches = 0

        n_point = len(np.where(self.getColValues("Type") == "POINT")[0])
        n_gaus = len(np.where(self.getColValues("Type") == "GAUSSIAN")[0])

        if n_patches == 1:
            plur = ""
        else:
            plur = "es"
        if use_log_info:
            log_call = self.log.info
        else:
            log_call = self.log.debug

        ref_ra, ref_dec = self._get_midpoint()
        tot_flux = np.sum(self.getColValues("I", units="Jy"))

        info = (
            f"Model contains {len(self.table)} sources in {n_patches} "
            f"patch{plur} of which:\n"
            f"    {n_point} are type POINT\n"
            f"    {n_gaus} are type GAUSSIAN\n"
            f"    Associated beam MS: {self.beamMS}\n"
            f"    Approximate RA, Dec of center: {ref_ra}, {ref_dec}\n"
            f"    Total flux: {tot_flux} Jy\n\n"
            f"    History:\n        " + "\n        ".join(self.history)
        )
        log_call(info)
        return info

    def info(self):
        """
        Prints information about the sky model.
        """
        _ = self._info(use_log_info=True)

    def copy(self):
        """
        Returns a copy of the sky model.
        """

        # The logger's stream handlers are not copyable with deepcopy, so copy
        # them by hand:
        self.log = None
        lsm_copy = copy.deepcopy(self)
        lsm_copy._updateGroups()
        lsm_copy.log = logging.getLogger("LSMTool")
        lsm_copy._add_history("COPY")
        self.log = logging.getLogger("LSMTool")

        return lsm_copy

    def more(
        self,
        colName=None,
        patchName=None,
        sourceName=None,
        sortBy=None,
        lowToHigh=False,
    ):
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

        >>> s.more(["Name", "I"], "bin0", sort_by=["I"])

        """
        col_name = colName
        patch_name = patchName
        source_name = sourceName
        sort_by = sortBy
        low_to_high = lowToHigh

        if patch_name is not None and source_name is not None:
            raise ValueError(
                "patch_name and source_name cannot both be specified."
            )

        table = self.table

        # Get columns
        col_name = self._verify_column_name(col_name)
        if col_name is not None:
            if isinstance(col_name, str):
                col_name = [
                    col_name
                ]  # needed in order to get a table instead of a column
            table = table[col_name]

        # Get patches
        if patch_name is not None:
            pindx = self._get_name_index(patch_name, patch=True)
            if pindx is not None:
                table = table.groups[pindx]

        # Get sources
        if source_name is not None:
            sindx = self._get_name_index(source_name)
            if sindx is not None:
                table = table[sindx]

        # Sort if desired
        if sort_by is not None:
            col_name = self._verify_column_name(sort_by)
            indx = table.argsort(col_name)
            if not low_to_high:
                indx = indx[::-1]
            table = table[indx]

        table.more(show_unit=True)

    def _verify_column_name(self, col_name, only_existing=True, quiet=False):
        """
        Verifies that column(s) exist and returns correctly formatted string or
        list of strings suitable for accessing the data table.

        Parameters
        ----------
        col_name : str, list of str
            Name of column or columns
        only_existing : bool, optional
            If True, only columns that exist in the table are allowed. If False,
            columns that are valid but do not exist in the table are returned
            without error.
        quiet : bool, optional
            If True, errors will be suppressed

        Returns
        -------
        col_name : str, None
            Properly formatted name of column or None if col_name not found

        """
        if isinstance(col_name, str):
            col_name_lower = col_name.lower()
            if col_name_lower not in tableio.ALLOWED_COLUMN_NAMES:
                if not quiet:
                    raise ValueError(
                        f"Column name {col_name!r} is not a valid makesourcedb "
                        "column."
                    )
                return None
            col_name_key = tableio.ALLOWED_COLUMN_NAMES[col_name_lower]
            if col_name_key not in self.table.keys() and only_existing:
                if not quiet:
                    raise ValueError(
                        f"Column name {col_name!r} not found in sky model."
                    )
                return None

        elif isinstance(col_name, list):
            col_name_lower = [c.lower() for c in col_name]
            for name in col_name_lower[:]:
                bad_names = []
                if name not in tableio.ALLOWED_COLUMN_NAMES:
                    bad_names.append(name)
                    col_name_lower.remove(name)
                else:
                    col_name_key = tableio.ALLOWED_COLUMN_NAMES[name]
                    if col_name_key not in self.table.keys():
                        bad_names.append(name)
                        col_name_lower.remove(name)

            if len(bad_names) > 0:
                if len(bad_names) == 1:
                    plur = ""
                else:
                    plur = "s"
                if not quiet:
                    self.log.warning(
                        "Column name%s %r not recognized. Ignoring.",
                        plur,
                        ",".join(bad_names),
                    )
            if len(col_name_lower) == 0:
                return None
            col_name_key = [
                tableio.ALLOWED_COLUMN_NAMES[n] for n in col_name_lower
            ]
        else:
            col_name_key = None

        return col_name_key

    def getPatchPositions(
        self,
        patchName=None,
        asArray=False,
        method=None,
        applyBeam=False,
        perPatchProjection=True,
    ):
        """
        Returns arrays or a dict of patch positions (as {'patchName':(RA,
        Dec)}).

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
            - 'wmean' => calculate the flux-weighted mean RA and Dec of the
               patch
            - None => current patch positions are returned
            Note that the mid, mean, and wmean positions are calculated from
            TAN- projected values.
        applyBeam : bool, optional
            If True, fluxes used as weights will be attenuated by the beam.
        perPatchProjection : bool, optional
            If True, a different projection center is used per patch. If False,
            a single projection center is used for all patches.

        Returns
        -------
        positions : numpy.ndarray or dict
            (RA, Dec) arrays (if as_array is False) of patch positions or a
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

        >>> s.getPatchPositions(method="wmean", asArray=True)

        """
        patch_name = patchName
        as_array = asArray
        apply_beam = applyBeam
        per_patch_projection = perPatchProjection

        if self.hasPatches:
            if patch_name is None:
                patch_name = self.getPatchNames()
            if type(patch_name) not in (list, np.ndarray):
                patch_name = [patch_name]
            if method is None:
                patch_dict = {}
                for patch in patch_name:
                    if patch in self.table.meta:
                        patch_dict[patch] = self.table.meta[patch]
                    else:
                        patch_dict[patch] = [
                            Angle(0.0, unit=u.deg),
                            Angle(0.0, unit=u.deg),
                        ]
            else:
                patch_dict = {}

                # Add projected x and y columns.
                if per_patch_projection:
                    # Each patch has a different projection center
                    x_all = []  # has length = num of sources
                    y_all = []
                    wcs_all = []
                    for name in patch_name:
                        patch_indices = self.getRowIndex(name)
                        patch_ra = self.table["Ra"][patch_indices]
                        patch_dec = self.table["Dec"][patch_indices]
                        x, y, wcs = self._get_xy(patch_ra, patch_dec)
                        x_all.extend(x)
                        y_all.extend(y)
                        wcs_all.append(wcs)
                else:
                    x_all, y_all, wcs = self._get_xy()
                    wcs_all = [wcs] * len(patch_name)

                x_col = Column(name="X", data=x_all)
                y_col = Column(name="Y", data=y_all)
                self.table.add_column(x_col)
                self.table.add_column(y_col)

                positions = []
                if method == "mid":
                    min_x = self._get_min_column("X")
                    max_x = self.get_max_column("X")
                    min_y = self._get_min_column("Y")
                    max_y = self.get_max_column("Y")
                    mid_x = min_x + (max_x - min_x) / 2.0
                    mid_y = min_y + (max_y - min_y) / 2.0
                    for i, _ in enumerate(patch_name):
                        ra, dec = wcs_all[i].wcs_pix2world(
                            mid_x[i], mid_y[i], 0
                        )
                        positions.append((ra.item(), dec.item()))
                elif method in {"mean", "wmean"}:
                    if method == "mean":
                        weight = False
                    else:
                        weight = True
                    mean_x = self._get_averaged_column(
                        "X", apply_beam=apply_beam, weight=weight
                    )
                    mean_y = self._get_averaged_column(
                        "Y", apply_beam=apply_beam, weight=weight
                    )
                    for i, _ in enumerate(patch_name):
                        ra, dec = wcs_all[i].wcs_pix2world(
                            mean_x[i], mean_y[i], 0
                        )
                        positions.append((ra.item(), dec.item()))
                if positions:
                    ra_norm, dec_norm = tableio.RADec2Angle(
                        *map(list, zip(*positions, strict=True))
                    )
                    patch_dict = dict(
                        zip(
                            patch_name,
                            zip(ra_norm, dec_norm, strict=True),
                            strict=True,
                        )
                    )

                self.table.remove_column("X")
                self.table.remove_column("Y")

            if as_array:
                ra = []
                dec = []
                for patch in patch_name:
                    ra.append(patch_dict[patch][0].value)
                    dec.append(patch_dict[patch][1].value)
                return np.array(ra), np.array(dec)
            return patch_dict

        return None

    def setPatchPositions(
        self,
        patchDict=None,
        method="mid",
        applyBeam=False,
        perPatchProjection=True,
    ):
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
            If no patch_dict is given, this parameter specifies the method used
            to set the patch positions:
            - 'mid' => the position is set to the midpoint of the patch
            - 'mean' => the position is set to the mean RA and Dec of the patch
            - 'wmean' => the position is set to the flux-weighted mean RA and
            Dec of the patch
            - 'zero' => set all positions to [0.0, 0.0]

            Note that the mid, mean, and wmean positions are calculated from
            TAN- projected values.
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

        >>> s.setPatchPositions(method="wmean")

        Set new position for the 'bin0' patch only::

        >>> s.setPatchPositions({"bin0": [123.231, 23.4321]})

        """
        patch_dict = patchDict
        apply_beam = applyBeam
        per_patch_projection = perPatchProjection

        if self.hasPatches:
            if method not in ["mid", "mean", "wmean", "zero"]:
                raise ValueError("Invalid method parameter")

            if patch_dict is None:
                # Delete any previous patch positions
                patch_names = self.getPatchNames()
                for patch_name in patch_names:
                    if patch_name in self.table.meta:
                        self.table.meta.pop(patch_name)
                if method == "zero":
                    patch_dict = {}
                    for n in patch_names:
                        patch_dict[n] = [
                            Angle(0.0, unit=u.deg),
                            Angle(0.0, unit=u.deg),
                        ]
                else:
                    patch_dict = self.getPatchPositions(
                        method=method,
                        applyBeam=apply_beam,
                        perPatchProjection=per_patch_projection,
                    )
            else:
                # Get positions for those patches that need them
                patch_names = [
                    patch for patch, pos in patch_dict.items() if pos is None
                ]
                patch_dict_no_pos = self.getPatchPositions(
                    method=method,
                    applyBeam=apply_beam,
                    patchName=patch_names,
                    perPatchProjection=False,
                )
                patch_dict.update(patch_dict_no_pos)

            for patch, pos in patch_dict.items():
                if isinstance(pos[0], str) or isinstance(pos[0], float):
                    ra, dec = tableio.RADec2Angle(*pos)
                    # Each patch stores scalar Angles, not length-one arrays.
                    self.table.meta[patch] = [ra[0], dec[0]]
                else:
                    self.table.meta[patch] = list(pos)

            self._add_history(f"SETPATCHPOSITIONS (method = {method!r})")
        else:
            raise RuntimeError("Sky model does not have patches.")

    def _get_midpoint(self, ra=None, dec=None, *, crdelt=None):
        """
        Compute the midpoint of the given RA and Dec coordinate lists.

        Parameters
        ----------
        ra : list or numpy.ndarray of float, optional
            Right ascension values in degrees. Normalisation is not required.
            If None, use the values from the sources in the sky model.
        dec : list or numpy.ndarray of float, optional
            Declination values in degrees, normalised to the range [-90, 90].
            If None, use the values from the sources in the sky model.
        crdelt: float, optional
            Delta in degrees for sky grid.

        Returns
        -------
        ra_midpoint, dec_midpoint : float
            Midpoint RA and Dec values.
        """
        ra = self.table["Ra"] if ra is None else ra
        dec = self.table["Dec"] if dec is None else dec

        if len(ra) == 0:
            return 0, 0
        if len(ra) == 1:
            ra_midpoint, dec_midpoint = ra[0], dec[0]
        else:
            wcs = make_wcs(ra[0], dec[0], crdelt=crdelt)
            x, y = wcs.wcs_world2pix(ra, dec, 0)
            xind = np.argsort(x)
            yind = np.argsort(y)
            xsorted = x[xind]
            ysorted = y[yind]
            xmid = 0.5 * (xsorted[0] + xsorted[-1])
            ymid = 0.5 * (ysorted[0] + ysorted[-1])
            midxind = np.searchsorted(xsorted, xmid)
            midyind = np.searchsorted(ysorted, ymid)
            ra_midpoint, dec_midpoint = ra[xind[midxind]], dec[yind[midyind]]

        return normalize_ra_dec(ra_midpoint, dec_midpoint)

    def _get_xy(self, ra=None, dec=None, *, crdelt=None):
        """
        Returns lists of projected x and y values.

        Parameters
        ----------
        ra : list or numpy.ndarray of float, optional
            Right ascension values in degrees. Normalisation is not required.
            If None, use the values from the sources in the sky model.
        dec : list or numpy.ndarray of float, optional
            Declination values in degrees, normalised to the range [-90, 90].
            If None, use the values from the sources in the sky model.
        crdelt: float, optional
            Delta in degrees for sky grid.

        Returns
        -------
        x, y : numpy.ndarray
            Arrays of x and y values
        wcs : astropy.wcs.WCS
            WCS object used for the projection.
        """
        ra = self.table["Ra"] if ra is None else ra
        dec = self.table["Dec"] if dec is None else dec

        if len(ra) == 0:
            return [], [], None

        ra_midpoint, dec_midpoint = self._get_midpoint(ra, dec, crdelt=crdelt)
        wcs = make_wcs(ra_midpoint, dec_midpoint, crdelt=crdelt)
        x, y = wcs.wcs_world2pix(ra, dec, 0)

        return x, y, wcs

    def getDefaultValues(self):
        """
        Returns dict of {col_name:default} values for all columns with defaults.

        Returns
        -------
        default_dict : dict
            Dict of {col_name:default} values

        """
        col_names = self.getColNames()
        default_dict = {}
        for col_name in col_names:
            if col_name in self.table.meta:
                default_dict[col_name] = self.table.meta[col_name]
        return default_dict

    def setDefaultValues(self, colDict):
        """
        Sets default column values.

        Parameters
        ----------
        colDict : dict
            Dict specifying column names and default values as
            {'col_name':value} where the value is in the units accepted by
            makesourcedb (e.g., Hz for 'Reference_frequency').

        Examples
        --------
        Set new default value for ReferenceFrequency::

        >>> s.setDefaultValues({"ReferenceFrequency": 140e6})

        """
        for col_name, default in colDict.items():
            self.table.meta[col_name] = default

    def ungroup(self):
        """
        Removes all patches from the sky model.

        Examples
        --------
        Remove all patches::

        >>> s.ungroup()

        """
        if self.hasPatches:
            for patch_name in self.getPatchNames():
                if patch_name in self.table.meta:
                    self.table.meta.pop(patch_name)
            self.table.remove_column("Patch")
            self._updateGroups()
            self._add_history("UNGROUP")
            self._info()

    def getColNames(self):
        """
        Returns a list of all available column names.

        Returns
        -------
        col_names : list
            List of all column names

        Examples
        --------
        Get column names::

        >>> s.getColNames()

        """
        return self.table.keys()

    def getColValues(
        self, colName, units=None, aggregate=None, applyBeam=False
    ):
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

            When the column has an array value, the aggregation function is
            applied independently to each element of the array. For example,
            when the values are [1, -1] and [5, -5], the sum is [6, -6].

            Note that, in some cases, certain aggregation functions will not
            produce meaningful results. For example, asking for the sum of the
            MajorAxis values per patch will not give a good indication of the
            size of the patch (to get the sizes, use the getPatchSizes()
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
        col_values : numpy.ndarray
            Independent array of column values. Modifying it does not change
            the sky model. None is returned if column is not found.

        Examples
        --------
        Get Stokes I fluxes in Jy::

        >>> s.getColValues("I")
        array([ 60.4892,   1.2413,   1.216 , ...,   1.12  ,   1.25  ,   1.16  ])

        Get Stokes I fluxes in mJy::

        >>> s.getColValues("I", units="mJy")
        array([ 60489.2,   1241.3,   1216. , ...,   1120. ,   1250. ,   1160. ])

        Get total Stokes I flux for the patches::

        >>> s.getColValues("I", aggregate="sum")
        array([ 61.7305,   1.216 ,   3.9793, ...,   1.12  ,   1.25  ,   1.16  ])

        Get flux-weighted average RA and Dec for the patches. As noted above,
        the getColValues() method is not appropriate for use with RA or Dec,
        so we must use getPatchPositions() instead::

        >>> ra, dec = s.getPatchPositions(method="wmean", asArray=True)

        """
        col_name = colName
        apply_beam = applyBeam

        col_name = self._verify_column_name(col_name)
        if col_name is None:
            return None
        if isinstance(col_name, list):
            if len(col_name) > 1:
                raise ValueError("Only one column can be specified.")
            col_name = col_name[0]

        allowed_fcns = ["sum", "mean", "wmean", "min", "max"]
        if aggregate not in allowed_fcns and aggregate is not None:
            raise ValueError("Value of parameter 'aggregate' not understood.")
        if aggregate in allowed_fcns and self.hasPatches:
            col = self.get_aggregated_column(
                col_name, aggregate, apply_beam=apply_beam
            )
        else:
            col = self._get_column(col_name, apply_beam=apply_beam)

        if col is None:
            return None

        # Filling a masked column already creates independent storage.
        # Aggregation and beam attenuation also produce owned columns; only
        # an unmodified table column needs an explicit copy here.
        if hasattr(col, "filled"):
            outcol = col.filled()
        elif col is self.table[col_name]:
            outcol = col.copy()
        else:
            outcol = col

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
        values : list, numpy.ndarray, or dict
            Array of values or dict of {source_name:value} pairs. If list or
            array, the length must match the number of rows in the table. If
            dict, missing values will be masked unless already present. Values
            are assumed to be in units required by makesourcedb.
        mask : list or numpy.ndarray of bool, optional
            If values is a list or array, a mask can be specified (True means
            the value is masked).
        index : int, optional
            Index that specifies the column position in the table, if column is
            not already present in the table.

        Examples
        --------
        Set Stokes I fluxes::

        >>> s.setColValues(
        ...     "I",
        ...     [1.0, 1.1, 1.2, 0.0, 1.3],
        ...     mask=[False, False, False, True, False],
        ... )

        """
        col_name = colName

        col_name = self._verify_column_name(col_name, only_existing=False)
        if col_name is None:
            return
        if isinstance(col_name, list):
            if len(col_name) > 1:
                raise ValueError("Only one column can be specified.")
            col_name = col_name[0]

        if isinstance(values, dict):
            if col_name in self.table.keys():
                data = self.table[col_name].data
                mask = self.table[col_name].mask
            else:
                data = [0] * len(self.table)
                mask = [True] * len(self.table)
            for source_name, value in values.items():
                indx = self._get_name_index(source_name)
                if col_name in {"Ra", "Dec"}:
                    val = Angle(value, unit=u.deg)
                else:
                    val = value
                data[indx] = val
                mask[indx] = False
        elif len(values) != len(self.table):
            raise ValueError(
                "Length of input values must match length of table."
            )
        else:
            if col_name in {"Ra", "Dec"}:
                vals = Angle(values, unit=u.deg)
            else:
                vals = values
            data = vals

        if mask is not None:
            data = np.ma.masked_array(data, mask)
        else:
            data = np.array(data)
        if col_name in self.table.keys():
            units = self.table.columns[col_name].unit
            self.table[col_name] = data
            self.table.columns[col_name].unit = units
        else:
            if col_name == "Patch":
                # Specify length of 50 characters
                new_col = Column(name=col_name, data=data, dtype="U50")
            else:
                new_col = Column(name=col_name, data=data)
            self.table.add_column(new_col, index=index)
        if col_name == "Patch":
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
        row_values : astropy table or row
            Table (if more than one source) or row (if one source). None is
            returned if source is not found.

        Examples
        --------
        Get row values for the source 'src1'::

        >>> rows = s.getRowValues("src1")

        Sum over the fluxes of sources in the 'bin1' patch::

        >>> tot = 0.0
        ... for row in s.getRowValues("bin1"):
        ...     tot += row["I"]

        """
        row_name = rowName

        # Check first for the row_name as a patch name. If no patch matches (or
        # the model is not grouped into patches), Try it as a source name. This
        # logic should work even if a row has the same name for the source and
        # its patch, as in this case the patch and source row index are
        # identical (since such a patch can have only one member source)
        if self.hasPatches and row_name in self.getPatchNames():
            pindx = self._get_name_index(row_name, patch=True)
            table = self.table.groups[pindx]
            return table.group_by("Patch")  # ensure that grouping is preserved

        if row_name in self.getColValues("Name"):
            indx = self._get_name_index(row_name)
            return self.table.filled()[indx]

        raise ValueError(f"Row name {row_name!r} not recognized.")

    def getRowIndex(self, rowName):
        """
        Returns a row selector for the specified source or patch.

        Parameters
        ----------
        rowName : str
            Exact name of the source or patch (wildcards are not interpreted).
            Patch names take precedence over source names.

        Returns
        -------
        indices : slice or numpy.ndarray
            Slice for a patch, or an integer array of matching source indices.
            Use directly to index a table or column. Patch slices select views
            without copying data or allocating one index per member source.
            Selectors refer to the current row order and must be obtained again
            after regrouping or adding/removing rows.
            ValueError is raised if neither a patch nor a source matches.

        Examples
        --------
        Get row index for the source 'src1'::

        >>> s.getRowIndex("src1")
        array([0])

        Get row indices for the patch 'bin1' and verify the patch name::

        >>> ind = s.getRowIndex("bin1")
        ... print(s.getColValues("patch")[ind])
        ['bin1', 'bin1', 'bin1']

        """
        row_name = rowName

        # Patch members occupy contiguous rows in the grouped table.
        if self.hasPatches:
            patch_names = self.table.groups.keys["Patch"]
            patch_ind = np.where(patch_names == row_name)[0]

            if len(patch_ind) > 0:
                group_ind = patch_ind[0]
                start = self.table.groups.indices[group_ind]
                end = self.table.groups.indices[group_ind + 1]
                return slice(start, end)

        indices = np.flatnonzero(self.table["Name"] == row_name)
        if indices.size:
            return indices
        raise ValueError(f"Row name {row_name!r} not recognized.")

    def setRowValues(self, values):
        """
        Sets values for a single row.

        If a row with the given name already exists, its values are
        updated. If not, a new row is made and appended to the table.

        Parameters
        ----------
        values : list, numpy.ndarray, or dict
            Array of values or dict of {col_name:value} pairs. If list or
            array, the length must match the number and order of the columns in
            the table. If dict, missing values will be masked unless already
            present.

        Examples
        --------
        Set row values for the source 'src1' (which can be a new source or an
        existing source)::

        >>> s.setRowValues(
        ...     {
        ...         "Name": "src1",
        ...         "Ra": 213.123,
        ...         "Dec": 23.1232,
        ...         "I": 23.2,
        ...         "Type": "POINT",
        ...     }
        ... )

        The RA and Dec values can be in degrees (as above) or in makesourcedb
        format. E.g.::

        >>> s.setRowValues(
        ...     {
        ...         "Name": "src1",
        ...         "Ra": "12:22:21.1",
        ...         "Dec": "+14.46.31.5",
        ...         "I": 23.2,
        ...         "Type": "POINT",
        ...     }
        ... )

        """
        # Read model into astropy table object
        temp_skymodel = SkyModel(values)

        # Concatenate tables
        self.concatenate(
            temp_skymodel, matchBy="name", keep="from2", inheritPatches=False
        )

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
        data : numpy.ndarray
            Array of patch sizes. None is returned if the sky model
            does not have patches

        """
        if self.hasPatches:
            col = self._get_size_column(weight=weight, apply_beam=applyBeam)
            if units is not None:
                col.convert_unit_to(units)
            return col.data
        return None

    def getPatchNames(self):
        """
        Returns array of all patch names in the sky model, with duplicates
        removed.

        Note: use getColValues('Patch') if you want the patch names for each
        source in the sky model.

        Returns
        -------
        names : numpy.ndarray
            Array of patch names. None is returned if the sky model does not
            have patches

        """
        if self.hasPatches:
            col = self.table.groups.keys["Patch"]
            if hasattr(col, "filled"):
                outcol = col.filled().copy()
            else:
                outcol = col.copy()
            return outcol.data
        return None

    def _get_name_index(self, name, patch=False):
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

        if patch:
            if self.hasPatches:
                names = self.getPatchNames().tolist()
            else:
                return None
        else:
            names = self.getColValues("Name").tolist()

        if isinstance(name, str) or isinstance(name, np.bytes_):
            indx = [
                i for i, item in enumerate(names) if fnmatch.fnmatch(item, name)
            ]
            if len(indx) == 0:
                return None
            return indx
        if isinstance(name, list):
            indx = []
            for n in name:
                bad_names = []
                nindx = [
                    i
                    for i, item in enumerate(names)
                    if fnmatch.fnmatch(item, n)
                ]
                if len(nindx) == 0:
                    bad_names.append(n)
                else:
                    indx += nindx
            if len(bad_names) > 0:
                if len(bad_names) == 1:
                    plur = ""
                else:
                    plur = "s"
                self.log.warning(
                    "Name%s %r not recognized. Ignoring.",
                    plur,
                    ",".join(bad_names),
                )
            if len(indx) == 0:
                raise ValueError("None of the specified names were found.")
            return indx
        return None

    def _get_column(self, col_name, apply_beam=False):
        """
        Returns the appropriate column (nonaggregated).

        Parameters
        ----------
        col_name : str
            Name of column to get. If not found, None is returned
        apply_beam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Nonaggregated Column object. Shares storage with the table unless
            beam attenuation is applied; callers must copy before modifying it.

        """
        col_name = self._verify_column_name(col_name)
        if col_name is None:
            return None

        col = self.table[col_name]

        if apply_beam and col_name in ["I", "Q", "U", "V"]:
            col = self._apply_beam_to_col(col.copy())

        return col

    def get_aggregated_column(
        self, col_name, aggregate="sum", apply_beam=False
    ):
        """
        Returns the appropriate column aggregated by group.

        Parameters
        ----------
        col_name : str
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
        apply_beam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Column object with aggregated values

        """
        col_name = self._verify_column_name(col_name)
        if col_name is None:
            return None

        if aggregate == "mean":
            col = self._get_averaged_column(
                col_name, weight=False, apply_beam=apply_beam
            )
        elif aggregate == "wmean":
            col = self._get_averaged_column(
                col_name, weight=True, apply_beam=apply_beam
            )
        elif aggregate == "sum":
            col = self._get_summed_column(col_name, apply_beam=apply_beam)
        elif aggregate == "min":
            col = self._get_min_column(col_name, apply_beam=apply_beam)
        elif aggregate == "max":
            col = self.get_max_column(col_name, apply_beam=apply_beam)
        else:
            raise ValueError("Aggregation function not understood.")
        return col

    def _apply_beam_to_col(self, col, patch=False):
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

        if not self._has_beam:
            self.log.warning(
                "No beam MS has been specified. No beam attenuation applied."
            )
            return col

        if patch:
            if self._patch_method is not None:
                # Try to get patch positions from the meta data
                ra_deg, dec_deg = self.getPatchPositions(asArray=True)
            else:
                # If patch positions are not set, use weighted mean positions
                ra_deg = self.getColValues(
                    "Ra", applyBeam=True, aggregate="wmean"
                )
                dec_deg = self.getColValues(
                    "Dec", applyBeam=True, aggregate="wmean"
                )
        else:
            ra_deg = self.getColValues("Ra")
            dec_deg = self.getColValues("Dec")

        flux = col.data
        vals = apply_beam_operation(
            self.beamMS, flux, ra_deg, dec_deg, timeIndx=self.beamTime
        )
        col[:] = vals

        return col

    def _get_filled_column(self, col_name):
        """
        For a given column, returns the filled column data, grouped by patch.

        Parameters
        ----------
        col_name : str
            Column name

        Returns
        -------
        col : astropy Column
            Column object with filled values
        """
        col = self.table[col_name]
        if hasattr(self.table[col_name], "filled"):
            col = self.table[col_name].filled()
            col = col.group_by(self.table["Patch"])
        return col

    def _get_summed_column(self, col_name, apply_beam=False):
        """
        Returns column summed by group.

        Parameters
        ----------
        col_name : str
            Column name
        apply_beam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Column object with aggregated sum of values
        """

        def npsum(array):
            return np.sum(array, axis=0)

        col = self._get_filled_column(col_name)
        col = col.groups.aggregate(npsum)

        if apply_beam and col_name in ["I", "Q", "U", "V"]:
            col = self._apply_beam_to_col(col, patch=True)

        return col

    def _get_min_column(self, col_name, apply_beam=False):
        """
        Returns column minimum value by group.

        Parameters
        ----------
        col_name : str
            Column name
        apply_beam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Column object with aggregated min values
        """

        def npmin(array):
            return np.min(array, axis=0)

        col = self._get_filled_column(col_name)
        col = col.groups.aggregate(npmin)

        if apply_beam and col_name in ["I", "Q", "U", "V"]:
            col = self._apply_beam_to_col(col, patch=True)

        return col

    def get_max_column(self, col_name, apply_beam=False):
        """
        Returns column maximum value by group.

        Parameters
        ----------
        col_name : str
            Column name
        apply_beam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Column object with aggregated max values

        """

        def npmax(array):
            return np.max(array, axis=0)

        col = self._get_filled_column(col_name)
        col = col.groups.aggregate(npmax)

        if apply_beam and col_name in ["I", "Q", "U", "V"]:
            col = self._apply_beam_to_col(col, patch=True)

        return col

    def _get_averaged_column(self, col_name, weight=True, apply_beam=False):
        """
        Returns column averaged by group.

        Parameters
        ----------
        col_name : str
            Column name
        weight : bool, optional
            If True, return average weighted by flux
        apply_beam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Column object with averaged values
        """
        if weight:
            # Using 'axis=0' makes the aggregation support array-valued columns.
            # For scalar-valued columns, a plain np.sum would suffice.
            # Note that this is an undocumented feature of astropy tables:
            # https://docs.astropy.org/en/stable/table/operations.html#aggregation
            # says the aggregation function should return a scalar value.
            def npsum(array):
                return np.sum(array, axis=0)

            if hasattr(self.table[col_name], "filled"):
                vals = self.table[col_name].filled().data
            else:
                vals = self.table[col_name].data
            if weight:
                weights = self.getColValues("I", applyBeam=apply_beam)
                if weights.shape != vals.shape:
                    weights = np.resize(weights, vals.shape)
                weight_col = Column(name="Weight", data=weights)
                val_weight_col = Column(name="ValWeight", data=vals * weights)
                self.table.add_column(val_weight_col)
                self.table.add_column(weight_col)
                numer = self.table["ValWeight"].groups.aggregate(npsum).data
                denom = self.table["Weight"].groups.aggregate(npsum).data
                self.table.remove_column("ValWeight")
                self.table.remove_column("Weight")
            else:
                val_col = Column(name="Val", data=vals)
                self.table.add_column(val_col)
                numer = self.table["Val"].groups.aggregate(npsum).data
                self.table.remove_column("Val")

            return Column(
                name=col_name,
                data=np.array(numer / denom),
                unit=self.table[col_name].unit,
            )

        # Using 'axis=0' makes the aggregation support array-valued columns.
        # For scalar-valued columns, a plain np.average would suffice.
        def npavg(c):
            return np.average(c, axis=0)

        col = self._get_filled_column(col_name)
        return col.groups.aggregate(npavg)

    def _get_size_column(self, weight=True, apply_beam=False):
        """
        Returns column of source largest angular sizes.

        Parameters
        ----------
        weight : bool, optional
            If True, return size weighted by flux
        apply_beam : bool, optional
            If True, fluxes will be attenuated by the beam

        Returns
        -------
        col : astropy Column
            Column object with sizes from Major_axis or from aggregated values
            if the model has patches

        """
        if weight:
            method = "wmean"
        else:
            method = "mean"

        if self.hasPatches:
            # Get patch positions
            ra_avg, dec_avg = self.getPatchPositions(
                method=method, asArray=True, applyBeam=apply_beam
            )

            # Fill out the columns by repeating the average value over the
            # entire group
            ra_avg_full = np.zeros(len(self.table), dtype=float)
            dec_avg_full = np.zeros(len(self.table), dtype=float)
            for i, ind in enumerate(self.table.groups.indices[1:]):
                ra_avg_full[self.table.groups.indices[i] : ind] = ra_avg[i]
                dec_avg_full[self.table.groups.indices[i] : ind] = dec_avg[i]

            dist = self._calculate_separation(
                self.table["Ra"], self.table["Dec"], ra_avg_full, dec_avg_full
            )
            if weight:
                if apply_beam and self._has_beam:
                    app_fluxes = self.getColValues("I", applyBeam=True)
                    weight_col = Column(name="Weight", data=app_fluxes)
                    val_weight_col = Column(
                        name="ValWeight", data=dist * app_fluxes
                    )
                else:
                    weight_col = Column(
                        name="Weight", data=self.table["I"].data
                    )
                    val_weight_col = Column(
                        name="ValWeight", data=dist * self.table["I"].data
                    )
                self.table.add_column(val_weight_col)
                self.table.add_column(weight_col)
                numer = (
                    self.table["ValWeight"].groups.aggregate(np.sum).data * 2.0
                )
                denom = self.table["Weight"].groups.aggregate(np.sum).data
                self.table.remove_column("ValWeight")
                self.table.remove_column("Weight")
                col = Column(name="Size", data=numer / denom, unit="degree")
            else:
                val_col = Column(name="Val", data=dist)
                self.table.add_column(val_col)
                size = self.table["Val"].groups.aggregate(np.max).data * 2.0
                self.table.remove_column("Val")
                col = Column(name="Size", data=size, unit="degree")
        elif "majoraxis" in self.table.colnames:
            col = self.table["Major_axis"]
        else:
            col = Column(
                name="Size", data=np.zeros(len(self.table)), unit="degree"
            )

        if hasattr(col, "filled"):
            outcol = col.filled(fill_value=0.0)
        else:
            outcol = col
        outcol.convert_unit_to("arcsec")

        return outcol

    def _calculate_separation(self, ra1, dec1, ra2, dec2):
        """
        Returns angular separation between two coordinates (all in degrees)

        Parameters
        ----------
        ra1 : float or numpy.ndarray
            RA of coordinate 1 in degrees
        dec1 : float or numpy.ndarray
            Dec of coordinate 1 in degrees
        ra2 : float
            RA of coordinate 2 in degrees
        dec2 : float
            Dec of coordinate 2 in degrees

        Returns
        -------
        separation : astropy.coordinates.Angle or numpy.ndarray
            Angular separation in degrees

        """

        return calculateSeparation(ra1, dec1, ra2, dec2)

    def getDistance(self, ra, dec, byPatch=False, units=None):
        """
        Returns angular distance for each source or patch to specified position

        Parameters
        ----------
        ra : float or str
            RA of position to which the distance is desired (in degrees or
            makesourcedb format)
        dec : float or str
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

        >>> s.getDistance(94.0, 42.0, units="arcmin")

        Find distance to patch centers:

        >>> s.setPatchPositions(method="mid")
        ... s.getDistance(94.0, 42.0, by_patch=True)

        """
        if byPatch and self.hasPatches:
            # Get patch positions
            s_ra, s_dec = self.getPatchPositions(asArray=True)
        else:
            s_ra = self.getColValues("RA")
            s_dec = self.getColValues("Dec")

        ra, dec = tableio.RADec2Angle(ra, dec)

        dist = self._calculate_separation(s_ra, s_dec, ra, dec)
        if units is not None:
            return dist.to(units).value
        return dist.value

    def write(
        self,
        fileName=None,
        format="makesourcedb",  # noqa
        clobber=False,
        *,
        sortBy=None,
        lowToHigh=False,
        addHistory=True,
        applyBeam=False,
        invertBeam=False,
        width=None,
    ):
        """
        Writes the sky model to a file.

        Parameters
        ----------
        fileName : str
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
                - 'facet' (ds9 region file of Voronoi facets; model must have
                  patches)
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
        width : float, optional
            The width in degrees of the total extent of the output facet
            regions. Only used when format = 'facet'. If not given, the width
            will be set to fully cover the extent of the model

        Examples
        --------
        Write the model to a makesourcedb sky model file suitable for use with
        BBS::

        >>> s.write("modsky.model")

        Write to a fits catalog::

        >>> s.write("sky.fits", format="fits")

        Write to a ds9 region file (point sources are indicated by points and
        Gaussians by ellipses)::

        >>> s.write("sky.reg", format="ds9")

        Write to a WSClean/ds9 facet region file (regions define Voronoi facets
        around patch positions)::

        >>> s.write("facets.reg", format="facet")

        """
        filename = fileName
        overwrite = clobber
        sort_by = sortBy
        low_to_high = lowToHigh
        add_history = addHistory
        apply_beam = applyBeam
        invert_beam = invertBeam

        if filename is None:
            if self._filename is None:
                raise IOError("No file name specified.")
            filename = self._filename

        if os.path.exists(filename):
            if overwrite:
                os.remove(filename)
            else:
                raise IOError(
                    f"The output file {filename!r} exists and clobber = False."
                )

        table = self.table.copy()

        # Apply beam attenuation
        if apply_beam:
            i_orig = self.getColValues("I")
            ra_deg = self.getColValues("Ra")
            dec_deg = self.getColValues("Dec")
            i_adj = apply_beam_operation(
                self.beamMS,
                i_orig,
                ra_deg,
                dec_deg,
                timeIndx=self.beamTime,
                invert=invert_beam,
            )
            units = self.table.columns["I"].unit
            table["I"] = i_adj
            table.columns["I"].unit = units

        # Sort if desired. For 'factor' output, save the order of patches in the
        # table meta
        if sort_by is not None:
            col_name = self._verify_column_name(sort_by)
            if format.lower() == "factor" and self.hasPatches:
                indx = np.argsort(self.getColValues("I", aggregate="sum"))
                if not low_to_high:
                    indx = indx[::-1]
                table.meta["patch_order"] = indx
            else:
                indx = table.argsort(col_name)
                if not low_to_high:
                    indx = indx[::-1]
                table = table[indx]

        if add_history:
            table.meta["History"] = self.history

        # Add patch sizes in degrees
        if format.lower() == "factor" and self.hasPatches:
            table.meta["patch_size"] = self.getPatchSizes(units="deg")

        # Add patch fluxes in m_jy
        if format.lower() == "factor" and self.hasPatches:
            table.meta["patch_flux"] = self.getColValues(
                "I", aggregate="sum", units="mJy"
            )

        # And reference coordinates and width in degrees
        if format.lower() == "facet":
            if not self.hasPatches:
                raise ValueError(
                    "Model must be grouped into patches when format = 'facet'."
                )

            ref_ra, ref_dec = self._get_midpoint()
            table.meta["refRA"] = ref_ra
            table.meta["refDec"] = ref_dec

            if width is not None:
                table.meta["width"] = width
            else:
                # Find the approximate width in RA and Dec that the model covers
                # and add 20% padding
                source_coord = SkyCoord(
                    ra=table["Ra"].value * u.degree,
                    dec=table["Dec"].value * u.degree,
                )
                ref_coord = SkyCoord(
                    ra=table.meta["refRA"] * u.degree,
                    dec=table.meta["refDec"] * u.degree,
                )
                separation = ref_coord.separation(source_coord)
                max_distance = np.max(
                    np.array([sep.value for sep in separation])
                )
                table.meta["width"] = 2 * max_distance * 1.2

        # Clean up as needed
        if format.lower() not in ["makesourcedb", "factor", "facet"]:
            # Make sure the metadata is empty when not needed
            table.meta = {}
        if format.lower() == "fits":
            # Remove custom formaters
            table.columns["Ra"].format = None
            table.columns["Dec"].format = None
            table.columns["I"].format = None

        table.write(filename, format=format.lower())

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

        tfile = tempfile.NamedTemporaryFile()
        self.table.write(tfile, format="votable")
        tableio.broadcastTable(tfile.name)
        tfile.close()

    def _clean(self):
        """
        Removes duplicate entries.
        """
        names = self.getColValues("Name")
        name_set = set(names)
        if len(names) == len(name_set):
            return

        filt_names = []
        filt_indices = []
        for i, name in enumerate(self.getColValues("Name")):
            if name in filt_names:
                filt_indices.append(i)
            else:
                filt_names.append(name)
        n_rows_orig = len(self.table)
        self.table = self.table[filt_indices]
        n_rows_new = len(self.table)
        if n_rows_orig - n_rows_new > 0:
            self.log.info(
                "Removed %s duplicate sources.", n_rows_orig - n_rows_new
            )
        self._updateGroups()

    def select(
        self,
        filterExpression,
        aggregate=None,
        applyBeam=False,
        useRegEx=False,
        force=True,
    ):
        """
        Filters the sky model, keeping all sources that meet the given
        expression.

        After filtering, thqion is true.

        Parameters
        ----------
        filterExpression : str, dict, list, or numpy.ndarray

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
            - If `numpy.ndarray`:
              The indices to filter on can be specified directly as a numpy
              array of row or patch indices such as:
              ``np.array([ 0,  2, 19, 20, 31, 37])``
              or as a numpy array of bools with the same length as the sky
              model. If a numpy array is given and the indices correspond to
              patches, then set aggregate=True.
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

              Units are optional and must be specified as required by
              astropy.units.
        aggregate : str, optional
            If set, the selection will be done on values aggregated
            over the patch members. The following aggregation functions are
            available:

                - 'sum': sum of patch values
                - 'mean': mean of patch values
                - 'wmean': Stokes I weighted mean of patch values
                - 'min': minimum of patch values
                - 'max': maximum of patch values
                - True: only valid when the filter indices are specified
                  directly as a numpy array. If True, filtering is done on
                  patches instead of sources.

        applyBeam : bool, optional
            If True, apparent fluxes will be used.
        useRegEx : bool, optional
            If True, string matching will use regular expression matching. If
            False, string matching uses Unix filename matching.
        force : bool, optional
            If True, selections that result in empty sky models are allowed. If
            False, such selections are not applied and the sky model is
            unaffected.

        Examples
        --------
        Filter on column 'I' (Stokes I flux). This filter will select all
        sources with Stokes I flux greater than 1.5 Jy::

        >>> s.select("I > 1.5 Jy")
        INFO: Kept 1102 sources.

        If the sky model has patches and the filter is desired per patch, use
        ``aggregate = function``. For example, to select on the sum of the patch
        fluxes::

        >>> s.select("I > 1.5 Jy", aggregate="sum")

        Or, to filter on patches smaller than 5 arcmin in size::

        >>> sizes = s.getPatchSizes(units="arcmin")
        ... s.select(sizes < 5.0, aggregate=True)

        Filter on source names, keeping those that match "src*_1?"::

        >>> s.select("Name == src*_1?")

        Use a CASA clean mask image named 'clean_mask.mask' to select sources
        that lie in masked regions::

        >>> s.select("clean_mask.mask == True")

        """
        operations.select.select(
            self,
            filterExpression,
            aggregate=aggregate,
            applyBeam=applyBeam,
            useRegEx=useRegEx,
            force=force,
        )

    def remove(
        self,
        filterExpression,
        aggregate=None,
        applyBeam=None,
        useRegEx=False,
        force=True,
    ):
        """
        Filters the sky model, removing all sources that meet the given
        expression.

        After filtering, the sky model contains only those sources for which the
        given filter expression is false.

        Parameters
        ----------
        filterExpression : str, dict, list, or numpy.ndarray

            - If string:
              A string specifying the filter expression in the form:
              '<property> <operator> <value> [<units>]'
              (e.g., 'I <= 10.5 Jy').
            - If dict:
              The filter can also be given as a dictionary in the
              form: {'filterProp':property, 'filterOper':operator,
              'filterVal':value, 'filterUnits':units}
            - If list:
              The filter can also be given as a list of:
              [property, operator, value] or
              [property, operator, value, units]
            - If `numpy.ndarray`:
              The indices to filter on can be specified directly as a numpy
              array of row or patch indices such as:
              ``array([ 0,  2, 19, 20, 31, 37])``
              or as a numpy array of bools with the same length as the sky
              model. If a numpy array is given and the indices correspond to
              patches, then set ``aggregate=True``.
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

            Units are optional and must be specified as required by
            astropy.units.

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
        Filter on column 'I' (Stokes I flux). This filter will remove all
        sources with Stokes I flux greater than 1.5 Jy::

        >>> s.remove("I > 1.5 Jy")
        INFO: Removed 1102 sources.

        If the sky model has patches and the filter is desired per patch, use
        ``aggregate = function``. For example, to filter on the sum of the patch
        fluxes::

        >>> s.remove("I > 1.5 Jy", aggregate="sum")

        Or, to filter on patches smaller than 5 arcmin in size::

        >>> sizes = s.getPatchSizes(units="arcmin")
        >>> s.remove(sizes < 5.0, aggregate=True)

        Filter on source names, removing those that match "src*_1?" (e.g.,
        'src2345_15', 'src_b2_1a', etc.)::

        >>> s.remove("Name == src*_1?")

        Use a CASA clean mask image named 'clean_mask.mask' to remove sources
        that lie in masked regions::

        >>> s.remove("clean_mask.mask == True")

        """
        operations.remove.remove(
            self,
            filterExpression,
            aggregate=aggregate,
            applyBeam=applyBeam,
            useRegEx=useRegEx,
            force=force,
        )

    def group(
        self,
        algorithm,
        targetFlux=None,
        patchNames=None,
        weightBySize=False,
        numClusters=100,
        *,
        FWHM=None,
        threshold=0.1,
        applyBeam=False,
        root="Patch",
        pad_index=False,
        method="mid",
        facet="",
        byPatch=False,
        kernelSize=0.1,
        nIterations=100,
        lookDistance=0.2,
        groupingDistance=0.01,
    ):
        """
        Groups sources into patches.

        Parameters
        ----------
        algorithm : str
            Algorithm to use for grouping:

            - 'single' => all sources are grouped into a single patch
            - 'every' => every source gets a separate patch named 'source_patch'
            - 'cluster' => SAGECAL clustering algorithm that groups sources into
              specified number of clusters (specified by the num_clusters
              parameter)
            - 'tessellate' => group into tiles whose total flux approximates the
              target flux (specified by the target_flux parameter)
            - 'threshold' => group by convolving the sky model with a Gaussian
              beam and then thresholding to find islands of emission (NOTE: all
              sources are currently considered to be point sources of flux
              unity)
            - 'facet' => group by facets using as an input a fits file. It
              requires the use of the additional parameter 'facet' to enter the
              name of the fits file.
            - 'voronoi' => given a previously grouped sky model, Voronoi
              tessellate using the patch positions for patches above the target
              flux (specified by the target_flux parameter) or whose names match
              the input names (specified by the patch_names parameter)
            - 'meanshift' => use the meanshift clustering algorithm
            - the filename of a mask image => group by masked regions (where
              mask = True). Sources outside of masked regions are given patches
              of their own

        targetFlux : str or float, optional
            Target flux for 'tessellate' (the total flux of each tile will be
            close to this value) and 'voronoi' algorithms. The target flux can
            be specified as either a float in Jy or as a string with units
            (e.g., '25.0 m_jy')
        patchNames : list, optional
            List of patch names to use for the 'voronoi' algorithm. If both
            patch_names and target_flux are given, the target_flux selection is
            applied first
        weightBySize : bool, optional
            If True, fluxes are weighted by patch size (as median_size / size)
            when the target_flux criterion is applied. Patches with sizes below
            the median (flux-weighted) size are upweighted and those above the
            mean are downweighted
        numClusters : int, optional
            Number of clusters for clustering. Sources are grouped around the
            num_clusters brightest sources.
        FWHM : str or float, optional
            FWHM of convolving Gaussian used for thresholding. The FWHM can
            be specified as either a float in degrees or as a string with units
            (e.g., '25.0 arcsec')
        threshold : float, optional
            Value between 0 and 1 above which emission is considered for
            thresholding
        apply_beam : bool, optional
            If True, fluxes will be attenuated by the beam.
        root : str, optional
            Root string from which patch names are constructed. For 'single',
            the patch name will be set to root; for the other grouping
            algorithms, the patch names will be 'root_INDX', where INDX is an
            integer ranging from (0:n_patches).
        pad_index : bool, optional
            If True, pad the INDX is used in the patch names. E.g.,
            facet_patch_001 instead of facet_patch_1
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
            For the 'tessellate' or 'meanshift' algorithms, use patches instead
            of sources
        kernel_size : float, optional
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

        >>> s.group("tessellate", target_flux=30.0)

        """
        operations.group.group(
            self,
            algorithm,
            targetFlux=targetFlux,
            patchNames=patchNames,
            weightBySize=weightBySize,
            numClusters=numClusters,
            FWHM=FWHM,
            threshold=threshold,
            applyBeam=applyBeam,
            root=root,
            pad_index=pad_index,
            method=method,
            facet=facet,
            byPatch=byPatch,
            kernelSize=kernelSize,
            nIterations=nIterations,
            lookDistance=lookDistance,
            groupingDistance=groupingDistance,
        )

    def transfer(self, patch_skymodel, matchBy="name", radius=0.1):
        """
        Transfer patches from the input sky model.

        Sources matching those in patch_sky_model will be grouped into the
        patches defined in patch_sky_model. Sources that do not appear in
        patch_sky_model will be placed into separate patches (one per source).
        Patch positions are not transferred (as they may no longer be
        appropriate after transfer).

        Parameters
        ----------
        patch_sky_model : str or Sky_model
            Input sky model from which to transfer patches.
        matchBy : str, optional
            Determines how matching sources are determined:

            - 'name' => matches are identified by name
            - 'position' => matches are identified by radius. Sources within the
              radius specified by the radius parameter are considered matches

        radius : float or str, optional
            Radius in degrees (if float) or 'value unit' (if str; e.g., '30
            arcsec') for matching when matchBy='position'

        Examples
        --------
        Transfer patches from one sky model to another and set their positions
        (matching sources are identified by name)::

        >>> s.transfer("master_sky.model")
        >>> s.setPatchPositions(method="mid")

        """
        operations.transfer.transfer(
            self, patch_skymodel, matchBy=matchBy, radius=radius
        )

    def move(self, name, position=None, shift=None):
        """
        Move or shift a source or sources.

        If both a position and a shift are specified, a source is moved to the
        new position and then shifted. Note that only a single source can be
        moved to a new position. However, multiple sources can be shifted.

        If an xyshift is specified, a FITS file must also be specified to define
        the WCS system. If a position, a shift, and an xyshift are all
        specified, a source is moved to the new position, shifted in RA and Dec,
        and then shifted in x and y.

        Parameters
        ----------
        name : str or list
            Source name or list of names (can include wildcards)
        position : list, optional
            A list specifying a new position as [RA, Dec] in either makesourcedb
            format (e.g., ['12:23:43.21', '+22.34.21.2']) or in degrees (e.g.,
            [123.2312, 23.3422])
        shift : list, optional
            A list specifying the shift as [RAShift, DecShift] in degrees (e.g.,
            [0.02312, 0.00342])

        Examples
        --------
        Move source '1609.6+6556' to a new position::

        >>> s.move("1609.6+6556", position=["16:10:00", "+65.57.00"])

        Shift the source by 10 arcsec in Dec::

        >>> s.move("1609.6+6556", shift=[0.0, 10.0 / 3600.0])

        Shift all sources by 10 pixels in x::

        >>> s.move("*", xyshift=[10, 0], fitsFile="image.fits")

        """
        operations.move.move(self, name, position=position, shift=shift)

    def add(self, col_names_vals):
        """
        Add a source to the sky model.

        Parameters
        ----------
        col_namesVals : dict
            A dictionary that specifies the column values for the source to be
            added

        Examples
        --------
        Add a point source::

        >>> source = {
        ...     "Name": "src1",
        ...     "Type": "POINT",
        ...     "Ra": "12:32:10.1",
        ...     "Dec": "23.43.21.21",
        ...     "I": 2.134,
        ... }
        ... s.add(source)

        """
        operations.add.add(self, col_names_vals)

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

        >>> s.merge(["bin0", "bin1", "bin2"], "binmerged")

        """
        operations.merge.merge(self, patches, name=name)

    def concatenate(
        self,
        skymodel2,
        matchBy="name",
        radius=0.1,
        keep="all",
        inheritPatches=False,
    ):
        """
        Concatenate two sky models.

        Parameters
        ----------
        skymodel2 : str or SkyModel
            Secondary sky model to concatenate with the parent sky model
        matchBy : str, optional
            Determines how duplicate sources are determined:

            - 'name' => duplicates are identified by name
            - 'position' => duplicates are identified by radius. Sources within
              the radius specified by the radius parameter are considered
              duplicates

        radius : float or str, optional
            Radius in degrees (if float) or 'value unit' (if str; e.g., '30
            arcsec') for matching when matchBy='position'
        keep : str, optional
            Determines how duplicates are treated:

            - 'all' => all duplicates are kept; those with identical names are
              re- named
            - 'from1' => duplicates kept are those from sky model 1 (the parent)
            - 'from2' => duplicates kept are those from sky model 2 (the
              secondary)

        inheritPatches : bool, optional
            If True, duplicates inherit the patch name from the parent sky
            model. If False, duplicates keep their own patch names.

        Examples
        --------
        Concatenate two sky models, identifying duplicates by matching to the
        source names. When duplicates are found, keep the source from the parent
        sky model and discard the duplicate from secondary sky model (this might
        be useful when merging two gsm.py sky models that have some overlap)::

        >>> skymodel2 = lsmtool.load("gsm_sky2.model")
        ... s.concatenate(skymodel2, matchBy="name", keep="from1")

        Concatenate two sky models, identifying duplicates by matching to the
        source positions within a radius of 10 arcsec. When duplicates are
        found, keep the source from the secondary sky model and discard the
        duplicate from the parent sky model (this might be useful when replacing
        parts of a low-resolution sky model with a high-resolution one)::

        >>> skymodel2 = lsmtool.load("high_res_sky.model")
        ... s.concatenate(
        ...     skymodel2,
        ...     matchBy="position",
        ...     radius=10.0 / 3600.0,
        ...     keep="from2",
        ... )

        """
        if isinstance(skymodel2, str):
            skymodel2 = SkyModel(skymodel2)
        operations.concatenate.concatenate(
            self,
            skymodel2,
            matchBy=matchBy,
            radius=radius,
            keep=keep,
            inheritPatches=inheritPatches,
        )

    def compare(
        self,
        skymodel2,
        radius="10 arcsec",
        *,
        outDir=".",
        labelBy=None,
        ignoreSpec=None,
        excludeMultiple=True,
        excludeByFlux=False,
        name1=None,
        name2=None,
        format="pdf",  # noqa
        make_plots=True,
    ):
        """
        Compare two sky models.

        Comparison plots and a text file with statistics are written out to the
        an output directory. Plots are made for:

          - flux ratio vs. radius from sky model center
          - flux ratio vs. sky position
          - flux ratio vs flux
          - position offsets

        The following statistics are saved to 'stats.txt' in the output
        directory:

            - mean and standard deviation of flux ratio
            - mean and standard deviation of RA offsets
            - mean and standard deviation of Dec offsets

        These statistics are also returned as a dictionary.

        Parameters
        ----------
        skymodel2 : SkyModel
            Secondary sky model to compare to the parent sky model
        radius : float or str, optional
            Radius in degrees (if float) or 'value unit' (if str; e.g., '30
            arcsec') for matching
        outDir : str, optional
            Plots are saved to this directory
        labelBy : str, optional
            One of 'source' or 'patch': label points using source names
            ('source') or patch names ('patch')
        ignoreSpec : float, optional
            Ignore sources with this spectral index
        excludeMultiple : bool, optional
            If True, sources with multiple matches are excluded. If False, the
            nearest of the multiple matches will be used for comparison
        excludeByFlux : bool, optional
            If True, matches whose predicted fluxes differ from the parent model
            fluxes by 25% are excluded from the positional offset plot.
        name1 : str, optional
            Name to use in the plots for the primary sky model. If None, 'Model
            1' is used.
        name2 : str, optional
            Name to use in the plots for skymodel2. If None, 'Model 2' is used.
        format : str, optional
            Format of plot files.
        make_plots : bool, optional
            If True, the plots described above are made.

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

        >>> skymodel2 = lsmtool.load("sky2.model")
        ... s.compare(skymodel2, outDir="comparison_results/")

        Compare a LOFAR sky model to a global sky model made from VLSS+TGSS+NVSS
        (where refRA and refDec are the approximate center of the LOFAR sky
        model coverage)::

        >>> skymodel2 = lsmtool.load(
        ...     "GSM", VOPosition=[refRA, refDec], VORadius="5 deg"
        ... )
        ... s.compare(
        ...     skymodel2,
        ...     radius="30 arcsec",
        ...     excludeMultiple=True,
        ...     outDir="comparison_results/",
        ...     name1="LOFAR",
        ...     name2="GSM",
        ...     format="png",
        ... )
        """

        if isinstance(skymodel2, str):
            skymodel2 = SkyModel(skymodel2)
        return operations.compare.compare(
            self,
            skymodel2,
            radius=radius,
            outDir=outDir,
            labelBy=labelBy,
            ignoreSpec=ignoreSpec,
            excludeMultiple=excludeMultiple,
            excludeByFlux=excludeByFlux,
            name1=name1,
            name2=name2,
            format=format,
            make_plots=make_plots,
        )

    def plot(self, filename=None, labelBy=None):
        """
        Shows a simple plot of the sky model.

        The circles in the plot are scaled with flux. If the sky model is
        grouped into patches, sources are colored by patch and the patch
        positions are indicated with stars.

        Parameters
        ----------
        filename : str, optional
            If given, the plot is saved to a file instead of displayed.
        labelBy : str, optional
            One of 'source' or 'patch': label points using source names
            ('source') or patch names ('patch')

        Examples
        --------
        Plot and display to the screen::

        >>> s.plot()

        Plot and save to a PDF file::

        >>> s.plot("sky_plot.pdf")

        """
        operations.plot.plot(self, filename, labelBy=labelBy)

    def rasterize(
        self, cellsize, fileRoot=None, writeRegionFile=False, clobber=False
    ):
        """
        Rasterize the sky model to FITS images (one image per spectral term).

        The resulting images can be used with DDECal in DP3 for prediction using
        IDG. If the sky model is grouped into contiguous patches, a ds9 region
        file defining the Voronoi patches can also written (this file is
        required for use with IDG predict).

        Note: currently, when writing the FITS images, only sky models with
        LogarithmicSI = False are supported.

        Parameters
        ----------
        cellsize : float
            The cellsize in degrees for the output image.
        fileRoot : str, optional
            Filename root for the output FITS images. The images will be named
            file_root + '_0.fits', file_root + '_1.fits', etc. (one for each
            spectral term in the sky model). If write_region_file is True, a ds9
            region file is also written as file_root + '.reg'. If not given, the
            root is taken from the filename of the input sky model (with its
            extension, if any, removed), if available, and otherwise is set to
            'skymodel'
        writeRegionFile : bool, optional
            If True and the sky model is grouped into contiguous patches, a ds9
            region file defining the Voronoi patches will be written (this file
            is required for DDECal)
        clobber : bool, optional
            If True, existing files are overwritten.
        """

        file_root = fileRoot
        write_region_file = writeRegionFile
        overwrite = clobber

        # TODO: Fix circular import and move to module scope
        from lsmtool.facet import tessellate  # noqa

        # Check inputs
        if write_region_file and not self.hasPatches:
            raise ValueError(
                "writeRegionFile = True but sky model is not grouped into "
                "patches."
            )
        if file_root is None:
            if self._filename is None:
                file_root = "skymodel"
            else:
                file_root = os.path.splitext(self._filename)[0]

        # Make a blank image for each spectral term
        reference_frequency = self.getColValues("ReferenceFrequency")
        ref_freq = reference_frequency[0]  # TODO: allow per-source ref freq
        fluxes = self.getColValues("I")
        types = self.getColValues("Type")
        nsources = len(fluxes)
        if "SpectralIndex" in self.getColNames():
            spectral_indices = self.getColValues("SpectralIndex")
        else:
            spectral_indices = [[]] * nsources
        nterms = len(spectral_indices[0]) + 1

        # Check that LogarithmicSI = False for all entries
        if nterms > 1:
            logsi = self.getColValues("LogarithmicSI")
            if np.any(logsi == "true"):
                raise RuntimeError(
                    "Sky model has one or more sources with "
                    "LogarithmicSI = True. Only sky models with "
                    "LogarithmicSI = False are supported at this time."
                )

        image_names = [f"{file_root}_{i}.fits" for i in range(nterms)]
        for image_name in image_names:
            if os.path.exists(image_name):
                if overwrite:
                    os.remove(image_name)
                else:
                    raise IOError(
                        f"The output file {image_name!r} exists and "
                        "clobber = False."
                    )

        x, y, wcs = self._get_xy(crdelt=cellsize)
        if "GAUSSIAN" in types:
            fwhm = np.max(
                self.getColValues("MajorAxis", units="degree") * cellsize
            )
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
        ref_ra, ref_dec = wcs.wcs_pix2world(xcen, ycen, 0)
        ra = self.getColValues("Ra")
        dec = self.getColValues("Dec")

        for image_name in image_names:
            make_template_image(
                image_name,
                ref_ra,
                ref_dec,
                ref_freq,
                ximsize=xsize,
                yimsize=ysize,
                cellsize_deg=cellsize,
            )

        # Build each image, one at a time (to minimize memory usage)
        for t, image_name in enumerate(image_names):
            # Read in image data
            hdu = pyfits.open(image_name, memmap=False)
            imdata = hdu[0].data
            w = WCS(hdu[0].header)

            # Loop over sources, adding them to the images (note that the
            # spectral terms sum together when summing polynomials, just as the
            # flux densities do)
            if t == 0:
                # Flux densities
                itervalues = fluxes
            else:
                # Spectral terms
                itervalues = spectral_indices
            for i, (ra_src, dec_src, val, type_) in enumerate(
                zip(ra, dec, itervalues, types, strict=True)
            ):
                if t > 0:
                    v = val[t - 1]
                    const = True
                else:
                    v = val
                    const = False
                ra_dec = np.array([[ra_src, dec_src, 0.0, 0.0]])
                xs, ys = (
                    w.wcs_world2pix(ra_dec, 0)[0][0],
                    w.wcs_world2pix(ra_dec, 0)[0][1],
                )
                if type_ == "POINT":
                    imdata[0, 0, int(np.round(ys)), int(np.round(xs))] += v
                elif type_ == "GAUSSIAN":
                    s1 = (
                        self.getColValues("MajorAxis", units="degree")[i]
                        / cellsize
                    )  # pixels
                    s2 = (
                        self.getColValues("MinorAxis", units="degree")[i]
                        / cellsize
                    )  # pixels
                    th = self.getColValues("Orientation")[i]  # degrees
                    c1, c2 = ys, xs
                    b = np.ceil(s1 * 2.5)
                    bbox = np.s_[
                        max(0, int(c1 - b)) : min(xsize, int(c1 + b + 1)),
                        max(0, int(c2 - b)) : min(ysize, int(c2 + b + 1)),
                    ]
                    x_ax, y_ax = np.mgrid[bbox]
                    g = [v, c1, c2, s1, s2, th]
                    ffimg = gaussian_fcn(g, x_ax, y_ax, const=const)
                    imdata[0, 0, :, :][bbox] += ffimg
            hdu[0].data = imdata
            hdu.writeto(image_name, overwrite=True)

        # Make region file if needed
        if write_region_file and self.hasPatches:
            ra, dec = self.getPatchPositions(asArray=True)
            patch_names = self.getPatchNames()
            x_pix_list = []
            y_pix_list = []
            for ra_src, dec_src in zip(ra, dec, strict=True):
                ra_dec = np.array([[ra_src, dec_src, 0.0, 0.0]])
                y_pix, x_pix = (
                    w.wcs_world2pix(ra_dec, 0)[0][0],
                    w.wcs_world2pix(ra_dec, 0)[0][1],
                )
                x_pix_list.append(x_pix)
                y_pix_list.append(y_pix)
            dist_pix = np.sqrt(xsize**2 + ysize**2)
            width_pix = dist_pix * cellsize
            _, vertices = tessellate(
                SkyCoord(ra, dec, unit="deg"),
                SkyCoord(ref_ra[0], ref_dec[0], unit="deg"),
                [width_pix, width_pix],
            )
            lines = []
            lines.append(
                "# Region file format: DS9 version 4.0\nglobal color=green "
                'font="helvetica 10 normal" select=1 highlite=1 edit=1 '
                "move=1 delete=1 include=1 fixed=0 source=1\nfk5\n"
            )
            for verts, pname in zip(vertices, patch_names, strict=True):
                xylist = []
                varray = np.array(verts).T
                ras = varray[0]
                decs = varray[1]
                for x, y in zip(ras, decs, strict=True):
                    xylist.append(f"{x}, {y}")
                lines.append(
                    f"polygon({', '.join(xylist)}) # text={{{pname}}}\n"
                )

            outputfile = f"{file_root}.reg"
            if os.path.exists(outputfile):
                if overwrite:
                    os.remove(outputfile)
                else:
                    raise IOError(
                        f"The output file {outputfile!r} exists and "
                        "clobber = False."
                    )
            with open(outputfile, "w") as f:
                f.writelines(lines)
