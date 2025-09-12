"""
Module for filtering and grouping a sky model based on sources found by
`pyBDSF`_ in an image. This file was originally copied from the `rapthor
repository`_. Also includes substantial changes introduced in the `SKA self
calibration pipeline`_.

.. _pyBDSF: https://pybdsf.readthedocs.io/en/stable/
.. _rapthor repository:
    https://git.astron.nl/RD/rapthor/-/blob/544ddf/rapthor/scripts/filter_skymodel.py
.. _SKA self calibration pipeline:
    https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-wflow-selfcal/-/blob/3be896/src/ska_sdp_wflow_selfcal/pipeline/support/filter_skymodel.py
"""

import logging
import numbers
import os
from ast import literal_eval
from pathlib import Path
from typing import List, Tuple, Union

import bdsf
import numpy as np
from astropy.io import fits as pyfits
from astropy.wcs import WCS
from casacore.tables import table as casa_table

from ..io import (
    ListOfPathLike,
    PathLike,
    PathLikeOptional,
    PathLikeOrListOptional,
    load,
    read_vertices_ra_dec,
    temp_storage,
    validate_paths,
)
from ..skymodel import SkyModel
from ..utils import format_coordinates, rasterize, transfer_patches

# Module logger
logger = logging.getLogger(__name__)

# type aliases
ListOfCoords = List[Tuple[numbers.Real, numbers.Real]]


def filter_skymodel(
    flat_noise_image: PathLike,
    true_sky_image: PathLikeOptional,
    input_true_skymodel: PathLikeOptional,
    input_apparent_skymodel: PathLikeOptional,
    output_apparent_sky: PathLike,
    output_true_sky: PathLike,
    vertices_file: PathLike,
    beam_ms: PathLikeOrListOptional = None,
    input_bright_skymodel: PathLikeOptional = None,
    *,  # remaining parameters are keyword-only
    thresh_isl: numbers.Real = 5.0,
    thresh_pix: numbers.Real = 7.5,
    rmsbox: Tuple[numbers.Integral] = (150, 50),
    rmsbox_bright: Tuple[numbers.Integral] = (35, 7),
    adaptive_rmsbox: bool = True,
    adaptive_thresh: numbers.Real = 75.0,
    filter_by_mask: bool = True,
    output_catalog: PathLikeOptional = "",
    output_flat_noise_rms: PathLikeOptional = "",
    output_true_rms: PathLikeOptional = "",
    ncores: int = 8,
) -> int:
    """
    Filters the input sky model using PyBDSF.

    If no islands of emission are detected in the input image, a blank sky
    model is made. If any islands are detected in the input image, filtered
    true-sky and apparent-sky models are made, as well as a FITS clean mask
    (with the filename input_image+'.mask').

    Parameters
    ----------
    flat_noise_image, true_sky_image, input_true_skymodel
        See :py:func:`lsmtool.filter_skymodel.filter_skymodel` for the
        meaning of the positional parameters.
    input_apparent_skymodel, output_apparent_sky, output_true_sky, beam_ms
        See :py:func:`lsmtool.filter_skymodel.filter_skymodel` for the
        meaning of the positional parameters.
    vertices_file, input_bright_skymodel
        See :py:func:`lsmtool.filter_skymodel.filter_skymodel` for the
        meaning of the positional parameters.

    Other Parameters
    ----------------
    thresh_isl : float, optional
        Value of thresh_isl PyBDSF parameter.
    thresh_pix : float, optional
        Value of thresh_pix PyBDSF parameter.
    rmsbox : tuple of float, optional
        Value of rms_box PyBDSF parameter.
    rmsbox_bright : tuple of float, optional
        Value of `rms_box_bright` PyBDSF parameter.
    adaptive_rmsbox : bool, optional
        Value of `adaptive_rms_box` PyBDSF parameter.
    adaptive_thresh : float, optional
        If `adaptive_rmsbox` is True, this value sets the threshold above which
        a source will use the small rms box.
    filter_by_mask : bool, optional
        If True, filter the input sky model by the PyBDSF-derived mask,
        removing sources that lie in unmasked regions.
    output_catalog: str or pathlib.Path, optional
        The filename for source catalog. If not provided, do not create it.
    output_flat_noise_rms: str or pathlib.Path, optional
        The filename for the flat noise root-mean-square (RMS) image. If not
        provied, do not create it. Creating this image may require an
        additional PyBDSF call and thereby slow down this function
        significantly.
    output_true_rms: str or pathlib.Path, optional
        The filename for the true sky RMS image. If not provied, do not create
        it.
    ncores : int
        Specify the number of cores that BDSF should use. Defaults to 8.

    Raises
    ------
    FileNotFoundError
        If none of the following input files is defined: `input_true_skymodel`,
        `input_apparent_skymodel` or `input_bright_skymodel`.

    Returns
    -------
    n_sources : int
        The number of sources detected by pyBDSF.
    """

    # Check that the input paths are valid before attempting any work.
    # For parameters that are expected to be paths:
    # - If they are required, ensure that they exist
    # - If they are not required, and are not null, ensure that they exist
    validate_paths(
        flat_noise_image=flat_noise_image,
        vertices_file=vertices_file,
        required=True,
    )
    validate_paths(
        true_sky_image=true_sky_image,
        input_true_skymodel=input_true_skymodel,
        input_apparent_skymodel=input_apparent_skymodel,
        input_bright_skymodel=input_bright_skymodel,
        required=False,
    )

    rmsbox = parse_rmsbox(rmsbox)
    rmsbox_bright = parse_rmsbox(rmsbox_bright)

    # set the TMPDIR environmental variable for temporary data storate
    with temp_storage():
        img_true_sky = process_images(
            flat_noise_image,
            true_sky_image,
            beam_ms,
            output_catalog,
            output_flat_noise_rms,
            output_true_rms,
            mean_map="zero",
            rms_box=rmsbox,
            thresh_pix=thresh_pix,
            thresh_isl=thresh_isl,
            thresh="hard",
            adaptive_rms_box=adaptive_rmsbox,
            adaptive_thresh=adaptive_thresh,
            rms_box_bright=rmsbox_bright,
            rms_map=True,
            quiet=True,
            ncores=ncores,
        )
    # NOTE: The TMPDIR environmental variable is set back to its original value
    # once the with block above exits

    # Save number of sources found by PyBDSF for later use
    n_sources = img_true_sky.nsrc

    # Filter the sky model (if it was given) and any sources were detected
    if img_true_sky.nisl > 0 and (input_true_skymodel or input_bright_skymodel):
        filter_sources(
            img_true_sky,
            vertices_file,
            input_true_skymodel,
            input_apparent_skymodel,
            input_bright_skymodel,
            beam_ms,
            filter_by_mask,
            output_true_sky,
            output_apparent_sky,
        )
    else:
        create_dummy_skymodel(
            img_true_sky, output_true_sky, output_apparent_sky
        )

    return n_sources


def parse_rmsbox(rmsbox: Union[str, None]):
    """Parses the rmsbox parameter."""
    return literal_eval(rmsbox) if isinstance(rmsbox, str) else rmsbox


def process_images(
    flat_noise_image: PathLike,
    true_sky_image: PathLike,
    beam_ms: PathLikeOrListOptional,
    output_catalog: PathLikeOptional,
    output_flat_noise_rms: PathLikeOptional,
    output_true_rms: PathLikeOptional,
    **config,
):
    """
    Processes images using PyBDSF and generates output files.

    This function runs PyBDSF on either the true sky image or the flat noise
    image, depending on whether a beam measurement set is provided. It then
    optionally generates a source catalog and RMS maps.

    Parameters
    ----------
    flat_noise_image : str or pathlib.Path
        Path to the flat noise image.
    true_sky_image : str or pathlib.Path
        Path to the true sky image.
    beam_ms : str or pathlib.Path or list of str or list of Path or None
        Path to the beam measurement set.
    output_catalog: str or pathlib.Path, optional
        The filename for source catalog. If empty, do not create it.
    output_flat_noise_rms: str or pathlib.Path, optional
        The filename for the flat noise RMS image. If empty, do not create it.
        Creating this image may require an additional PyBDSF call and thereby
        slow down this function significantly.
    output_true_rms: str or pathlib.Path, optional
        The filename for the true sky RMS image. If empty, do not create it.
    **config
        Additional keyword arguments passed to the bdsf.process_image call.
    """

    # Run PyBDSF first on the true-sky image to determine its properties and
    # measure source fluxes.
    img_true_sky = bdsf.process_image(
        true_sky_image if beam_ms else flat_noise_image,
        atrous_do=True,
        atrous_jmax=3,
        **config,
    )

    if output_catalog:
        # Write the catalog of sources detected in the input true sky image.
        # This will write the catalog to file, even if no sources were found.
        img_true_sky.write_catalog(
            outfile=str(output_catalog),
            format="fits",
            catalog_type="srl",
            clobber=True,
            force_output=True,
        )

    if output_true_rms:
        img_true_sky.export_image(
            outfile=output_true_rms, img_type="rms", clobber=True
        )

    if not output_flat_noise_rms:
        return img_true_sky

    if beam_ms:
        # Run PyBDSF again on the flat-noise image and save the RMS map for
        # later use in the image diagnostics step
        img_flat_noise = bdsf.process_image(
            flat_noise_image,
            **config,
            stop_at="isl",
        )
        img_flat_noise.export_image(
            outfile=output_flat_noise_rms, img_type="rms", clobber=True
        )
        del img_flat_noise  # helps reduce memory usage

    else:
        # No beam, output_flat_noise_rms is the same as img_true_sky
        img_true_sky.export_image(
            outfile=output_flat_noise_rms, img_type="rms", clobber=True
        )

    return img_true_sky


def filter_sources(
    img_true_sky: PathLike,
    vertices_file: PathLike,
    input_true_skymodel: PathLikeOptional,
    input_apparent_skymodel: PathLikeOptional,
    input_bright_skymodel: PathLikeOptional,
    beam_ms: PathLikeOrListOptional,
    filter_by_mask: bool,
    output_true_sky: PathLikeOptional,
    output_apparent_sky: PathLikeOptional,
) -> SkyModel:
    """
    Filter and group sources based on a mask and other criteria.

    This function filters the input sky model based on a mask generated from
    the true sky image. It also handles adding bright sources, removing
    negative components, and grouping sources by mask islands.

    Parameters
    ----------
    img_true_sky : bdsf.image.Image
        The PyBDSF image object.
    vertices_file : str or pathlib.Path
        Filename of file with vertices, which determine the imaging field.
    input_true_skymodel : str or pathlib.Path, optional
        Filename of input makesourcedb sky model with true fluxes.
    input_apparent_skymodel : str or Path, optional
        Filename of input makesourcedb sky model with apparent fluxes.
    input_bright_skymodel : str or pathlib.Path, optional
        Filename of input makesourcedb sky model of bright sources only.
    beam_ms : str or pathlib.Path, optional
        The filename of the MS for deriving the beam attenuation.
    filter_by_mask : bool, optional
        If True, filter the input sky model by the PyBDSF-derived mask.
    output_true_sky : str or pathlib.Path
        Output file name for the generated true sky model.
    output_apparent_sky : str or pathlib.Path
        Output file name for the generated apparent sky model.
    """

    mask_file = f"{img_true_sky.filename}.mask"
    img_true_sky.export_image(
        outfile=mask_file, clobber=True, img_type="island_mask"
    )
    del img_true_sky  # helps reduce memory usage

    # Construct polygon needed to trim the mask to the sector
    trim_mask(mask_file, vertices_file)

    # Select the best measurement set for beam attenuation.
    beam_ms = select_midpoint(beam_ms) if beam_ms else None

    # Load input true sky model and add bright source catalogue if provided
    if input_true_skymodel:
        # Load the sky model with the associated beam MS.
        true_skymodel = load(input_true_skymodel, beamMS=beam_ms)
        if input_bright_skymodel:
            # If bright sources were peeled before imaging, add them back
            add_bright_sources(true_skymodel, input_bright_skymodel)
    else:
        # If no input true skymodel given, use the input bright sources
        # skymodel
        true_skymodel = load(input_bright_skymodel)

    # Do final filtering and write out the sky models
    if filter_by_mask:
        # Keep only those sources in PyBDSF masked regions
        true_skymodel.select(f"{mask_file} == True")

    # Group the sky model by mask islands
    true_skymodel.group(mask_file)

    if input_apparent_skymodel:
        apparent_skymodel = load(input_apparent_skymodel)
        # Match the filtering and grouping of the filtered model
        matches = np.isin(
            apparent_skymodel.getColValues("Name"),
            true_skymodel.getColValues("Name"),
        )
        apparent_skymodel.select(matches)
        transfer_patches(
            true_skymodel,
            apparent_skymodel,
            patch_dict=true_skymodel.getPatchPositions(),
        )
        apparent_skymodel.write(
            output_apparent_sky, clobber=True, applyBeam=False
        )
    else:
        # If the apparent sky model is not given, attenuate the true-sky
        # one by applying the beam. If beam_ms is None, the apparent and
        # true skymodels will be equal.
        true_skymodel.write(
            output_apparent_sky, clobber=True, applyBeam=bool(beam_ms)
        )

    # Write out true sky model
    true_skymodel.write(output_true_sky, clobber=True)

    # Remove the mask file
    os.remove(mask_file)

    return true_skymodel


def trim_mask(mask_file: PathLike, vertices_file: PathLike):
    """
    Trim the mask file to the given vertices.

    This function opens the mask file, creates a polygon from the vertices
    using the file's WCS, rasterizes the polygon, and overwrites the mask file
    with the rasterized polygon data.

    Parameters
    ----------
    mask_file: str or pathlib.Path:
        Path to the mask file.
    vertices_file: str or pathlib.Path:
        Path to the file containing vertices.
    """
    hdu = pyfits.open(mask_file, memmap=False)
    vertices = read_vertices_ra_dec(vertices_file)
    # Construct polygon needed to trim the mask to the sector
    verts = create_polygon(WCS(hdu[0].header), vertices)

    # Rasterize the poly
    data = hdu[0].data
    data[0, 0, :, :] = rasterize(verts, data[0, 0, :, :])
    hdu.writeto(mask_file, overwrite=True)


def select_midpoint(beam_ms: ListOfPathLike) -> str:
    """
    Select the best measurement set for beam attenuation.

    Selects the measurement set (MS) that is closest to the median time of all
    provided data. This is intended to choose a representative beam for
    attenuation calculations.

    Parameters
    ----------
    beam_ms : list of str
        List of measurement set filenames.

    Returns
    -------
    str
        The filename of the selected measurement set.
    """
    if isinstance(beam_ms, (str, Path)):
        return beam_ms

    n = len(beam_ms)
    ms_times = np.empty(n)
    for i, ms in enumerate(beam_ms):
        with casa_table(str(ms), ack=False) as table:
            ms_times[i] = np.mean(table.getcol("TIME"))

    ms_times = sorted(ms_times)
    mid_time = ms_times[n // 2]
    beam_ind = ms_times.index(mid_time)
    return beam_ms[beam_ind]


def create_polygon(wcs: WCS, vertices: ListOfCoords) -> ListOfCoords:
    """
    Create a polygon from vertices in world coordinates.

    This function converts vertices to pixel coordinates using the provided
    World Coordinate System (WCS), and returns them as a list of tuples.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS object for coordinate transformation.
    vertices : list
        Vertices of the polygon in world coordinates.

    Returns
    -------
    coordinates: list of tuple
        A list of (x, y) pixel coordinates representing the polygon vertices.
    """
    return list(generate_polygon(wcs, vertices))


def generate_polygon(wcs: WCS, vertices: ListOfCoords):
    """
    Generate pixel coordinates for polygon vertices.
    See ::py:func:`create_polygon`.
    """
    ra_ind = wcs.axis_type_names.index("RA")
    dec_ind = wcs.axis_type_names.index("DEC")

    for ra_vert, dec_vert in vertices:
        ra_dec = np.array([[0.0, 0.0, 0.0, 0.0]])
        ra_dec[0][ra_ind] = ra_vert
        ra_dec[0][dec_ind] = dec_vert
        yield (
            wcs.wcs_world2pix(ra_dec, 0)[0][ra_ind],
            wcs.wcs_world2pix(ra_dec, 0)[0][dec_ind],
        )


def add_bright_sources(
    input_skymodel: SkyModel, input_bright_skymodel: PathLike
):
    """
    Add bright sources to the input sky model.

    Loads the bright source sky model, renames the sources to remove any
    previous sector prefixes, and appends it to the input sky model.

    Parameters
    ----------
    input_skymodel : SkyModel
        The input sky model to which bright sources will be added.
    input_bright_skymodel : str or pathlib.Path
        The filename of the bright source sky model.
    """

    bright_skymodel = load(input_bright_skymodel)
    # Rename the bright sources, removing the '_sector_*' added previously
    # (otherwise the '_sector_*' text will be added every iteration, eventually
    # making for very long source names)
    new_names = [
        name.split("_sector")[0]
        for name in bright_skymodel.getColValues("Name")
    ]
    # str.split can be relaced with str.removeprefix once python 3.8 is dropped
    bright_skymodel.setColValues("Name", new_names)
    input_skymodel.concatenate(bright_skymodel)
    return bright_skymodel


def create_dummy_skymodel(
    img_true_sky: bdsf.image.Image,
    output_true_sky: PathLike,
    output_apparent_sky: PathLike,
):
    """
    Create a dummy sky model if no islands of emission are detected.

    Parameters
    ----------
    img_true_sky : bdsf.image.Image
        The PyBDSF image object.
    output_true_sky : str or pathlib.Path
        Output file name for the true sky model.
    output_apparent_sky : str or pathlib.Path
        Output file name for the apparent sky model.

    """
    ra, dec = img_true_sky.pix2sky(
        (img_true_sky.shape[-2] / 2.0, img_true_sky.shape[-1] / 2.0)
    )
    del img_true_sky  # helps reduce memory usage

    ra, dec = format_coordinates(ra, dec)
    dummy_text = (
        "Format = Name, Type, Patch, Ra, Dec, I, SpectralIndex, "
        "LogarithmicSI, ReferenceFrequency='100000000.0', MajorAxis, "
        "MinorAxis, Orientation\n"
        f",,p1,{ra},{dec}\n"
        f"s0c0,POINT,p1,{ra},{dec},0.00000001,[0.0,0.0],false,"
        "100000000.0,,,\n"
    )
    for filename in (output_true_sky, output_apparent_sky):
        Path(filename).write_text(dummy_text, encoding="utf-8")
