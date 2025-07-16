"""
Script to filter and group a sky model with an image.

This file was originally copied from the Rapthor repository:
https://git.astron.nl/RD/rapthor/-/blob/544ddf/rapthor/scripts/filter_skymodel.py

Also includes substantial changes introduced in the SKA self calibration
pipeline at
https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-wflow-selfcal/-/blob/3be896/src/ska_sdp_wflow_selfcal/pipeline/support/filter_skymodel.py
"""

import logging
import os
from ast import literal_eval
from pathlib import Path

import bdsf
import numpy as np
from astropy.io import fits as pyfits
from astropy.wcs import WCS

from ..io import load, read_vertices_ra_dec, temp_storage
from ..utils import format_coordinates, rasterize

# Module logger
logger = logging.getLogger(__name__)


def filter_skymodel(
    # pylint: disable=too-many-locals,too-many-statements
    flat_noise_image,
    true_sky_image,
    input_skymodel,
    output_apparent_sky,
    output_true_sky,
    vertices_file,
    beam_ms="",
    input_bright_skymodel=None,
    thresh_isl=5.0,
    thresh_pix=7.5,
    rmsbox=(150, 50),
    rmsbox_bright=(35, 7),
    adaptive_rmsbox=True,
    adaptive_thresh=75.0,
    filter_by_mask=True,
    remove_negative=False,
    output_catalog="",
    output_flat_noise_rms="",
    output_true_rms="",
    ncores=8,
):
    """
    Filters the input sky model using PyBDSF.

    Note: If no islands of emission are detected in the input image, a
    blank sky model is made. If any islands are detected in the input image,
    filtered true-sky and apparent-sky models are made, as well as a FITS clean
    mask (with the filename input_image+'.mask').

    Parameters
    ----------
    flat_noise_image : str
        Filename of input image to use to detect sources for filtering.
        It should be a flat-noise / apparent sky image (without primary-beam
        correction).
    true_sky_image : str, optional
        Filename of input image to use to determine the true flux of sources.
        It should be a true flux image (with primary-beam correction).
        If beam_ms is not empty, this argument must be supplied. Otherwise,
        filter_skymodel ignores it and uses the flat_noise_image instead.
    input_skymodel : str
        Filename of input makesourcedb sky model.
        If beam_ms is empty, it should be an apparent sky model, without
        primary-beam correction.
        If beam_ms is not empty, it should be a true sky model, with
        primary-beam correction.
    output_apparent_sky: str
        Output file name for the generated apparent sky model, without
        primary-beam correction.
    output_true_sky : str
        Output file name for the generated true sky model, with
        primary-beam correction.
    vertices_file : str
        Filename of file with vertices, which determine the imaging field.
    beam_ms : str, optional
        The filename of the MS for deriving the beam attenuation and
        theoretical image noise. If empty, the generated apparent and true sky
        models will be equal.
    input_bright_skymodel : str, optional
        Filename of input makesourcedb sky model of bright sources only.
        If supplied, filter_skymodel adds the sources in this skymodel.
        If beam_ms is empty, it should be an apparent sky model, without
        primary-beam correction.
        If beam_ms is not empty, it should be a true sky model, with
        primary-beam correction.
    thresh_isl : float, optional
        Value of thresh_isl PyBDSF parameter
    thresh_pix : float, optional
        Value of thresh_pix PyBDSF parameter
    rmsbox : tuple of floats, optional
        Value of rms_box PyBDSF parameter
    rmsbox_bright : tuple of floats, optional
        Value of rms_box_bright PyBDSF parameter
    adaptive_rmsbox : bool, optional
        Value of adaptive_rms_box PyBDSF parameter
    use_adaptive_threshold : bool, optional
        If True, use an adaptive threshold estimated from the negative values
        in the image
    adaptive_thresh : float, optional
        If adaptive_rmsbox is True, this value sets the threshold above
        which a source will use the small rms box
    comparison_skymodel : str, optional
        The filename of the sky model to use for flux scale and astrometry
        comparisons
    filter_by_mask : bool, optional
        If True, filter the input sky model by the PyBDSF-derived mask,
        removing sources that lie in unmasked regions
    remove_negative : bool, optional
        If True, remove negative sky model components
    output_catalog: str, optional
        The filename for source catalog. If empty, do not create it.
    output_flat_noise_rms: str, optional
        The filename for the flat noise RMS image. If empty, do not create it.
        Creating this image may require an additional PyBDSF call and thereby
        slow down this function significantly.
    output_true_rms: str, optional
        The filename for the true sky RMS image. If empty, do not create it.
    ncores : int, optional
        Specify the number of cores that BDSF should use. Defaults to 8.
    """

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

    if img_true_sky.nisl > 0:
        filter_sources(
            img_true_sky,
            vertices_file,
            input_skymodel,
            input_bright_skymodel,
            beam_ms,
            filter_by_mask,
            remove_negative,
            output_true_sky,
            output_apparent_sky,
        )
    else:
        create_dummy_skymodel(
            img_true_sky, output_true_sky, output_apparent_sky
        )


def parse_rmsbox(rmsbox):
    """Parses the rmsbox parameter."""
    return literal_eval(rmsbox) if isinstance(rmsbox, str) else rmsbox


def process_images(
    flat_noise_image,
    true_sky_image,
    beam_ms,
    output_catalog,
    output_flat_noise_rms,
    output_true_rms,
    **config,
):
    """
    Processes images using PyBDSF and generates output files.

    This function runs PyBDSF on either the true sky image or the
    flat noise image, depending on whether a beam measurement set
    is provided. It then optionally generates a source catalog and
    RMS maps.

    Parameters
    ----------
    flat_noise_image : str or Path
        Path to the flat noise image.
    true_sky_image : str or Path
        Path to the true sky image.
    beam_ms : str or Path
        Path to the beam measurement set.
    output_catalog: str, optional
        The filename for source catalog. If empty, do not create it.
    output_flat_noise_rms: str, optional
        The filename for the flat noise RMS image. If empty, do not create it.
        Creating this image may require an additional PyBDSF call and thereby
        slow down this function significantly.
    output_true_rms: str, optional
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
        img_true_sky.write_catalog(
            outfile=str(output_catalog),
            format="fits",
            catalog_type="srl",
            clobber=True,
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
    img_true_sky,
    vertices_file,
    input_skymodel,
    input_bright_skymodel,
    beam_ms,
    filter_by_mask,
    remove_negative,
    output_true_sky,
    output_apparent_sky,
):
    """
    Filter and group sources based on a mask and other criteria.

    This function filters the input sky model based on a mask
    generated from the true sky image. It also handles adding
    bright sources, removing negative components, and grouping
    sources by mask islands.

    Parameters
    ----------
    img_true_sky : bdsf.image.Image
        The PyBDSF image object.
    vertices_file : str or Path
        Filename of file with vertices, which determine the imaging field.
    input_skymodel : str or Path
        Filename of input makesourcedb sky model.
    input_bright_skymodel : str or Path, optional
        Filename of input makesourcedb sky model of bright sources only.
    beam_ms : str or Path, optional
        The filename of the MS for deriving the beam attenuation.
    filter_by_mask : bool, optional
        If True, filter the input sky model by the PyBDSF-derived mask.
    remove_negative : bool, optional
        If True, remove negative sky model components.
    output_true_sky : str or Path
        Output file name for the generated true sky model.
    output_apparent_sky : str or Path
        Output file name for the generated apparent sky model.

    """

    mask_file = f"{img_true_sky.filename}.mask"
    img_true_sky.export_image(
        outfile=mask_file, clobber=True, img_type="island_mask"
    )

    # Construct polygon needed to trim the mask to the sector
    trim_mask(mask_file, vertices_file)

    # Load the sky model with the associated beam MS.
    input_skymodel = load(str(input_skymodel), beamMS=str(beam_ms))

    # If bright sources were peeled before imaging, add them back
    if input_bright_skymodel:
        add_bright_sources(input_skymodel, input_bright_skymodel)

    # Do final filtering and write out the sky models
    if remove_negative:
        # Keep only those sources with positive flux densities
        input_skymodel.select("I > 0.0")

    if input_skymodel and filter_by_mask:
        # Keep only those sources in PyBDSF masked regions
        input_skymodel.select(f"{mask_file} == True")

    # Write out apparent- and true-sky models
    input_skymodel.group(mask_file)  # group the sky model by mask islands
    input_skymodel.write(output_true_sky, clobber=True)
    input_skymodel.write(
        output_apparent_sky,
        clobber=True,
        applyBeam=bool(beam_ms),
    )
    os.remove(mask_file)


def trim_mask(mask_file, vertices_file):
    """
    Trim the mask file to the given vertices.

    This function opens the mask file, creates a polygon from the vertices
    using the file's WCS, rasterizes the polygon, and overwrites the
    mask file with the rasterized polygon data.

    Args:
        mask_file: Path to the mask file.
        vertices_file: Path to the file containing vertices.
    """
    hdu = pyfits.open(mask_file, memmap=False)
    vertices = read_vertices_ra_dec(vertices_file)
    verts = create_polygon(WCS(hdu[0].header), vertices)

    # Rasterize the poly
    data = hdu[0].data
    data[0, 0, :, :] = rasterize(verts, data[0, 0, :, :])
    hdu.writeto(mask_file, overwrite=True)


def create_polygon(wcs, vertices):
    """
    Create a polygon from vertices in world coordinates.

    This function converts vertices to pixel coordinates using the
    provided WCS, and returns them as a list of tuples.

    Args:
        wcs (astropy.wcs.WCS):
            The WCS object for coordinate transformation.
        vertices (list):
            Vertices of the polygon in world coordinates.

    Returns:
        A list of (x, y) pixel coordinates representing the polygon vertices.
    """

    # Construct polygon needed to trim the mask to the sector
    return list(generate_polygon(wcs, vertices))


def generate_polygon(wcs, vertices):

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


def add_bright_sources(input_skymodel, input_bright_skymodel):

    s_bright = load(str(input_bright_skymodel))
    # Rename the bright sources, removing the '_sector_*' added previously
    # (otherwise the '_sector_*' text will be added every iteration, eventually
    # making for very long source names)
    new_names = [
        name.split("_sector")[0] for name in s_bright.getColValues("Name")
    ]
    s_bright.setColValues("Name", new_names)
    input_skymodel.concatenate(s_bright)


def create_dummy_skymodel(img_true_sky, output_true_sky, output_apparent_sky):
    """
    Create a dummy sky model if no islands of emission are detected.

    Parameters
    ----------
    img_true_sky : bdsf.image.Image
        The PyBDSF image object.
    output_true_sky : str or Path
        Output file name for the true sky model.
    output_apparent_sky : str or Path
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
