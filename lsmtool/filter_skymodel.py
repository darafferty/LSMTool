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
from typing import Union

import bdsf
import numpy as np
import sofia2
from astropy.io import fits as pyfits
from astropy.table import Table
from astropy.units import Quantity
from astropy.wcs import WCS

from .correct_gaussian_orientation import compute_absolute_orientation
from .io import load, read_vertices_ra_dec, temp_storage
from .utils import (
    format_coordinates,
    rasterize,
    rotation_matrix_2d,
    table_to_array,
)

# Module logger
logger = logging.getLogger(__name__)


# Type aliases for paths-like objects
PathLike = Union[str, Path]
PathLikeOptional = Union[PathLike, None]

# conversion factor between sofia and makeshourcedb parameterisations
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))

KNOWN_SOURCE_FINDERS = {"sofia", "bdsf"}


def filter_skymodel(
    flat_noise_image,
    true_sky_image,
    input_skymodel,
    output_apparent_sky,
    output_true_sky,
    beam_ms,
    source_finder="bdsf",
    **kws,
):
    """
    Filters a sky model based on a source finder.

    This function filters a sky model using either SoFiA-2 or PyBDSF,
    based on the `source_finder` parameter.  It applies the chosen
    source finder to generate a filtered sky model.

    Parameters
    ----------
    flat_noise_image : str or Path
        Filename of input image to use to detect sources for filtering.
        It should be a flat-noise / apparent sky image (without primary-beam
        correction).
    true_sky_image : str or Path, optional
        Filename of input image to use to determine the true flux of sources.
        It should be a true flux image (with primary-beam correction).
        If beam_ms is not empty, this argument must be supplied. Otherwise,
        filter_skymodel ignores it and uses the flat_noise_image instead.
    input_skymodel : str or Path
        Filename of input makesourcedb sky model.
        If beam_ms is empty, it should be an apparent sky model, without
        primary-beam correction.
        If beam_ms is not empty, it should be a true sky model, with
        primary-beam correction.
    output_apparent_sky : str or Path
        Output file name for the generated apparent sky model.
    output_true_sky : str or Path
        Output file name for the generated true sky model.
    beam_ms : str or Path, optional
        The filename of the MS for deriving the beam attenuation.
    source_finder : str, optional
        The source finder to use, either "sofia" or "bdsf". Defaults to "bdsf".
    **kws
        Additional keyword arguments to pass to the source finder function.

    """

    runners = {"bdsf": filter_skymodel_bdsf, "sofia": filter_skymodel_sofia}
    source_finder = resolve_source_finder(source_finder)
    runner = runners[source_finder]

    runner(
        flat_noise_image,
        true_sky_image,
        input_skymodel,
        output_apparent_sky,
        output_true_sky,
        beam_ms=beam_ms,
        **kws,
    )


def resolve_source_finder(name, fallback="bdsf", emit=logger.warning):
    """
    Resolve which source finder to use.

    This function checks the given source finder name against valid options
    ("sofia" or "bdsf"). If the name is invalid, it falls back to the
    default algorithm and emits a warning message. Custom message handling is
    possible by passing a callable as the `emit` parameter.

    Parameters
    ----------
    name : str or bool or None
        The source finder name to resolve.  If True or "on", the fallback
        is used.  If None, False, "off", or "none", None is returned. If an
        invalid string is passed, this is reported by the `emit` function, and
        the fallback value is returned (if emit does not raise an exception).
    fallback : str, optional
        The default source finder algorithm to use if the given name is
        invalid. Defaults to "bdsf".
    emit : callable, optional
        The function to use for emitting messages. Defaults to
        `logger.warning`.

    Returns
    -------
    str or None
        The resolved source finder name, or None if no source finder should
        be used.

    Raises
    ------
    TypeError
        If the input `name` is not a string, boolean, or None.
    """

    if name in {None, False, "off", "none"}:
        return None

    if name in {True, "on"}:
        name = fallback

    if not isinstance(name, str):
        raise TypeError(f"Invalid source finder: {name!r}.")

    source_finder = name.lower()
    if source_finder in KNOWN_SOURCE_FINDERS:
        return source_finder

    emit = emit or logger.warning

    emit(
        f"{source_finder!r} is not a valid value for 'source_finder'. Valid "
        f"options are {KNOWN_SOURCE_FINDERS}. Falling back to the default algorithm: "
        f"{fallback!r}.",
    )
    return fallback


def filter_skymodel_bdsf(
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

    rmsbox = _bdsf_parse_rmsbox(rmsbox)
    rmsbox_bright = _bdsf_parse_rmsbox(rmsbox_bright)

    # set the TMPDIR environmental variable for temporary data storate
    with temp_storage():
        img_true_sky = _bdsf_process_images(
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
        _bdsf_filter_sources(
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
        _bdsf_create_dummy_skymodel(
            img_true_sky, output_true_sky, output_apparent_sky
        )


def _bdsf_parse_rmsbox(rmsbox):
    """Parses the rmsbox parameter."""
    if rmsbox is not None and isinstance(rmsbox, str):
        return literal_eval(rmsbox)
    return rmsbox


def _bdsf_process_images(
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


def _bdsf_filter_sources(
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
    _bdsf_trim_mask(mask_file, vertices_file)

    # Load the sky model with the associated beam MS.
    input_skymodel = load(str(input_skymodel), beamMS=str(beam_ms))

    # If bright sources were peeled before imaging, add them back
    if input_bright_skymodel:
        _bdsf_add_bright_sources(input_skymodel, input_bright_skymodel)

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


def _bdsf_trim_mask(mask_file, vertices_file):
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
    verts = _bdsf_create_polygon(WCS(hdu[0].header), vertices)

    # Rasterize the poly
    data = hdu[0].data
    data[0, 0, :, :] = rasterize(verts, data[0, 0, :, :])
    hdu.writeto(mask_file, overwrite=True)


def _bdsf_create_polygon(wcs, vertices):
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
    return list(_bdsf_generate_polygon(wcs, vertices))


def _bdsf_generate_polygon(wcs, vertices):

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


def _bdsf_add_bright_sources(input_skymodel, input_bright_skymodel):

    s_bright = load(str(input_bright_skymodel))
    # Rename the bright sources, removing the '_sector_*' added previously
    # (otherwise the '_sector_*' text will be added every iteration, eventually
    # making for very long source names)
    new_names = [
        name.split("_sector")[0] for name in s_bright.getColValues("Name")
    ]
    s_bright.setColValues("Name", new_names)
    input_skymodel.concatenate(s_bright)


def _bdsf_create_dummy_skymodel(
    img_true_sky, output_true_sky, output_apparent_sky
):
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


def filter_skymodel_sofia(
    # pylint: disable=too-many-arguments,too-many-locals
    flat_noise_image: PathLike,
    true_sky_image: PathLikeOptional,
    output_apparent_sky: PathLike,
    output_true_sky: PathLike,
    beam_ms: PathLikeOptional = None,
    output_dir: PathLikeOptional = None,
    output_prefix: str = "sofia",
    ncores: int = 0,
    **kws,
):
    """
    Filter the sources in a FITS image using SoFiA-2.

    In order to measure the source parameters (size and orientation), SoFiA
    does a spatial moment analysis of the image data. This method of obtaining
    the source parameters is typically much faster fitting gaussians to the
    image data. It is advised that users assess the quality of the source
    parameterisation when choosing SoFia for filtering sources from the sky
    model.

    Note: SoFiA-2 only supports input beam in FITS format (BDSF uses
    Measurement Set format).

    Note: This function produces the following output files:
    - SoFiA source catalogue: "<output_prefix>_cat.xml".
    - output_apparent_sky: Generated apparent sky model in CSV format, without
      primary-beam correction.
    - output_true_sky: Generated true sky model in CSV format, with
      primary-beam correction. Will be identical to the output_apparent_sky if
      no beam measurement set is provided.

    The SoFiA-2 documentation is available at:
    https://gitlab.com/SoFiA-Admin/SoFiA-2/-/wikis/documents/SoFiA-2_User_Manual.pdf

    Parameters
    ----------
    flat_noise_image : str or Path
        Filename of input image to use to detect sources for filtering.
        It should be a flat-noise / apparent sky image (without primary-beam
        correction).
    true_sky_image : str or Path, optional
        Filename of input image to use to determine the true flux of sources.
        It should be a true flux image (with primary-beam correction).
        If beam_ms is not given or is None, this argument must be supplied.
        Otherwise, filter_skymodel ignores it and uses the flat_noise_image
        instead.
    output_apparent_sky: str or Path
        Output file name for the generated apparent sky model, without
        primary-beam correction.
    output_true_sky : str or Path
        Output file name for the generated true sky model, with primary-beam
        correction.
    beam_ms : str or Path, optional
        The filename of the MS for deriving the beam attenuation and
        theoretical image noise. If empty, the generated apparent and true sky
        models will be equal.
    output_dir : str or Path, optional
        Full path to the directory to which all output files will be written.
        If unset, the parent directory of the input data cube will be used by
        default.
    output_prefix: str, optional
        File name prefix to be used as the template for all output files. For
        example, if output_prefix = my_data, then the output files will be
        named my_data_cat.xml, my_data_mom0.fits, etc. If unset, the name of
        the input data cube will be used as the file name template by default.
    ncores : int, optional
        Specify the number of cores that SoFiA-2 should use. Defaults to 0,
        which will use the value of the environment variable OMP_NUM_THREADS.
    **kws
        Additional keyword-only arguments that will be passed to SoFiA (values
        should be strings). This can be used to overwrite any of the default
        source finding parameters for the SoFiA run.

    """

    # Settings that may need to be adjusted to improve quality of
    # output models:
    # - Reduce `scfind.threshold` and use reliability filtering
    # - `output.writeMoments=true` to export the model image
    input_image = Path(true_sky_image if beam_ms else flat_noise_image)
    sofia_args = {
        "input.data": str(input_image),  # Input file
        "pipeline.threads": str(ncores),
        "scfind.enable": "true",  # Use the default S+C find algorithm
        "scfind.statistic": "gauss",  # Gaussian statistics seem to give
        # better output models
        "scfind.kernelsZ": "0",  # Required for 2D images
        "linker.radiusZ": "1",  # Required for 2D images
        "linker.minSizeZ": "1",  # Required for 2D images
        # TODO: should below be used if beam corrections already applied
        # (ie if `input_image` set to `true_sky_image`), or will this
        # apply beam corrections twice?
        "parameter.physical": "true",  # Adds beam correction to output `f_sum`
        "parameter.wcs": "true",  # Attempts to use WCS for centroid
        # positions (`x`, `y`, `z`)
        "output.overwrite": "true",  # Default to overwrite outputs
        "output.writeCatXML": "true",  # Output high precision catalogue
        "output.writeCatASCII": "false",  # Ignore inaccurate txt catalogue,
        **kws,
    }

    # Add output directory if specified
    if output_dir:
        sofia_args["output.directory"] = str(output_dir)
    else:
        output_dir = input_image.parent

    # Add output filename prefix if specified
    if output_prefix:
        sofia_args["output.filename"] = output_prefix

    # Run sofia
    # pylint: disable=c-extension-no-member # pylint cannot inspect sofia2.
    sofia2.process_image(*map("=".join, sofia_args.items()))

    # Read output source catalog
    catalog_path = Path(output_dir) / f"{output_prefix}_cat.xml"
    catalog_table = _sofia_validate_catalog(catalog_path)
    _sofia_rename_sources(catalog_table)

    # Read image header
    image_header = pyfits.getheader(input_image, 0)
    _sofia_validate_image(image_header)

    # Convert source parameters
    source_parameters = _sofia_get_source_parameters(
        image_header, catalog_table
    )

    # Create additional columns needed by BBS format.
    # Details of the expected format fo the BBS source catalog can be found at:
    # https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb
    _sofia_write_skymodel(output_true_sky, catalog_table, source_parameters)

    # Pull the true sky model file through LSMTool.
    output_true_sky = str(output_true_sky)
    skymodel = load(output_true_sky, beamMS=beam_ms)
    skymodel.write(output_true_sky, clobber=True, applyBeam=False)

    # Generate apparent sky model.
    skymodel.write(output_apparent_sky, clobber=True, applyBeam=bool(beam_ms))


def _sofia_validate_catalog(catalog_path):
    # Read output source catalog
    catalog_table = Table.read(catalog_path, format="votable")

    # Check units of required columns are as expected
    assert catalog_table["ra"].unit == "deg"
    assert catalog_table["dec"].unit == "deg"
    assert catalog_table["freq"].unit == "Hz"
    assert catalog_table["f_sum"].unit == "Jy"
    # NOTE: The fork of SoFiA-2 being used in this implementation:
    # https://gitlab.com/milhazes/SoFiA-2  converts the flux units to Jy.

    # Get gaussian orientation
    orientations = catalog_table["ell_pa"]
    # Only support double precision for now.
    if orientations.dtype.type is not np.float64:
        raise TypeError(
            f"The dtype of the 'ell_pa' column of {catalog_path} "
            "is not float64 (double)."
        )

    return catalog_table


def _sofia_rename_sources(catalog_table):
    # Remove whitespace from source names (incompatible with BBS format)
    names = catalog_table["name"]
    for i, name in enumerate(names):
        names[i] = name.replace(" ", "_")


def _sofia_validate_image(image_header):
    # Check units are as expected
    for i, dimension in enumerate(("width", "height"), 1):
        unit = image_header[f"CUNIT{i}"]
        if unit != "deg":
            raise ValueError(
                f"Unit of the image {dimension} is {unit}. Only 'deg' is "
                "supported."
            )


def _sofia_get_source_parameters(image_header, catalog_table):
    """
    Get source parameters from SoFiA-2 catalog and image header.

    Extracts source positions, axes, orientation, and fluxes
    from the SoFiA-2 catalog and image header.  Orientations are corrected to
    be with respect to the North Celestial Pole (NCP).

    Parameters
    ----------
    image_header : astropy.io.fits.Header
        The FITS image header.
    catalog_table : astropy.table.Table
        The SoFiA-2 source catalog.

    Returns
    -------
    tuple
        A tuple containing: RA strings, Dec strings, semimajor and semiminor
        axes in arcseconds, orientations wrt NCP in degrees, and fluxes in the
        image units.
    """

    orientations = get_corrected_gaussian_orientations(
        image_header, catalog_table
    )

    semimajor_arcsec, semiminor_arcsec = _sofia_get_source_fwhm(
        image_header, catalog_table, np.radians(orientations)
    )

    # Get fluxes
    fluxes = catalog_table["f_sum"]

    # Create arrays of strings that adheres to the BBS format:
    ra_strings, dec_strings = format_coordinates(
        catalog_table["ra"], catalog_table["dec"]
    )

    return {
        "Ra": ra_strings,
        "Dec": dec_strings,
        "MajorAxis": semimajor_arcsec,
        "MinorAxis": semiminor_arcsec,
        "Orientation": np.degrees(orientations),
        "I": fluxes,
    }


def _sofia_get_source_fwhm(image_header, catalog_table, orientations):
    """
    Compute the Full-Width Half-Maximum (FWHM) parameters for the detected
    sources in arcseconds.

    SoFiA-2 computes the full major and minor axes of the source ellipses
    following approach of [Banks+]_. This is equivalent to measuring twice the
    standard deviation along each axes of the Gaussian sources. The
    makesourcedb format expects the axes to be in units of the FWHM of the
    Gaussians. The conversion between the two parametrisations is handled by
    this function. In addition, converting from pixel values to arcseconds in
    the case of non-square pixels is also handled.

    Parameters
    ----------
    image_header : astropy.io.fits.Header
        The FITS image header.
    catalog_table : astropy.table.Table
        The SoFiA-2 source catalog.
    orientations : numpy.ndarray
        The corrected Gaussian orientations in radians, defined with respect
        to the NCP.


    Returns
    -------
    numpy.ndarray
        The ellipse semimajor and semiminor axes in arcseconds.

    .. [Banks+]: https://doi.org/10.1093/mnras/272.4.821
    """
    # Get gaussian major and minor axes in arcseconds
    ellipse_fwhm_pixels = (
        table_to_array(catalog_table["ell_maj", "ell_min"])
        * FWHM_PER_SIGMA
        / 2
    )
    # Get pixel sizes (width, height) in arcseconds
    pixel_size_arcsec = (
        Quantity(
            [[image_header["CDELT1"], image_header["CDELT2"]]], unit="deg"
        )
        .to("arcsec")
        .value
    )

    # Check if pixels are square
    if np.ptp(np.abs(pixel_size_arcsec)):
        return ellipse_fwhm_pixels * abs(pixel_size_arcsec[0])

    # Handle non-square pixels:
    # Compute scales. In general this will be different for each source if the
    # pixels are not square and the orientations are different for each source.
    rotation_matrix = np.moveaxis(rotation_matrix_2d(orientations), -1, 0)
    pixel_unit_vectors_wrt_sources = rotation_matrix * pixel_size_arcsec
    metric_scalars = np.sqrt(np.square(pixel_unit_vectors_wrt_sources).sum(1))
    return (metric_scalars * ellipse_fwhm_pixels).T


def get_corrected_gaussian_orientations(image_header, catalog_table):
    """
    Corrects Gaussian orientations to be absolute.

    Applies a correction to the Gaussian orientations obtained from SoFiA-2
    to ensure they are defined with respect to the North Celestial Pole (NCP),
    as required by the BBS sky model format. SoFiA-2 outputs orientations
    relative to the vertical axis of the image, which is not necessarily
    aligned with the NCP.

    Parameters
    ----------
    image_header : astropy.io.fits.Header
        The FITS image header, used to extract the image center coordinates.
    catalog_table : astropy.table.Table
        The SoFiA-2 source catalog, containing the source coordinates and
        orientations.

    Returns
    -------
    numpy.ndarray
        The corrected Gaussian orientations in radians, defined with respect
        to the NCP.

    """
    # Get right angle and declination values of centroid position. Table
    # columns are masked arrays. We want all values (valid and invalid) so
    # convert to array
    ra_source = np.array(catalog_table["ra"])
    dec_source = np.array(catalog_table["dec"])

    # Get gaussian orientation
    orientations = np.radians(catalog_table["ell_pa"])

    # Apply orientation correction function [1] required because of
    # 'relative' vs 'absolute' orientation [2]
    # https://github.com/darafferty/LSMTool/blob/master/lsmtool/correct_gaussian_orientation.py
    # https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb#orientation_of_gaussian_sources
    ra_centre, dec_centre = image_header["CRVAL1"], image_header["CRVAL2"]
    return compute_absolute_orientation(
        orientations,
        *np.radians([ra_source, dec_source]),
        *np.radians([ra_centre, dec_centre]),
    )


def _sofia_write_skymodel(output_true_sky, catalog_table, source_parameters):
    """
    Writes the source catalog to a file, ensuring that it adheres to the BBS
    format expected by downstream toolchain (wsclean, dp3).

    Constructs an astropy Table with the necessary columns for the BBS format
    and writes it to a CSV file. Then, it modifies the header of the CSV file
    to match the BBS format specifications. Details of the source catalog
    format can be found at:
    https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb

    See also: https://wsclean.readthedocs.io/en/latest/component_list.html

    Parameters
    ----------
    output_true_sky : str or Path
        Output file name for the skymodel.
    catalog_table : astropy.table.Table
        The source catalog table.
    source_parameters : tuple
        Tuple containing source parameters (RA, Dec, axes, orientation, flux).
    """

    # Get reference frequency
    reference_freq, *non_unique = set(catalog_table["freq"])
    # Check that we have only one unique frequency (this is continuum imaging)
    if non_unique:
        raise ValueError(
            "Multiple frequencies detected in a continuum image source "
            "catalogue."
        )

    # Add columns to new table
    csv_table = Table()
    num_rows = len(catalog_table)
    csv_table["Name"] = catalog_table["name"]
    csv_table["Type"] = np.full(num_rows, "GAUSSIAN")
    csv_table["Ra"] = source_parameters["Ra"]
    csv_table["Dec"] = source_parameters["Dec"]
    csv_table["I"] = source_parameters["I"]
    csv_table["SpectralIndex"] = np.full(num_rows, "[-0.8]")
    csv_table["LogarithmicSI"] = np.full(num_rows, False)
    csv_table["ReferenceFrequency"] = reference_freq
    csv_table["MajorAxis"] = source_parameters["MajorAxis"]
    csv_table["MinorAxis"] = source_parameters["MinorAxis"]
    csv_table["Orientation"] = source_parameters["Orientation"]

    with output_true_sky.open("w", encoding="utf-8") as stream:
        # Write the format specifier line for BBS makesourcedb format
        stream.write(
            "Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, "
            f"ReferenceFrequency='{reference_freq:.5e}', "
            "MajorAxis, MinorAxis, Orientation\n\n"
        )

        # Output table data (no header) to file in CSV format
        csv_table.write(
            stream,
            format="ascii.no_header",
            delimiter=",",
            formats={"I": "%.15e", "ReferenceFrequency": "%.5e"},
        )
