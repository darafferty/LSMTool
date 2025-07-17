import numbers
from pathlib import Path
from typing import Sequence

import numpy as np
import sofia2
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity

from ..correct_gaussian_orientation import compute_absolute_orientation
from ..io import PathLike, PathLikeOptional, PathLikeOrListOptional, load
from ..utils import format_coordinates, rotation_matrix_2d, table_to_array

# conversion factor between sofia and makeshourcedb parameterisations
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def filter_skymodel(
    # pylint: disable=too-many-arguments,too-many-locals
    flat_noise_image: PathLike,
    true_sky_image: PathLikeOptional,
    output_apparent_sky: PathLike,
    output_true_sky: PathLike,
    beam_ms: PathLikeOrListOptional = None,
    output_dir: PathLikeOptional = None,
    output_prefix: str = "sofia",
    ncores: numbers.Integral = 0,
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
    catalog_table = validate_catalog(catalog_path)
    rename_sources(catalog_table)

    # Read image header
    image_header = fits.getheader(input_image, 0)
    validate_image(image_header)

    # Convert source parameters
    source_parameters = get_source_parameters(image_header, catalog_table)

    # Create additional columns needed by BBS format.
    # Details of the expected format fo the BBS source catalog can be found at:
    # https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb
    write_skymodel(output_true_sky, catalog_table, source_parameters)

    # Pull the true sky model file through LSMTool.
    output_true_sky = str(output_true_sky)
    skymodel = load(output_true_sky, beamMS=beam_ms)
    skymodel.write(output_true_sky, clobber=True, applyBeam=False)

    # Generate apparent sky model.
    skymodel.write(output_apparent_sky, clobber=True, applyBeam=bool(beam_ms))


def validate_catalog(catalog_path: PathLike) -> Table:
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


def rename_sources(catalog_table: Table):
    # Remove whitespace from source names (incompatible with BBS format)
    names = catalog_table["name"]
    for i, name in enumerate(names):
        names[i] = name.replace(" ", "_")


def validate_image(image_header: fits.header.Header):
    # Check units are as expected
    for i, dimension in enumerate(("width", "height"), 1):
        unit = image_header[f"CUNIT{i}"]
        if unit != "deg":
            raise ValueError(
                f"Unit of the image {dimension} is {unit}. Only 'deg' is "
                "supported."
            )


def get_source_parameters(
    image_header: fits.header.Header, catalog_table: Table
) -> dict:
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

    semimajor_arcsec, semiminor_arcsec = get_source_fwhm(
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


def get_source_fwhm(
    image_header: fits.header.Header,
    catalog_table: Table,
    orientations: Sequence,
) -> np.ndarray:
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


def get_corrected_gaussian_orientations(
    image_header: fits.header.Header, catalog_table: Table
) -> np.ndarray:
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


def write_skymodel(
    output_true_sky: PathLike, catalog_table: Table, source_parameters: tuple
):
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
