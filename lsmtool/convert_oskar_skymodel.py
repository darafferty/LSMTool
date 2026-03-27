"""
Script for converting an OSKAR CSV skymodel files to makesourcedb format, with
optional filtering of sources based on flux density and size.

Context: OSKAR is a tool for simulating radio interferometer observations, and
supports skymodels in CSV format. Since the LOFAR toolchain expects skymodels
in the makesourcedb format, this script can be used to do the conversion.

The OSKAR skymodel format is documented in `<the OSKAR user manual>
https://ska-telescope.gitlab.io/sim/oskar/python/sky.html`_.
Info about the makesourcedb format can be found on
`<the LOFAR wiki>
https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb`_
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from lsmtool.utils import format_coordinates

# ---------------------------------------------------------------------------- #
# Init logger
logger = logging.getLogger(__name__)

# Module constants
HEADER_FORMAT_LINE = (
    "FORMAT = Name, Type, Ra, Dec, I, Q, U, V, "
    "MajorAxis, MinorAxis, Orientation, "
    "ReferenceFrequency, SpectralIndex='[]', RotationMeasure\n\n"
)
SOURCE_NAME_PREFIX = "src"
OUTPUT_FORMAT = {
    "Name": "%s",
    "Type": "%s",
    "RA": "%s",
    "Dec": "%s",
    "I": "%.9f",
    "Q": "%.1f",
    "U": "%.1f",
    "V": "%.1f",
    "MajorAxis": "%f",
    "MinorAxis": "%f",
    "Orientation": "%f",
    "ReferenceFrequency": "%.1f",
    "SpectralIndex": "[%.1f]",
    "RotationMeasure": "%.1f",
}
OUTPUT_ORDER = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]


# ---------------------------------------------------------------------------- #


def is_header_line(line):
    """
    Check if a input line is a comment header.

    Parameters
    ----------
    line : str
        Input line from the skymodel file.

    Returns
    -------
    bool
        True if the line is a header line, False otherwise.
    """
    return line.startswith("#") or line.isspace()


def get_header(filename):
    """
    Get the header lines from a skymodel file.

    Parameters
    ----------
    filename : str
        Path to the skymodel file.

    Yields
    ------
    str
        Header line from the skymodel file.
    """
    with Path(filename).open() as f:
        for line in f:
            if is_header_line(line):
                yield line.strip()
            else:
                break


def read_data(input_file):
    """
    Load the OSKAR skymodel using numpy's genfromtxt function, which can handle
    large files efficiently. Also get the header lines as a list.

    Parameters
    ----------
    input_file : Path, str
        Path to the input skymodel file.

    Returns
    -------
    header: list
        A list of header lines from the input file.
    data: np.ndarray
        A numpy structured array containing the data from the input file. The
        columns of the array can be accessed by their descriptions, which are
        taken from the second line of the input file.
    """
    data = np.genfromtxt(
        input_file,
        delimiter=",",
        names=True,
        encoding="utf-8",
        skip_header=1,
    )
    header = list(get_header(input_file))
    return header, data


def filter_sources(
    data,
    point_size_threshold,
    min_flux_point,
    min_flux_extended,
):
    """
    Filter sources from the skymodel based on flux density and size.

    Sources are catagorised as either point sources or extended sources based
    on their size. Data rows containing sources that are below the flux
    threshold for their category are removed.

    Parameters
    ----------
    data : np.ndarray
        Numpy structured array containing the data from the input skymodel
        file.
    point_size_threshold : float
        Threshold for categorising a source as a point source or extended
        source.
    min_flux_point : float
        Minimum flux density in Jy for a point source to be included.
    min_flux_extended : float
        Minimum flux density in Jy for an extended source to be included.

    Returns
    -------
    filtered_data : np.ndarray
        Data for the output skymodel containing only sources above the given
        flux threshold.
    is_point_source : np.ndarray
        Boolean array indicating which sources in the filtered data are point
        sources and which are extended sources.
    n_point_removed : int
        Number of point sources that were removed based on the flux threshold.
    n_extended_removed : int
        Number of extended sources that were removed based on the flux
        threshold.
    """

    # Determine flux cutoff based on source size
    major = data["FWHM_major_arcsec"]
    is_point_source = major <= point_size_threshold

    # Filter sources based on flux density and size criteria
    threshold = np.array([min_flux_extended, min_flux_point])[
        is_point_source.astype(int)
    ]
    keep = data["I_Jy"] > threshold
    remove = ~keep

    removed_point_sources = is_point_source[remove]
    n_point_removed = removed_point_sources.sum()
    n_extended_removed = (~removed_point_sources).sum()

    return (
        data[keep],
        is_point_source[keep],
        n_point_removed,
        n_extended_removed,
    )


def convert(header, data, point_sources):
    """
    Convert data and header to makesourcedb format.

    Parameters
    ----------
    header : list[str]
        Header lines from the input skymodel.
    data : np.ndarray
        Input data from OSKAR skymodel.
    point_sources : np.ndarray
        Boolean array indicating which sources are point sources.

    Yields
    -------
    str
        Header lines for output skymodel.
    np.ndarray
        Data values for output skymodel.
    """

    # Get name and type columns
    n_output_sources = len(data)
    names = np.char.add(
        SOURCE_NAME_PREFIX, np.arange(n_output_sources).astype(str)
    )
    source_type = np.array(["GAUSSIAN", "POINT"])[point_sources.astype(int)]

    # Get header info
    column_descriptions = header[1].strip("# \n").split(", ")
    column_descriptions = np.take(column_descriptions, OUTPUT_ORDER)

    # header
    yield (
        f"# Number of sources: {n_output_sources}",
        f"# {', '.join(column_descriptions)}",
        HEADER_FORMAT_LINE,
    )

    # Get the output column order
    column_names = np.take(list(data.dtype.fields.keys()), OUTPUT_ORDER[2:])
    ra, dec = format_coordinates(
        data["RA_deg"], data["Dec_deg"], precision=5, pad=True
    )

    # data
    yield np.array(
        [names, source_type, ra, dec, *(data[_] for _ in column_names)], object
    ).T


def convert_oskar_skymodel(
    input_file,
    output_file,
    point_size_threshold=1e-8,
    min_flux_point=0.005,
    min_flux_extended=0.02,
):
    """
    Convert an OSKAR skymodel to makesourcedb format, optionally filtering
    sources based on flux density and size.

    Parameters
    ----------
    input_file : str or Path
        Path to input OSKAR skymodel file.
    output_file : str or Path
        Path to output skymodel file.
    point_size_threshold : float, optional
        Threshold in arcseconds for the size of the major axis of gaussian
        sources below which they are considered point sources, and above which
        they are considered extended sources. Default: 1e-8 arcseconds.
    min_flux_point : float, optional
        Minimum flux density in Jy for point sources. Default: 0.005
    min_flux_extended : float, optional
        Minimum flux density in Jy for extended sources. Default: 0.02
    """

    # Read
    header, data = read_data(input_file)
    n_input_sources = len(data)

    # Filter
    data, is_point_source, n_point_removed, n_extended_removed = filter_sources(
        data,
        point_size_threshold,
        min_flux_point,
        min_flux_extended,
    )
    n_output_sources = len(data)

    # Convert
    header, data = convert(header, data, is_point_source)

    # Write
    write(output_file, header, data)

    # Summary
    logger.info(
        "Conversion complete.\n"
        "Total input sources: %i\n"
        "Total output sources: %i\n"
        "Point sources removed: %i\n"
        "Extended sources removed: %i",
        n_input_sources,
        n_output_sources,
        n_point_removed,
        n_extended_removed,
    )


def write(
    output_file,
    header,
    data,
    **kws,
):
    """
    Write the output makesourcedb skymodel file from header lines and data
    array.

    Parameters
    ----------
    output_file : str or Path
        Path to the output makesourcedb skymodel file.
    header : list of str
        Header lines to be written to the top of the output skymodel file.
    data : np.ndarray
        Source data to write to file.
    **kws
        Additional keyword arguments to pass to `numpy.savetxt`.
    """

    with Path(output_file).open("w") as file:
        # write header
        file.write("\n".join(header))

        # write data
        kws = {"fmt": tuple(OUTPUT_FORMAT.values()), "delimiter": ", ", **kws}
        np.savetxt(file, data, **kws)


def main():
    """
    Main function for converting OSKAR skymodel to makesourcedb format.

    Parse the command line arguments, and call the convert_skymodel function
    to do the conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert OSKAR skymodel to makesourcedb format. Optionally "
        "filter sources based on flux density and size."
    )
    parser.add_argument("input_file", help="Path to input OSKAR skymodel file")
    parser.add_argument(
        "output_file", help="Path to output makesourcedb skymodel file"
    )
    parser.add_argument(
        "--point-size-threshold",
        type=float,
        default=1e-8,
        help="Threshold in arcseconds for the size of the major axis of "
        "gaussian sources below which they are considered point sources, and "
        "above which they are considered extended sources. "
        "Default: 1e-8 arcseconds",
    )
    parser.add_argument(
        "--min-flux-point",
        type=float,
        default=0.005,
        help="Minimum total intensity (Stokes I) flux density in Jy for point "
        "sources. Point sources with flux density below this threshold will be "
        "removed. Default: 0.005 Jy",
    )
    parser.add_argument(
        "--min-flux-extended",
        type=float,
        default=0.02,
        help="Minimum total intensity (Stokes I) flux density in Jy for "
        "extended sources. Extended sources with flux density below this "
        "threshold will be removed. Default: 0.02 Jy",
    )

    args = parser.parse_args()

    convert_oskar_skymodel(
        args.input_file,
        args.output_file,
        args.point_size_threshold,
        args.min_flux_point,
        args.min_flux_extended,
    )
