
import argparse
import itertools as itt
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #

HEADER_FORMAT_LINE = (
    "FORMAT = Name, Type, Ra, Dec, I, Q, U, V, "
    "MajorAxis, MinorAxis, Orientation, "
    "ReferenceFrequency, SpectralIndex='[]', RotationMeasure\n\n"
)


# ---------------------------------------------------------------------------- #
def deg_to_hms(ra_deg):
    ra_hours = ra_deg / 15.0
    h = int(ra_hours)
    m = int((ra_hours - h) * 60)
    s = (ra_hours - h - m / 60) * 3600
    return f"{h:02d}:{m:02d}:{s:08.5f}"


def deg_to_dms(dec_deg):
    sign = "-" if dec_deg < 0 else ""
    dec_deg = abs(dec_deg)
    d = int(dec_deg)
    m = int((dec_deg - d) * 60)
    s = (dec_deg - d - m / 60) * 3600
    return f"{sign}{d:02d}.{m:02d}.{s:07.4f}"


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


def _read_data(input_file):
    """Generator that yields lines from the input file, stripping whitespace."""
    with Path(input_file).open("r") as file:
        for line in file:
            yield line.strip()


def read_data(input_file):
    """
    Reads lines from the input file and group them into header lines and data
    lines. This function returns a pair of iterables for the header and data
    respectively.

    Parameters
    ----------
    input_file : Path, str
        Path to the input skymodel file.

    Returns
    -------
    header_lines: list
        A list of header lines from the input file.
    data_lines: generator
        Generator that yields data lines from the input file, one at a time.
        Each line is stripped of leading and trailing whitespace.
    """
    # Create line generator
    lines = _read_data(input_file)
    # Split header lines
    grouped = itt.groupby(lines, key=is_header_line)
    # Consume the iterable of header lines so the file pointer moves to the
    # start of the data lines
    _, header_lines = next(grouped)  # First group is header lines
    header_lines = list(header_lines)
    # Get the data lines as a generator
    _, data_lines = next(grouped)  # Second group is data lines
    return header_lines, data_lines


def filter_sources(
    data_lines,
    point_size_threshold,
    min_flux_point,
    min_flux_extended,
    source_counts=None,
):
    """
    Generator that filter sources from the skymodel and converts the remaining
    source data to lists of floats.

    This function loops through the input data lines (sources) from the input
    skymodel, catagorises the source as either a point source or an extended
    source based on its size, and filters the lines that have sources that are
    below the flux threshold for their category. Finally, it converts the
    data to a list of floats and yields them one at a time.

    Parameters
    ----------
    data_lines : iterable
        Iterable of lines from the input skymodel file, where each line
        corresponds to a source.
    point_size_threshold : float
        Threshold for categorising a source as a point source or extended
        source.
    min_flux_point : float
        Minimum flux density in Jy for a point source to be included.
    min_flux_extended : float
        Minimum flux density in Jy for an extended source to be included.
    source_counts : dict, optional
        Dict of integers for keeping track of the number of removed sources. If
        provided, it should have the following form:
        {"n_input_sources": 0, "n_point_removed": 0, "n_extended_removed": 0}.
        These counts will be updated in place as the sources are processed. You
        do not need to specify this if you don't care about knowing the number
        of sources that are read or removed.

    Yields
    -------
    list of float
        Row data for filtered output skymodel.
    """
    if source_counts is None:
        # If no source_counts provided from outside, create a local
        # source_counts dict. These counts will not be available outside the
        # function, but we define it nonetheless to avoid having unnecessary
        # conditionals in the loop below.
        source_counts = defaultdict(int)

    for line in data_lines:
        source_counts["n_input_sources"] += 1
        # ra, dec, I, Q, U, V, ref_freq, spix, rm, major, minor, orient
        cells = line.split(",")
        total_intensity = float(cells[2])
        major = float(cells[9])

        # Apply filtering based on flux and size
        is_point_source = abs(major) < point_size_threshold
        threshold = min_flux_point if is_point_source else min_flux_extended
        if total_intensity < threshold:
            category = "point" if is_point_source else "extended"
            source_counts[f"n_{category}_removed"] += 1
            continue

        # reformat
        yield convert_row(line)


def convert_row(line):
    """
    Convert a row of data from the input OSKAR skymodel to makesourcedb format.

    Parameters
    ----------
    line : str
        Comma separated line of data values. Data values should be convertable
        to float.

    Returns
    -------
    list of float
        Row of data values
    """
    return [float(x.strip()) for x in line.split(",")]


def format_row(source_name, data):
    """
    Format data values for the output makesourcedb skymodel.

    Parameters
    ----------
    source_name : str
        Source name.
    data : list of float
        Row of data values. The order of the data values should be:
        RA, Dec, I, Q, U, V, ReferenceFrequency, SpectralIndex, RotationMeasure,
        MajorAxis, MinorAxis, Orientation

    Returns
    -------
    str
        Formatted data line for output makesourcedb skymodel.
    """
    (ra, dec, i, q, u, v, ref_freq, spix, rm, major, minor, orient) = data
    ra_str = deg_to_hms(ra)
    dec_str = deg_to_dms(dec)
    return (
        f"{source_name}, GAUSSIAN, {ra_str}, {dec_str}, "
        f"{i}, {q}, {u}, {v}, "
        f"{major}, {minor}, {orient}, "
        f"{ref_freq}, [{spix}], {rm}\n"
    )


def convert_skymodel(
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
        they are considered extended sources. Default: 1e-8 arcseconds
        (effectively treating all sources as extended sources).
    min_flux_point : float, optional
        Minimum flux density in Jy for point sources. Default: 0.005
    min_flux_extended : float, optional
        Minimum flux density in Jy for extended sources. Default: 0.02
    """

    # Read
    header_lines, data_lines = read_data(input_file)

    # Filter
    source_counts = {
        "n_input_sources": 0,
        "n_extended_removed": 0,
        "n_point_removed": 0,
    }
    filtered_data = filter_sources(
        data_lines,
        point_size_threshold,
        min_flux_point,
        min_flux_extended,
        source_counts,
    )

    # Write
    output_source_count = write(output_file, header_lines, filtered_data)

    # Summary
    logger.info(
        "Conversion complete.\n"
        "Total input sources: %i\n"
        "Total output sources: %i\n"
        "Point sources removed: %i\n"
        "Extended sources removed: %i",
        source_counts["n_input_sources"],
        output_source_count,
        source_counts["n_point_removed"],
        source_counts["n_extended_removed"],
    )


def write(output_file, header_lines, filtered_data):
    """
    Write the output makesourcedb skymodle file from header lines and data
    values.

    Parameters
    ----------
    output_file : str or Path
        Path to the output makesourcedb skymodel file.
    header_lines : iterable of str
        Header lines from the input OSKAR skymodel file.
    filtered_data : iterable of list of float
        Filtered and converted data values for each source.

    Returns
    -------
    int
        Number of sources written to the output file.
    """

    with open(output_file, "w") as file:
        # Write placeholder for "# Number of sources:" line, which will be
        # updated later with the actual number of sources after writing the
        # data lines. The placeholder should be long enough to accomodate the
        # actual number of sources.
        n_sources_line = "# Number of sources: {: <7}\n"
        file.write(n_sources_line.format(0))

        # Write header lines
        for line in header_lines:
            if line.startswith("# Number of sources:"):
                continue  # Skip original number of sources line
            file.write(f"{line}\n")

        # Write required FORMAT line
        file.write(HEADER_FORMAT_LINE)

        # Write filtered + converted rows
        for i, data in enumerate(filtered_data):
            file.write(format_row(f"src{i}", data))

        # Move back to the start of the file to update the source count
        n_sources = i + 1
        file.seek(0)
        file.write(n_sources_line.format(n_sources))

    return n_sources


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
        "Default: 1e-8 arcseconds (effectively treating all sources as extended"
        " sources)",
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

    convert_skymodel(
        args.input_file,
        args.output_file,
        args.point_size_threshold,
        args.min_flux_point,
        args.min_flux_extended,
    )
