import shlex
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from conftest import TEST_DATA_PATH

from lsmtool.convert_oskar_skymodel import (
    MAKESOURCEDB_FORMAT_STRING,
    OSKAR_DEFAULT_COLUMN_NAMES,
    convert_oskar_skymodel,
    convert_to_makesourcedb,
    filter_sources,
    get_column_names,
    is_header_line,
    main,
    read_oskar_skymodel,
    write_to_makesourcedb,
)

# ---------------------------------------------------------------------------- #
# Module constants

TEST_RESOURCES_PATH = TEST_DATA_PATH
TEST_DATA_PATH = TEST_RESOURCES_PATH / Path(__file__).stem

OSKAR_NUMPY_DTYPE = np.dtype(
    [
        ("RA", "<f8"),
        ("Dec", "<f8"),
        ("I", "<f8"),
        ("Q", "<f8"),
        ("U", "<f8"),
        ("V", "<f8"),
        ("ReferenceFrequency", "<f8"),
        ("SpectralIndex", "<f8"),
        ("RotationMeasure", "<f8"),
        ("MajorAxis", "<f8"),
        ("MinorAxis", "<f8"),
        ("Orientation", "<f8"),
    ]
)

# ---------------------------------------------------------------------------- #
# Helper functions


def generate_oskar_skymodel_data(n_sources, rng):
    """
    Generate a random sample of sources for testing skymodel conversion.

    Data are generated from uniform distributions within reasonable ranges for
    each parameter, with the exception of Q, U, V, and rotation measure which
    are set to zero for all sources, and reference frequency which is set to
    144 MHz for all sources.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    n_sources : int
        Number of sources to generate.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_sources, 12) containing the generated source data in
        the order: RA, Dec, I, Q, U, V, Ref. freq., Spectral index, Rotation
        measure, FWHM major, FWHM minor, Position angle.
    """

    ra = rng.uniform(0, 360, n_sources)
    dec = rng.uniform(-90, 90, n_sources)
    i = rng.uniform(0.001, 20, n_sources)
    q = u = v = np.zeros(n_sources)
    ref_freq = np.full(n_sources, 1.44e8)
    spectral_index = rng.uniform(-1, 0, n_sources)
    rotation_measure = np.zeros(n_sources)
    fwhm_major = rng.uniform(0.01, 20, n_sources)
    fwhm_minor = rng.uniform(0, 1, n_sources) * fwhm_major
    position_angle = rng.uniform(0, 180, n_sources)
    return np.column_stack(
        (
            ra,
            dec,
            i,
            q,
            u,
            v,
            ref_freq,
            spectral_index,
            rotation_measure,
            fwhm_major,
            fwhm_minor,
            position_angle,
        )
    ).view(OSKAR_NUMPY_DTYPE)


def random_skymodel(n_sources, rng):
    """Generate a random skymodel dataset and header for testing."""
    return (
        [
            f"# Number of sources: {n_sources}",
            "# RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), "
            "Ref. freq. (Hz), Spectral index, Rotation measure (rad/m^2), "
            "FWHM major (arcsec), FWHM minor (arcsec), Position angle (deg)",
            MAKESOURCEDB_FORMAT_STRING,
        ],
        generate_oskar_skymodel_data(n_sources, rng),
    )


# ---------------------------------------------------------------------------- #
# Fixtures


@pytest.fixture(scope="session")
def rng():
    """Random number generator fixture for reproducibility."""
    return np.random.default_rng(seed=881726)


@pytest.fixture()
def sample_csv_text():
    """Sample OSKAR CSV data as header text for testing."""
    return {
        "header": [
            "# Number of sources: 5",
            "# RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), "
            "Ref. freq. (Hz), Spectral index, Rotation measure (rad/m^2), "
            "FWHM major (arcsec), FWHM minor (arcsec), Position angle (deg)",
        ],
        "data": (
            "134.316584681925, -34.806858824585, 7.12299476324136e-01, "
            "0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, "
            "1.44000000000000e+08,-7.00000000000000e-01, 0.00000000000000e+00, "
            "2.552896197845,   1.705990053656, 115.020339486069\n"
            "123.357884389652, -36.922300562525, 5.05840198469155e-01, "
            "0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, "
            "1.44000000000000e+08,-7.00000000000000e-01, 0.00000000000000e+00, "
            "3.381746257518,   2.676308575899,  37.727775657383\n"
            "135.273082634371, -29.029007646589, 5.04116042438541e-01, "
            "0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, "
            "1.44000000000000e+08,-7.00000000000000e-01, 0.00000000000000e+00, "
            "14.702668401253,   1.319160550191,  28.175171030289\n"
            "128.770281841595, -26.176730148372, 2.58824974285288e-01, "
            "0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, "
            "1.44000000000000e+08,-7.00000000000000e-01, 0.00000000000000e+00, "
            "12.861909842975,   2.246703463196,  74.249587537251\n"
            "125.806642113852, -36.775065379826, 5.243444339899304, "
            "0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, "
            "1.44000000000000e+08,-7.00000000000000e-01, 0.00000000000000e+00, "
            "0.0, 0.0, 0.0"
        ),
    }


@pytest.fixture()
def sample_csv_path(tmp_path, sample_csv_text):
    """Sample CSV file path containing sample csv text header and data."""
    path = tmp_path / "sample.csv"
    path.write_text(
        "\n".join((*sample_csv_text["header"], sample_csv_text["data"]))
    )
    return path


@pytest.fixture()
def expected_sample_data():
    """
    Expected data for the sample CSV file after conversion to list of floats.
    """
    return (
        np.array(
            [
                (
                    134.316584681925,
                    -34.806858824585,
                    0.712299476324136,
                    0.0,
                    0.0,
                    0.0,
                    1.44e8,
                    -0.7,
                    0.0,
                    2.552896197845,
                    1.705990053656,
                    115.020339486069,
                ),
                (
                    123.357884389652,
                    -36.922300562525,
                    0.505840198469155,
                    0.0,
                    0.0,
                    0.0,
                    1.44e8,
                    -0.7,
                    0.0,
                    3.381746257518,
                    2.676308575899,
                    37.727775657383,
                ),
                (
                    135.273082634371,
                    -29.029007646589,
                    0.504116042438541,
                    0.0,
                    0.0,
                    0.0,
                    1.44e8,
                    -0.7,
                    0.0,
                    14.702668401253,
                    1.319160550191,
                    28.175171030289,
                ),
                (
                    128.770281841595,
                    -26.176730148372,
                    0.258824974285288,
                    0.0,
                    0.0,
                    0.0,
                    1.44e8,
                    -0.7,
                    0.0,
                    12.861909842975,
                    2.246703463196,
                    74.249587537251,
                ),
                (
                    125.806642113852,
                    -36.775065379826,
                    5.243444339899304,
                    0.0,
                    0.0,
                    0.0,
                    1.44e8,
                    -0.7,
                    0.0,
                    0,
                    0,
                    0,
                ),
            ],
        )
        .view(OSKAR_NUMPY_DTYPE)
        .squeeze()
    )


@pytest.fixture()
def expected_converted_data():
    return np.array(
        [
            [
                "src0",
                "GAUSSIAN",
                "08:57:15.98032",
                "-34.48.24.69177",
                0.712299476324136,
                0.0,
                0.0,
                0.0,
                2.552896197845,
                1.705990053656,
                115.020339486069,
                144000000.0,
                -0.7,
                0.0,
            ],
            [
                "src1",
                "GAUSSIAN",
                "08:13:25.89225",
                "-36.55.20.28203",
                0.505840198469155,
                0.0,
                0.0,
                0.0,
                3.381746257518,
                2.676308575899,
                37.727775657383,
                144000000.0,
                -0.7,
                0.0,
            ],
            [
                "src2",
                "GAUSSIAN",
                "09:01:05.53983",
                "-29.01.44.42753",
                0.504116042438541,
                0.0,
                0.0,
                0.0,
                14.702668401253,
                1.319160550191,
                28.175171030289,
                144000000.0,
                -0.7,
                0.0,
            ],
            [
                "src3",
                "GAUSSIAN",
                "08:35:04.86764",
                "-26.10.36.22853",
                0.258824974285288,
                0.0,
                0.0,
                0.0,
                12.861909842975,
                2.246703463196,
                74.249587537251,
                144000000.0,
                -0.7,
                0.0,
            ],
            [
                "src4",
                "POINT",
                "08:23:13.59411",
                "-36.46.30.23537",
                5.243444339899304,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                144000000.0,
                -0.7,
                0.0,
            ],
        ],
        dtype=object,
    )


@pytest.fixture()
def expected_output_header():
    """
    Expected header lines in the converted makesourcedb skymodel for the sample
    CSV data.
    """
    return (
        "# Number of sources: 5",
        "# RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), "
        "FWHM major (arcsec), FWHM minor (arcsec), Position angle (deg), "
        "Ref. freq. (Hz), Spectral index, Rotation measure (rad/m^2)",
        MAKESOURCEDB_FORMAT_STRING,
    )


@pytest.fixture()
def expected_output_lines():
    """
    Expected lines in the converted makesourcedb skymodel for the sample CSV.
    data.
    """
    return [
        "src0, GAUSSIAN, 08:57:15.98032, -34.48.24.69177, 0.712299476,"
        " 0.0, 0.0, 0.0, 2.552896, 1.705990, 115.020339, "
        "144000000.0, [-0.7], 0.0\n",
        "src1, GAUSSIAN, 08:13:25.89225, -36.55.20.28203, 0.505840198,"
        " 0.0, 0.0, 0.0, 3.381746, 2.676309, 37.727776, "
        "144000000.0, [-0.7], 0.0\n",
        "src2, GAUSSIAN, 09:01:05.53983, -29.01.44.42753, 0.504116042,"
        " 0.0, 0.0, 0.0, 14.702668, 1.319161, 28.175171, "
        "144000000.0, [-0.7], 0.0\n",
        "src3, GAUSSIAN, 08:35:04.86764, -26.10.36.22853, 0.258824974,"
        " 0.0, 0.0, 0.0, 12.861910, 2.246703, 74.249588, "
        "144000000.0, [-0.7], 0.0\n",
        "src4, POINT, 08:23:13.59411, -36.46.30.23537, 5.243444340,"
        " 0.0, 0.0, 0.0, 0.000000, 0.000000, 0.000000, "
        "144000000.0, [-0.7], 0.0\n",
    ]


# ---------------------------------------------------------------------------- #
# Tests


@pytest.mark.parametrize(
    "line, expected",
    [
        ("# This is a comment line", True),
        ("   # This is a comment line with leading whitespace", True),
        ("Format = RA, Dec, I", True),
        ("   Format= (Ra, Dec, I, Q, U, V)    ", True),
        ("#\t\t(RA,Dec,I,Q,U)\t\t=\tformat\t", True),
        ("\n", True),
        ("   \n", True),
        (
            "# (Name, Type, Ra, Dec, I, ReferenceFrequency='100e6', "
            "SpectralIndex, Q, U, MajorAxis, MinorAxis, Orientation, V)"
            " = format",
            True,
        ),
        ("RA,Dec,I,Q,U,V", False),
        ("src0, GAUSSIAN, 08:57:15.98032, -34.48.24.69177,", False),
    ],
)
def test_is_header_line(line, expected):
    """Test identification of header lines."""
    assert is_header_line(line) == expected


@pytest.mark.parametrize(
    "header_lines, expected_column_names",
    [
        (["Format = RA, Dec, I"], ["RA", "Dec", "I"]),
        (["# format = RA Dec I"], ["RA", "Dec", "I"]),
        (["Format= (Ra, Dec, I, Q, U, V)"], ["Ra", "Dec", "I", "Q", "U", "V"]),
        (["# (RA,Dec,I,Q,U) = format"], ["RA", "Dec", "I", "Q", "U"]),
        (["                   Format=RA, Dec, I"], ["RA", "Dec", "I"]),
        (["     #format=RA Dec I"], ["RA", "Dec", "I"]),
        (
            ["        Format	=	(Ra, Dec, I, Q, U, V)    "],
            ["Ra", "Dec", "I", "Q", "U", "V"],
        ),
        (["#\t\t(RA,Dec,I,Q,U)\t\t=\tformat\t"], ["RA", "Dec", "I", "Q", "U"]),
        (
            [
                "# RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), Ref. "
                "freq. (Hz), Spectral index, Rotation measure (rad/m^2), FWHM "
                "major (arcsec), FWHM minor (arcsec), Position angle (deg)"
            ],
            OSKAR_DEFAULT_COLUMN_NAMES,
        ),
        (
            [
                "# Number of sources: 3",
                "# (Name, Type, Ra, Dec, I, ReferenceFrequency='100e6', "
                "SpectralIndex, Q, U, MajorAxis, MinorAxis, Orientation, V) = "
                "format",
            ],
            [
                "Name",
                "Type",
                "Ra",
                "Dec",
                "I",
                "ReferenceFrequency",
                "SpectralIndex",
                "Q",
                "U",
                "MajorAxis",
                "MinorAxis",
                "Orientation",
                "V",
            ],
        ),
    ],
)
def test_get_column_names(header_lines, expected_column_names):
    """Test parsing the column names from the OSKAR file header."""
    assert get_column_names(header_lines) == expected_column_names


def test_read_oskar_skymodel(
    sample_csv_path, sample_csv_text, expected_sample_data
):
    """Test reading header and data lines from input file."""
    header, data = read_oskar_skymodel(sample_csv_path)
    assert header == sample_csv_text["header"]
    assert np.all(data == expected_sample_data)


@pytest.mark.parametrize(
    "input_file, expected_data",
    [
        (
            TEST_DATA_PATH / "oskar_example_basic.csv",
            np.ma.masked_array(
                data=[
                    (
                        "s1",
                        "POINT",
                        20.0,
                        -30.0,
                        1,
                        0,
                        [-0.7],
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ),
                    (
                        "s2",
                        "GAUSSIAN",
                        20.0,
                        -30.5,
                        3,
                        0,
                        [-0.7],
                        2,
                        2,
                        600,
                        50,
                        45,
                        0,
                    ),
                    (
                        "s3",
                        "GAUSSIAN",
                        20.5,
                        -30.5,
                        3,
                        0,
                        [-0.7],
                        0,
                        0,
                        700,
                        10,
                        -10,
                        2,
                    ),
                ],
                mask=[
                    (
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        [False],
                        False,
                        False,
                        True,
                        True,
                        True,
                        False,
                    ),
                    (
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        [False],
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ),
                    (
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        [False],
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ),
                ],
                dtype=[
                    ("Name", "<U100"),
                    ("Type", "<U100"),
                    ("Ra", "<f8"),
                    ("Dec", "<f8"),
                    ("I", "<i8"),
                    ("ReferenceFrequency", "<i8"),
                    ("SpectralIndex", "<f8", (1,)),
                    ("Q", "<i8"),
                    ("U", "<i8"),
                    ("MajorAxis", "<i8"),
                    ("MinorAxis", "<i8"),
                    ("Orientation", "<i8"),
                    ("V", "<i8"),
                ],
            ),
        ),
        (
            TEST_DATA_PATH / "oskar_example_missing_data.txt",
            np.array(
                [
                    [
                        (
                            0.0,
                            70.0,
                            1.1,
                            0.0,
                            0.0,
                            0.0,
                            1.0e08,
                            -0.7,
                            0.0,
                            200.0,
                            150.0,
                            23.0,
                        )
                    ],
                    [
                        (
                            0.0,
                            71.2,
                            2.3,
                            1.0,
                            0.0,
                            0.0,
                            1.0e08,
                            -0.7,
                            0.0,
                            90.0,
                            40.0,
                            -10.0,
                        )
                    ],
                    [
                        (
                            0.1,
                            69.8,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        )
                    ],
                ],
                dtype=[
                    ("RA", "<f8"),
                    ("Dec", "<f8"),
                    ("I", "<f8"),
                    ("Q", "<f8"),
                    ("U", "<f8"),
                    ("V", "<f8"),
                    ("ReferenceFrequency", "<f8"),
                    ("SpectralIndex", "<f8"),
                    ("RotationMeasure", "<f8"),
                    ("MajorAxis", "<f8"),
                    ("MinorAxis", "<f8"),
                    ("Orientation", "<f8"),
                ],
            ),
        ),
    ],
)
def test_read_oskar_skymodel_example(input_file, expected_data):
    """
    Test that `read_oskar_skymodel` can read a makesourcedb-formatted file.
    """
    _, data = read_oskar_skymodel(input_file)
    assert (data == expected_data).all()


@pytest.mark.parametrize(
    "point_size_threshold, min_flux_point, min_flux_extended, "
    "expected_counts, expected_indices_remain",
    [
        # Treat all sources as extended, test that none are filtered for small
        # `min_flux_point`
        (0, 0.005, 0.02, [0, 0], [0, 1, 2, 3, 4]),
        # Treat all sources as extended, test that all source are filtered for
        # large `min_flux_extended`
        (-1, 0, 10, [0, 5], []),
        # Treat all sources as point sources, test that none are filtered for
        # small `min_flux_point`
        (15, 0, 0, [0, 0], [0, 1, 2, 3, 4]),
        # Treat all sources as point sources, test that all are filtered for
        # large `min_flux_point`
        (15, 10, 0, [5, 0], []),
        # Test that only sources above size threshold are filtered
        (10, 0, 10, [0, 2], [0, 1, 4]),
        # Test that only sources below size threshold are filtered
        (10, 10, 0, [3, 0], [2, 3]),
    ],
)
def test_filter_sources(
    expected_sample_data,
    point_size_threshold,
    min_flux_point,
    min_flux_extended,
    expected_counts,
    expected_indices_remain,
    caplog,
):
    """Test source filtering"""
    result, point_sources = filter_sources(
        expected_sample_data,
        point_size_threshold,
        min_flux_point,
        min_flux_extended,
    )

    # Check counter have correct values
    caplog_text = caplog.text
    assert f"Removing {expected_counts[0]} point sources" in caplog_text
    assert f"Removing {expected_counts[1]} extended sources" in caplog_text

    # Check that the expected lines remain in the output
    assert np.all(result == expected_sample_data[expected_indices_remain])


@pytest.mark.parametrize(
    "expected_point_sources", [np.array([False, False, False, False, True])]
)
def test_convert_to_makesourcedb(
    sample_csv_text,
    expected_sample_data,
    expected_point_sources,
    expected_converted_data,
):
    """Test full conversion of data from input format to output format."""
    header, data = convert_to_makesourcedb(
        sample_csv_text["header"],
        expected_sample_data,
        expected_point_sources,
    )

    # Check header lines
    assert header == [
        "# Number of sources: 5",
        "# RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), FWHM major "
        "(arcsec), FWHM minor (arcsec), Position angle (deg), Ref. freq. (Hz), "
        "Spectral index, Rotation measure (rad/m^2)",
        MAKESOURCEDB_FORMAT_STRING,
    ]

    # Check data lines
    assert np.all(data == expected_converted_data)


def test_write_to_makesourcedb(
    tmp_path,
    expected_converted_data,
    expected_output_header,
    expected_output_lines,
):
    """Test writing of output file."""
    output_path = tmp_path / "test_write_sample.skymodel"
    write_to_makesourcedb(
        output_path, expected_output_header, expected_converted_data
    )

    assert output_path.exists()

    lines = output_path.read_text().splitlines(keepends=True)
    header_lines, data_lines = lines[:3], lines[4:]
    assert np.all(
        np.char.strip(header_lines) == np.char.strip(expected_output_header)
    )

    assert list(data_lines) == expected_output_lines


def test_convert_oskar_skymodel(caplog, tmp_path, sample_csv_path):
    """
    Test full conversion from input file to output file on small sample dataset,
    check that the expected messages are printed.
    """

    output_path = tmp_path / "converted_sample.skymodel"

    convert_oskar_skymodel(
        sample_csv_path,
        output_path,
        point_size_threshold=1e-8,
        min_flux_point=0.005,
        min_flux_extended=0.02,
    )

    assert all(
        msg in caplog.text
        for msg in (
            "Conversion complete.",
            "Total input sources: 5",
            "Total output sources: 5",
        )
    )


@pytest.mark.parametrize(
    "input_oskar_file, expected_makesourcedb_file",
    [
        (
            TEST_DATA_PATH / "oskar_sky_model_example.csv",
            TEST_DATA_PATH / "makesourcedb_sky_model_example.csv",
        )
    ],
)
def test_convert_example_oskar_skymodel(
    tmp_path, input_oskar_file, expected_makesourcedb_file
):
    """
    Test full conversion from input file to output file on example dataset and
    compare output to expected output file.
    """

    output_path = tmp_path / "makesourcedb_result.csv"

    convert_oskar_skymodel(
        input_oskar_file,
        output_path,
        point_size_threshold=1e-8,
        min_flux_point=1,
        min_flux_extended=1,
    )

    result_text = output_path.read_text()
    expected_text = expected_makesourcedb_file.read_text()
    assert result_text.strip() == expected_text.strip()


@pytest.mark.parametrize(
    "command, expected_args",
    [
        (
            "convert_oskar_skymodel input/test.skymodel output/test.skymodel",
            ["input/test.skymodel", "output/test.skymodel", 1e-8, 0.005, 0.02],
        ),
        (
            "convert_oskar_skymodel input/test.skymodel output/test.skymodel "
            "--point-size-threshold 12 --min-flux-point 1 "
            "--min-flux-extended 2",
            ["input/test.skymodel", "output/test.skymodel", 12, 1, 2],
        ),
    ],
)
def test_cli(command, expected_args):
    """Test command-line interface."""

    sys.argv[:] = shlex.split(command)

    with mock.patch(
        "lsmtool.convert_oskar_skymodel.convert_oskar_skymodel"
    ) as mock_convert_skymodel:
        main()

    # Assert
    mock_convert_skymodel.assert_called_once_with(*expected_args)


def test_performance(tmp_path, rng, n_sources=10_000, time_limit=1):
    """
    Test that we can process a certain number sources within a time limit in
    seconds.
    """
    output_file = tmp_path / "performance_test.skymodel"
    # mock the `read_oskar_skymodel` function to return generated data. In this
    # way we avoid first having to write a large file to disk before running
    # the test
    with mock.patch(
        "lsmtool.convert_oskar_skymodel.read_oskar_skymodel"
    ) as mock_read_oskar_skymodel:
        mock_read_oskar_skymodel.return_value = random_skymodel(n_sources, rng)

        # Time execution
        t0 = time.time()
        convert_oskar_skymodel(
            None,
            output_file,
            point_size_threshold=0.1,
            min_flux_point=1,
            min_flux_extended=1,
        )
        t1 = time.time()

    # Check that runtime is below time limit
    time_taken = t1 - t0
    print(
        f"Performance test took {time_taken:.2f} seconds for {n_sources} "
        "sources."
    )
    assert time_taken < time_limit, (
        "Performance test took too long: "
        f"{time_taken:.2f} seconds > {time_limit} seconds"
    )
