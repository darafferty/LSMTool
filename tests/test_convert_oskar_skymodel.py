import shlex
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from conftest import TEST_DATA_PATH

from lsmtool.convert_oskar_skymodel import (
    HEADER_FORMAT_LINE,
    convert_row,
    convert_skymodel,
    filter_sources,
    format_row,
    main,
    read_data,
    write,
)

TEST_RESOURCES_PATH = TEST_DATA_PATH
TEST_DATA_PATH = TEST_RESOURCES_PATH / Path(__file__).stem


@pytest.fixture()
def sample_csv_text():
    """Sample CSV data as text for testing."""
    return {
        "header": (
            "# RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), "
            "Ref. freq. (Hz), Spectral index, Rotation measure (rad/m^2), "
            "FWHM major (arcsec), FWHM minor (arcsec), Position angle (deg)"
        ),
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
            "125.806642113852, -36.775065379826, 2.43444339899304e-01, "
            "0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, "
            "1.44000000000000e+08,-7.00000000000000e-01, 0.00000000000000e+00, "
            "8.852988859902,   6.332216807758,  96.100299655753"
        ),
    }


@pytest.fixture()
def sample_csv_path(tmp_path, sample_csv_text):
    """Sample CSV file path containing sample csv text header and data."""
    path = tmp_path / "sample.csv"
    path.write_text("\n".join(sample_csv_text.values()))
    return path


@pytest.fixture()
def expected_sample_data():
    """
    Expected data for the sample CSV file after conversion to list of floats.
    """
    return [
        (
            134.316584681925,
            -34.806858824585,
            7.12299476324136e-01,
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
            5.05840198469155e-01,
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
            5.04116042438541e-01,
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
            2.58824974285288e-01,
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
            2.43444339899304e-01,
            0.0,
            0.0,
            0.0,
            1.44e8,
            -0.7,
            0.0,
            8.852988859902,
            6.332216807758,
            96.100299655753,
        ),
    ]


@pytest.fixture()
def expected_makesourcedb_lines():
    """
    Expected lines in the converted makesourcedb skymodel for the sample CSV.
    data.
    """
    return [
        "src0, GAUSSIAN, 08:57:15.98032, -34.48.24.6918, 0.712299476324136,"
        " 0.0, 0.0, 0.0, 2.552896197845, 1.705990053656, 115.020339486069, "
        "144000000.0, [-0.7], 0.0\n",
        "src1, GAUSSIAN, 08:13:25.89225, -36.55.20.2820, 0.505840198469155,"
        " 0.0, 0.0, 0.0, 3.381746257518, 2.676308575899, 37.727775657383, 1"
        "44000000.0, [-0.7], 0.0\n",
        "src2, GAUSSIAN, 09:01:05.53983, -29.01.44.4275, 0.504116042438541,"
        " 0.0, 0.0, 0.0, 14.702668401253, 1.319160550191, 28.175171030289, "
        "144000000.0, [-0.7], 0.0\n",
        "src3, GAUSSIAN, 08:35:04.86764, -26.10.36.2285, 0.258824974285288,"
        " 0.0, 0.0, 0.0, 12.861909842975, 2.246703463196, 74.249587537251, "
        "144000000.0, [-0.7], 0.0\n",
        "src4, GAUSSIAN, 08:23:13.59411, -36.46.30.2354, 0.243444339899304,"
        " 0.0, 0.0, 0.0, 8.852988859902, 6.332216807758, 96.100299655753, 1"
        "44000000.0, [-0.7], 0.0\n",
    ]


def test_read_data(sample_csv_path, sample_csv_text):
    """Test reading header and data lines from input file."""
    header_lines, data = read_data(sample_csv_path)
    assert list(header_lines) == [sample_csv_text["header"]]
    assert list(data) == sample_csv_text["data"].splitlines()


@pytest.fixture
def counters(request):
    """
    Fixture for providing counters dict for testing `filter_sources` function.
    """
    if request.param:
        return {
            "n_input_sources": 0,
            "n_extended_removed": 0,
            "n_point_removed": 0,
        }
    return None


@pytest.mark.parametrize(
    "point_size_threshold, min_flux_point, min_flux_extended, "
    "expected_counts, expected_indices_remain",
    [
        # Treat all sources as extended, test that none are filtered for small
        # `min_flux_point`
        (1e-8, 0.005, 0.02, [0, 0], [0, 1, 2, 3, 4]),
        # Test that all extended source are filtered for large `min_flux_point`
        (1e-8, 0, 1, [5, 0], []),
        # Test that all point sources are filtered for large
        # `min_flux_extended`
        (15, 1, 0, [0, 5], []),
        # Test that extended sources can be filtered
        (10, 0, 1, [2, 0], [0, 1, 4]),
        # Test that point sources can be filtered
        (10, 1, 0, [0, 3], [2, 3]),
    ],
)
@pytest.mark.parametrize(
    "counters",
    [False, True],
    indirect=True,
)
def test_filter_sources(
    sample_csv_text,
    point_size_threshold,
    min_flux_point,
    min_flux_extended,
    counters,
    expected_counts,
    expected_indices_remain,
):
    """Test source filtering"""
    data_lines = sample_csv_text["data"].splitlines(keepends=True)
    with mock.patch(
        "lsmtool.convert_oskar_skymodel.convert_row",
        side_effect=lambda line: line,
    ):
        result = list(
            filter_sources(
                data_lines,
                point_size_threshold,
                min_flux_point,
                min_flux_extended,
                source_counts=counters,
            )
        )

    # Check counter have correct values
    if counters:
        expected_counts = {
            "n_input_sources": 5,
            "n_extended_removed": expected_counts[0],
            "n_point_removed": expected_counts[1],
        }
        assert counters == expected_counts

    # Check that the expected lines remain in the output
    assert result == [data_lines[i] for i in expected_indices_remain]


def test_convert_row(sample_csv_text, expected_sample_data):
    """Test data conversion."""
    data_lines = sample_csv_text["data"].splitlines()
    result = list(map(convert_row, data_lines))
    assert np.allclose(result, expected_sample_data)


def test_format_row(expected_sample_data, expected_makesourcedb_lines):
    """Test formatting of data for makesourcedb skymodel output."""
    result = [
        format_row(f"src{i}", row_data)
        for i, row_data in enumerate(expected_sample_data)
    ]
    assert result == expected_makesourcedb_lines


def test_write(
    tmp_path, sample_csv_text, expected_sample_data, expected_makesourcedb_lines
):
    """Test writing of output file."""
    output_path = tmp_path / "test_write_sample.skymodel"
    write(output_path, [sample_csv_text["header"]], expected_sample_data)

    assert output_path.exists()
    lines = output_path.read_text().splitlines(keepends=True)
    assert lines[0].strip() == "# Number of sources: 5"
    assert lines[1].strip() == sample_csv_text["header"]
    assert lines[2].strip() == HEADER_FORMAT_LINE.strip()
    assert lines[4:] == expected_makesourcedb_lines


def test_convert_skymodel(capsys, tmp_path, sample_csv_path):
    """
    Test full conversion from input file to output file on small sample dataset,
    check that the expected messages are printed.
    """

    output_path = tmp_path / "converted_sample.skymodel"

    convert_skymodel(
        sample_csv_path,
        output_path,
        point_size_threshold=1e-8,
        min_flux_point=0.005,
        min_flux_extended=0.02,
    )

    captured = capsys.readouterr()
    assert all(
        msg in captured.out
        for msg in (
            "Conversion complete.",
            "Total input sources: 5",
            "Total output sources: 5",
            "Point sources removed: 0",
            "Extended sources removed: 0",
        )
    )


def test_convert_example_oskar_skymodel(tmp_path):
    """
    Test full conversion from input file to output file on example dataset and
    compare output to expected output file.
    """
    output_path = tmp_path / "makesourcedb_result.csv"

    convert_skymodel(
        TEST_DATA_PATH / "oskar_sky_model_example.csv",
        output_path,
        point_size_threshold=1e-8,
        min_flux_point=0.005,
        min_flux_extended=0.02,
    )

    result_text = output_path.read_text()
    expected_path = TEST_DATA_PATH / "makesourcedb_sky_model_example.csv"
    expected_text = expected_path.read_text()
    assert result_text == expected_text


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
        "lsmtool.convert_oskar_skymodel.convert_skymodel"
    ) as mock_convert_skymodel:
        main()

    # Assert
    mock_convert_skymodel.assert_called_once_with(*expected_args)
