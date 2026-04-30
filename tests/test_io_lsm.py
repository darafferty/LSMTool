import ast
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as _np
import pytest

from lsmtool.skymodel import SkyModel
from lsmtool.tableio import (
    loadAstropyTableFromLSM,
    loadTableFromLSM,
    validateLSMFormat,
)


class InvalidLSMFormat(Exception):
    pass


def _data_rows(path: Path):
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            yield line


def _get_lsm_header(path: Path):
    with path.open() as f:
        for line in f:
            if "format" in line:
                try:
                    header_columns, *_ = (
                        line.replace("#", "").replace(" ", "").split("=")
                    )
                    header_columns = header_columns.lstrip("(").rstrip(")")
                    return header_columns.split(",")
                except ValueError as e:
                    raise InvalidLSMFormat(f"Invalid header {line}") from e
        else:
            raise InvalidLSMFormat("Format line not provided in {path}")


def _get_sky_header(path: Path):
    with path.open() as f:
        for line in f:
            if "format" in line.lower():
                try:
                    _, header_columns = line.split("=", maxsplit=1)
                except ValueError as e:
                    raise InvalidLSMFormat(f"Invalid header {line}") from e

                columns = [
                    part.strip().split("=", maxsplit=1)[0]
                    for part in header_columns.split(",")
                ]
                return columns

    raise InvalidLSMFormat(f"Format line not provided in {path}")


def _split_sky_row(line: str):
    parts = []
    token = []
    bracket_depth = 0

    for ch in line:
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)

        if ch == "," and bracket_depth == 0:
            parts.append("".join(token).strip())
            token = []
            continue

        token.append(ch)

    if token:
        parts.append("".join(token).strip())

    return parts


def _skymodel_rows(path: Path):
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            if "format" in s.lower():
                continue
            if s.startswith(","):
                continue
            yield s


_EXPECTED_LSM_COLUMN_NAMES = [
    "component_id",
    "source_id",
    "ra_deg",
    "dec_deg",
    "a_arcsec",
    "b_arcsec",
    "pa_deg",
    "spec_idx",
    "log_spec_idx",
    "i_pol_jy",
    "ref_freq_hz",
    "epoch",
]

_EXPECTED_SKYMODEL_COLUMN_NAMES = [
    "Name",
    "Type",
    "Patch",
    "Ra",
    "Dec",
    "I",
    "SpectralIndex",
    "LogarithmicSI",
    "ReferenceFrequency",
    "MajorAxis",
    "MinorAxis",
    "Orientation",
]


def validate_lsm_format(skymodel_path):
    """Validate that a skymodel matches the LSM format.

    Expected columns per row:
    (component_id, ra_deg, dec_deg, i_pol_jy, a_arcsec, b_arcsec, pa_deg,
     ref_freq_hz, spec_idx, log_spec_idx)
    """
    p = Path(skymodel_path)
    assert p.exists(), f"Missing file: {p}"
    lsm_header = _get_lsm_header(skymodel_path)

    for idx, (column_name, expected_column_name) in enumerate(
        zip(lsm_header, _EXPECTED_LSM_COLUMN_NAMES, strict=True)
    ):
        assert column_name == expected_column_name, (
            f"column {idx} name mismatch"
        )

    reader = csv.reader(_data_rows(p), delimiter=",", quotechar='"')
    rows = list(reader)
    assert rows, f"No data rows found in {p.name}"

    for n, row in enumerate(rows, start=1):
        component_id = row[0].strip()
        source_id = row[1].strip()
        assert component_id, f"Row {n} empty component_id"
        assert source_id, f"Row {n} empty component_id"

        # numeric conversions
        try:
            ra = float(row[2])
            dec = float(row[3])
            a = float(row[4])
            b = float(row[5])
            pa = float(row[6])

            i_pol = float(row[9])
            ref_freq = float(row[10])
        except ValueError as exc:
            raise AssertionError(
                f"Row {n} numeric parse error: {exc} (row={row})"
            )

        assert 0.0 <= ra < 360.0, f"Row {n} ra out of range: {ra}"
        assert -90.0 <= dec <= 90.0, f"Row {n} dec out of range: {dec}"
        assert i_pol >= 0.0, f"Row {n} i_pol negative: {i_pol}"

        # spec_idx: quoted CSV field containing a python-style list
        spec_field = row[7].strip()
        try:
            spec_list = ast.literal_eval(spec_field)
        except Exception as exc:
            raise AssertionError(
                f"Row {n} spec_idx parse error: (value={spec_field})"
            ) from exc

        assert isinstance(spec_list, (list, tuple)), (
            f"Row {n} spec_idx is not a list: {spec_list}"
        )
        for v in spec_list:
            assert isinstance(v, (int, float)), (
                f"Row {n} spec_idx element not numeric: {v}"
            )

        log_field = row[8].strip().lower()
        assert log_field in ("true", "false"), (
            f"Row {n} log_spec_idx not boolean: {row[9]}"
        )


def validate_skymodel_format(skymodel_path):
    p = Path(skymodel_path)
    assert p.exists(), f"Missing file: {p}"
    sky_header = _get_sky_header(p)

    for idx, (column_name, expected_column_name) in enumerate(
        zip(sky_header, _EXPECTED_SKYMODEL_COLUMN_NAMES, strict=True)
    ):
        assert column_name == expected_column_name, (
            f"column {idx} name mismatch"
        )

    for n, line in enumerate(_skymodel_rows(p), start=1):
        row = _split_sky_row(line)
        assert len(row) == len(_EXPECTED_SKYMODEL_COLUMN_NAMES), (
            f"Row {n} has wrong number of columns: {len(row)} (row={row})"
        )

        name = row[0].strip()
        source_type = row[1].strip()
        patch = row[2].strip()
        ra = row[3].strip()
        dec = row[4].strip()

        assert name, f"Row {n} empty Name"
        assert source_type in ("POINT", "GAUSSIAN"), (
            f"Row {n} unknown Type: {source_type}"
        )
        assert patch, f"Row {n} empty Patch"
        assert ra, f"Row {n} empty Ra"
        assert dec, f"Row {n} empty Dec"

        try:
            _ = float(row[5])
            _ = float(row[8])
            _ = float(row[9])
            _ = float(row[10])
            _ = float(row[11])
        except ValueError as exc:
            raise AssertionError(
                f"Row {n} numeric parse error: {exc} (row={row})"
            ) from exc

        try:
            spec_list = ast.literal_eval(row[6])
        except Exception as exc:
            raise AssertionError(
                f"Row {n} spec_idx parse error: (value={row[6]})"
            ) from exc

        assert isinstance(spec_list, (list, tuple)), (
            f"Row {n} spec_idx is not a list: {spec_list}"
        )
        for v in spec_list:
            assert isinstance(v, (int, float)), (
                f"Row {n} spec_idx element not numeric: {v}"
            )

        log_field = row[7].strip().lower()
        assert log_field in ("true", "false"), (
            f"Row {n} LogarithmicSI not boolean: {row[7]}"
        )


@pytest.fixture()
def lsm_skymodel(request):
    return request.config.resource_dir / "skymodel.lsm"


@pytest.fixture()
def apparent_skymodel(request):
    return request.config.resource_dir / "apparent.sky"


def test_lsm_format(lsm_skymodel):
    validate_lsm_format(lsm_skymodel)


def test_skymodel_format(apparent_skymodel):
    validate_skymodel_format(apparent_skymodel)


def test_load_lsm_with_astropy(lsm_skymodel):
    table = loadAstropyTableFromLSM(lsm_skymodel)
    assert set(table.colnames) ^ set(_EXPECTED_LSM_COLUMN_NAMES) == set()
    assert len(table) == 3


@pytest.fixture()
def expected_lsm_content():
    return {
        "Name": ["J000011-000001", "J000011-000002", "J000011-000003"],
        "Patch": ["J000011", "J000011", "J000011"],
        "ReferenceFrequency": [1.01e8, 1.02e8, 1.03e8],
        "I": [10.0, 20.0, 30.0],
        "MajorAxis": [100.0, 200.0, 300.0],
        "MinorAxis": [10.0, 20.0, 30.0],
        "Orientation": [1.0, 2.0, 3.0],
        "SpectralIndex": [
            [-0.7, 0.01, 0.123],
            [-0.7, 0.02, 0.123],
            [-0.7, 0.03, 0.123],
        ],
        "LogarithmicSI": ["true", "false", "true"],
    }


def test_load_table_from_lsm(lsm_skymodel, expected_lsm_content):
    table = loadTableFromLSM(lsm_skymodel)
    for key, expected_values in expected_lsm_content.items():
        if key == "SpectralIndex":
            for idx, (value, expected_value) in enumerate(
                zip(table[key], expected_values, strict=True)
            ):
                assert list(value) == expected_value, (
                    f"{idx} mismatch for {key}"
                )
        else:
            assert list(table[key]) == list(expected_values), f"{key}"


def test_validation_succeed(lsm_skymodel, apparent_skymodel):
    assert validateLSMFormat(lsm_skymodel)
    assert not validateLSMFormat(apparent_skymodel)


def assert_tables_equal(t1, t2):
    assert t1.colnames == t2.colnames
    for col in t1.colnames:
        c1, c2 = t1[col], t2[col]
        if isinstance(c1[0], _np.ndarray):
            # ndarray elements: compare element-wise
            for v1, v2 in zip(c1, c2):
                assert _np.allclose(v1, v2), f"Failed comparison for {col}"

        elif _np.issubdtype(c1.dtype, _np.floating):
            # Float columns: use allclose for tolerance
            assert _np.allclose(c1, c2), f"Failed comparison for {col}"
        else:
            # Other types: exact comparison
            assert _np.all(c1 == c2), f"Failed comparison for {col}"


def test_instantiate_lsm_skymodel_from_file(lsm_skymodel):
    skymodel = SkyModel(str(lsm_skymodel))
    expected_table = loadTableFromLSM(lsm_skymodel)
    assert_tables_equal(skymodel.table, expected_table)


def test_instantiate_lsm_skymodel_store_skymodel(lsm_skymodel, tmpdir):
    skymodel = SkyModel(str(lsm_skymodel))

    # Write to temporary file
    output_path = str(Path(tmpdir) / "saved_skymodel.sky")
    skymodel.write(output_path)

    validate_skymodel_format(output_path)
    # Read back and verify
    loaded_skymodel = SkyModel(output_path)
    assert_tables_equal(skymodel.table, loaded_skymodel.table)


def test_instantiate_lsm_skymodel_store_lsm(lsm_skymodel, tmpdir):
    skymodel = SkyModel(str(lsm_skymodel))

    # Write to temporary file
    output_path = str(Path(tmpdir) / "saved_skymodel.sky")
    skymodel.write(output_path, format="lsm")

    validate_lsm_format(Path(output_path))
    # Read back and verify
    loaded_skymodel = SkyModel(output_path)
    assert_tables_equal(skymodel.table, loaded_skymodel.table)
