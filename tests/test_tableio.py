from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from lsmtool.skymodel import SkyModel
from lsmtool.tableio import (
    loadAstropyTableFromLSM,
    loadTableFromLSM,
    skyModelReader,
    validateLSMFormat,
)

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


@pytest.fixture()
def lsm_skymodel(test_data_path):
    return test_data_path / "skymodel.lsm"


@pytest.fixture()
def lsm_skymodel_partial_spectral_index(test_data_path):
    return test_data_path / "sky_model_target.lsm.csv"


@pytest.fixture()
def expected_lsm_skymodel_partial_spectral_index(test_data_path):
    return test_data_path / "expected_sky_model_target.lsm_stored.csv"


@pytest.fixture()
def apparent_skymodel(request):
    return request.config.resource_dir / "apparent.sky"


def test_load_lsm_with_astropy(lsm_skymodel):
    table = loadAstropyTableFromLSM(lsm_skymodel)
    assert set(table.colnames) ^ set(_EXPECTED_LSM_COLUMN_NAMES) == set()
    assert len(table) == 3


@pytest.fixture()
def expected_lsm_content():
    return {
        "Name": ["J000011-000001", "J000011-000002", "J000011-000003"],
        "Type": ["POINT", "GAUSSIAN", "GAUSSIAN"],
        "Patch": ["J000011", "J000011", "J000011"],
        "ReferenceFrequency": [1.01e8, 1.02e8, 1.03e8],
        "I": [10.0, 20.0, 30.0],
        "MajorAxis": [0.0, 200.0, 300.0],
        "MinorAxis": [0.0, 20.0, 30.0],
        "Orientation": [0.0, 2.0, 3.0],
        "SpectralIndex": [
            [-0.7, 0.01, 0.123],
            [-0.7, 0.02, 0.123],
            [-0.7, 0.03, 0.123],
        ],
        "LogarithmicSI": ["true", "false", "true"],
    }


def test_load_table_from_lsm(lsm_skymodel, expected_lsm_content):
    """
    Verifies that can load the table from LSM/GSM format
    """
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


def test_validation_succeed(
    lsm_skymodel,
    apparent_skymodel,
    lsm_skymodel_partial_spectral_index,
    expected_lsm_skymodel_partial_spectral_index,
):
    """
    Verifies that the validation function work properly
    """
    assert validateLSMFormat(lsm_skymodel)
    assert validateLSMFormat(lsm_skymodel_partial_spectral_index)
    assert validateLSMFormat(expected_lsm_skymodel_partial_spectral_index)
    assert not validateLSMFormat(apparent_skymodel)


def assert_tables_equal(t1, t2):
    assert t1.colnames == t2.colnames
    for col in t1.colnames:
        c1, c2 = t1[col], t2[col]
        if isinstance(c1[0], np.ndarray):
            # ndarray elements: compare element-wise
            for v1, v2 in zip(c1, c2, strict=True):
                assert np.allclose(v1, v2), f"Failed comparison for {col}"

        elif np.issubdtype(c1.dtype, np.floating):
            # Float columns: use allclose for tolerance
            assert np.allclose(c1, c2), f"Failed comparison for {col}"
        else:
            # Other types: exact comparison
            assert np.all(c1 == c2), f"Failed comparison for {col}"


def test_instantiate_lsm_skymodel_from_file(lsm_skymodel):
    """
    Verifies that you can load the SkyModel object
    from LSM skymodel
    """
    skymodel = SkyModel(str(lsm_skymodel))
    expected_table = loadTableFromLSM(lsm_skymodel)
    assert_tables_equal(skymodel.table, expected_table)


def test_instantiate_lsm_skymodel_store_skymodel(lsm_skymodel, tmpdir):
    """
    Verifies that you can load the SkyModel object
    from LSM skymodel and store the makesourcedb format skymodel
    """
    skymodel = SkyModel(str(lsm_skymodel))

    # Write to temporary file
    output_path = str(Path(tmpdir) / "saved_skymodel.sky")
    skymodel.write(output_path)

    # Read back and verify
    loaded_skymodel = SkyModel(output_path)
    assert_tables_equal(skymodel.table, loaded_skymodel.table)


def test_instantiate_lsm_skymodel_store_lsm(lsm_skymodel, tmpdir):
    """
    Verifies that you can load the SkyModel object
    from LSM skymodel and store the lsm format skymodel
    """
    skymodel = SkyModel(str(lsm_skymodel))

    # Write to temporary file
    output_path = str(tmpdir / "saved_skymodel.sky")
    skymodel.write(output_path, format="lsm", clobber=True)
    assert Path(output_path).exists()
    # Read back and verify
    loaded_skymodel = SkyModel(output_path)
    assert_tables_equal(skymodel.table, loaded_skymodel.table)


def test_skymodelreader_emptyfile(tmp_path):
    """Test skyModelReader raises IOError for empty file"""

    # Create a fake existing sky model file to test functionality
    skymodel_path = tmp_path / "empty.sky"
    skymodel_path.touch()

    with pytest.raises(IOError):
        skyModelReader(str(skymodel_path))


def test_skymodelreader_headeronly(tmp_path):
    """Test skyModelReader returns empty table for header-only file"""

    # Create a fake existing sky model file to test functionality
    skymodel_path = tmp_path / "empty.sky"
    skymodel_path.write_text("FORMAT = Name, Type, Ra, Dec, I\n")

    table = skyModelReader(str(skymodel_path))

    assert isinstance(table, Table)
    assert len(table) == 0


def test_lsm_skymodel_read_incomplete_spectral_index(
    lsm_skymodel_partial_spectral_index,
    expected_lsm_skymodel_partial_spectral_index,
    tmp_path,
):
    """Test that SkyModel can parse partially populated spectral index"""
    skymodel = SkyModel(str(lsm_skymodel_partial_spectral_index))

    assert len(skymodel.table) == 9
    assert skymodel.table[8]["SpectralIndex"] == [-0.7, 0.1, 0.3, 0.1, 0.3]

    row = skymodel.table[8]
    assert row["Name"] == "Component 000008"
    assert row["Patch"] == "Patch_1"
    assert row["Ra"] == pytest.approx(123.246689256631)
    assert row["Dec"] == pytest.approx(-32.720251347259)
    assert row["I"] == pytest.approx(2.39699273933944)
    assert row["ReferenceFrequency"] == pytest.approx(144000000)
    assert row["MajorAxis"] == pytest.approx(1.7381665486575)
    assert row["MinorAxis"] == pytest.approx(1.3767510279985)
    assert row["Orientation"] == pytest.approx(143.662170729077)
    assert list(row["SpectralIndex"]) == [-0.7, 0.1, 0.3, 0.1, 0.3]
    assert bool(row["LogarithmicSI"])

    generated_lsm = tmp_path / "lsm.csv"
    skymodel.write(generated_lsm, format="lsm", clobber=True)
    assert generated_lsm.exists()
    assert (
        generated_lsm.read_text()
        == expected_lsm_skymodel_partial_spectral_index.read_text()
    )


@pytest.mark.parametrize("as_angles", [False, True])
def test_radec_normalization_matches_scalar(as_angles):
    """Vector normalization agrees at poles, wraps, and random coordinates."""
    from astropy.coordinates import Angle

    from lsmtool.operations_lib import normalize_ra_dec
    from lsmtool.tableio import RADec2Angle

    rng = np.random.default_rng(42)
    ra = np.concatenate(
        ([0, 360, -360, 720, -1, 180, 90], rng.uniform(-2000, 2000, 1000))
    )
    dec = np.concatenate(
        ([90, -90, 180, -180, 270, -270, 0], rng.uniform(-2000, 2000, 1000))
    )
    expected = np.array(
        [normalize_ra_dec(r, d) for r, d in zip(ra, dec, strict=True)]
    )
    if as_angles:
        ra = Angle(ra, unit="deg").to("rad")
        dec = Angle(dec, unit="deg").to("rad")
    actual_ra, actual_dec = RADec2Angle(ra, dec)
    np.testing.assert_allclose(actual_ra.degree, expected[:, 0], atol=1e-12)
    np.testing.assert_allclose(actual_dec.degree, expected[:, 1], atol=1e-12)


@pytest.mark.parametrize(
    ("ra", "dec", "expected_ra", "expected_dec"),
    [
        (370.0, 100.0, [190], [80]),
        ([370.0, -10.0], [100.0, -100.0], [190, 170], [80, -80]),
        ("12:00:00", "-30.00.00", [180], [-30]),
        (["12:00:00", "06:00:00"], ["-30:00:00", "45:00:00"],
         [180, 90], [-30, 45]),
        ([], [], [], []),
        ([10, 20], [30], [10], [30]),
    ],
)
def test_radec_input_formats(ra, dec, expected_ra, expected_dec):
    from lsmtool.tableio import RADec2Angle

    actual_ra, actual_dec = RADec2Angle(ra, dec)
    np.testing.assert_allclose(actual_ra.degree, expected_ra)
    np.testing.assert_allclose(actual_dec.degree, expected_dec)
