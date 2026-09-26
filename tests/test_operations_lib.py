"""
Tests for the `lsmtool.operations_lib` module.
"""

import filecmp

import numpy as np
import pytest
from astropy.coordinates import Angle
from numpy.testing import assert_allclose, assert_array_equal

from lsmtool.constants import WCS_PIXEL_SCALE
from lsmtool.operations_lib import apply_beam, make_wcs, normalize_ra_dec


@pytest.mark.parametrize("invert", [False, True])
def test_apply_beam(
    tmp_path, test_data_path, test_ms_lofar_hba, lofar_hba_skymodel, invert
):
    """Test `apply_beam` function"""
    filename = f"test_apply_beam{'_invert' * invert}.out"
    output_path = tmp_path / filename
    reference_path = test_data_path / filename

    result = apply_beam(
        str(test_ms_lofar_hba),
        lofar_hba_skymodel.getColValues("I"),
        lofar_hba_skymodel.getColValues("RA"),
        lofar_hba_skymodel.getColValues("Dec"),
        invert=invert,
    )

    np.savetxt(output_path, result, fmt="%.6f")
    assert filecmp.cmp(reference_path, output_path, shallow=False)


@pytest.mark.parametrize(
    "coords, expected",
    [((450.0, 95.0), (270.0, 85.0)), ((190.75, -115.34), (10.75, -64.66))],
)
def test_normalize_ra_dec(coords, expected):
    """
    Test `normalize_ra_dec` function
    """
    result = normalize_ra_dec(*coords)
    assert np.isscalar(result.ra)
    assert np.isscalar(result.dec)
    assert_allclose((result.ra, result.dec), expected)


@pytest.mark.parametrize("unit", [None, "deg", "rad"])
def test_normalize_ra_dec_arrays(unit):
    """Normalize wraps and pole crossings without changing input arrays."""
    ra = np.array([0.0, 360.0, -360.0, 720.0, -1.0, 180.0, 90.0, 450.0, 450.0])
    dec = np.array([90.0, -90.0, 180.0, -180.0, 270.0, -270.0, 0.0, 95.0, -95.0])
    if unit is not None:
        ra = Angle(ra, unit="deg").to(unit)
        dec = Angle(dec, unit="deg").to(unit)
    original_ra, original_dec = ra.copy(), dec.copy()

    result = normalize_ra_dec(ra, dec)

    assert_allclose(result.ra, [0, 0, 180, 180, 359, 180, 90, 270, 270])
    assert_allclose(result.dec, [90, -90, 0, 0, -90, 90, 0, 85, -85])
    assert_array_equal(ra, original_ra)
    assert_array_equal(dec, original_dec)


@pytest.mark.parametrize("unit", ["deg", "rad"])
def test_normalize_ra_dec_scalar_angles(unit):
    result = normalize_ra_dec(
        Angle(450, unit="deg").to(unit), Angle(95, unit="deg").to(unit)
    )
    assert np.isscalar(result.ra)
    assert np.isscalar(result.dec)
    assert_allclose(result, (270, 85))


def test_normalize_ra_dec_broadcasting():
    result = normalize_ra_dec([[0], [90]], [0, 100, -100])
    assert_array_equal(result.ra, [[0, 180, 180], [90, 270, 270]])
    assert_array_equal(result.dec, [[0, 80, -80], [0, 80, -80]])


def test_normalize_ra_dec_empty():
    result = normalize_ra_dec([], [])
    assert result.ra.shape == (0,)
    assert result.dec.shape == (0,)


def test_make_wcs_default():
    """Test `make_wcs` with default parameters."""
    ref_ra = 10
    ref_dec = -42
    crdelt = WCS_PIXEL_SCALE
    w = make_wcs(ref_ra, ref_dec)
    assert w is not None
    assert w.naxis == 2
    assert_array_equal(w.wcs.crpix, [1000, 1000])
    assert_array_equal(w.wcs.cdelt, [-crdelt, crdelt])
    assert_array_equal(w.wcs.crval, [ref_ra, ref_dec])
    assert_array_equal(w.wcs.ctype, ["RA---TAN", "DEC--TAN"])


def test_make_wcs_custom():
    """Test `make_wcs` with a custom crdelt parameter."""
    ref_ra = -10
    ref_dec = 42
    crdelt = 0.42
    w = make_wcs(ref_ra, ref_dec, crdelt)
    assert w is not None
    assert w.naxis == 2
    assert_array_equal(w.wcs.crpix, [1000, 1000])
    assert_array_equal(w.wcs.cdelt, [-crdelt, crdelt])
    assert_array_equal(w.wcs.crval, [ref_ra, ref_dec])
    assert_array_equal(w.wcs.ctype, ["RA---TAN", "DEC--TAN"])
