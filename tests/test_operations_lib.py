"""
Tests for the `lsmtool.operations_lib` module.
"""

import filecmp

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

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
    assert_allclose((result.ra, result.dec), expected)


def test_make_wcs_default():
    """Test `make_wcs` with default parameters."""
    ref_ra = 10
    ref_dec = -42
    crdelt = 0.066667
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
