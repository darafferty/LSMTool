#!/usr/bin/env python3

import filecmp
import numpy as np
import pathlib
import requests
import tarfile
import tempfile
import unittest

import lsmtool
from lsmtool.operations_lib import apply_beam, make_wcs, normalize_ra_dec


class TestOperationsLib(unittest.TestCase):
    """
    Test class for the module `operations_lib`

    For testing a Measurement Set will be downloaded, if not present in the
    `tests/resources` directory. The skymodel was generated, using the pointing
    information in this MS, using the following commands:
    ```
    s = lsmtool.load('TGSS', VOPosition=["08:13:36.067800", "+48.13.02.58100"], VORadius=1.0)
    s.write('tests/resources/LOFAR_HBA_MOCK.sky')
    ```
    """

    @classmethod
    def setUpClass(cls):
        """
        Download and unpack Measurement Set used for testing
        """
        # Path to directory containing test resources
        cls.resource_path = pathlib.Path(__file__).parent / "resources"

        # Path to the input Measurement Set (will be downloaded if non-existent)
        cls.ms_path = cls.resource_path / "LOFAR_HBA_MOCK.ms"

        # URL to download location of input Measurement Set
        cls.ms_url = "https://support.astron.nl/software/ci_data/EveryBeam/L258627-one-timestep.tar.bz2"

        # Path to the input Sky Model (will be created if non-existent)
        cls.skymodel_path = cls.resource_path / "LOFAR_HBA_MOCK.sky"

        # Create a temporary test directory
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = pathlib.Path(cls.temp_dir.name)

        # Download MS if it's not available in resource directory
        if not cls.ms_path.exists():
            cls.download_ms()

    @classmethod
    def download_ms(cls):
        """
        Download and unpack MS in our temporary test directory
        """
        # Filter to strip leading component from file names (like `tar --strip-components=1`)
        filter = lambda member, path: member.replace(
            name=pathlib.Path(*pathlib.Path(member.path).parts[1:])
        )
        with requests.get(cls.ms_url, stream=True) as req:
            with tarfile.open(fileobj=req.raw, mode="r|bz2") as tarobj:
                tarobj.extractall(path=cls.ms_path, filter=filter)

    def setUp(self):
        self.skymodel = lsmtool.load(str(self.skymodel_path))
        self.ra = self.skymodel.getColValues("RA")
        self.dec = self.skymodel.getColValues("Dec")
        self.flux = self.skymodel.getColValues("I")

    def test_apply_beam(self):
        """
        Test `apply_beam` function
        """
        outfile = str(self.temp_path / "test_apply_beam.out")
        reffile = str(self.resource_path / "test_apply_beam.out")
        result = apply_beam(
            str(self.ms_path),
            self.flux,
            self.ra,
            self.dec,
        )
        np.set_printoptions(precision=6)
        with open(outfile, "w") as f:
            f.write(str(result))
        assert filecmp.cmp(reffile, outfile, shallow=False)

    def test_apply_beam_invert(self):
        """
        Test `apply_beam` function with inverted beam
        """
        outfile = str(self.temp_path / "test_apply_beam_invert.out")
        reffile = str(self.resource_path / "test_apply_beam_invert.out")
        result = apply_beam(
            str(self.ms_path),
            self.flux,
            self.ra,
            self.dec,
            invert=True
        )
        np.set_printoptions(precision=6)
        with open(outfile, "w") as f:
            f.write(str(result))
        assert filecmp.cmp(reffile, outfile, shallow=False)

    def test_normalize_ra_dec(self):
        """
        Test `normalize_ra_dec` function
        """
        ra = 450.0
        dec = 95.0
        result = normalize_ra_dec(ra, dec)
        assert (result.ra == 270.0 and result.dec == 85.0)


def test_make_wcs_default():
    """Test `make_wcs` with default parameters."""
    ref_ra = 10
    ref_dec = -42
    crdelt = 0.066667
    w = make_wcs(ref_ra, ref_dec)
    assert w is not None
    assert w.naxis == 2
    assert (w.wcs.crpix == np.array([1000, 1000])).all()
    assert (w.wcs.cdelt == np.array([-crdelt, crdelt])).all()
    assert (w.wcs.crval == np.array([ref_ra, ref_dec])).all()
    assert (w.wcs.ctype == np.array(["RA---TAN", "DEC--TAN"])).all()


def test_make_wcs_custom():
    """Test `make_wcs` with a custom crdelt parameter."""
    ref_ra = -10
    ref_dec = 42
    crdelt = 0.42
    w = make_wcs(ref_ra, ref_dec, crdelt)
    assert w is not None
    assert w.naxis == 2
    assert (w.wcs.crpix == np.array([1000, 1000])).all()
    assert (w.wcs.cdelt == np.array([-crdelt, crdelt])).all()
    assert (w.wcs.crval == np.array([ref_ra, ref_dec])).all()
    assert (w.wcs.ctype == np.array(["RA---TAN", "DEC--TAN"])).all()


if __name__ == "__main__":
    unittest.main(verbosity=2)
