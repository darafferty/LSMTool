#!/usr/bin/env python3

import filecmp
import numpy
import pathlib
import requests
import tarfile
import tempfile
import unittest

import lsmtool
from lsmtool.operations_lib import attenuate


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

    def test_attenuate(self):
        """
        Test `attenuate` function
        """
        outfile = str(self.temp_path / "test_attenuate.out")
        reffile = str(self.resource_path / "test_attenuate.out")
        result = attenuate(
            str(self.ms_path),
            self.flux,
            self.ra,
            self.dec,
        )
        numpy.set_printoptions(precision=6)
        with open(outfile, "w") as f:
            f.write(str(result))
        assert filecmp.cmp(reffile, outfile, shallow=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
