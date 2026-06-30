"""
Configuration for python tests.
"""

import contextlib
import shutil
import tarfile
from collections.abc import MutableMapping, Sequence
from dataclasses import InitVar, asdict, dataclass
from pathlib import Path

import astropy.units as u
import mocpy
import numpy as np
import pytest
import requests
from astropy.coordinates import Latitude, Longitude
from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats.distributions import rv_frozen, uniform

from lsmtool.io import PathLike, PathLikeOptional, check_file_exists, load
from lsmtool.utils import format_coordinates

# ---------------------------------------------------------------------------- #
# Module constants

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "resources"

# Path to the LOFAR HBA mock measurement set
LOFAR_HBA_URL = "https://support.astron.nl/software/ci_data/EveryBeam/L258627-one-timestep.tar.bz2"

# Random number generator seed for reproducibility
RNG_SEED = 881726
RNG = np.random.default_rng(seed=RNG_SEED)

# ---------------------------------------------------------------------------- #


def pytest_configure(config):
    config.resource_dir = TEST_DATA_PATH


# ---------------------------------------------------------------------------- #
# Helper functions


def untar(
    filename: PathLike,
    destination: PathLikeOptional = None,
    remove_archive: bool = False,
):
    """
    Uncompress the measurement set in the tgz file.

    Parameters
    ----------
    filename:  str or pathlib.Path
        Name of the tar file.
    destination:  str or pathlib.Path
        Path to extract the tar file to.
    """

    path = check_file_exists(filename)

    # Default output folder is the same as the input folder.
    destination = destination or path.parent

    # Uncompress the tgz file.
    with tarfile.open(path, "r:gz") as file:
        file.extractall(destination, filter="data")

    # Remove the compressed archive if requested
    if remove_archive:
        path.unlink()


def copy_test_data(files_to_copy, target):
    """
    Copy a single file to a target path, or a list of files into target folder.
    The function emulates GNU's `cp` command-line program.
    The source files are assumed to be in the test data folder.
    """

    if isinstance(files_to_copy, (str, Path)):
        # copy single file
        files_to_copy = [files_to_copy]
    elif not target.is_dir():
        # copy multiple files - ensure target is a directory
        raise NotADirectoryError("Copy target is not a valid directory.")

    for filename in files_to_copy:
        path = check_file_exists(TEST_DATA_PATH / filename)
        shutil.copy(path, target)


def get_context(expected, **kws):
    """
    Get the appropriate runtime context for executing test code based on
    whether the expected result is an exception or not.

    Parameters
    ----------
    expected : Exception or object
        The expected result of the test. If this object is an exception class,
        the context manager will be `pytest.raises(expected, **kws)`. Otherwise,
        it will be a null context manager.

    Examples
    --------
    For tests that are expected to succeed:
    >>> @pytest.mark.parametrize("expected", [1])
    ... def test_success(expected):
    ...     with get_context(expected):
    ...         assert expected == 1

    For tests that are expected to fail:
    The following example will raise an IndexError, which will get caught by
    the `pytest.raises` context manager, leading to a successful test
    >>> @pytest.mark.parametrize("expected", [LookupError])
    ... def test_expected_failure(expected):
    ...     with get_context(expected):
    ...         [][1]

    Returns
    -------
    contextlib.AbstractContextManager
    """
    if isinstance(expected, type):
        if isinstance(expected, contextlib.AbstractContextManager):
            return expected

        if issubclass(expected, BaseException):
            return pytest.raises(expected, **kws)

    return contextlib.nullcontext(expected)


# ---------------------------------------------------------------------------- #
# Helper classes for generating random skymodel data


class rv_constant:
    """
    A constant value "random variable" that emulates the
    `scipy.stats.distributions` API.
    """

    def __init__(self, value):
        self.value = value

    def rvs(self, n, *args, **kws):
        return np.full(n, self.value)


RVType = rv_frozen | rv_constant | None


@dataclass
class SkyModelGenerator:
    """
    Class for generating random skymodel data for testing purposes.

    SkyModelGenerator objects should be initialized with random variable
    distributions from `scipy.stats.distributions`) for each parameter of the
    skymodel, or with the custom `rv_constant` class for parameters that should
    take a constant value.
    - If a parameter is set to `None`, it will be ignored and not included in
      the generated skymodel.
    - If the method `get_<parameter>` is defined for a parameter, it will
      be used to generate the values for that parameter.

    Attributes
    ----------
    ra : RVType
        Distribution for Right ascension (in degrees). Default: uniform(0, 360).
    dec : RVType
        Distribution for Declination (in degrees). Default: uniform(-90, 90).
    i : RVType
        Distribution for Stokes I flux (in Jy). Default: uniform(0.001, 20).
    q : RVType
        Distribution for Stokes Q flux (in Jy). Default: constant(0).
    u : RVType
        Distribution for Stokes U flux (in Jy). Default: constant(0).
    v : RVType
        Distribution for Stokes V flux (in Jy). Default: constant(0).
    reference_frequency : RVType
        Distribution for Reference frequency (in Hz). Default: constant(1.44e8).
    spectral_index : RVType
        Distribution for Spectral index. Default: uniform(-1, 0).
    rotation_measure : RVType
        Distribution for Rotation measure. Default: constant(0).
    major_axis : RVType
        Distribution for Major axis in arcsec. Default: uniform(0.01, 20).
    minor_axis : RVType
        Distribution for Minor axis as a fraction of major axis. Default:
        uniform(0, 1).
    orientation : RVType
        Position angle in degrees. Default: uniform(0, 180).
    """

    ra: RVType = uniform(0, 360)
    dec: RVType = uniform(-90, 90)
    i: RVType = uniform(0.001, 20)
    q: RVType = rv_constant(0)
    u: RVType = rv_constant(0)
    v: RVType = rv_constant(0)
    reference_frequency: RVType = rv_constant(1.44e8)
    spectral_index: RVType = uniform(-1, 0)
    rotation_measure: RVType = rv_constant(0)
    major_axis: RVType = uniform(0.01, 20)
    minor_axis: RVType = uniform(0, 1)
    orientation: RVType = uniform(0, 180)

    def __call__(self, n_sources):
        """
        Generate a random skymodel.

        Parameters
        ----------
        n_sources : int
            The number of sources to generate in the skymodel.

        Returns
        -------
        samples : dict[str, numpy.ndarray]
            A dictionary with arrays of `n_sources` sampled values for each
            parameter.
        """
        samples = self.sample(n_sources)
        ra, dec = self.get_coords(n_sources, samples)
        samples.update(ra=ra, dec=dec)
        return {
            "name": self.get_names(n_sources, samples),
            "type": self.get_types(n_sources, samples),
            **samples,
        }

    def sample(self, n_sources):
        """
        Generate a random sample of sources from the specified distributions,
        returning a dictionary with parameter names as keys and arrays of
        sampled values as values.

        Parameters
        ----------
        n_sources : int
            The number of sources to generate in the skymodel.

        Returns
        -------
        samples : dict[str, numpy.ndarray]
            A dictionary with arrays of `n_sources` sampled values for each
            parameter.
        """
        samples = {}
        for name, dist in asdict(self).items():
            if dist is None:
                continue

            if sampler := getattr(self, f"get_{name}", None):
                samples[name] = sampler(n_sources, samples)
            else:
                samples[name] = dist.rvs(n_sources, random_state=RNG)

        return samples

    def get_coords(self, n_sources, state):
        """
        Generate the RA and Dec coordinates for the sources in the skymodel.
        """
        return format_coordinates(state["ra"], state["dec"], pad=True)

    def get_names(self, n_sources, state):
        """
        Generate unique source names for the specified number of sources.

        Each name is generated from the RA and Dec of the source using
        J-coordinate formatting, eg J012345+012345.

        Parameters
        ----------
        n_sources : int
            The number of sources to generate in the skymodel.
        state : dict
            A dictionary containing the random samples of other parameters.

        Returns
        -------
        names : numpy.ndarray
            An array of unique source names as strings.
        """
        ra, dec = np.array(
            np.char.rsplit([state["ra"], state["dec"]], ".", 1).tolist()
        )[..., 0]
        return np.char.add(
            "J",
            np.char.add(
                np.char.replace(ra, ":", ""),
                np.char.replace(dec, ".", ""),
            ),
        )

    def get_types(self, n_sources, state):
        """
        Generate source types for the specified number of sources.

        The source types are generated as "GAUSSIAN" for all sources, but this
        method can be modified to generate different types if needed.

        Parameters
        ----------
        n_sources : int
            The number of sources to generate in the skymodel.
        state : dict
            A dictionary containing the random samples of other parameters.

        Returns
        -------
        types : numpy.ndarray
            An array of source types as strings.
        """
        return np.full(n_sources, "GAUSSIAN")

    def get_minor_axis(self, n_sources, state):
        """
        Generate values for the minor axis of the sources, ensuring that they
        are smaller than the corresponding major axis values.

        Parameters
        ----------
        n_sources : int
            The number of sources to generate in the skymodel.
        state : dict
            A dictionary containing the random samples of other parameters.

        Returns
        -------
        minor_axis : numpy.ndarray
            An array of values for the minor axis of the sources.
        """
        major_axis = state["major_axis"]
        return major_axis * self.minor_axis.rvs(len(major_axis))

    def get_header(self, state):
        """
        Generate the makesourcedb format string for the header.

        Parameters
        ----------
        state : dict
            A dictionary containing the random samples of other parameters.

        Returns
        -------
        header : str
            The makesourcedb format string for the header.
        """
        return "FORMAT = " + ", ".join(
            np.char.replace(np.char.title(list(state.keys())), "_", "")
        )

    def to_file(self, filename, n_sources):
        """
        Generate a random skymodel and save it to a file.

        Parameters
        ----------
        filename : str or pathlib.Path
            The path to the file where the generated skymodel should be saved.
        n_sources : int
            The number of sources to generate in the skymodel.
        """
        samples = self(n_sources)
        np.savetxt(
            filename,
            np.column_stack(list(samples.values())),
            header=self.get_header(samples),
            delimiter=", ",
            fmt="%s",
        )




# ---------------------------------------------------------------------------- #
# Fixtures


@pytest.fixture(scope="module")
def test_data_path(request):
    """Path to the test data subfolder for the test module."""

    test_module_name = request.node.module.__name__
    test_data_path = request.config.resource_dir / test_module_name
    return (
        test_data_path
        if test_data_path.exists()
        else request.config.resource_dir
    )


@pytest.fixture
def midbands_ms(tmp_path, test_data_path):
    """Uncompresses test_midbands.ms into a temporary directory."""
    ms_name = "test_midbands.ms"
    untar(test_data_path / f"{ms_name}.tgz", tmp_path)
    return tmp_path / ms_name


@pytest.fixture(scope="module")
def test_ms_lofar_hba(test_data_path):
    """
    Fixture that provides the path to the LOFAR HBA mock measurement set. If
    the file does not exist, it will be downloaded and extracted from the
    specified URL.
    """
    path = test_data_path / "LOFAR_HBA_MOCK.ms"
    if path.exists():
        return path

    def filter_(member, _):
        return member.replace(name=Path(*Path(member.path).parts[1:]))

    with requests.get(LOFAR_HBA_URL, stream=True) as req:
        with tarfile.open(fileobj=req.raw, mode="r|bz2") as tarobj:
            tarobj.extractall(path=path, filter=filter_)

    return path


@pytest.fixture(scope="module")
def lofar_hba_skymodel(test_data_path):
    """Fixture that loads the skymodel data from the test data path."""
    return load(test_data_path / "LOFAR_HBA_MOCK.sky")


@pytest.fixture(scope="module")
def test_image_wcs(test_data_path):
    return WCS(fits.getheader(test_data_path / "test_image.fits"))


@pytest.fixture
def mock_moc():
    """Fixture that provides a mock MOC object for testing."""
    lon = Longitude([5, -5, -5, 5], u.deg)
    lat = Latitude([5, 5, -5, -5], u.deg)
    return mocpy.MOC.from_polygon(lon, lat)


@pytest.fixture
def cone_params():
    """Fixture that provides cone search parameters for testing."""
    return {"ra": 190.0, "dec": 44.0, "radius": 1.0}
