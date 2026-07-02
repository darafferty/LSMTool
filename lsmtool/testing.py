"""
Utility functions used for testing.
"""

import contextlib
from dataclasses import asdict, dataclass

import numpy as np
import pytest
from astropy.coordinates import Angle
from scipy.stats.distributions import rv_frozen, uniform

from .io import load
from .utils import format_coordinates


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


def check_skymodels_equal(
    left_filename, right_filename, check_patch_names_sizes=True
):
    """
    Compares the contents of two skymodels to check for equality.

    This function loads two skymodels and compares their contents, ignoring
    comments since they contain log messages which vary depending on run time.

    Parameters
    ----------
    left_filename : str or pathlib.Path
        Path to the first skymodel file.
    right_filename : str or pathlib.Path
        Path to the second skymodel file.
    check_patch_names_sizes : bool
        Whether to check patch names and sizes.
    """
    left = load(str(left_filename))
    right = load(str(right_filename))

    # Check the default (static) values
    return bool(
        left.getDefaultValues() == right.getDefaultValues()
        and check_columns_equal(left, right, check_patch_names_sizes)
        and check_patches_equal(left, right, check_patch_names_sizes)
    )


def check_columns_equal(left, right, check_patch_names_sizes):
    """
    Checks the columns of two skymodels for equality.

    Parameters
    ----------
    left : Skymodel
        The first skymodel to compare.
    right : Skymodel
        The second skymodel to compare.
    check_patch_names_sizes : bool
        Whether to check patch names and sizes.

    Returns
    -------
    bool
        True if the columns are considered equal, False otherwise.
    """
    # Check column names (ignoring the Patch column if needed)
    ignore = set() if check_patch_names_sizes else {"Patch"}
    left_column_names = set(left.getColNames()) - ignore
    right_column_names = set(right.getColNames()) - ignore
    if left_column_names != right_column_names:
        return False

    for name in left_column_names:
        left_values = left.getColValues(name)
        right_values = right.getColValues(name)
        equals = (
            np.isclose
            if np.issubdtype(left_values.dtype, np.inexact)
            else np.equal
        )
        if not equals(left_values, right_values).all():
            return False

    return True


def check_patches_equal(left, right, check_patch_names_sizes):
    """
    Checks the patches of two skymodels for equality.

    Parameters
    ----------
    left : Skymodel
        The first skymodel to compare.
    right : Skymodel
        The second skymodel to compare.
    check_patch_names_sizes : bool
        Whether to check patch names and sizes.

    Returns
    -------
    bool
        True if the patches are considered equal, False otherwise.
    """

    # Check patch positions. If they are defined (not None), check if they are
    # the same shape, and if so, if they are approximately equal to within some
    # tolerance
    patch_positions_equal = True
    if (left_patch_pos := left.getPatchPositions()) and (
        right_patch_pos := right.getPatchPositions()
    ):
        patch_positions_equal = len(left_patch_pos) == len(
            right_patch_pos
        ) and np.allclose(
            # Need to convert list of lists of Angle object to Angle array for
            # element-wise comparison to work as expected here
            Angle(left_patch_pos.values()),
            Angle(right_patch_pos.values()),
        )
    if patch_positions_equal and check_patch_names_sizes:
        # Check patch names and sizes
        return np.all(left.getPatchNames() == right.getPatchNames()) and np.all(
            left.getPatchSizes() == right.getPatchSizes()
        )
    return patch_positions_equal


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


def uniform_range(a, b):
    """
    Return a frozen uniform distribution with the specified range.

    Parameters
    ----------
    a : float
        The lower bound of the uniform distribution.
    b : float
        The upper bound of the uniform distribution.

    Returns
    -------
    rv_frozen
        A frozen uniform distribution object.
    """
    return uniform(loc=a, scale=b - a)


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

    ra: RVType = uniform_range(0, 360)
    dec: RVType = uniform_range(-90, 90)
    i: RVType = uniform_range(0.001, 20)
    q: RVType = rv_constant(0)
    u: RVType = rv_constant(0)
    v: RVType = rv_constant(0)
    reference_frequency: RVType = rv_constant(1.44e8)
    spectral_index: RVType = uniform_range(-1, 0)
    rotation_measure: RVType = rv_constant(0)
    major_axis: RVType = uniform_range(0.01, 20)
    minor_axis: RVType = uniform_range(0, 1)
    orientation: RVType = uniform_range(0, 180)

    def __call__(self, n_sources, random_state=None):
        """
        Generate a random skymodel.

        Parameters
        ----------
        n_sources : int
            The number of sources to generate in the skymodel.
        random_state : int, RandomState instance or None, optional
            The random state to use for reproducibility. If None (or
            np.random), the numpy.random.RandomState singleton is used.

        Returns
        -------
        samples : dict[str, numpy.ndarray]
            A dictionary with arrays of `n_sources` sampled values for each
            parameter.
        """
        samples = self.sample(n_sources, random_state)
        ra, dec = self.get_coords(n_sources, samples)
        samples.update(ra=ra, dec=dec)
        return {
            "name": self.get_names(n_sources, samples),
            "type": self.get_types(n_sources, samples),
            **samples,
        }

    def sample(self, n_sources, random_state=None):
        """
        Generate a random sample of sources from the specified distributions,
        returning a dictionary with parameter names as keys and arrays of
        sampled values as values.

        Parameters
        ----------
        n_sources : int
            The number of sources to generate in the skymodel.
        random_state : int, RandomState instance or None, optional
            The random state to use for reproducibility. If None (or
            np.random), the numpy.random.RandomState singleton is used.

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
                samples[name] = dist.rvs(n_sources, random_state=random_state)

        return samples

    def get_coords(self, n_sources, samples):
        """
        Generate the RA and Dec coordinates for the sources in the skymodel.
        """
        return format_coordinates(samples["ra"], samples["dec"], pad=True)

    def get_names(self, n_sources, samples):
        """
        Generate unique source names for the specified number of sources.

        Each name is generated from the RA and Dec of the source using
        J-coordinate formatting, eg J012345+012345.

        Parameters
        ----------
        n_sources : int
            The number of sources to generate in the skymodel.
        samples : dict
            A dictionary containing the random samples of other parameters.

        Returns
        -------
        names : numpy.ndarray
            An array of unique source names as strings.
        """
        ra, dec = np.array(
            np.char.rsplit([samples["ra"], samples["dec"]], ".", 1).tolist()
        )[..., 0]
        return np.char.add(
            "J",
            np.char.add(
                np.char.replace(ra, ":", ""),
                np.char.replace(dec, ".", ""),
            ),
        )

    def get_types(self, n_sources, samples):
        """
        Generate source types for the specified number of sources.

        The source types are generated as "GAUSSIAN" for all sources, but this
        method can be modified to generate different types if needed.

        Parameters
        ----------
        n_sources : int
            The number of sources to generate in the skymodel.
        samples : dict
            A dictionary containing the random samples of other parameters.

        Returns
        -------
        types : numpy.ndarray
            An array of source types as strings.
        """
        return np.full(n_sources, "GAUSSIAN")

    def get_minor_axis(self, n_sources, samples):
        """
        Generate values for the minor axis of the sources, ensuring that they
        are smaller than the corresponding major axis values.

        Parameters
        ----------
        n_sources : int
            The number of sources to generate in the skymodel.
        samples : dict
            A dictionary containing the random samples of other parameters.

        Returns
        -------
        minor_axis : numpy.ndarray
            An array of values for the minor axis of the sources.
        """
        major_axis = samples["major_axis"]
        return major_axis * self.minor_axis.rvs(len(major_axis))

    def get_header(self, samples):
        """
        Generate the makesourcedb format string for the header.

        Parameters
        ----------
        samples : dict
            A dictionary containing the random samples of other parameters.

        Returns
        -------
        header : str
            The makesourcedb format string for the header.
        """
        return "FORMAT = " + ", ".join(
            np.char.replace(np.char.title(list(samples.keys())), "_", "")
        )

    def to_file(self, filename, n_sources, random_state=None):
        """
        Generate a random skymodel and save it to a file.

        Parameters
        ----------
        filename : str or pathlib.Path
            The path to the file where the generated skymodel should be saved.
        n_sources : int
            The number of sources to generate in the skymodel.
        random_state : int, RandomState instance or None, optional
            The random state to use for reproducibility. If None (or
            np.random), the numpy.random.RandomState singleton is used.
        """
        samples = self(n_sources, random_state)
        np.savetxt(
            filename,
            np.column_stack(list(samples.values())),
            header=self.get_header(samples),
            delimiter=", ",
            fmt="%s",
        )
