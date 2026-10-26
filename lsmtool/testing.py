"""
Utility functions used for testing.
"""

import contextlib
import io
import itertools as itt
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.coordinates import Angle
from astropy.utils.decorators import lazyproperty
from matplotlib.collections import EllipseCollection
from scipy.spatial.distance import cdist
from scipy.stats import binned_statistic_2d
from scipy.stats._distn_infrastructure import rv_sample
from scipy.stats.distributions import (
    cosine,
    rv_continuous,
    rv_frozen,
    uniform,
)

from .io import PathLike, load
from .skymodel import SkyModel
from .utils import format_coordinates


def get_context(expected: Exception | object, **kws):
    """
    Get the appropriate runtime context for executing test code based on
    whether the expected result is an exception or not.

    Parameters
    ----------
    expected : Exception or object
        The expected result of the test. If this object is an exception class,
        the context manager will be `pytest.raises(expected, **kws)`.
        Otherwise, it will be a do-nothing context manager
        `contextlib.nullcontext(expected)` that simply returns the input value.
        If `expected` is already a context manager, it will be returned as-is.

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
    context: contextlib.AbstractContextManager or object
        Object that acts like a context manager.
    """
    # pass exisiting context managers through unchanged
    if hasattr(expected, "__enter__") and hasattr(expected, "__exit__"):
        return expected

    if isinstance(expected, type):
        if issubclass(expected, Warning):
            # For warning classes, return a pytest.warns context manager
            return pytest.warns(expected, **kws)

        if issubclass(expected, BaseException):
            # For exception classes, return a pytest.raises context manager
            return pytest.raises(expected, **kws)

    return contextlib.nullcontext(expected)


def check_skymodels_equal(
    left_filename: PathLike,
    right_filename: PathLike,
    check_patch_names_sizes: bool = True,
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
    left = load(left_filename)
    right = load(right_filename)

    # Check the default (static) values
    return bool(
        left.getDefaultValues() == right.getDefaultValues()
        and check_columns_equal(left, right, check_patch_names_sizes)
        and check_patches_equal(left, right, check_patch_names_sizes)
    )


def check_columns_equal(
    left: SkyModel, right: SkyModel, check_patch_names_sizes: bool
):
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


def check_patches_equal(
    left: SkyModel, right: SkyModel, check_patch_names_sizes: bool
):
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
# Utilities for generating random skymodel data


class Constant(rv_sample):
    """
    A frozen constant distribution that always returns the same value.
    """

    def __init__(self, value: float):
        super().__init__(values=([value], [1]))


class Uniform(rv_continuous):
    def __new__(cls, a: float, b: float):
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
        return uniform(loc=a, scale=(b - a))


RandomState = Optional[int | np.random.RandomState | np.random.Generator]
RandomVariate = Optional[rv_frozen | Constant]


class Sampler:
    """
    Mixin class for sampling random variables defined as dataclass fields.
    """

    def sample(
        self,
        size: int,
        random_state: RandomState = None,
        ignore: list[str] = None,
        **kws,
    ):
        """
        Generate a random sample of numbers by looping over the distributions
        that are fields of the class. Retrun a dictionary keyed on the field
        names, with arrays of sampled numbers as values.

        Parameters
        ----------
        size : int
            The number of samples to generate.
        random_state : int, RandomState instance or None, optional
            The random state to use for reproducibility. If None (or np.random),
            the numpy.random.RandomState singleton is used.
        ignore : list of str, optional
            A list of parameter names to ignore when sampling.
        **kws
            Additional keyword arguments to pass to custom sampler methods.

        Returns
        -------
        samples : dict[str, numpy.ndarray]
            A dictionary with arrays of `n` sampled values for each parameter.
        """

        samples = {}
        fields = list(self.__dataclass_fields__)
        for name in sorted(set(fields) - set(ignore or []), key=fields.index):
            dist = getattr(self, name)
            if dist is None:
                continue

            samples[name] = dist.rvs(size=size, random_state=random_state)

            if sampler := getattr(self, f"get_{name}", None):
                # Custom sampler method defined for variable, use it to update
                # the sample
                samples[name] = sampler(samples, **kws)

        return samples


def _squeeze(samples):
    return {key: np.squeeze(datum) for key, datum in samples.items()}


class ConditionalSampler(Sampler):
    """
    Mixin class for sampling random variables and optionally rejecting samples
    based on user-defined conditions.
    """

    def sample(
        self,
        size: int,
        max_rejections: int = 1000,
        **kws,
    ):
        index = 0
        counter = itt.count()
        samples = np.full(size, np.nan, self.dtype())
        while index < size:
            count = next(counter)
            # Generate a new sample and check if it is valid. If valid, store
            # it. Otherwise, count as a rejection.
            sample = Sampler.sample(self, 1, **kws)

            # Since the samples are generated as arrays, we need to extract the
            # element for each key to get a single sample point.
            sample = _squeeze(sample)

            if self.validate(sample, samples):
                samples[index] = tuple(
                    sample[key] for key in samples.dtype.fields
                )
                index += 1

            elif count >= max_rejections:
                warnings.warn(
                    "Maximum number of sample rejections reached before "
                    "reaching requested number of samples.",
                    stacklevel=2,
                )
                break

        if index == 0:
            raise RuntimeError(
                f"No valid samples could be generated in {index + count} "
                "iterations."
            )

        if index < size:
            warnings.warn(
                f"Only {index} valid samples could be generated (out of "
                f"requested {size}) in {count} iterations.",
                stacklevel=2,
            )

        # Keep only the valid samples
        samples = samples[:index]

        # convert to dict
        return {key: samples[key] for key in self.__dataclass_fields__}

    def dtype(self, **dtypes):
        return [
            (key, dtypes.get(key, float)) for key in self.__dataclass_fields__
        ]

    def validate(self, _sample, _samples):
        return True


@dataclass
class SkyModelGenerator(Sampler):
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
    ra : RandomVariate
        Distribution for Right ascension (in degrees). Default: uniform(0, 360).
    dec : RandomVariate
        Distribution for Declination (in degrees). Default: uniform(-90, 90).
    i : RandomVariate
        Distribution for Stokes I flux (in Jy). Default: uniform(0.001, 20).
    q : RandomVariate
        Distribution for Stokes Q flux (in Jy). Default: constant(0).
    u : RandomVariate
        Distribution for Stokes U flux (in Jy). Default: constant(0).
    v : RandomVariate
        Distribution for Stokes V flux (in Jy). Default: constant(0).
    reference_frequency : RandomVariate
        Distribution for Reference frequency (in Hz). Default: constant(1.44e8).
    spectral_index : RandomVariate
        Distribution for Spectral index. Default: uniform(-1, 0).
    rotation_measure : RandomVariate
        Distribution for Rotation measure. Default: constant(0).
    major_axis : RandomVariate
        Distribution for Major axis in arcsec. Default: uniform(0.01, 20).
    minor_axis : RandomVariate
        Distribution for Minor axis as a fraction of major axis. Default:
        uniform(0, 1).
    orientation : RandomVariate
        Position angle in degrees. Default: uniform(0, 180).
    """

    ra: RandomVariate = Uniform(0, 360)
    dec: RandomVariate = Uniform(-90, 90)
    i: RandomVariate = Uniform(0.001, 20)
    q: RandomVariate = Constant(0)
    u: RandomVariate = Constant(0)
    v: RandomVariate = Constant(0)
    reference_frequency: RandomVariate = Constant(1.44e8)
    spectral_index: RandomVariate = Uniform(-1, 0)
    rotation_measure: RandomVariate = Constant(0)
    major_axis: RandomVariate = Uniform(0.01, 20)
    minor_axis: RandomVariate = Uniform(0, 1)
    orientation: RandomVariate = Uniform(0, 180)

    @property
    def column_order(self):
        """
        Return the order of the fields, including name, type, and patch first.
        """
        return ["name", "type", "patch", *self.__dataclass_fields__.keys()]

    def __call__(
        self,
        n_sources: int,
        n_patches: Optional[int] = None,
        **kws,
    ):
        """
        Generate a random skymodel.

        Parameters
        ----------
        n_sources : int
            The number of sources to generate in the skymodel.
        n_patches : int, optional
            The number of patches to divide the sky into. If None, no patching
            is applied.

        Returns
        -------
        samples : dict[str, numpy.ndarray]
            A dictionary with arrays of `n_sources` sampled values for each
            parameter.
        patches : np.ndarray
            The patch definitions as a (n, 5) array of strings.
        """
        samples, patches = self.sample(n_sources, n_patches=n_patches, **kws)

        ra, dec = self.get_coords(samples)
        samples.update(ra=ra, dec=dec)

        # Update the samples dictionary with the source names and types
        samples.update(
            name=self.get_names(samples), type=self.get_types(samples)
        )
        return samples, patches

    def sample(
        self,
        n_sources: int,
        n_patches: Optional[int] = None,
        **kws,
    ):
        """
        Generate a random sample of sources from the specified distributions,
        returning a dictionary with parameter names as keys and arrays of
        sampled values as values.

        Parameters
        ----------
        n_sources : int
            The number of sources to generate in the skymodel.
        n_patches : int, optional
            The number of patches to divide the sky into. If None, no patching
            is applied.

        Returns
        -------
        samples : dict[str, numpy.ndarray]
            A dictionary with arrays of `n_sources` sampled values for each
            parameter.
        patches : np.ndarray
            The patch definitions as a (n, 5) array of strings.
        """
        samples = Sampler.sample(self, n_sources, **kws)

        if n_patches:
            # Get patch definitions
            patch_defs, samples["patch"] = self.get_patches(samples, n_patches)
        else:
            patch_defs = None

        return samples, patch_defs

    def get_coords(self, samples: dict):
        """
        Generate the RA and Dec coordinates for the sources in the skymodel.

        Parameters
        ----------
        samples : dict
            A dictionary containing the random samples of other parameters.

        Returns
        -------
        coords : tuple[numpy.ndarray, numpy.ndarray]
            Arrays of RA and Dec coordinates for the sources.
        """
        return format_coordinates(samples["ra"], samples["dec"], pad=True)

    def get_names(self, samples: dict):
        """
        Generate unique source names for the specified number of sources.

        Each name is generated from the RA and Dec of the source using
        J-coordinate formatting, eg J012345+012345.

        Parameters
        ----------
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

    def get_types(self, samples: dict):
        """
        Generate source types for the specified number of sources.

        The source types are generated as "GAUSSIAN" for all sources, but this
        method can be modified to generate different types if needed.

        Parameters
        ----------
        samples : dict
            A dictionary containing the random samples of other parameters.

        Returns
        -------
        types : numpy.ndarray
            An array of source types as strings.
        """
        n_sources = len(samples["ra"])
        return np.full(n_sources, "GAUSSIAN")

    def get_patches(self, samples: dict, n_patches):
        """
        Get the patch definitions and the patch column for the given samples.

        Parameters`
        ----------
        samples : dict
            A dictionary containing the source samples with keys "ra" and "dec".
        n_patches : tuple of int
            The number of patches along each axis.

        Returns
        -------
        patch_defs : np.ndarray
            The patch definitions as a (n, 5) array of strings.
        patch_col : ndarray
            The patch column corresponding to each source.
        """
        stat, xedge, yedge, binnumber = binned_statistic_2d(
            (ra := samples["ra"]),
            (dec := samples["dec"]),
            np.ones(ra.shape),
            "count",
            n_patches,
            expand_binnumbers=True,
        )

        n_patches_total = np.prod(n_patches)
        patch_names = np.char.add(
            "Patch", np.arange(n_patches_total).astype(str)
        )

        xcenter = xedge[:-1] + np.diff(xedge) / 2
        ycenter = yedge[:-1] + np.diff(yedge) / 2
        ra_patch, dec_patch = np.array(list(itt.product(xcenter, ycenter))).T
        ra_patch, dec_patch = self.get_coords(
            {"ra": ra_patch, "dec": dec_patch}
        )

        patch_defs = np.empty((n_patches_total, 5), object)
        patch_defs[:, 0] = " "
        patch_defs[:, 1] = ""
        patch_defs[:, 2] = patch_names
        patch_defs[:, 3] = ra_patch
        patch_defs[:, 4] = dec_patch

        patch_col = patch_names.reshape(n_patches)[tuple(binnumber - 1)]

        return patch_defs, patch_col

    def get_minor_axis(self, samples: dict):
        """
        Generate values for the minor axis of the sources, ensuring that they
        are smaller than the corresponding major axis values.

        Parameters
        ----------
        samples : dict
            A dictionary containing the random samples of other parameters.

        Returns
        -------
        minor_axis : numpy.ndarray
            An array of values for the minor axis of the sources.
        """
        major_axis = samples["major_axis"]
        return major_axis * self.minor_axis.rvs(size=major_axis.size)

    def get_header(self, samples: dict):
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
        keys = [key for key in self.column_order if key in samples]
        return "FORMAT = " + ", ".join(
            np.char.replace(np.char.title(keys), "_", "")
        )

    def to_array(self, samples: dict):
        """
        Convert the samples dictionary to a numpy array by stacking the columns
        in the correct order.

        Parameters
        ----------
        samples : dict
            A dictionary containing the random samples of other parameters.

        Returns
        -------
        array : numpy.ndarray
            A 2D array where each column corresponds to a parameter in the
            specified column order.
        """
        # ensure correct column order
        return np.column_stack(
            [samples[key] for key in self.column_order if key in samples]
        )

    def to_file(
        self,
        filename: PathLike,
        n_sources: int,
        n_patches: Optional[int] = None,
        random_state: RandomState = None,
        **kws,
    ):
        """
        Generate a random skymodel and save it to a file.

        Parameters
        ----------
        filename : str or pathlib.Path
            The path to the file where the generated skymodel should be saved.
            File will be saved as text file in CSV format.
        n_sources : int
            The number of sources to generate in the skymodel.
        n_patches : int, optional
            The number of patches to divide the sky into. If None, no patching
            is applied.
        random_state : int, RandomState instance or None, optional
            The random state to use for reproducibility. If None (or np.random),
            the numpy.random.RandomState singleton is used.
        **kws
            Additional keyword arguments passed to the sampler.
        """

        # Get samples
        samples, patches = self(
            n_sources, n_patches, random_state=random_state, **kws
        )

        # Save file
        np.savetxt(
            filename,
            self.to_array(samples),
            header=self.get_header(samples),
            delimiter=", ",
            fmt="%s",
        )

        # Add patch definitions if available
        if patches is not None:
            lines = Path(filename).read_text().splitlines()
            patch_defs = io.StringIO()
            np.savetxt(patch_defs, patches, delimiter=", ", fmt="%s")

            content = "\n".join(
                [
                    *lines[:1],
                    "",
                    *patch_defs.getvalue().splitlines(),
                    *lines[1:],
                ]
            )
            Path(filename).write_text(content)


@dataclass
class CovarianceMatrixGenerator(ConditionalSampler):
    """
    Generator for covariance matrices with specified standard deviations and
    correlation. Ensures that generated matrices are non-singular.
    """

    std_ra: RandomVariate = Uniform(0.5, 1)
    std_dec: RandomVariate = Uniform(0.5, 1)
    cor: RandomVariate = cosine(scale=1 / np.pi)

    def validate(self, sample: dict, _samples: dict):
        sigma = self.get_matrix(**sample)
        return np.linalg.det(sigma) != 0

    def get_matrix(self, std_ra, std_dec, cor):

        cov = cor * std_ra * std_dec
        sigma = np.squeeze([[std_ra**2, cov], [cov, std_dec**2]])

        if sigma.ndim > 2:
            sigma = np.moveaxis(sigma, -1, 0)

        return sigma

    def rvs(self, size: int, *args, **kws):
        samples = Sampler.sample(self, size, *args, **kws)
        # All information is contained in the covariance matrix
        return self.get_matrix(**samples)


@dataclass
class ClusterGenerator(ConditionalSampler):
    """
    Generator for cluster positions with associated covariance matrices.
    Ensures that cluster centers are sufficiently far apart based on the
    Mahalanobis distance.
    """

    MIN_DISTANCE_SIGMA = 6.0

    ra: RandomVariate = Uniform(0, 360)
    dec: RandomVariate = Uniform(-90, 90)
    cov: RandomVariate = field(default_factory=CovarianceMatrixGenerator)

    def sample(self, n: int, *args, **kws):
        samples = super().sample(n, *args, **kws)
        samples["pos"] = np.array([samples.pop("ra"), samples.pop("dec")]).T
        return samples

    def dtype(self):
        return super().dtype(cov=(float, (2, 2)))

    def validate(self, sample: dict, samples: dict):
        # ensure cluster centres are far enough appart
        positions = np.array([samples["ra"], samples["dec"]]).T
        not_yet_generated = np.isnan(positions).any(1)
        if not_yet_generated.all():
            return True

        positions = positions[~not_yet_generated]
        covariances = samples["cov"][~not_yet_generated]

        pos = np.array([[sample["ra"], sample["dec"]]])
        cov = sample["cov"]

        # Check distance of the new sample to existing cluster centers using
        # Mahalanobis distance
        mahalanobis_distance = cdist(
            pos, positions, metric="mahalanobis", VI=np.linalg.inv(covariances)
        )
        if (mahalanobis_distance < self.MIN_DISTANCE_SIGMA).any():
            return False

        # Check distance of existing cluster centers to the new sample
        mahalanobis_distance = cdist(
            pos, positions, metric="mahalanobis", VI=np.linalg.inv(cov)
        )
        return (mahalanobis_distance >= self.MIN_DISTANCE_SIGMA).all()


class Cluster:
    """
    Represents a single cluster for generating sources for a sky model.
    """

    def __init__(self, pos, cov):
        """
        Initialize the `Cluster` source generator.

        Sources are generated from a multivariate normal probability
        distribution centered a at the given position and with given
        covariance matrix.

        Parameters
        ----------
        pos : array-like
            The position of the cluster center.
        cov : array-like
            The 2x2 covariance matrix of the cluster.
        """
        self.position = np.array(pos)
        self.covariance = np.array(cov)

    @property
    def position(self):
        """The position of the cluster centre in RA, Dec coordinates."""
        return self._position

    @position.setter
    def position(self, position):
        position = np.array(position)
        if position.shape != (2,):
            raise ValueError("Position must be a 2-element array.")
        self._position = position

    @lazyproperty
    def covariance(self):
        """The 2x2 covariance matrix of the source distribution."""
        return self.sigma

    @covariance.setter
    def covariance(self, sigma):

        covariance_matrix = np.array(sigma, dtype=float)
        if covariance_matrix.shape != (2, 2):
            raise ValueError("Covariance matrix must be 2x2")

        if np.linalg.det(covariance_matrix) == 0:
            raise ValueError("Covariance matrix is singular")

        self.sigma = covariance_matrix

        # Invalidate cached properties that depend on the covariance matrix
        del self.semi_axes
        del self.angle

    @property
    def distribution(self):
        """The multivariate normal distribution representing the cluster."""
        return multivariate_normal(mean=self.position, cov=self.covariance)

    @property
    def rvs(self):
        return self.distribution.rvs

    @property
    def n_clusters(self):
        """The number of clusters."""
        return len(self.position)

    @property
    def var_ra(self):
        """Variance of the RA coordinate for all clusters."""
        return self.covariance[0, 0]

    @property
    def var_dec(self):
        """Variance of the Dec coordinate for all clusters."""
        return self.covariance[1, 1]

    @property
    def cov(self):
        """Covariance between the RA and Dec coordinates for all clusters."""
        return self.covariance[0, 1]

    @lazyproperty
    def semi_axes(self):
        """
        Compute the semi-major and semi-minor axes of the ellipses
        representing the clusters based on their covariance matrices.

        Returns
        -------
        a, b : np.ndarray
            Semi-major  and semi-minor axes for all cluster.
        """

        # There is some degeneracy in the covariance matrix representation vs
        # the shape parameter representation of the ellipse. The semi-major axis
        # is defined as the largest of the two axis, and the rotation angle of
        # the ellipse is the angle from the positive x-axis to the semi-major
        # axis. Therefore, if std_ra > std_dec, we may swap the two axes and
        # invert the angle to obtain an identical distribution.
        eigenvalues, _ = np.linalg.eigh(self.covariance)
        semi_axes = np.sqrt(eigenvalues)
        a = semi_axes.max()
        b = semi_axes.min()
        return a, b

    @property
    def semi_major_axis(self):
        """
        The semi-major axes of the iso-probability ellipse of the source
        distribution. This is defined as the larger of the two standard
        deviations of the source distribution.
        """
        a, _ = self.semi_axes
        return a

    @property
    def semi_minor_axis(self):
        """
        The semi-minr axes of the iso-probability ellipse of the source
        distribution. This is defined as the smaller of the two standard
        deviations of the source distribution.
        """
        _, b = self.semi_axes
        return b

    @lazyproperty
    def angle(self):
        """
        Rotation angle of the the iso-probability ellipse of the source
        distribution. This is the counter-clockwise angle from the positive
        x-axis to the semi-major axis of the ellipse.
        """
        return np.degrees(
            0.5 * np.arctan2(2 * self.cov, (self.var_ra - self.var_dec))
        )


class AttrVector:
    """
    Descriptor class for vectorised attribute lookup.
    """

    def __init__(self, name: str):
        self.name = name

    def __get__(self, instance: Optional["SourceClusters"], _):
        if instance is None:
            return self

        return np.array(
            [getattr(cluster, self.name) for cluster in instance.clusters]
        )


class SourceClusters:
    """
    A class representing clustered sky coordinates.
    """

    positions = AttrVector("position")
    covariances = AttrVector("covariance")
    semi_major_axes = AttrVector("semi_major_axis")
    semi_minor_axes = AttrVector("semi_minor_axis")
    angles = AttrVector("angle")

    @classmethod
    def generated(
        cls,
        n_clusters: int,
        probabilities=None,
        random_state: RandomState = None,
        **config,
    ):
        """
        Generates a random set of clusters using the `ClusterGenerator` class,
        and initializes a `SourceClusters` instance with the generated cluster
        parameters.

        Parameters
        ----------
        n_clusters : int
            _description_
        probabilities : list of float, optional
            The probability amplitude associated with each cluster. This
            determines the fraction of sources that will fall withing each
            cluster. If None, sources are equally likely to come from any
            cluster.
        random_state : int, RandomState instance or None, optional
            The random state to use for reproducibility. If None (or
            np.random), the numpy.random.RandomState singleton is used.

        Returns
        -------
        SourceClusters
            Object contain the generated source cluster definitions.
        """
        clusters = ClusterGenerator(**config)
        cluster_params = clusters.sample(n_clusters, random_state=random_state)

        # Generate source coordinates from clusters
        return cls(
            positions=cluster_params["pos"],
            covariances=cluster_params["cov"],
            probabilities=probabilities,
        )

    def __init__(self, positions, covariances, probabilities=None):
        """
        Initialize the clustered coordinates with centres and covariances.

        Parameters
        ----------
        positions : list of array-like
            The locations of the clusters. Each element in the sequence
            represents the mean of the 2-dimensional Gaussian distribution from
            which samples will be genrated from that cluster.
        covariances : list of array-like
            Sequence of symmetric positive (semi)definite covariance matrices
            for the multivariate normal distributions of the clusters. This
            sequence must have the same length as the centres sequence.
        probabilities : list of float, optional
            The probability amplitude associated with each cluster. This
            determines the fraction of sources that will fall withing each
            cluster. If None, sources are equally likely to come from any
            cluster.
        """

        if len(positions) != len(covariances):
            raise ValueError(
                "centres and covariances must have the same length (the "
                "number of clusters)."
            )
        if probabilities is not None and len(probabilities) != len(positions):
            raise ValueError(
                "probabilities must have the same length as positions."
            )

        self.clusters = [
            Cluster(pos, cov)
            for pos, cov in zip(positions, covariances, strict=True)
        ]
        self.probabilities = probabilities

    @property
    def n_clusters(self):
        return len(self.clusters)

    def sample(self, size: int, random_state: RandomState = None, **_kws):
        """
        Generate random samples for the clustered sources.

        Parameters
        ----------
        size : int
            The number of samples to generate.
        random_state : int, RandomState instance or None, optional
            The random state to use for reproducibility. If None (or np.random),
            the numpy.random.RandomState singleton is used.

        Returns
        -------
        coordinates : np.ndarray
            An array of shape (2, n) containing the generated coordinates.
        labels : np.ndarray
            An array of shape (n,) containing the cluster labels for each
            generated coordinate.
        """
        rng = np.random.default_rng(random_state)
        cluster_indices = rng.choice(
            self.n_clusters,
            p=self.probabilities,
            size=size,
        )
        n_points_per_cluster = np.unique_counts(cluster_indices)
        coords = np.empty((size, 2))
        labels = np.empty(size, dtype=int)
        index = 0
        for cluster_index, n_points in zip(*n_points_per_cluster, strict=True):
            if n_points:
                cluster = self.clusters[cluster_index]
                labels[index : index + n_points] = cluster_index
                coords[index : index + n_points] = cluster.rvs(size=n_points)
                index += n_points

        # Generate patch names from cluster labels
        labels = np.char.add("Patch", labels.astype(int).astype(str))

        return {"ra": coords[:, 0], "dec": coords[:, 1], "patch": labels}

    def plot(
        self,
        n_sigma: float = 3.0,
        axes: Optional[plt.Axes] = None,
        mark_centres: Optional[bool | dict] = None,
        **kws,
    ):
        """
        Plot the covariance ellipses of the clusters and optionally mark the
        cluster centres.

        Parameters
        ----------
        n_sigma : int, optional
            The contour level at which to draw the ellipse, by default 3.
        axes : matplotlib.axes.Axes, optional
            Axes on which to draw the ellipses. If None (the default), a new
            figure and axes will be created.
        mark_centres : dict or bool or None, optional
            Configuration for marking the cluster centres. If a dictionary is
            provided, it will be passed as keyword arguments to the plotting
            function. If True or None (the default), default marker settings
            will be used. If False, cluster centres will not be marked.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plotted ellipses.
        """

        if axes is None:
            fig, ax = plt.subplots()
        else:
            ax = axes
            fig = ax.figure

        # Plot centres
        if mark_centres is not False:
            mark_centres = mark_centres or {}
            mark_centres |= {
                "color": "k",
                "marker": ".",
                "linestyle": "",
            }
            ax.plot(*self.positions.T, **mark_centres)

        # Plot ellipses representing the covariance of each cluster
        kws |= {
            "linewidth": 1,
            "facecolor": "none",
        }
        ec = EllipseCollection(
            2 * self.semi_major_axes * n_sigma,
            2 * self.semi_minor_axes * n_sigma,
            self.angles,
            units="xy",
            offset_transform=ax.transData,
            offsets=self.positions,
            **kws,
        )
        ax.add_collection(ec)

        ax.set_aspect("equal")
        ax.grid()

        # #
        # for dist in self.distributions:
        #     xy = dist.rvs(100)
        #     ax.plot(*xy.T, ".", alpha=0.35)

        return fig


@dataclass(kw_only=True)
class ClusteredSkyModelGenerator(SkyModelGenerator):
    """
    A class for generating grouped sky models.

    This class extends the functionality of the basic SkyModelGenerator to
    support generating sources in clusters at various locations on the sky in
    order to fascilitate testing of clustering algorithms in LSMTool.
    """

    # SourceClusters object generates coordinates, so we can set the
    # RA and Dec generators to None
    clusters: SourceClusters

    ra = None
    dec = None

    def sample(
        self,
        n_sources: int,
        **kws,
    ):
        # Get coordinates by generating them from the SourceClusters object
        clustered_data = self.clusters.sample(n_sources, **kws)

        # Generate the remaining columns
        samples, _ = super().sample(n_sources, ignore=["clusters"], **kws)
        # Create patch definitions. Patch names column is already generated in
        # the clustered_data dict
        samples |= clustered_data
        patch_defs, _ = self.get_patches(samples)

        return samples, patch_defs

    def get_patches(self, samples: dict, _n_patches=None):

        patch_names = list(set(samples["patch"].tolist()))

        ra_patch, dec_patch = self.get_coords(
            {
                "ra": self.clusters.positions[:, 0],
                "dec": self.clusters.positions[:, 1],
            }
        )

        patch_defs = np.empty((len(patch_names), 5), object)
        patch_defs[:, 0] = " "
        patch_defs[:, 1] = ""
        patch_defs[:, 2] = patch_names
        patch_defs[:, 3] = ra_patch
        patch_defs[:, 4] = dec_patch

        return patch_defs, patch_names
