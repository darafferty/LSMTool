"""
Utility functions used for testing.
"""

import contextlib
import io
import itertools as itt
import numbers
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypeAlias, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.coordinates import Angle
from astropy.utils.decorators import lazyproperty
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Ellipse
from scipy.stats import binned_statistic_2d, multivariate_normal
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

# ---------------------------------------------------------------------------- #
# Type aliases

ST = TypeVar("ST")
ArrayLike1D: TypeAlias = Sequence[ST]
ArrayLike2D: TypeAlias = Sequence[ArrayLike1D]
ArrayLike3D: TypeAlias = Sequence[ArrayLike2D]

# The following array types are defined in numpy >=2.6
Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[ST]]
Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[ST]]
Array3D: TypeAlias = np.ndarray[tuple[int, int, int], np.dtype[ST]]

RandomVariate: TypeAlias = Optional[rv_frozen | rv_sample]
RandomState: TypeAlias = Optional[
    numbers.Integral | np.random.RandomState | np.random.Generator
]
SampleDict: TypeAlias = dict[str, np.ndarray]
SampleSize: TypeAlias = numbers.Integral | tuple[numbers.Integral, ...]

PatchDefinitions: TypeAlias = Optional[Array2D[str]]


# ---------------------------------------------------------------------------- #


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
) -> bool:
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
) -> bool:
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
) -> bool:
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

    def __init__(self, value: numbers.Real):
        super().__init__(values=([value], [1]))


class Uniform(rv_continuous):
    """
    A frozen uniform distribution over a specified range.
    """

    def __new__(cls, a: numbers.Real, b: numbers.Real):
        """
        Return a frozen uniform distribution with the specified range.

        Parameters
        ----------
        a : numbers.Real
            The lower bound of the uniform distribution.
        b : numbers.Real
            The upper bound of the uniform distribution.

        Returns
        -------
        rv_frozen
            A frozen uniform distribution object.
        """
        return uniform(loc=a, scale=(b - a))


class Sampler:
    """
    Mixin class for sampling random variables defined as dataclass fields.
    """

    def sample(
        self,
        size: SampleSize,
        random_state: RandomState = None,
        ignore: list[str] = None,
        **kws,
    ) -> SampleDict:
        """
        Generate samples for the distributions defined by the dataclass fields.

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
        dict[str, numpy.ndarray]
            Sampled arrays keyed by field name.
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


def _squeeze(samples: SampleDict) -> SampleDict:
    return {key: np.squeeze(datum) for key, datum in samples.items()}


class ConditionalSampler(Sampler):
    """
    Mixin class for sampling random variables and optionally rejecting samples
    based on user-defined conditions.
    """

    def sample(
        self,
        size: SampleSize,
        max_rejections: numbers.Integral = 100,
        random_state: RandomState = None,
        **kws,
    ) -> SampleDict:

        if isinstance(random_state, int):
            # Need to deal with integer random states here since we will reuse
            # the random state multiple times
            random_state = np.random.RandomState(random_state)

        index = 0
        counter = itt.count()
        samples = np.full(size, np.nan, self.dtype())
        while index < size:
            count = next(counter)
            # Generate a new sample and check if it is valid. If valid, store
            # it. Otherwise, count as a rejection.
            sample = Sampler.sample(self, 1, random_state=random_state, **kws)

            # Since the samples are generated as arrays, we need to extract the
            # element for each key to get a single sample point.
            sample = _squeeze(sample)

            if self.validate(sample, samples):
                samples[index] = tuple(
                    sample[key] for key in samples.dtype.fields
                )
                index += 1

            elif count - index >= max_rejections:
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

    def dtype(
        self, **dtypes: dict[str, np.typing.DTypeLike]
    ) -> list[tuple[str, type]]:
        return [
            (key, dtypes.get(key, float)) for key in self.__dataclass_fields__
        ]

    def validate(self, _sample: SampleDict, _samples: SampleDict) -> bool:
        return True


@dataclass(kw_only=True)
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
    def column_order(self) -> list[str]:
        """
        Return the order of the fields, including name, type, and patch first.
        """
        return ["name", "type", "patch", *self.__dataclass_fields__.keys()]

    def __call__(
        self,
        n_sources: numbers.Integral,
        n_patches: Optional[int] = None,
        **kws,
    ) -> tuple[SampleDict, PatchDefinitions]:
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
        n_sources: numbers.Integral,
        n_patches: Optional[numbers.Integral] = None,
        **kws,
    ) -> tuple[SampleDict, PatchDefinitions]:
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

    def get_coords(
        self, samples: SampleDict
    ) -> tuple[Array1D[str], Array1D[str]]:
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

    def get_names(self, samples: SampleDict) -> np.ndarray[str]:
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

    def get_types(self, samples: SampleDict) -> Array1D[str]:
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

    def get_patches(
        self, samples: SampleDict, n_patches: numbers.Integral
    ) -> tuple[PatchDefinitions, Array1D[str]]:
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
        patch_col = patch_names.reshape(n_patches)[tuple(binnumber - 1)]

        xcentre = xedge[:-1] + np.diff(xedge) / 2
        ycentre = yedge[:-1] + np.diff(yedge) / 2
        ra, dec = np.array(list(itt.product(xcentre, ycentre))).T
        ra, dec = self.get_coords({"ra": ra, "dec": dec})

        patch_defs = self._patch_definitions(
            n_patches_total, patch_names, ra, dec
        )

        return patch_defs, patch_col

    def _patch_definitions(
        self,
        n_patches: numbers.Integral,
        patch_names: Array1D[str],
        ra: Array1D[str],
        dec: Array1D[str],
    ) -> PatchDefinitions:
        patch_defs = np.empty((n_patches, 5), object)
        patch_defs[:, 0] = " "
        patch_defs[:, 1] = ""
        patch_defs[:, 2] = patch_names
        patch_defs[:, 3] = ra
        patch_defs[:, 4] = dec
        return patch_defs

    def get_minor_axis(self, samples: SampleDict) -> Array1D[float]:
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

    def get_header(self, samples: SampleDict) -> str:
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

    def to_array(self, samples: SampleDict) -> Array2D[object]:
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
        n_sources: numbers.Integral,
        n_patches: Optional[int] = None,
        random_state: RandomState = None,
        **kws,
    ) -> None:
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

    def validate(self, sample: SampleDict, _samples: SampleDict):
        sigma = self.get_matrix(**sample)
        return np.linalg.det(sigma) != 0

    def get_matrix(
        self,
        std_ra: ArrayLike1D[numbers.Real],
        std_dec: ArrayLike1D[numbers.Real],
        cor: ArrayLike1D[numbers.Real],
    ) -> Array2D[float] | Array3D[float]:

        cov = cor * std_ra * std_dec
        sigma = np.squeeze([[std_ra**2, cov], [cov, std_dec**2]])

        if sigma.ndim > 2:
            sigma = np.moveaxis(sigma, -1, 0)

        return sigma

    def rvs(self, size: SampleSize, *args, **kws) -> Array3D[float]:
        samples = super().sample(size, *args, **kws)
        # All information is contained in the covariance matrix
        return self.get_matrix(**samples)


def mahalanobis_distance(
    pos: Array1D[float],
    positions: Array2D[float],
    covariances: Array2D[float],
) -> Array1D[float]:
    """
    Fast computation of the Mahalanobis distances from point to a set of
    reference poistions (cluster centres), given their covariance matrices.

    Parameters
    ----------
    pos : array-like
        Coordinate position (RA, Dec) as array-like of size 2.
    positions : array-like
        Reference positions (RA, Dec) as array-like of size (N, 2).
    covariances : array-like
        Covariance matrices as array-like of of size (N, 2, 2).

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing the Mahalanobis distance of the reference
        points to the input position.
    """
    delta_ra, delta_dec = (np.atleast_2d(positions) - np.atleast_2d(pos)).T
    delta_ra[delta_ra > 180] -= 360
    r = np.array([delta_ra, delta_dec]).T
    precisions = np.linalg.inv(covariances)
    return np.sqrt((r * np.einsum("ijk,ij...->ik", precisions, r)).sum(-1))


@dataclass
class ClusterGenerator(ConditionalSampler):
    """
    Generator for cluster positions with associated covariance matrices.
    Ensures that cluster centres are sufficiently far apart based on the
    Mahalanobis distance.
    """

    MIN_DISTANCE_SIGMA = 6.0

    ra: RandomVariate = Uniform(0, 360)
    dec: RandomVariate = Uniform(-90, 90)
    cov: RandomVariate = field(default_factory=CovarianceMatrixGenerator)

    def sample(self, size: SampleSize, *args, **kws) -> SampleDict:
        samples = super().sample(size, *args, **kws)
        samples["pos"] = np.array([samples.pop("ra"), samples.pop("dec")]).T
        return samples

    def dtype(self) -> list[tuple[str, type]]:
        return super().dtype(cov=(float, (2, 2)))

    def validate(self, sample: SampleDict, samples: SampleDict) -> bool:
        # ensure cluster centres are far enough appart
        positions = np.array([samples["ra"], samples["dec"]]).T
        not_yet_generated = np.isnan(positions).any(1)
        if not_yet_generated.all():
            # This is the first sample
            return True

        # Check mahalanobis distance of the new sample to existing cluster
        # centres using covarinance matrices of the existing clusters
        pos = np.array([[sample["ra"], sample["dec"]]])
        positions = positions[~not_yet_generated]
        covariances = samples["cov"][~not_yet_generated]
        distances = mahalanobis_distance(pos, positions, covariances)
        if (distances < self.MIN_DISTANCE_SIGMA).any():
            return False

        # Check mahalanobis distance of existing cluster centres to the new
        # sample, with the covariance matrix of the new sample
        r = positions - pos
        p = np.linalg.inv(sample["cov"])
        distance = np.sqrt((r * (p @ r.T).T).sum(1))
        return (distance >= self.MIN_DISTANCE_SIGMA).all()


class Cluster:
    """
    Represents a single cluster for generating sources for a sky model.
    """

    _plot_style_defaults = {
        "centre": {
            "color": "k",
            "marker": ".",
            "linestyle": "",
        },
        "ellipse": {
            "linewidth": 1,
            "facecolor": "none",
        },
    }

    @classmethod
    def from_shape_parameters(
        cls,
        pos: Array2D,
        a: numbers.Real,
        b: numbers.Real,
        theta: numbers.Real = 0.0,
    ) -> "Cluster":
        """
        Initialize a `Cluster` object from the shape parameters of the
        covariance ellipse.

        Parameters
        ----------
        pos : array-like
            Position (RA, Dec) of the cluster centre as an array-like object of
            shape (2,).
        a : numbers.Real
            Semi-major axis of the covariance ellipse. For uncorrellated
            variables, this corresponds to the larger standard deviation of the
            two variables.
        b : numbers.Real
            Semi-minor axis of the covariance ellipse. For uncorrellated
            variables, this corresponds to the smaller standard deviation of the
            two variables.
        theta : int, optional
            Rotation angle of the covariance ellipse in degrees, by default 0.

        Returns
        -------
        Cluster
            The initialized `Cluster` object.
        """
        sin = np.sin(np.radians(theta))
        cos = np.cos(np.radians(theta))
        rotation = np.array([[cos, -sin], [sin, cos]])
        scale = np.array([[a**2, 0], [0, b**2]])
        covariance_matrix = rotation @ scale @ rotation.T

        return cls(pos, covariance_matrix)

    def __init__(
        self, pos: ArrayLike1D[numbers.Real], cov: ArrayLike2D[numbers.Real]
    ):
        """
        Initialize the `Cluster` source generator.

        Sources are generated from a multivariate normal probability
        distribution centred a at the given position and with given
        covariance matrix.

        Parameters
        ----------
        pos : array-like
            The position of the cluster centre.
        cov : array-like
            The 2x2 covariance matrix of the cluster.
        """
        self.position = np.array(pos)
        self.covariance = np.array(cov)

    @property
    def position(self) -> Array1D[float]:
        """The position of the cluster centre in RA, Dec coordinates."""
        return self._position

    @position.setter
    def position(self, position: ArrayLike1D[numbers.Real]):
        position = np.array(position)
        if position.shape != (2,):
            raise ValueError("Position must be a 2-element array.")
        self._position = position

        # Invalidate cached properties that depend on the position
        del self.distribution

    @property
    def covariance(self) -> Array2D[float]:
        """The 2x2 covariance matrix of the source distribution."""
        return self._sigma

    @covariance.setter
    def covariance(self, sigma: ArrayLike2D[numbers.Real]):
        # Check tht the covaraince matrix is valid. If it is valid, et the
        # `_sigma` attribute and invalidate the cached properties derived from
        # it.
        covariance_matrix = np.array(sigma, dtype=float)
        if covariance_matrix.shape != (2, 2):
            raise ValueError("Covariance matrix must be 2x2")

        if np.linalg.det(covariance_matrix) == 0:
            raise ValueError("Covariance matrix is singular")

        self._sigma = covariance_matrix

        # Invalidate cached properties that depend on the covariance matrix
        del self.precision
        del self.distribution
        del self.semi_axes
        del self.angle

    @lazyproperty
    def precision(self) -> Array2D[float]:
        """Precision matrix for the source distribution."""
        return np.linalg.inv(self.covariance)

    @lazyproperty
    def distribution(self) -> rv_continuous:
        """The multivariate normal distribution representing the cluster."""
        return multivariate_normal(mean=self.position, cov=self.covariance)

    def rvs(
        self, size: SampleSize, random_state: RandomState = None
    ) -> np.ndarray[float]:
        if size <= 0:
            return np.ndarray((0, 2), float)

        ra, dec = np.atleast_2d(
            self.distribution.rvs(size=size, random_state=random_state)
        ).T
        to_wrap = np.abs(dec) > 90
        dec[to_wrap] = (180 - np.abs(dec[to_wrap])) * np.sign(dec[to_wrap])
        ra[to_wrap] = ra[to_wrap] + 180
        ra[ra > 360] -= 360
        return np.array([ra, dec]).T

    @property
    def var_ra(self) -> numbers.Real:
        """Variance of the RA coordinate for all clusters."""
        return self.covariance[0, 0]

    @property
    def var_dec(self) -> numbers.Real:
        """Variance of the Dec coordinate for all clusters."""
        return self.covariance[1, 1]

    @property
    def cov(self) -> numbers.Real:
        """Covariance between the RA and Dec coordinates for all clusters."""
        return self.covariance[0, 1]

    @lazyproperty
    def semi_axes(self) -> tuple[numbers.Real, numbers.Real]:
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
    def semi_major_axis(self) -> numbers.Real:
        """
        The semi-major axes of the iso-probability ellipse of the source
        distribution. This is defined as the larger of the two standard
        deviations of the source distribution.
        """
        a, _ = self.semi_axes
        return a

    @property
    def semi_minor_axis(self) -> numbers.Real:
        """
        The semi-minr axes of the iso-probability ellipse of the source
        distribution. This is defined as the smaller of the two standard
        deviations of the source distribution.
        """
        _, b = self.semi_axes
        return b

    @lazyproperty
    def angle(self) -> numbers.Real:
        """
        Rotation angle of the the iso-probability ellipse of the source
        distribution. This is the counter-clockwise angle from the positive
        x-axis to the semi-major axis of the ellipse.
        """
        return np.degrees(
            0.5 * np.arctan2(2 * self.cov, (self.var_ra - self.var_dec))
        )

    def plot(
        self,
        axes: plt.Axes = None,
        n_sigma: numbers.Real = 3.0,
        mark_centre: Optional[bool | dict] = None,
        **kws,
    ) -> tuple[Ellipse, Optional[plt.Line2D]]:
        """
        Plot the covariance ellipse of the cluster and optionally mark the
        centre position.

        Parameters
        ----------
        axes : matplotlib.axes.Axes, optional
            Axes on which to draw the ellipses. If None (the default), a new
            figure and axes will be created.
        n_sigma : int, optional
            The contour level at which to draw the ellipse, by default 3.
        mark_centre : dict or bool or None, optional
            Configuration for marking the cluster centres. If a dictionary is
            provided, it will be passed as keyword arguments to the plotting
            function. If True or None (the default), default marker settings
            will be used. If False, cluster centres will not be marked.
        **kws
            Additional keyword arguments to be passed to the Ellipse
            constructor.

        Returns
        -------
        matplotlib.patches.Ellipse
            The ellipse artist.
        """
        if axes is None:
            _, axes = plt.subplots()

        # Plot centre
        centre_mark = None
        if mark_centre is not False:
            if mark_centre in (True, None):
                mark_centre = {}

            (centre_mark,) = axes.plot(
                *self.position.T,
                **(
                    self._plot_style_defaults["centre"]
                    | _normalize_plot_kws(mark_centre)
                ),
            )

        # Plot iso-probability ellipse
        ellipse = Ellipse(
            self.position,
            2 * self.semi_major_axis * n_sigma,
            2 * self.semi_minor_axis * n_sigma,
            angle=self.angle,
            **(self._plot_style_defaults["ellipse"] | _normalize_plot_kws(kws)),
        )
        axes.add_patch(ellipse)
        axes.set_aspect("equal")
        axes.autoscale_view()
        return ellipse, centre_mark


def _normalize_plot_kws(kws: dict[str, Any]) -> dict[str, Any]:
    remap = {
        "c": "color",
        "ec": "edgecolor",
        "fc": "facecolor",
        "ls": "linestyle",
        "lw": "linewidth",
        "m": "marker",
        "mew": "markeredgewidth",
        "mec": "markeredgecolor",
    }
    return {remap.get(k, k): v for k, v in kws.items()}


class AttrVector:
    """
    Descriptor class for vectorised attribute lookup on a collection of objects.

    Example
    -------
    ... class Named:
    ...    def __init__(self, name: str):
    ...        self.name = name
    ...
    ... class ObjectSequence(list):
    ...    names = AttrVector("name")
    ...
    ... seq = ObjectSequence([Named('hello'), Named('world')])
    ... print(seq.names)
    ['hello', 'world']
    """

    def __init__(self, name: str):
        self.name = name

    def __get__(
        self, instance: Optional["SourceClusters"], _owner: type = None
    ):
        if instance is None:
            return self

        return np.array(
            [getattr(cluster, self.name) for cluster in instance.clusters]
        )


class SourceClusters:
    """
    A class representing clustered sky coordinates.
    """

    # Vectorized attribute lookup on `Cluster` objects stored in the list at the
    # `clusters` attribute
    positions = AttrVector("position")
    covariances = AttrVector("covariance")
    precisions = AttrVector("precision")
    semi_major_axes = AttrVector("semi_major_axis")
    semi_minor_axes = AttrVector("semi_minor_axis")
    angles = AttrVector("angle")

    # Default plot style for the source clusters visualization
    _plot_style_defaults = {
        "centre": {
            "color": "k",
            "marker": ".",
            "linestyle": "",
        },
        "ellipse": {
            "linewidth": 1,
            "facecolor": "none",
        },
    }

    @classmethod
    def generate(
        cls,
        n_clusters: numbers.Integral,
        probabilities: Optional[Sequence[float]] = None,
        random_state: RandomState = None,
        **config,
    ) -> "SourceClusters":
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

    def __init__(
        self,
        positions: ArrayLike2D[numbers.Real],
        covariances: ArrayLike3D[numbers.Real],
        probabilities: Optional[ArrayLike1D[numbers.Real]] = None,
    ):
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

    def __len__(self):
        return len(self.clusters)

    def get_names(self, prefix: str = "Patch") -> Array1D[str]:
        """
        Generate names for each cluster based on the given prefix.

        Parameters
        ----------
        prefix : str, optional
            Prefix to add to the cluster number, by default 'Patch'.

        Returns
        -------
        np.ndarray
            Array of names corresponding to each cluster.
        """
        return np.char.add(prefix, np.arange(len(self)).astype(str))

    def sample(
        self, size: SampleSize, random_state: RandomState = None, **_kws
    ) -> SampleDict:
        """
        Generate random samples for the clustered sources.

        Parameters
        ----------
        n : int
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
        if random_state is None:
            # Use numpy's shared internal random state
            random_state = np.random
        elif isinstance(random_state, int):
            # Need to deal with integer random states here since we will reuse
            # the random state multiple times
            random_state = np.random.RandomState(random_state)

        coords = np.empty((size, 2))
        labels = np.empty(size, dtype=int)
        cluster_indices = random_state.choice(
            len(self),
            p=self.probabilities,
            size=size,
        )
        index = 0
        for cluster_index, n_points in zip(
            *np.unique(cluster_indices, return_counts=True),
            strict=True,
        ):
            if n_points:
                cluster = self.clusters[cluster_index]
                labels[index : index + n_points] = cluster_index
                coords[index : index + n_points] = cluster.rvs(
                    size=n_points,
                    random_state=random_state,
                )
                index += n_points

        # Generate patch names from cluster labels
        labels = np.char.add("Patch", labels.astype(int).astype(str))

        return {"ra": coords[:, 0], "dec": coords[:, 1], "patch": labels}

    def plot(
        self,
        axes: Optional[plt.Axes] = None,
        n_sigma: numbers.Real = 3.0,
        mark_centres: Optional[bool | dict[str, Any]] = None,
        **kws,
    ) -> plt.Figure:
        """
        Plot the covariance ellipses of the clusters and optionally mark the
        cluster centres.

        Parameters
        ----------
        axes : matplotlib.axes.Axes, optional
            Axes on which to draw the ellipses. If None (the default), a new
            figure and axes will be created.
        n_sigma : int, optional
            The contour level at which to draw the ellipse, by default 3.
        mark_centres : dict or bool or None, optional
            Configuration for marking the cluster centres. If a dictionary is
            provided, it will be passed as keyword arguments to the plotting
            function. If True or None (the default), default marker settings
            will be used. If False, cluster centres will not be marked.
        **kws
            Additional keyword arguments to be passed to the EllipseCollection
            constructor.

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
        mark_centres = kws.pop("mark_centre", mark_centres)
        if mark_centres is not False:
            if mark_centres in (True, None):
                mark_centres = {}
            ax.plot(
                *self.positions.T,
                **(
                    self._plot_style_defaults["centre"]
                    | _normalize_plot_kws(mark_centres)
                ),
            )

        # Plot iso-probability ellipses
        ec = EllipseCollection(
            2 * self.semi_major_axes * n_sigma,
            2 * self.semi_minor_axes * n_sigma,
            self.angles,
            units="xy",
            offset_transform=ax.transData,
            offsets=self.positions,
            **(self._plot_style_defaults["ellipse"] | _normalize_plot_kws(kws)),
        )
        ax.add_collection(ec)

        # Update axes limits
        deltas = n_sigma * self.semi_major_axes[np.newaxis].T
        maxima = (self.positions + deltas).max(0)
        minima = (self.positions - deltas).min(0)
        xlim, ylim = np.array([minima, maxima]).T
        ax.set(xlim=xlim, ylim=ylim)

        ax.set_aspect("equal")
        ax.grid()

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
        n_sources: numbers.Integral,
        **kws,
    ) -> tuple[SampleDict, PatchDefinitions]:
        """
        Generate a random sample of sources (with clustered positions),
        returning a dictionary with parameter names as keys and arrays of
        sampled values as values.

        Parameters
        ----------
        n_sources : int
            The number of sources to generate in the skymodel.
        **kws
            Additional keyword arguments passed to the sampling functions of the
            underlying clusters and sky model generator.

        Returns
        -------
        samples : dict[str, numpy.ndarray]
            A dictionary with arrays of `n_sources` sampled values for each
            parameter.
        patches : np.ndarray
            The patch definitions as a (n, 5) array of strings.
        """
        # Get coordinates by generating them from the SourceClusters object
        kws["n_patches"] = None
        clustered_data = self.clusters.sample(n_sources, **kws)

        # Generate the remaining columns
        samples, _ = super().sample(n_sources, ignore=["clusters"], **kws)
        # Create patch definitions. Patch names column is already generated in
        # the clustered_data dict
        samples |= clustered_data
        patch_defs, _ = self.get_patches(samples)

        return samples, patch_defs

    def get_patches(
        self, _samples: SampleDict, _n_patches: int = None
    ) -> tuple[PatchDefinitions, Array1D[str]]:

        patch_names = self.clusters.get_names()
        ra, dec = self.get_coords(
            {
                "ra": self.clusters.positions[:, 0],
                "dec": self.clusters.positions[:, 1],
            }
        )

        patch_defs = np.empty((len(patch_names), 5), object)
        patch_defs[:, 0] = " "
        patch_defs[:, 1] = ""
        patch_defs[:, 2] = patch_names
        patch_defs[:, 3] = ra
        patch_defs[:, 4] = dec

        return patch_defs, patch_names
