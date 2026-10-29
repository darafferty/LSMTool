"""
Tests for skymodel assertion helpers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import kstest
from scipy.stats.distributions import norm as normal
from scipy.stats.distributions import uniform

from lsmtool import load
from lsmtool.testing import (
    Cluster,
    ClusteredSkyModelGenerator,
    SkyModelGenerator,
    SourceClusters,
    Uniform,
    check_skymodels_equal,
    mahalanobis_distance,
)

# ---------------------------------------------------------------------------- #
TEST_PATH = Path(__file__).parent
RESOURCES_PATH = TEST_PATH / "resources"

# ---------------------------------------------------------------------------- #
# Fixtures


@pytest.fixture(scope="module")
def input_skymodel(test_data_path):
    return test_data_path / "expected.true_sky.txt"


@pytest.fixture
def modified_skymodel(request, tmp_path):
    # Use the input_skymodel fixture value
    input_skymodel = request.getfixturevalue("input_skymodel")
    return _modify_patches(input_skymodel, *request.param, tmp_path)


def _modify_patches(input_skymodel, update_patch_names, regroup, tmp_path):
    """
    Modify the patch definitions in the skymodel and save as a temporary file,
    returning the filename.
    """
    skymodel = load(input_skymodel)

    if update_patch_names:
        _update_patch_names(skymodel)

    if regroup:
        skymodel.group("every")

    suffix = f"{int(update_patch_names)}{int(regroup)}"
    filename = tmp_path / f"skymodel_testcase_{suffix}.sky"
    skymodel.write(filename, format="makesourcedb")
    return filename


def _update_patch_names(skymodel):
    last_patch_name = skymodel.getPatchNames()[-1]
    new_name = "Whatever"

    patch_pos = skymodel.getPatchPositions()
    patch_pos[new_name] = patch_pos.pop(last_patch_name)
    skymodel.setPatchPositions(patch_pos)

    patches = skymodel.table["Patch"]
    patches[patches == last_patch_name] = new_name

    patches = skymodel.table.groups.keys["Patch"]
    patches[patches == last_patch_name] = new_name


# ---------------------------------------------------------------------------- #
# Tests


@pytest.mark.parametrize(
    "modified_skymodel, check_patch_names_sizes, expected_equal",
    [
        pytest.param(
            # (update_patch_names, regroup),
            (False, True),
            False,  # check_names_sizes
            False,  # expected equal
            id="different_groups_check_names",
        ),
        pytest.param(
            (False, True),
            True,
            False,
            id="different_groups_ignore_names",
        ),
        pytest.param(
            (True, False),
            True,
            False,
            id="different_patch_names_check",
        ),
        pytest.param(
            (True, False),
            False,
            True,
            id="different_patch_names_ignore",
        ),
        pytest.param(
            (False, False),
            True,
            True,
            id="no_edit",
        ),
    ],
    indirect=["modified_skymodel"],
)
def test_check_skymodels_equal_patches(
    input_skymodel, modified_skymodel, check_patch_names_sizes, expected_equal
):
    """
    Test check_skymodels_equal in the case that the skymodels are identical,
    except for the patch sizes or names.
    """

    # Test that differences in patch names and sizes are checked when asked for
    assert (
        check_skymodels_equal(
            input_skymodel, modified_skymodel, check_patch_names_sizes
        )
        is expected_equal
    )


@pytest.mark.parametrize(
    "left_filename, right_filename, check_patch_names_sizes, expected_equal",
    [
        # Equal cases
        pytest.param(
            "single_point.sky",
            "single_point.sky",
            True,
            True,
            id="single_point_model",
        ),
        pytest.param(
            "single_spectralindx.sky",
            "single_spectralindx.sky",
            True,
            True,
            id="multi_point_single_si",
        ),
        pytest.param(
            "nans.sky",
            "nans.sky",
            True,
            True,
            id="patches_contains_nans",
        ),
        pytest.param(
            "transfer_patches_from.sky",
            "to_patched.sky",
            True,
            True,
            id="patch_positions_close_but_not_exact",
        ),
        pytest.param(
            "transfer_patches_from.sky",
            "transfer_patches_to.sky",
            False,
            True,
            id="ignore_patches",
        ),
        # Unequal cases
        pytest.param(
            "transfer_patches_from.sky",
            "transfer_patches_to.sky",
            True,
            False,
            id="check_patches",
        ),
        pytest.param(
            "single_point.sky",
            "single_spectralindx.sky",
            False,
            False,
            id="not_equal",
        ),
    ],
)
def test_check_skymodels_equal(
    test_data_path,
    left_filename,
    right_filename,
    check_patch_names_sizes,
    expected_equal,
):
    # Assert
    assert (
        check_skymodels_equal(
            test_data_path / left_filename,
            test_data_path / right_filename,
            check_patch_names_sizes,
        )
        is expected_equal
    )


class TestSkyModelGenerator:
    """
    Test the SkyModelGenerator class.
    """

    def test_minimal(self, tmp_path, rng):
        """
        Test that we can generate a minimal random skymodel and that we can
        load it.
        """

        # create skymodel generator and sample 100 sources
        generator = SkyModelGenerator(
            q=None,
            u=None,
            v=None,
            reference_frequency=None,
            spectral_index=None,
            rotation_measure=None,
            major_axis=None,
            minor_axis=None,
            orientation=None,
        )
        # check that we can write and read the skymodel without errors
        path = tmp_path / "test_skymodel_generator_minimal.txt"
        generator.to_file(path, 10, random_state=rng)

        skymodel = load(path)
        assert skymodel.getColNames() == ["Name", "Type", "Ra", "Dec", "I"]
        assert len(skymodel) == 10

    @pytest.fixture(
        params=[
            pytest.param({}, id="default"),
            pytest.param(
                {"ra": Uniform(0, 45), "dec": Uniform(-45, 45)},
                id="custom",
            ),
        ],
    )
    def config(self, request):
        """Fixture to supply config for SkyModelGenerator tests."""
        return request.param

    def test_skymodel_generator_samples(self, config, rng):
        """
        Test the random number generation from SkyModelGenerator.
        """

        # create skymodel generator and draw a random sample of sources
        generator = SkyModelGenerator(**config)
        samples, _ = generator.sample(n_sources=100, random_state=rng)
        ra, dec = samples["ra"], samples["dec"]

        # Check that the samples are within the expected ranges and have the
        # expected distribution.
        ra0 = generator.ra.kwds["loc"]
        ra1 = ra0 + generator.ra.kwds["scale"]
        dec0 = generator.dec.kwds["loc"]
        dec1 = dec0 + generator.dec.kwds["scale"]

        assert np.all((ra0 < ra) & (ra < ra1))
        assert np.all((dec0 < dec) & (dec < dec1))
        assert not any(
            map(len, np.nonzero([samples["q"], samples["u"], samples["v"]]))
        )
        assert np.all(samples["reference_frequency"] == 1.44e8)
        assert np.all(samples["minor_axis"] < samples["major_axis"])

        # test that samples are drawn from the correct distributions
        assert kstest(ra, uniform(ra0, ra1 - ra0).cdf).pvalue > 0.05
        assert kstest(dec, uniform(dec0, dec1 - dec0).cdf).pvalue > 0.05

    @pytest.mark.parametrize("n_sources", [1, 1000])
    def test_read_write(self, tmp_path, config, n_sources, rng):
        """
        Test that we can write a skymodel to a file and read it back correctly.
        """

        # create skymodel generator and draw a random sample of sources
        generator = SkyModelGenerator(**config)
        generator.to_file(
            path := tmp_path / "test_skymodel_generator.txt",
            n_sources=n_sources,
            random_state=rng,
        )
        assert path.exists()

        skymodel = load(path)
        assert skymodel.getColNames() == [
            "Name",
            "Type",
            "Ra",
            "Dec",
            "I",
            "Q",
            "U",
            "V",
            "ReferenceFrequency",
            "SpectralIndex",
            "RotationMeasure",
            "MajorAxis",
            "MinorAxis",
            "Orientation",
        ]
        assert len(skymodel) == n_sources

    @pytest.mark.parametrize(
        "n_sources, n_patches",
        [
            [1, (2, 2)],
            [10, (2, 2)],
            [100, (2, 3)],
            [1000, (11, 9)],
            [10_000, (20, 20)],
        ],
    )
    def test_patches(self, tmp_path, rng, n_sources, n_patches):
        """
        Test that we can generate a random skymodel with patches and that we can
        load it.
        """
        x_patches, y_patches = n_patches
        path = tmp_path / f"skymodel_{n_sources}_{x_patches}x{y_patches}.txt"

        # Generate the skymodel file with the specified number of sources and
        # patches
        generator = SkyModelGenerator()
        generator.to_file(path, n_sources, n_patches, rng)

        # Check that we can load the skymodel
        assert path.exists()
        skymodel = load(path)

        # check the number of sources
        assert len(skymodel) == n_sources

        # check number of patches
        assert "Patch" in skymodel.getColNames()
        assert len(skymodel.table.meta) == x_patches * y_patches

        # check number of sources per patch
        n_sources_per_patch = [
            sum(skymodel.table["Patch"] == patch_name)
            for patch_name in skymodel.table.meta
        ]
        assert sum(n_sources_per_patch) == n_sources


class TestCluster:
    """
    Test the functionality of the Cluster class and derived properties or the
    iso-probability ellipse of the underlying multivariate normal distribution.
    """

    def test_properties(self):
        """
        Test that the properties derived from the covariance matrix are
        initialized correctly.
        """
        # Arrange
        position = [12.0, -4.0]
        covariance = [[1.0, 0.5], [0.5, 1.0]]

        # Act
        cluster = Cluster(position, covariance)

        # Assert
        np.testing.assert_array_equal(cluster.position, position)
        np.testing.assert_array_equal(cluster.covariance, covariance)
        np.testing.assert_array_equal(cluster.distribution.mean, position)
        np.testing.assert_array_equal(cluster.distribution.cov, covariance)

        assert cluster.var_ra == 1.0
        assert cluster.var_dec == 1.0
        assert cluster.cov == 0.5
        assert cluster.semi_major_axis == np.sqrt(1.5)
        assert cluster.semi_minor_axis == np.sqrt(0.5)
        assert cluster.angle == 45

        # Recreate the cluster using shape parameters and check covariance
        # consistency
        cluster = Cluster.from_shape_parameters(
            position,
            cluster.semi_major_axis,
            cluster.semi_minor_axis,
            cluster.angle,
        )
        assert np.allclose(cluster.covariance, covariance)

    def test_covariance_update_refreshes_derived_properties(self):
        """
        Test that the properties derived from the covariance matrix are
        updated correctly when the covariance matrix changes.
        """
        # Arrange
        cluster = Cluster([0.0, 0.0], [[1.0, 0.0], [0.0, 4.0]])

        # Check initial derived properties
        assert cluster.semi_major_axis == 2.0
        assert cluster.angle == 90.0

        # Act: update the covariance
        cluster.covariance = [[4.0, 0.0], [0.0, 1.0]]

        # Assert: check updated derived properties
        assert cluster.semi_major_axis == 2.0
        assert cluster.semi_minor_axis == 1.0
        assert cluster.angle == 0.0

    @pytest.mark.parametrize(
        "position, covariance, message",
        [
            ([1.0], [[1.0, 0.0], [0.0, 1.0]], "Position must be"),
            ([1.0, 2.0], [[1.0, 0.0]], "Covariance matrix must be 2x2"),
            ([1.0, 2.0], [[1.0, 0.0], [0.0, 0.0]], "matrix is singular"),
        ],
    )
    def test_invalid_inputs(self, position, covariance, message):
        """
        Test that initializing Cluster with invalid inputs raises an exception
        with the appropriate message.
        """
        with pytest.raises(ValueError, match=message):
            Cluster(position, covariance)

    def test_rvs(self):
        """
        Test the random number generation (of source positions) from the
        multivariate normal distribution of the cluster.
        """

        # Arrange
        cluster = Cluster([2.0, -3.0], [[1.0, 0.2], [0.2, 2.0]])
        # Act
        samples = cluster.rvs(size=500, random_state=1)
        # Assert
        assert samples.shape == (500, 2)

        # test that samples are drawn from the correct distributions
        assert kstest(samples[:, 0], normal(2.0, 1.0).cdf).pvalue > 0.05
        assert (
            kstest(samples[:, 1], normal(-3.0, np.sqrt(2.0)).cdf).pvalue > 0.05
        )

    @pytest.mark.parametrize("pole", [1, -1])
    def test_cluster_at_pole(self, pole):
        """
        Test that the cluster correctly handles positions near the celestial
        poles.
        """
        # Arrange
        cluster = Cluster([0.0, pole * 90.0], [[1.0, 0.0], [0.0, 1.0]])

        # Act
        samples = cluster.rvs(size=100, random_state=1)

        # Assert
        assert samples.shape == (100, 2)
        assert np.all(np.abs(samples[:, 1]) <= 90)

    @pytest.mark.mpl_image_compare(
        baseline_dir=RESOURCES_PATH / "plotting" / "baseline_images",
        results_dir=RESOURCES_PATH / "plotting" / "generated_images",
        remove_text=True,
    )
    def test_plot_cluster(self):
        """
        Test the plotting of a single cluster with various options. Resulting
        figure is compared against the baseline image.
        """
        # Arrange
        cluster = Cluster([0.34, -0.22], [[1.5, 0.5], [0.5, 1.5]])

        fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True)
        cluster.plot(ax0, n_sigma=1, mark_centre=True)
        cluster.plot(ax0, n_sigma=3, mark_centre=False, ls="--")
        cluster.plot(ax0, n_sigma=5, mark_centre=False, ls="-.")
        ax0.plot(
            *cluster.rvs(100, random_state=13).T,
            ls="",
            marker="o",
            color="orangered",
            alpha=0.3,
        )
        cluster.plot(
            ax1,
            n_sigma=1,
            mark_centre={
                "m": "x",
                "c": "orange",
                "ms": 10,
                "mew": 2,
            },
            ls="-",
            linewidth=5,
            edgecolor="purple",
            facecolor="c",
            alpha=0.25,
        )
        return fig


class TestSourceClusters:
    """
    Test the functionality of the SourceClusters class and derived properties.
    """

    @pytest.mark.parametrize(
        "pos, cov, prob, message",
        [
            pytest.param(
                [1.0],
                (cov := [[1.0, 0.0], [0.0, 1.0]]),
                None,
                "centres and covariances must have the same length",
            ),
            pytest.param(
                [1.0, 2.0],
                cov,
                [1],
                "probabilities must have the same length as positions",
            ),
            pytest.param(
                [(1.0, 2.0)],
                [[1.0, 0.0]],
                None,
                "Covariance matrix must be 2x2",
            ),
        ],
    )
    def test_invalid_input(self, pos, cov, prob, message):
        """
        Test that initializing the SourceClusters with invalid inputs raises an
        exception with the appropriate message.
        ."""
        with pytest.raises(
            ValueError,
            match=message,
        ):
            SourceClusters(pos, cov, prob)

    def test_distance_wrapping(self):
        positions = [(1, 0), (359, 0)]
        covariances = np.tile(4 * np.eye(2), (2, 1, 1))
        assert np.all(
            mahalanobis_distance([0, 0], positions, covariances) == 0.5
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=RESOURCES_PATH / "plotting" / "baseline_images",
        results_dir=RESOURCES_PATH / "plotting" / "generated_images",
        remove_text=True,
    )
    def test_plot_clusters(self):
        """
        Test the plotting of multiple clusters with various options. Resulting
        figure is compared against the baseline image.
        """
        clusters = SourceClusters.generate(
            n_clusters=12,
            probabilities=None,
            random_state=11,
            ra=Uniform(15, 60),
            dec=Uniform(-15, 15),
        )

        fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True)

        clusters.plot(ax0, n_sigma=1, mark_centres=True, ls="-")
        clusters.plot(ax0, n_sigma=3, mark_centres=False, ls=":")
        clusters.plot(ax0, n_sigma=5, mark_centres=False, ls="-")

        sample = clusters.sample(120, random_state=12)
        labels = np.char.lstrip(sample["patch"], "Patch").astype(int)

        # plot samples
        ax0.scatter(
            sample["ra"],
            sample["dec"],
            c=labels,
            cmap="tab20",
            alpha=0.3,
        )
        clusters.plot(
            ax1,
            n_sigma=3,
            mark_centres={
                "m": "x",
                "c": "orange",
                "ms": 5,
                "mew": 1,
            },
            ls="-",
            linewidth=1,
            edgecolor="purple",
            alpha=0.5,
        )
        ax0.grid(False)
        ax1.grid(False)
        return fig


class TestClusteredSkymodelGenerator:
    """
    Test generating skymodels from clustered source distribution.
    """

    @pytest.mark.parametrize("n_sources, n_clusters", [(100, 20), (1_000, 40)])
    def test_read_write(self, tmp_path, n_sources, n_clusters, rng):
        """
        Test that we can generate a grouped skymodel, read it, and that it has
        the expected structure.
        """

        path = (
            tmp_path / f"test_clustered_skymodel_{n_sources}_{n_clusters}.txt"
        )

        # Generate cluster parameters
        clusters = SourceClusters.generate(
            n_clusters=n_clusters,
            probabilities=None,
            random_state=rng,
            ra=Uniform(15, 60),
            dec=Uniform(-15, 15),
        )
        generator = ClusteredSkyModelGenerator(clusters=clusters)
        generator.to_file(path, n_sources, random_state=rng)

        assert path.exists()
        skymodel = load(path)

        assert skymodel.getColNames() == [
            "Name",
            "Type",
            "Patch",
            "Ra",
            "Dec",
            "I",
            "Q",
            "U",
            "V",
            "ReferenceFrequency",
            "SpectralIndex",
            "RotationMeasure",
            "MajorAxis",
            "MinorAxis",
            "Orientation",
        ]

        assert len(skymodel) == n_sources
