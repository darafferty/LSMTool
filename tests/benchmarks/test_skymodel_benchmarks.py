"""
Benchmark tests for SkyModel functions using pytest-benchmark.
"""

import math

import pytest

import lsmtool
from lsmtool.testing import SkyModelGenerator

# ---------------------------------------------------------------------------- #
# Fixtures


@pytest.fixture(params=[1, 100, 1000, 10_1000, 100_000], scope="session")
def n_sources(request):
    return request.param


@pytest.fixture(params=[(10, 10)], scope="session")
def n_patches(request):  # (2, 2), (10, 10), (25, 25), (50, 50)
    return request.param


@pytest.fixture(scope="session")
def output_dir(pytestconfig):
    path = pytestconfig.resource_dir / "generated_skymodels"
    path.parent.mkdir(exist_ok=True)
    return path


@pytest.fixture
def generated_skymodel_path(output_dir, n_sources, n_patches, rng):
    """
    Generate a skymodel file with the specified number of sources and patches.
    """
    xp, yp = n_patches
    path = output_dir / f"skymodel_{n_sources}_{xp}x{yp}.txt"
    if not path.exists():
        generator = SkyModelGenerator()
        generator.to_file(path, n_sources, n_patches, rng)
    return path


@pytest.fixture
def skymodel(generated_skymodel_path):
    """
    Load the generated skymodel from the specified path.
    """
    return lsmtool.load(generated_skymodel_path)


# ---------------------------------------------------------------------------- #
# Benchmark tests


@pytest.mark.benchmark(group="SkyModel.getPatchPositions", min_rounds=1)
# @pytest.mark.parametrize("per_patch_projection", [True])
def test_get_patch_positions_benchmark(
    benchmark, skymodel, n_sources, n_patches, per_patch_projection=True
):
    """Benchmark `SkyModel.getPatchPositions`"""
    benchmark.extra_info["n_sources"] = n_sources
    benchmark.extra_info["n_patches"] = math.prod(n_patches)

    benchmark(
        skymodel.getPatchPositions,
        perPatchProjection=per_patch_projection,
        method="wmean",
    )


# @pytest.mark.benchmark(group="SkyModel.group", min_rounds=1)
# def test_group_benchmark(benchmark, skymodel, n_sources, n_patches):
#     """Benchmark `SkyModel.group`"""
#     benchmark.extra_info["n_sources"] = n_sources
#     benchmark.extra_info["n_patches"] = math.prod(n_patches)

#     benchmark(
#         skymodel.group,
#         "meanshift",
#         byPatch=True,
#         applyBeam=False,
#         lookDistance=0.075,
#         groupingDistance=0.01,
#     )
