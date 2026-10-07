"""
Benchmark tests for SkyModel functions using pytest-benchmark.
"""

import math
from pathlib import Path

import pytest

import lsmtool
from lsmtool.testing import SkyModelGenerator

# ---------------------------------------------------------------------------- #

output_root = Path(__file__).parent.parent
skymodels_path = output_root / "resources" / 'generated_skymodels'


# ---------------------------------------------------------------------------- #
# Benchmarks


@pytest.fixture(params=[1, 100, 1000, 10_000, 50_000, 100_000], scope="session")
def n_sources(request):
    return request.param


@pytest.fixture(params=[(10, 10)], scope="session")
def n_patches(request):  # (2, 2), (10, 10), (25, 25), (50, 50)
    return request.param


@pytest.fixture
def generated_skymodel_path(n_sources, n_patches, rng):
    """
    Generate a skymodel file with the specified number of sources and patches.
    """
    xp, yp = n_patches
    path = skymodels_path / f"skymodel_{n_sources}_{xp}x{yp}.txt"
    if not path.exists():
        generator = SkyModelGenerator()
        generator.to_file(path, n_sources, n_patches, rng)
    return path


@pytest.fixture
def skymodel(generated_skymodel_path):
    """
    Load the generated skymodel from the specified path.
    """
    skymodel = lsmtool.load(generated_skymodel_path)
    return skymodel


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


@pytest.mark.benchmark(group="SkyModel.group", min_rounds=1)
def test_group_benchmark(benchmark, skymodel, n_sources, n_patches):
    """Benchmark `SkyModel.group`"""
    benchmark.extra_info["n_sources"] = n_sources
    benchmark.extra_info["n_patches"] = math.prod(n_patches)

    benchmark(
        skymodel.group,
        "meanshift",
        byPatch=True,
        applyBeam=False,
        lookDistance=0.075,
        groupingDistance=0.01,
    )
