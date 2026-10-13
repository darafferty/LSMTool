"""
Benchmark tests for SkyModel functions using pytest-benchmark.
"""

import math

import pytest

import lsmtool
from lsmtool.testing import SkyModelGenerator

try:
    import pytest_benchmark  # noqa
except ImportError:
    pytest.skip("pytest-benchmark is not installed", allow_module_level=True)


# ---------------------------------------------------------------------------- #
# Fixtures


@pytest.fixture(params=[1, 100, 1000, 10_000, 100_000], scope="session")
def n_sources(request):
    print(f"n_sources={request.param}")
    return request.param


@pytest.fixture(
    params=[(10, 10), (100, 100)], scope="session", ids="{0[0]}x{0[1]}".format
)
def n_patches_xy(request):
    return request.param


@pytest.fixture(scope="session")
def output_dir(pytestconfig):
    path = pytestconfig.resource_dir / "generated_skymodels"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture
def generated_skymodel_path(output_dir, n_sources, n_patches_xy, rng):
    """
    Generate a skymodel file with the specified number of sources and patches.
    """
    xp, yp = n_patches_xy
    path = output_dir / f"skymodel_{n_sources}_{xp}x{yp}.txt"
    if not path.exists():
        generator = SkyModelGenerator()
        generator.to_file(path, n_sources, n_patches_xy, rng)
    return path


@pytest.fixture
def skymodel(generated_skymodel_path):
    """
    Load the generated skymodel from the specified path.
    """
    return lsmtool.load(generated_skymodel_path)


# ---------------------------------------------------------------------------- #
# Benchmark Tests


@pytest.mark.benchmark(group="SkyModel.getPatchPositions", min_rounds=1)
def test_get_patch_positions_benchmark(
    benchmark, skymodel, n_sources, n_patches_xy, per_patch_projection=True
):
    """Benchmark `SkyModel.getPatchPositions`"""
    benchmark.extra_info["n_sources"] = n_sources
    benchmark.extra_info["n_patches"] = math.prod(n_patches_xy)

    benchmark(
        skymodel.getPatchPositions,
        perPatchProjection=per_patch_projection,
        method="wmean",
    )
