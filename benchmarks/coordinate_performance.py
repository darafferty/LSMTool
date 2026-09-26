"""Compare coordinate workloads with selected functions from a Git revision."""

import argparse
import ast
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Always measure this checkout, even when invoked from another directory.
REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import lsmtool
import lsmtool.operations._meanshift as meanshift
import lsmtool.skymodel as skymodel
import lsmtool.tableio as tableio


def revision_function(revision, path, name, namespace):
    """Load one historical function using the current module's dependencies."""
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=REPOSITORY, text=True
    )
    tree = ast.parse(source)
    node = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    scope = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


def measure(sky_model):
    start = time.perf_counter()
    model = lsmtool.load(str(sky_model))
    loaded = time.perf_counter()
    positions = model.getPatchPositions(
        perPatchProjection=True, method="wmean", asArray=True
    )
    positioned = time.perf_counter()
    model.group(
        "meanshift", byPatch=True, applyBeam=False,
        lookDistance=0.075, groupingDistance=0.01,
    )
    grouped = time.perf_counter()
    results = (
        positions,
        np.array(model.table["Patch"]),
        model.getPatchPositions(asArray=True),
    )
    timings = (loaded - start, positioned - loaded, grouped - positioned)
    return results, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sky_model", type=Path, help="Input model with patches")
    parser.add_argument(
        "--baseline", default="770343b",
        help="Trusted Git revision for the baseline functions (default: 770343b)",
    )
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if not args.sky_model.is_file():
        parser.error(f"Input file does not exist: {args.sky_model}")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    # The distance change affects the Python implementation. Do not silently
    # measure the optional compiled backend, which group() otherwise prefers.
    backend_name = "lsmtool.operations._meanshiftc"
    saved_backend = sys.modules.get(backend_name)
    sys.modules[backend_name] = None

    current = (
        tableio.RADec2Angle,
        skymodel.SkyModel.getPatchPositions,
        meanshift.Grouper.euclid_distance,
    )
    baseline = (
        revision_function(
            args.baseline, "lsmtool/tableio.py", "RADec2Angle", vars(tableio)
        ),
        revision_function(
            args.baseline, "lsmtool/skymodel.py", "getPatchPositions",
            vars(skymodel),
        ),
        revision_function(
            args.baseline, "lsmtool/operations/_meanshift.py", "euclid_distance",
            vars(meanshift),
        ),
    )
    print(f"Checkout: {REPOSITORY}\nBaseline: {args.baseline}", flush=True)
    print("Backend: Python mean-shift; times exclude import and printing",
          flush=True)
    try:
        for repeat in range(args.repeats):
            results = []
            for label, functions in (("before", baseline), ("after", current)):
                (
                    tableio.RADec2Angle,
                    skymodel.SkyModel.getPatchPositions,
                    meanshift.Grouper.euclid_distance,
                ) = functions
                result, timings = measure(args.sky_model)
                results.append(result)
                print(
                    f"Run {repeat + 1} {label}: load={timings[0]:.6f}s "
                    f"positions={timings[1]:.6f}s group={timings[2]:.6f}s",
                    flush=True,
                )
            for before, after in zip(*results, strict=True):
                np.testing.assert_array_equal(before, after)
            print(
                "Patch positions, group memberships, and group positions "
                "are exactly identical.", flush=True,
            )

        ra = np.linspace(-720, 720, 100_000).tolist()
        dec = np.linspace(-360, 360, 100_000).tolist()
        numeric_results = []
        for label, functions in (("before", baseline), ("after", current)):
            start = time.perf_counter()
            numeric_results.append(functions[0](ra, dec))
            elapsed = time.perf_counter() - start
            print(f"{label}: 100000 numeric pairs={elapsed:.6f}s", flush=True)
        for before, after in zip(*numeric_results, strict=True):
            np.testing.assert_array_equal(before.degree, after.degree)
    finally:
        (
            tableio.RADec2Angle,
            skymodel.SkyModel.getPatchPositions,
            meanshift.Grouper.euclid_distance,
        ) = current
        if saved_backend is None:
            sys.modules.pop(backend_name, None)
        else:
            sys.modules[backend_name] = saved_backend


if __name__ == "__main__":
    main()
