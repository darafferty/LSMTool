# Coordinate performance benchmark

`coordinate_performance.py` consolidates the temporary scripts used for the
RADec2Angle and mean-shift investigation. It runs the same workloads and exact
result comparisons, with configurable input, baseline revision, and repetition.

From an environment with LSMTool's dependencies installed:

```sh
python benchmarks/vectorize-radec-normalization/coordinate_performance.py tests/sector_1.apparent_sky.txt --repeats 2
```

The large sky model is not included in Git. Supply its path, or use a smaller
tracked model for a smoke test:

```sh
python benchmarks/vectorize-radec-normalization/coordinate_performance.py tests/resources/apparent.sky
```

The default baseline is `770343b`, the first vectorization commit. To compare
with the implementation before that commit, use `--baseline 770343b^`.
The selected revision must be available in the local Git history.

The script measures model loading, weighted patch-position calculation with
per-patch projection, and mean-shift grouping separately. Grouping uses
`byPatch=True`, `applyBeam=False`, `lookDistance=0.075`, and
`groupingDistance=0.01`. Each run compares patch positions, source-to-group
assignments, and final group positions exactly. A final microbenchmark measures
100,000 numeric coordinate pairs supplied as lists and checks their equality.
A mismatch raises an assertion and exits unsuccessfully.

It imports LSMTool from this checkout and forces the Python mean-shift backend,
including when the optional compiled grouper is installed. Timing excludes
interpreter startup, imports, output printing, and equality checks. The baseline
runs first, followed by the working tree, in the same process. Results are
indicative local timings, not a statistically controlled benchmark.

This is a focused comparison, not a general benchmark of two complete checkouts:
it extracts `RADec2Angle`, `SkyModel.getPatchPositions`, and
`Grouper.euclid_distance` from the selected revision and runs them with the
current module dependencies. It is intended for the revisions in this
investigation. Historical `_getXY` calls are adapted to `_get_xy`, with
patch selections passed as RA and Dec arrays to match the current API.
Arbitrary revisions may require other historical helpers or
have incompatible interfaces. It executes code from the selected revision,
so use trusted revisions only. Monkey patches are restored on completion.

For profiling the whole comparison:

```sh
python -m cProfile -o /tmp/coordinate-performance.prof \
    benchmarks/vectorize-radec-normalization/coordinate_performance.py tests/sector_1.apparent_sky.txt
```

## What the equality checks do not establish

The explicit-position bug discussed in MR !154 is covered by
`test_set_patch_positions_stores_scalar_angles` in `tests/test_skymodel.py`:

```sh
python -m pytest tests/test_skymodel.py \
    -k test_set_patch_positions_stores_scalar_angles -q
```

The benchmark calculates patch positions and runs grouping; it does not set
explicit numeric or string coordinate dictionaries. Its before/after equality
checks therefore did not cover that setter path. A bug present in both
implementations would also pass an equality-only comparison.

Historical `RADec2Angle` already returned one-element Angle arrays for scalar
input. The bug was that `setPatchPositions` stored those arrays instead of
extracting scalar elements. The fix belongs in that setter, preserving the
array-return contract used by other callers. The regression tests check scalar
metadata and `(N,)` coordinate arrays for one and multiple updated patches,
including a mix of updated and untouched patches.

The script does not restore historical `setPatchPositions` or shared helpers
such as `normalize_ra_dec`. Both benchmark variants use the current versions
of those dependencies. Use the regression tests to verify the setter's contract;
use this script for the documented performance workloads.
