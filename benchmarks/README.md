# Coordinate performance benchmark

`coordinate_performance.py` consolidates the temporary scripts used for the
RADec2Angle and mean-shift investigation. It runs the same workloads and exact
result comparisons, with configurable input, baseline revision, and repetition.

From an environment with LSMTool's dependencies installed:

```sh
python benchmarks/coordinate_performance.py tests/sector_1.apparent_sky.txt --repeats 2
```

The large sky model is not included in Git. Supply its path, or use a smaller
tracked model for a smoke test:

```sh
python benchmarks/coordinate_performance.py tests/resources/apparent.sky
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
investigation; arbitrary revisions may require other historical helpers or
have incompatible interfaces. It executes code from the selected revision,
so use trusted revisions only. Monkey patches are restored on completion.

For profiling the whole comparison:

```sh
python -m cProfile -o /tmp/coordinate-performance.prof \
    benchmarks/coordinate_performance.py tests/sector_1.apparent_sky.txt
```

See [the investigation record](../docs/development/radec-vectorization.md) for
reported results and implementation rationale.
