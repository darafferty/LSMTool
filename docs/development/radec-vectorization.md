# RADec2Angle performance investigation

2026-09-26–2026-09-28 · Branch: `perf/vectorize-radec-normalization` ·
[MR !154](https://git.astron.nl/RD/LSMTool/-/merge_requests/154)

## Context and decisions

The user reported that MR !149's memory improvements also made the local
`test_get_patch_positions.py` and `test_group.py` programs 7–8× faster.
Their cProfile results identified `RADec2Angle` as the next bottleneck.
The requested investigation led to vectorization, then faster string parsing
and Python mean-shift distances. At the user's request, the work was recorded
on a separate branch and the benchmark scripts were added to Git.

| Commit | Change |
| --- | --- |
| `770343b` | Vectorized normalization and batched patch positions |
| `a23f10a` | Reproducible benchmark and usage notes |
| `7e7ac50` | Faster sexagesimal parsing and mean-shift distances |
| `a80be7d` | Shared normalization helper following review |
| `9f608f6` | Scalar patch-position fix, review record, and benchmark reorganization |

## Implementation rationale

- **Normalization:** Replace per-coordinate Astropy object creation with NumPy
  arithmetic. Following review, the algorithm lives in
  `operations_lib.normalize_ra_dec`, called once per batch by `RADec2Angle`.
  The helper retains its named-tuple return, returns scalars for scalar input,
  supports broadcasting and empty arrays, and does not mutate inputs. Explicit
  conversion of Angle inputs to degrees also fixes the old helper's treatment
  of radian `.value` as degrees.
- **Coordinate contract:** RA stays in [0, 360), Dec in [-90, 90]. Crossing a pole
  reflects Dec and shifts RA by 180°; coordinates exactly at a pole are unchanged.
  `RADec2Angle` preserves pairing up to the shorter input and returns Angle
  arrays, including one-element arrays for scalar input.
- **Calculated patch positions:** `getPatchPositions` collects projected
  coordinates and normalizes them in one batch, retaining per-patch projection
  where requested and scalar Angles in the returned mapping. Batches use lists
  because Astropy rejects tuples as ambiguous angle specifications.
- **String parsing:** A restricted fast path accepts colon-separated RA and
  colon- or dot-separated Dec with two-digit minutes/seconds, fractional seconds,
  and signed zero. It preserves Astropy's arithmetic order and unit conversion.
  Other formats, unsupported mixtures, invalid ranges, and warning boundaries
  fall back to Astropy, including 24 hours, 60 minutes/seconds, and fractional
  seconds that round to 60.
- **Python mean-shift:** Add the two squared floating-point components directly
  instead of using a general row reduction. Other dimensions and non-floating
  dtypes retain the reduction. Sequential in-place updates, thresholds, stopping
  rules, and cluster ordering remain unchanged; the compiled backend is unaffected.

No new dependencies were introduced.

## Performance evidence

Unprofiled local measurements used the external
`tests/sector_1.apparent_sky.txt`. Each stage had its own baseline run;
these are indicative timings, not statistically controlled results.

| Operation | Initial baseline → vectorization | Vectorization → parsing/distance changes |
| --- | ---: | ---: |
| Loading | 10.59 → 8.20 s | 7.96 → 2.21 s |
| Weighted patch positions, per-patch projection | 2.38 → 1.77 s | 1.74 → 1.78 s |
| Grouping, excluding loading | 24.85 → 24.44 s | 24.66 → 13.71 s |
| 100,000 numeric pairs supplied as lists | 3.89 → 1.72 s | Not targeted |

Vectorization improved loading and patch-position calculation about 1.3× and
numeric-list normalization 2.3×. The next stage improved loading 3.6× and grouping
1.8×, or about 2.4×/2.0× for loading plus patch positions/grouping respectively.
A repeat confirmed loading 8.05 → 2.26 s and grouping 24.45 → 13.97 s, with
patch-position time essentially unchanged. Both stages preserved exact patch
positions, group memberships, and group positions (2,119 groups); the initial
patch-position program's printed output was also byte-for-byte identical.

Under cProfile, the initial patch-position program fell from approximately
37 to 28 seconds; `RADec2Angle` originally accounted for about 28 seconds,
mostly during loading. These profiled times include instrumentation overhead.

## Validation and reproduction

Validation used `/home/marcel/code/rapthor/.tox/py/bin/python` with this checkout;
the system Python lacked Astropy. Historical focused runs passed:

| Stage | Tests passed | Coverage |
| --- | ---: | --- |
| Initial vectorization | 66 | Random coordinates, poles/wraps, radians, input formats, empty and unequal-length inputs |
| Parsing and distances | 82 | Parser equivalence/errors/warnings; distance boundaries, integers, exact float32/float64 trajectories and clusters |
| Shared helper review | 93 | Additionally: fixed expected results, scalar returns, broadcasting, and input immutability; two unrelated beam tests excluded |
| Scalar patch-position fix | 63 | Focused skymodel, tableio, and operation suites, including four new shape cases |

The shared-helper review used:

```sh
python -m pytest tests/test_operations_lib.py tests/test_tableio.py \
    tests/test_meanshift.py tests/test_skymodel.py tests/test_io.py \
    tests/test_lsmtool.py -k 'not apply_beam' -q --tb=short
```

The tracked [benchmark script](../../benchmarks/vectorize-radec-normalization/coordinate_performance.py)
and [usage notes](../../benchmarks/vectorize-radec-normalization/README.md)
reproduce the workloads without the original local programs:

```sh
python benchmarks/vectorize-radec-normalization/coordinate_performance.py \
    tests/resources/apparent.sky --baseline 7e7ac50 --repeats 2
```

That review-stage run preserved exact results for the small tracked model
(67 groups) and the numeric microbenchmark (1.675 → 1.662 s). Use the external
full-size model to reproduce the larger workloads above. The script defaults
to baseline `770343b`; `--baseline 770343b^` selects the pre-vectorization code.
The dedicated subdirectory, requested by the user, leaves room for independent
future investigations. Moving it required updating links, commands, and the
repository-root lookup; the relocated smoke test passed exact equality.

## Review follow-up: explicit patch positions must be scalar (2026-09-28)

The user supplied Greptile's report that explicit coordinates could produce
`(N, 1)` arrays from `getPatchPositions(asArray=True)`, questioning its claim
that vectorization introduced the issue. The shape bug was real, but the
attribution was incorrect: the old `RADec2Angle` also converted its temporary
lists into Angle arrays before returning them.

Loading `RADec2Angle` and `setPatchPositions` from `770343b^` into the current
environment reproduced shapes `[(1, 1), (1, 1)]` after setting
`{patch_name: [123.231, 23.4321]}`. The fixed setter returns `[(1,), (1,)]`.
This was a historical-function reproduction, not an entire historical checkout.

The fix extracts `ra[0]` and `dec[0]` when storing converted explicit numeric
or string positions. Each patch needs scalar metadata; extraction belongs in
`setPatchPositions`, preserving `RADec2Angle`'s established array-return contract
for callers that already index its output. Coordinate values and batched
calculation remain unchanged.

Four regression cases cover numeric/string inputs for one/two patches, checking
scalar metadata, values, selected-patch array shapes, and mixtures of updated
and untouched patches. All failed before the fix; this command then passed
63 tests:

```sh
python -m pytest tests/test_skymodel.py tests/test_tableio.py \
    tests/test_lsmtool.py -q -p no:cacheprovider --disable-warnings
```

**Benchmark limitation:** The benchmark calculates positions and groups models;
it does not supply explicit coordinate dictionaries to the setter. It restores
only historical `RADec2Angle`, `getPatchPositions`, and `euclid_distance`, using
current dependencies such as `setPatchPositions` and `normalize_ra_dec`.
Equality can preserve a bug shared by both variants. The shape tests therefore
check the API contract independently. No new performance measurements were
taken for this correctness fix.
