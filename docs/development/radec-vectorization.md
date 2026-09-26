# RADec2Angle performance investigation

Date: 2026-09-26

## Conversation record

The user reported that the memory improvements in MR RD/LSMTool!149 also
improved runtime by a factor of 7–8 when running the local programs
`tests/test_get_patch_positions.py` and `tests/test_group.py`. Profiling those
programs with `python -m cProfile` suggested that `RADec2Angle` in
`lsmtool/tableio.py` had become the bottleneck. The user asked whether further
speed improvements, particularly vectorization, were possible.

The assistant inspected the implementation and callers, collected baseline
profiles, implemented vectorized normalization and batched patch-position
normalization, and checked correctness and runtime. The assistant reported
roughly 1.3× faster loading and patch-position calculation, with little change
to mean-shift grouping time itself. Coordinate-string parsing and mean-shift
distance calculations were identified as possible subsequent targets.

The user then requested that these changes be committed on a new, separate
branch, together with a record of the conversation and the solution rationale.
This document records that discussion and the technical basis for the change;
it is a summary rather than a verbatim transcript.

## Technical rationale

`RADec2Angle` already parsed input coordinates in batches, but iterated over
Astropy Angle arrays to normalize each coordinate pair individually. Iteration
creates scalar Angle objects and incurs Python and Astropy overhead for every
pair. The replacement performs modulo arithmetic and pole reflection on NumPy
arrays in degrees, then constructs the two output Angle arrays once.

The normalization preserves RA in [0, 360) and Dec in [-90, 90]. Declinations
beyond a pole are reflected, and their corresponding right ascensions are
shifted by 180 degrees. Coordinates exactly at either pole are not reflected.
The shorter input determines the number of returned pairs, preserving the
previous zip behavior. Scalar inputs still return one-element Angle arrays;
array inputs and empty arrays are also supported. Existing string parsing,
including makesourcedb declination notation, remains in place.

`getPatchPositions` previously called `RADec2Angle` once for every calculated
patch position. It now collects the projected positions and normalizes them in
one call. Projection remains per patch where requested. The returned mapping
still contains individual RA and Dec Angle values for each patch.

An initial batching implementation passed tuples to Astropy, which rejects
those as ambiguous angle specifications. Tests exposed this, and the final
implementation explicitly converts the batches to lists.

The work deliberately retains Astropy's coordinate-string parser. Replacing
that parser would need separate investigation of accepted formats and error
handling. Mean-shift distance calculations are also a separate optimization:
they dominate grouping time after loading and are unaffected by this change.

## Validation and measurements

The local input was `tests/sector_1.apparent_sky.txt`. The two user-provided
programs and that input were already untracked before this work and are not
part of the implementation commit.

The system Python lacked Astropy, so validation used the existing environment
at `/home/marcel/code/rapthor/.tox/py/bin/python`, importing LSMTool from this
checkout.

Focused test suites:

- `tests/test_tableio.py`
- `tests/test_skymodel.py`
- `tests/test_io.py`
- `tests/test_lsmtool.py`

Together these passed 66 tests. Added tests cover random coordinate pairs,
wraps and pole crossings, Angle arrays expressed in radians, scalar and list
inputs, string formats, empty inputs, and unequal input lengths. Expected
normalization values are computed using the existing scalar
`normalize_ra_dec` implementation.

The patch-position program's printed output was byte-for-byte identical
before and after the change. A separate comparison loaded the original
functions from Git HEAD and ran the original and modified implementations
sequentially in the same process. Patch positions, group memberships, and
group positions were exactly identical; grouping produced 2,119 groups.

Unprofiled single-run elapsed times from that comparison:

| Operation | Before | After |
| --- | ---: | ---: |
| Model loading | 10.59 s | 8.20 s |
| Weighted patch positions, per-patch projection | 2.38 s | 1.77 s |
| Mean-shift grouping, excluding loading | 24.85 s | 24.44 s |
| Normalizing 100,000 numeric coordinate pairs supplied as lists | 3.89 s | 1.72 s |

These are indicative measurements on this machine, not statistical benchmark
results. Loading and patch-position calculation improved by about 1.3×;
the numeric-list normalization benchmark improved by about 2.3×.

Under cProfile, the patch-position program took approximately 37 seconds
before and 28 seconds after the change. In the original profile,
`RADec2Angle` accounted for approximately 28 seconds, predominantly during
loading. Profiling overhead makes these numbers unsuitable as substitutes for
unprofiled elapsed times.

Temporary benchmark scripts, logs, and profiles were stored under `/tmp` and
are not repository artifacts. No new dependencies were introduced.

## Follow-up: parsing and mean-shift distances

After commit `770343b`, the user asked to continue with the further optimization
opportunities. The follow-up remains on `perf/vectorize-radec-normalization`.

### Changes and rationale

A restricted fast path now parses ordinary sexagesimal strings with two-digit
minutes and seconds. It accepts colon-separated RA and colon- or dot-separated
Dec, including fractional seconds and signed zero. Conversion uses the same
arithmetic order and Astropy unit conversion as the general parser. Other
formats, mixed unsupported formats, invalid ranges, and boundary values that
produce Astropy warnings fall back to the original parser. In particular,
24 hours and 60 minutes/seconds retain their warnings; decimal seconds that
round to 60 also fall back. This avoids invoking Astropy's general grammar
parser and creating an Angle object for every ordinary coordinate string.

The Python mean-shift implementation now computes two-dimensional Euclidean
distances by adding the two squared components directly. This avoids the
cost of a general NumPy row reduction for each neighbourhood search. Other
array dimensions and non-floating dtypes retain the general reduction. The
algorithm's sequential, in-place coordinate updates, distance thresholds,
iteration stopping rule, and cluster ordering are unchanged. The optional
compiled grouper is unaffected.

### Validation

The five focused suites (`test_tableio`, `test_meanshift`, `test_skymodel`,
`test_io`, and `test_lsmtool`) passed 82 tests. New tests compare the fast parser
against the Astropy fallback for random and boundary coordinates, alternate
formats, warnings, and invalid inputs. Mean-shift tests check distances at the
neighbourhood boundary, integer inputs, and exact agreement of all saved
iteration coordinates and final clusters for float32 and float64 inputs.

The same full-model comparison used above was repeated against commit
`770343b`, additionally restoring its original mean-shift distance method for
the baseline. Patch positions, group memberships, and group positions remained
exactly identical, with 2,119 final groups.

Initial unprofiled timings for this follow-up:

| Operation | Commit 770343b | Follow-up |
| --- | ---: | ---: |
| Model loading | 7.96 s | 2.21 s |
| Weighted patch positions, per-patch projection | 1.74 s | 1.78 s |
| Mean-shift grouping, excluding loading | 24.66 s | 13.71 s |

Loading improved by about 3.6× and grouping by about 1.8×. Patch-position
calculation and numeric-coordinate normalization were not targeted in this
follow-up. Summing loading and the corresponding operation gives roughly 2.4×
for the patch-position workload and 2.0× for the grouping workload, excluding
interpreter startup and printing. These are local measurements, not guaranteed
speedups across machines, coordinate formats, or the optional compiled grouper.

A second run confirmed similar timings: loading 8.05 → 2.26 seconds,
patch positions 1.74 → 1.79 seconds, and grouping 24.45 → 13.97 seconds.
The full-model equality checks passed again.
