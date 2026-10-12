# Benchmark reference

This folder contains reference benchmark measurements for lsmtool regression
tests. The reference benchmark measurement was done on CI on:

- Date: 9 Oct 2026
- commit: 0adf6a48640a4342371ce345aca3635d730fe310

## Rotation

To produce a new reference measurement run the following commands:

```bash
rm -r tests/benchmarks/reference/Linux*
git checkout master
python -m pytest --benchmark-only --benchmark-storage tests/benchmarks/reference --benchmark-save reference
```