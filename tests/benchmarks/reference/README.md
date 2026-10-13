# Benchmark reference

This folder contains reference benchmark measurements for lsmtool regression
tests. The reference benchmark measurement was done on CI on:

- Date: 9 Oct 2026
- commit: 0adf6a48640a4342371ce345aca3635d730fe310

## Rotation

To replace the reference measurement with a more recent benchmark, do the following:

- Navigate to the [Pipelines section on LSMTool's gitlab](https://git.astron.nl/RD/LSMTool/-/pipelines)
  and click on the `run_benchmark_test` stage
- Download the job artifacts and extract them.
- The benchmark results will be named, e.g.: 
  `tests/benchmarks/results/Linux-CPython-3.13-64bit/0002_master.json`.
  Copy this file to the reference folder and rename it:

  ```bash
  cp tests/benchmarks/results/*/0002_master.json tests/benchmarks/reference/0001_reference.json
  ```

- Commit your updates and open an MR.
