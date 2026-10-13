# Benchmark results

This folder contains benchmark results for LSMTool regression tests. The
reference benchmark measurement file is also copied to this folder for
comparison.

To run the benchmark and compare with the reference measurement, execute the
`tox` environment and optionally pass a label that will be used in the
filename.

```bash
tox -e benchmark -- --benchmark-save=work
```

After running the benchmarks, the `results` folder will contain the measured
results under a folder describing the platform and python build version:

```text
results
├── 0001_reference.json         (Reference measurement for comparison)
├── benchmark_compare.png       (Plot comapring current measurements with reference)
├── Linux-CPython-3.10-64bit    (Benchmark results from previous run)
│   ├── 0002_work.json
│   └── ...                      
└── README.md                   (This file)
```
