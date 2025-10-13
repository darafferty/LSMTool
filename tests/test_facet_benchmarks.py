"""
Benchmarking different implementations of functions used in `tessellate`.
"""

import itertools as itt
from collections import defaultdict
import matplotlib.pyplot as plt
import json
from pathlib import Path

import numpy as np
import pytest


INDEX_OUTSIDE_DIAGRAM = -1
BBOX_SHAPE_FOR_XY_RANGES = (2, 2)


# ---------------------------------------------------------------------------- #


def in_box_v0(cal_coords, bounding_box):
    return np.logical_and(
        np.logical_and(
            bounding_box[0] <= cal_coords[:, 0],
            cal_coords[:, 0] <= bounding_box[1],
        ),
        np.logical_and(
            bounding_box[2] <= cal_coords[:, 1],
            cal_coords[:, 1] <= bounding_box[3],
        ),
    )


def in_box_v1(cal_coords, bounding_box):
    minx, maxx, miny, maxy = bounding_box
    minx, maxx = sorted([minx, maxx])
    miny, maxy = sorted([miny, maxy])
    x, y = np.transpose(cal_coords)
    return (minx <= x) & (x <= maxx) & (miny <= y) & (y <= maxy)


def in_box_v2(cal_coords, bounding_box):
    minx, maxx, miny, maxy = bounding_box
    minx, maxx = sorted([minx, maxx])
    miny, maxy = sorted([miny, maxy])
    x, y = cal_coords[..., 0], cal_coords[..., 1]
    return (minx <= x) & (x <= maxx) & (miny <= y) & (y <= maxy)


@pytest.mark.parametrize("version", range(3))
@pytest.mark.parametrize(
    "size", (10 ** np.linspace(1, 6, 11)).astype(int).tolist()
)
def test_in_box_benchmark(version, size, benchmark):
    # Act
    np.random.seed(0)
    bounding_box = [0, 1, 0, 1]
    data = np.random.rand(size, 2)
    in_box = eval(f"in_box_v{version}")
    benchmark(in_box, data, bounding_box)


# ---------------------------------------------------------------------------- #


def collect_results(filename):
    path = Path(filename)
    data = json.loads(path.read_text())

    db = defaultdict(lambda: defaultdict(list))
    for info in data["benchmarks"]:
        params = info["params"]
        stats = db[params["version"]]
        stats["n"].append(params["size"])
        for key, val in info["stats"].items():
            stats[key].append(val)

    return db


def plot_benchmark_results(ax, db, **kws):
    markers = itt.cycle("os^DvP*X")
    for version, stats in db.items():
        ax.errorbar(
            stats["n"],
            stats["median"],
            stats["iqr"],
            marker=next(markers),
            ms=5,
            capsize=0,
            lw=1,
            ls="-" if version == 0 else "",
            label=f"v={version}",
        )
    ax.legend()
    ax.grid()
    ax.set(xscale="log", yscale="log")


def main():
    test_folder = Path(__file__).parent
    path = test_folder / "benchmark/runtime-results/"
    files = sorted(path.rglob("*.json"))

    db = {}
    for json_file in files[-1:]:
        fig, axes = plt.subplots(figsize=(8, 5))
        name = json_file.stem[5:]

        # Collect results from benchmark
        stats = db[name] = collect_results(json_file)
        ax = axes  # [i]
        plot_benchmark_results(ax, stats)
        ax.set(title=name, xlabel="Number of facets", ylabel="Time [s]")

        versions = np.array(list(stats.keys()))
        fastest = np.argmin([s["median"] for s in stats.values()], 0)
        fastest_version_by_size, score = np.unique(
            versions[fastest], return_counts=True
        )
        print(name)
        print(
            "\n".join(
                map(
                    "{0:<9}|{1:<9}".format,
                    *zip(
                        *[
                            ("version", "score"),
                            *(zip(fastest_version_by_size, score)),
                        ]
                    ),
                )
            )
        )
        print()
        fig.savefig(test_folder / f"benchmark/{name}.pdf")


# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    np.random.seed(90909)
    main()
