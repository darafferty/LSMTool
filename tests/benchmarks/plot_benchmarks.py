"""
Plot benchmark results, comparing to a reference run.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- #

BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"


# ---------------------------------------------------------------------------- #


def read_benchmark(filename):
    """
    Read benchmark data from a JSON file and return x, y, and y_err arrays.

    Parameters
    ----------
    filename : Path or str
        Path to the JSON file containing the benchmark data.

    Returns
    -------
    data : dict
        A dictionary mapping the number of patches to a 3xN array containing
        the number of sources, mean runtimes, and standard deviation of the
        runtimes for each benchmark result.
    """
    data = json.loads(Path(filename).read_text())
    results = data["benchmarks"]
    data = defaultdict(list)
    for result in results:
        stats = result["stats"]
        n_patches = result["extra_info"]["n_patches"]
        data[n_patches].append(
            (
                result["extra_info"]["n_sources"],
                stats["mean"],
                stats["stddev"],
            )
        )

    return data


def plot_benchmarks(ax, stats, title, **kws):
    """
    Create an errorbar plot of the benchmark results in the file path on the
    given matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    stats : dict
        A dictionary mapping benchmark labels to their corresponding data.
    title : str
        Title for the axes.
    **kws
        Additional keyword arguments passed to `matplotlib.axes.Axes.errorbar`.
    """
    markers_sequence = iter(("o^vsdx"))
    markers = defaultdict(lambda: next(markers_sequence, None))
    colours = {"reference": "k"}
    legend = {}
    for label, stats_per_label in stats.items():
        for n_patches, data in stats_per_label.items():
            (n_sources, t, t_err) = zip(*data, strict=True)
            legend[n_patches, label] = art = ax.errorbar(
                n_sources,
                t,
                yerr=t_err,
                fmt=markers[n_patches],
                color=colours.get(label, None),
                label=f"{label}: {n_patches} patches",
                **kws,
            )
            colours[label] = art[0].get_color()

    ax.set(
        title=title,
        xlabel="Number of Sources",
        ylabel="Time (s)",
        xscale="log",
        yscale="log",
    )
    ax.grid()
    # Sort the legend entries by number of patches and whether the label starts
    # with "ref"
    _, artists = zip(
        *sorted(
            legend.items(),
            key=lambda x: (x[0][0], not x[0][1].startswith("ref")),
        ),
        strict=True,
    )
    ax.legend(handles=artists, ncols=2, loc="upper left")


def collect_benchmarks():
    """
    Collect benchmark results from the reference and results directories.

    Returns
    -------
    dict
        A dictionary mapping benchmark labels to their corresponding data.
    """
    benchmarks = {}
    counters = defaultdict(int)
    for result_path in RESULTS_DIR.rglob("*.json"):
        label = result_path.stem[5:]
        if label in benchmarks:
            counters[label] += 1
            label = f"{label}-{counters[label]}"

        benchmarks[label] = read_benchmark(result_path)

    return benchmarks


def main(filename=None):
    """
    Plot the benchmark results and save the figure if a filename is provided.

    Parameters
    ----------
    filename : Path or str, optional
        Path to save the figure to. If not provided, the plot will not be saved.
    """
    fig, ax = plt.subplots(1, 1, figsize=[10.5, 6.8], tight_layout=True)
    benchmarks = collect_benchmarks()
    plot_benchmarks(ax, benchmarks, title="SkyModel.getPatchPositions")

    if filename:
        fig.savefig(filename)
    else:
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
