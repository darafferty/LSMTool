"""
Plot benchmark results, comparing to a reference run.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------- #

BENCHMARK_DIR = Path(__file__).parent
REFERENCE_DIR = BENCHMARK_DIR / "reference"
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
    n_sources : np.ndarray
        Number of sources in the skymodel
    t_mean : np.ndarray
        Mean runtimes.
    t_std : np.ndarray
        Standard deviation of the runtimes.

    """
    data = json.loads(Path(filename).read_text())
    results = data["benchmarks"]
    n = len(results)
    n_sources, t_mean, t_std = np.empty((3, n))
    for i, result in enumerate(results):
        stats = result["stats"]
        n_sources[i] = result["extra_info"]["n_sources"]
        t_mean[i] = stats["mean"]
        t_std[i] = stats["stddev"]

    return n_sources, t_mean, t_std


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
    """
    for label, (x, y, y_err) in stats.items():
        ax.errorbar(x, y, yerr=y_err, fmt="o", label=label, **kws)

    ax.set(
        title=title,
        xlabel="Number of Sources",
        ylabel="Time (s)",
        xscale="log",
        yscale="log",
    )
    ax.grid()
    ax.legend()


def collect_benchmarks():
    """
    Collect benchmark results from the reference and results directories.

    Returns
    -------
    dict
        A dictionary mapping benchmark labels to their corresponding data.
    """
    benchmarks = {
        "reference": read_benchmark(next(REFERENCE_DIR.rglob("*.json")))
    }

    for result_path in RESULTS_DIR.rglob("*.json"):
        label = result_path.stem[5:]
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
    fig, ax = plt.subplots(1, 1)
    benchmarks = collect_benchmarks()
    plot_benchmarks(ax, benchmarks, title="SkyModel.getPatchPositions")

    if filename:
        fig.savefig(filename)
    else:
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
