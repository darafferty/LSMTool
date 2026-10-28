"""
Compare benchmark results for different commits listed in a YAML file.

Use this script as follows:

    >>> python compare_benchmarks.py benchmark_commits.yaml

The script expects a single argument, the path to a YAML file containing the
labelled commits to benchmark. For example, a file with contents:

    ```yaml
    master: master
    feature: 1234567890abcdef1234567890abcdef12345678
    ```

Will run the benchmark tests for the master branch as well as for the specified
feature commit. Note that the labels (here "master and "feature") have no
special meaning (i.e. can be anything the user chooses) and are only used for
labelling the results.
"""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml
from git import Repo

# ---------------------------------------------------------------------------- #
# Constants

BENCHMARK_DIRECTORY = Path(__file__).parent
RESULTS_DIRECTORY = BENCHMARK_DIRECTORY / "results"
REPO = Repo(BENCHMARK_DIRECTORY, search_parent_directories=True)


# ---------------------------------------------------------------------------- #


def main(labelled_commits_yaml=None, *pytest_args):
    """
    Run the benchmark tests for every commit in the labelled commits YAML file.

    Parameters
    ----------
    labelled_commits_yaml : str
        Path to a YAML file containing the labelled commits to benchmark.
    """

    if not labelled_commits_yaml:
        current = REPO.git.branch("--show-current")
        labelled_commits_yaml = {"master": "master", current: current}
    else:
        labelled_commits_yaml = read_yaml(labelled_commits_yaml)

    # Read the commits from yaml file
    return_codes = [
        run_benchmark(commit, label, *pytest_args)
        for label, commit in labelled_commits_yaml.items()
    ]
    return any(return_codes)


def read_yaml(filename):
    """
    Read a dict of named commits from a YAML file and verify their existence in
    the repository.

    Parameters
    ----------
    filename : str
        Path to the YAML file to read.

    Returns
    -------
    dict
        Dictionary mapping labels to commits.
    """
    labelled_commits_yaml = Path(filename).resolve()
    if not labelled_commits_yaml.exists():
        raise FileNotFoundError(f"YAML file not found: {labelled_commits_yaml}")

    with open(filename) as stream:
        commits = yaml.safe_load(stream)

    if not isinstance(commits, dict) or not commits:
        raise ValueError(
            "Expected a non-empty YAML mapping of labels to commits."
        )

    return commits


def run_benchmark(commit, label, *pytest_args):
    """
    Run benchmarks for a specific commit and label.

    Parameters
    ----------
    commit : str
        The commit hash or branch name (commit-ish) to benchmark.
    label : str
        Label to use for saving the benchmark results.
    """

    # Add a temporary working tree to isolate any changes made during the
    # benchmarking from the users work tree. This will create a new working tree
    # linked to the repo. After the benchmarks have run, this worktree will be
    # removed.
    with TemporaryDirectory(prefix="lsmtool-benchmark-") as temporary_directory:
        worktree = Path(temporary_directory) / "worktree"
        try:
            print("Adding temporary worktree at:", worktree)
            print("... and checking out the commit:", commit)
            print(os.getcwd())

            # Check out the commit in the temporary worktree
            REPO.git.worktree("add", "--detach", str(worktree), commit)

            # Since the benchmark scripts may not exist on the current branch we
            # are testing, we need to check out the benchmark scripts from
            # master as well.
            # REPO.git.checkout("master", "--", "tests/benchmarks")

            pytest_command = [
                "tests/benchmarks/test_skymodel_benchmarks.py",
                *pytest_args,
                "--benchmark-only",
                "--benchmark-storage",
                str(RESULTS_DIRECTORY),
                "--benchmark-save",
                label,
            ]
            if commit != "master":
                # Find the reference filename for comparison
                reference_file = sorted(
                    RESULTS_DIRECTORY.rglob("*_master.json")
                )[-1]

                pytest_command.extend(
                    [
                        "--benchmark-compare",
                        reference_file.stem,
                        "--benchmark-compare-fail",
                        "mean:10%",
                        "--benchmark-sort",
                        "name",
                        "--benchmark-columns",
                        "mean, stddev, rounds",
                        "--benchmark-group-by",
                        "param:n_patches_xy",
                    ]
                )

            print(f"Running benchmark: {label}, Commit: {commit}")
            return pytest.main(pytest_command)

        finally:
            # Remove the temporary worktree after benchmarking
            print("Removing temporary worktree at:", worktree)
            if worktree.exists():
                REPO.git.worktree("remove", "--force", str(worktree))


if __name__ == "__main__":
    _, *args = sys.argv
    sys.exit(main(*args))
