import contextlib as ctx
import os
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from conftest import TEST_DATA_PATH

from lsmtool.io import (
    _restore_tmpdir,
    _set_tmpdir,
    read_vertices_ra_dec,
    temp_storage,
)


@pytest.mark.parametrize(
    "trial_paths, expected_tmpdir, context",
    [
        # Nominal test cases
        pytest.param(["/tmp"], "/tmp", None, id="path_tmp"),
        pytest.param(
            ["/var/tmp", "/tmp"], "/var/tmp", None, id="path_var_tmp"
        ),
        pytest.param(
            ["/usr/tmp", "/var/tmp", "/tmp"],
            "/usr/tmp" if Path("/usr/tmp").exists() else "/var/tmp",
            None,
            id="path_usr_tmp",
        ),
        # Error test cases
        pytest.param(
            None,
            None,
            pytest.raises(NotADirectoryError),
            id="invalid_trial_paths_none",
        ),
        pytest.param(
            [],
            [],
            pytest.raises(NotADirectoryError),
            id="invalid_trial_paths_empty",
        ),
    ],
)
@patch("lsmtool.io._restore_tmpdir", wraps=_restore_tmpdir)
@patch("lsmtool.io._set_tmpdir", wraps=_set_tmpdir)
def test_temp_storage(
    mock_set_tmpdir, mock_restore_tmpdir, trial_paths, expected_tmpdir, context
):
    """Test the temp_storage context manager."""

    # Act
    with context or ctx.nullcontext():
        with temp_storage(trial_paths) as tmp:
            assert os.environ["TMPDIR"] == expected_tmpdir == str(tmp)

    # Assert
    mock_set_tmpdir.assert_called_once_with(trial_paths)
    mock_restore_tmpdir.assert_called_once()


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param(
            (TEST_DATA_PATH / "expected_sector_1_vertices.pkl"),
            id="path_input",
        ),
        pytest.param(
            str((TEST_DATA_PATH / "expected_sector_1_vertices.pkl")),
            id="string_input",
        ),
    ],
)
def test_read_vertices_ra_dec(filename):
    """Test reading vertices from pickle file."""
    verts = read_vertices_ra_dec(filename)
    expected = (
        (265.2866140036157, 53.393467021582275),
        (266.78226621292583, 61.02229999320357),
        (250.90915045307418, 61.02229999320357),
        (252.40480266238433, 53.393467021582275),
        (265.2866140036157, 53.393467021582275),
    )
    assert np.allclose(verts, expected)


@pytest.mark.parametrize(
    "contents", ["Invalid content", ["Invalid", "content"]]
)
def test_read_vertices_invalid(tmp_path, contents):
    """Test reading vertices from pickle file."""
    path = tmp_path / "test_read_vertices_invalid.pkl"
    path.unlink(missing_ok=True)
    with path.open("wb") as file:
        pickle.dump(contents, file)

    with pytest.raises(ValueError):
        read_vertices_ra_dec(path)


def test_read_vertices_non_existent():
    with pytest.raises(FileNotFoundError):
        read_vertices_ra_dec("/path/to/vertices.file")
