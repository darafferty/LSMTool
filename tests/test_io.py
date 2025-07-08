import os
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from conftest import TEST_DATA_PATH

from lsmtool.io import (
    TRIAL_TMP_PATHS,
    _restore_tmpdir,
    _set_tmpdir,
    read_vertices_ra_dec,
    temp_storage,
)


@pytest.mark.parametrize(
    "trial_paths, expected_tmpdir",
    [
        pytest.param(["/tmp"], "/tmp", id="path_tmp"),
        pytest.param(["/var/tmp", "/tmp"], "/var/tmp", id="path_var_tmp"),
        pytest.param(
            ["/usr/tmp", "/var/tmp", "/tmp"],
            "/usr/tmp" if Path("/usr/tmp").exists() else "/var/tmp",
            id="path_usr_tmp",
        ),
    ],
)
@patch("lsmtool.io._set_tmpdir", wraps=_set_tmpdir)
@patch("lsmtool.io._restore_tmpdir", wraps=_restore_tmpdir)
def test_temp_storage(
    mock_restore_tmpdir,
    mock_set_tmpdir,
    trial_paths,
    expected_tmpdir,
):
    """Test the temp_storage context manager."""

    # Act
    with temp_storage(trial_paths):
        assert os.environ["TMPDIR"] == expected_tmpdir

    # Assert
    mock_set_tmpdir.assert_called_once_with(trial_paths)
    mock_restore_tmpdir.assert_called_once()


@patch("lsmtool.io._set_tmpdir", side_effect=ValueError("tmpdir error"))
@patch("lsmtool.io._restore_tmpdir", wraps=_restore_tmpdir)
def test_temp_storage_set_tmpdir_error(mock_restore_tmpdir, mock_set_tmpdir):
    """Test temp_storage when _set_tmpdir raises an error."""

    # Act
    with pytest.raises(ValueError, match="tmpdir error"):
        with temp_storage():
            pass

    # Assert
    mock_set_tmpdir.assert_called_once_with(TRIAL_TMP_PATHS)
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


def test_read_vertices_invalid(tmp_path):
    """Test reading vertices from pickle file."""
    path = tmp_path / "test_read_vertives_invalid.pkl"
    with path.open("wb") as file:
        pickle.dump("Invalid content", file)

    with pytest.raises(ValueError):
        read_vertices_ra_dec(path)
