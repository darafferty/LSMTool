import pickle

import numpy as np
import pytest
from conftest import TEST_DATA_PATH

from lsmtool.io import read_vertices_ra_dec


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

