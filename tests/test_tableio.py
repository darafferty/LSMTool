from astropy.table import Table
import pytest

from lsmtool.tableio import skyModelReader

def test_skymodelreader_emptyfile(tmp_path):
    """ Test skyModelReader raises IOError for empty file """

    # Create a fake existing sky model file to test functionality
    skymodel_path = tmp_path/ "empty.sky"
    skymodel_path.touch()

    with pytest.raises(IOError):
        skyModelReader(str(skymodel_path))

def test_skymodelreader_headeronly(tmp_path):
    """ Test skyModelReader returns empty table for header-only file """

    # Create a fake existing sky model file to test functionality
    skymodel_path = tmp_path/ "empty.sky"
    skymodel_path.write_text(
        "FORMAT = Name, Type, Ra, Dec, I\n"
    )

    table = skyModelReader(str(skymodel_path))

    assert isinstance(table,Table)
    assert len(table) == 0