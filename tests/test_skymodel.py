"""Unit tests for :mod:`lsmtool.skymodel`."""

import pytest

import lsmtool
from lsmtool.skymodel import SkyModel


@pytest.fixture
def grouped_skymodel():
    """A sky model with several multi-source patches."""
    return lsmtool.load("tests/resources/patches.sky")


def test_get_row_index_returns_the_contiguous_patch_members(grouped_skymodel):
    """Patch indices identify precisely the rows in the grouped table."""
    indices = grouped_skymodel.getRowIndex("bin1")

    assert indices == list(range(3, 11))
    assert set(grouped_skymodel.table["Patch"][indices]) == {"bin1"}


def test_set_patch_positions_calculates_expected_patch_midpoint(grouped_skymodel):
    """Midpoint positions are calculated correctly for multi-source patches."""
    grouped_skymodel.setPatchPositions(method="mid")

    position = grouped_skymodel.getPatchPositions()["bin0"]
    assert position[0].deg == pytest.approx(242.76155336)
    assert position[1].deg == pytest.approx(65.98411845)


def test_set_patch_positions_avoids_copying_helpers(
    grouped_skymodel, monkeypatch
):
    """Setting positions reads grouped column slices instead of copied columns."""
    def fail_if_called(*args, **kwargs):
        raise AssertionError("The patch-position path must not use copying helpers")

    monkeypatch.setattr(SkyModel, "getRowIndex", fail_if_called)
    monkeypatch.setattr(SkyModel, "getColValues", fail_if_called)

    grouped_skymodel.setPatchPositions(method="mid")

    assert grouped_skymodel.getPatchPositions()["bin1"][0].deg == pytest.approx(
        246.43713949
    )
