#!/usr/bin/env python3

"""Compute the absolute Gaussian position angle for a skymodel file"""

import logging
from argparse import ArgumentParser

import astropy.units as u
from astropy.coordinates import Angle
from astropy.table import Table
from numpy import arctan2, cos, sin

logger = logging.getLogger(__name__)


def compute_absolute_orientation(
    relative_orientation, ra_source, dec_source, ra_center, dec_center
):
    """
    Compute the absolute position angle from a relative one, where the relative
    one was created from an image with center ra_center, dec_center.

    Parameters
    ----------
    relative_orientation : astropy.units.Quantity
        Position angle of the source with respect to the image.
    ra_source, dec_source : astropy.units.Quantity
        Right ascension and declination of the source.
    ra_center, dec_center : astropy.units.Quantity
        Right ascension and declination of the image center from which the
        relative position angle is measured.

    Returns
    -------
    astropy.units.Quantity
        Position angles with respect to the North Celestial Pole.
    """
    ra_diff = ra_source - ra_center
    dl_ddec = -sin(dec_source) * sin(ra_diff)
    # fmt: off
    dm_ddec = (
        cos(dec_source) * cos(dec_center) +
        sin(dec_source) * sin(dec_center) * cos(ra_diff)
    )  # fmt: on
    angle_towards_ncp = arctan2(dl_ddec, dm_ddec)
    return relative_orientation - angle_towards_ncp


def add_absolute_orientation(
    skymodel_filename_in, skymodel_filename_out, ra_center, dec_center
):
    """
    Add absolute orientation to Gaussian sources in the sky model.

    This function reads a sky model, computes the absolute orientation for
    Gaussian sources, updates the sky model, and saves it to a new file.

    Parameters
    ----------
    skymodel_filename_in : str or Path
        Input sky model filename.
    skymodel_filename_out : str or Path
        Output sky model filename.
    ra_center : astropy.units.Quantity
        Right ascension of the image center.
    dec_center : astropy.units.Quantity
        Declination of the image center.
    """
    skymodel_table = Table.read(
        str(skymodel_filename_in), format="makesourcedb"
    )
    if "Orientation" not in skymodel_table.columns:
        raise RuntimeError(
            f"No Orientation column present in {skymodel_filename_in}"
        )

    if "OrientationIsAbsolute" not in skymodel_table.columns:
        skymodel_table["OrientationIsAbsolute"] = False

    for row in skymodel_table:
        if row["Type"] == "GAUSSIAN":
            if row["OrientationIsAbsolute"]:
                logger.info("Orientation is already absolute, skipping.")
            else:
                absolute_orientation = compute_absolute_orientation(
                    row["Orientation"] * u.deg,
                    row["Ra"] * u.deg,
                    row["Dec"] * u.deg,
                    ra_center,
                    dec_center,
                )
            row["Orientation"] = absolute_orientation.to(u.deg).value
            row["OrientationIsAbsolute"] = True

    skymodel_table.write(str(skymodel_filename_out), format="makesourcedb")


def main():
    parser = ArgumentParser(
        description="Add absolute Gaussian position angle to a skymodel file"
    )
    parser.add_argument("skymodelfile", help="Skymodel file")
    parser.add_argument(
        "ra_center",
        help="Right ascension of center, e.g. '32.2deg' or '3h35m23.2'",
        type=str,
    )
    parser.add_argument(
        "dec_center",
        help="Declination of center, e.g. '12.5deg' or '5d35m17.8'",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output filename (default: derived from input filename)",
        type=str,
    )
    args = parser.parse_args()

    skymodel_filename_out = args.output
    if skymodel_filename_out is None:
        if args.skymodelfile.endswith((".skymodel", ".txt", ".sky")):
            base, suffix = args.skymodelfile.rsplit(".", 1)
            skymodel_filename_out = f"{base}_absolute_orientation{suffix}"
        else:
            skymodel_filename_out = f"{args.skymodelfile}.absolute_orientation"

    if args.ra_center.count(":") == 2:
        # Handle casacore format in e.g. msoverview output
        args.ra_center = args.ra_center.replace(":", "h", 1).replace(
            ":", "m", 1
        )

    if args.dec_center.count(".") == 3:
        args.dec_center = args.dec_center.replace(".", "d", 1).replace(
            ".", "m", 1
        )
    ra_center = Angle(args.ra_center)
    dec_center = Angle(args.dec_center)
    add_absolute_orientation(
        args.skymodelfile, skymodel_filename_out, ra_center, dec_center
    )
    logger.info(f"Saved corrected skymodel to {skymodel_filename_out}")
