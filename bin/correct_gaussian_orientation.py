#!/usr/bin/env python3

"""Compute the absolute Gaussian position angle for a skymodel file"""

from astropy.table import Table
from argparse import ArgumentParser
from astropy.coordinates import Angle
import astropy.units as u
from lsmtool import tableio  # Registers makesourcedb reader and writer with astropy.tables
from numpy import sin, cos, arctan2
import logging


logger = logging.getLogger(__name__)

def compute_absolute_orientation(
    relative_orientation, ra_source, dec_source, ra_center, dec_center
):
    """Compute the absolute position angle from a relative one, where the relative one was created from an image with center ra_center, dec_center.

    Args:
        relative_orientation (astropy Quantity): position angle with respect to Norh in the image
        ra_center (astropy Quantity): right ascension of the center of the image from which the relative position angle was deduced
        dec_center (astropy Quantity): declination of the center of the image from which the relative position angle was deduced
    """
    ra_diff = ra_source - ra_center
    dl_ddec = -sin(dec_source) * sin(ra_diff)
    dm_ddec = cos(dec_source) * cos(dec_center) + sin(dec_source) * sin(dec_center) * cos(ra_diff)
    angle_towards_ncp = arctan2(dl_ddec, dm_ddec)
    return relative_orientation - angle_towards_ncp


def add_absolute_orientation(skymodel_filename_in, skymodel_filename_out, ra_center, dec_center):
    """Add absolute position for all gaussian sources"""
    skymodel_table = Table.read(skymodel_filename_in, format="makesourcedb")
    orientation_is_absolute_column_present = "OrientationIsAbsolute" in skymodel_table.columns
    if "Orientation" not in skymodel_table.columns:
        raise RuntimeError("No Orientation column present in " + skymodel_filename_in)
    if not orientation_is_absolute_column_present:
        skymodel_table["OrientationIsAbsolute"] = False
    for row in skymodel_table:
        if row["Type"] == "GAUSSIAN":
            if row["OrientationIsAbsolute"]:
                print("Orientation is already absolute, skipping")
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
    skymodel_table.write(skymodel_filename_out, format="makesourcedb")


if __name__ == "__main__":
    parser = ArgumentParser(description="Add absolute Gaussian position angle to a skymodel file")
    parser.add_argument("skymodelfile", help="Skymodel file")
    parser.add_argument(
        "ra_center", help="Right ascension of center, e.g. '32.2deg' or '3h35m23.2'", type=str
    )
    parser.add_argument(
        "dec_center", help="Declination of center, e.g. '12.5deg' or '5d35m17.8'", type=str
    )
    parser.add_argument(
        "-o", "--output", help="Output filename (default: derived from input filename)", type=str
    )
    args = parser.parse_args()

    skymodel_filename_out = args.output
    if skymodel_filename_out is None:
        if args.skymodelfile.endswith(".skymodel") or args.skymodelfile.endswith(".txt"):
            filename_parts = args.skymodelfile.rsplit(".")
            skymodel_filename_out = filename_parts[0] + "_absolute_orientation." + filename_parts[1]
        else:
            skymodel_filename_out = filename + ".absolute_orientation"
    if args.ra_center.count(':') == 2:  # Handle casacore format in e.g. msoverview output
        args.ra_center = args.ra_center.replace(":", "h", 1).replace(":", "m", 1)
    if args.dec_center.count(".") == 3:
        args.dec_center = args.dec_center.replace(".", "d", 1).replace(".", "m", 1)
    ra_center = Angle(args.ra_center)
    dec_center = Angle(args.dec_center)
    add_absolute_orientation(args.skymodelfile, skymodel_filename_out, ra_center, dec_center)
    logger.info("Saved corrected skymodel to " + skymodel_filename_out)
