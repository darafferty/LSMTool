#!/usr/bin/env python
"""
Generate and plot Voronoi facets from an LSMTool test sky model.

The script:

1. loads a test sky model (default: ``tests/resources/apparent.sky``);
2. groups the sources into patches with the ``tessellate`` algorithm
   (unless ``--keep-patches`` is given);
3. generates Voronoi facets around the patch positions (or directly around
   the individual sources with ``--seed source``) using
   :func:`lsmtool.facet.tessellate`;
4. refines the facets by splitting any that are too large using :func:`lsmtool.facet.split_large_facets`
5. plots the facets, sources, patch centres and any new facet centres created
   by splitting over a FITS background image
   (default: ``tests/resources/test_image.fits``).

Examples
--------
Run with the defaults and write ``facets_plot.png``::

    python plot_facets.py

Use a brighter target flux (fewer, larger facets)::

    python plot_facets.py --target-flux "1.0 Jy" --output facets_1Jy.png

Generate facets directly from individual source positions::

    python plot_facets.py --seed source

Plot without a background image and display the result interactively::

    python plot_facets.py --no-image --show
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Make sure the local LSMTool package is used when the script is run from a
# checkout (rather than an installed copy).
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from lsmtool import load  # noqa: E402
from lsmtool.facet import Facet, split_large_facets, tessellate  # noqa: E402

DEFAULT_SKYMODEL = REPO_ROOT / "tests" / "resources" / "apparent.sky"
DEFAULT_IMAGE = REPO_ROOT / "tests" / "resources" / "test_image.fits"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Voronoi facets from an LSMTool sky model and plot them "
            "over a background image."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skymodel",
        type=Path,
        default=DEFAULT_SKYMODEL,
        help="Input sky model (makesourcedb format).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="FITS image to use as the background.",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Do not plot a background image.",
    )
    parser.add_argument(
        "--seed",
        choices=["patch", "source"],
        default="patch",
        help="Seed the Voronoi tessellation with patch or source positions.",
    )
    parser.add_argument(
        "--target-flux",
        default="0.5 Jy",
        help=(
            "Target flux for the 'tessellate' grouping step. Only used when "
            "the model is (re)grouped."
        ),
    )
    parser.add_argument(
        "--keep-patches",
        action="store_true",
        help=(
            "Use the patches already present in the sky model instead of "
            "regrouping the sources."
        ),
    )
    parser.add_argument(
        "--flux-threshold",
        type=float,
        default=0.01,
        help=(
            "Minimum Stokes I flux in Jy for a source to be used when "
            "splitting large facets."
        ),
    )
    parser.add_argument(
        "--bbox-pad",
        type=float,
        default=0.2,
        help="Fractional padding around the seeds when making the facet grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("facets_plot.png"),
        help="Output plot filename.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image resolution.",
    )
    parser.add_argument(
        "--label-patches",
        action="store_true",
        help="Label the patch centres with their names.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot instead of only saving it.",
    )
    return parser.parse_args()


def group_sources(lsm, target_flux, keep_patches):
    """Group sources into patches if needed."""
    if keep_patches:
        if not lsm.hasPatches:
            raise ValueError(
                "The sky model has no patches, so --keep-patches cannot be "
                "used with --seed patch. Omit --keep-patches to group the "
                "sources, or use --seed source."
            )
        print(f"Using the {len(lsm.getPatchNames())} patches in the model.")
        return

    print(f"Grouping sources with target flux {target_flux} ...")
    lsm.group("tessellate", targetFlux=target_flux, method="mid")
    print(f"Grouped into {len(lsm.getPatchNames())} patches.")


def get_seeds(lsm, seed):
    """Return the seed coordinates and their names."""
    if seed == "patch":
        if not lsm.hasPatches:
            raise ValueError("The sky model must be grouped into patches.")
        ra, dec = lsm.getPatchPositions(asArray=True)
        names = np.asarray(lsm.getPatchNames())
    else:
        ra = lsm.getColValues("Ra")
        dec = lsm.getColValues("Dec")
        names = np.asarray(lsm.getColValues("Name"))

    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)

    # Discard any invalid positions
    valid = np.isfinite(ra) & np.isfinite(dec)
    if not np.all(valid):
        print(f"Ignoring {np.count_nonzero(~valid)} seed(s) with invalid "
              "positions.")
        ra = ra[valid]
        dec = dec[valid]
        names = names[valid]

    if len(ra) == 0:
        raise ValueError("No valid seed positions found.")

    return SkyCoord(ra, dec, unit="deg"), names


def find_new_centres(points, original_points, atol=1e-8):
    """Return the points that are not present in the original positions."""
    points = np.asarray(points, dtype=float)
    original_points = np.asarray(original_points, dtype=float)
    if len(points) == 0 or len(original_points) == 0:
        return points

    known = np.isclose(
        points[:, None, 0], original_points[None, :, 0], atol=atol
    ) & np.isclose(
        points[:, None, 1], original_points[None, :, 1], atol=atol
    )
    return points[~known.any(axis=1)]


def make_facets(lsm, seed, bbox_pad, flux_threshold):
    """Generate facets around the selected seed positions and refine them."""
    seeds, seed_names = get_seeds(lsm, seed)

    # Centre the tessellation on the mean seed position and make a bounding
    # box that comfortably encloses all the seeds.
    centre = SkyCoord(
        float(np.mean(seeds.ra.deg)),
        float(np.mean(seeds.dec.deg)),
        unit="deg",
    )
    max_separation = float(np.max(centre.separation(seeds)).deg)
    width = 2.0 * max_separation * (1.0 + bbox_pad)
    if width <= 0.0:
        width = 1.0
        print("All seeds are at the same position; using a 1 deg bounding "
              "box.")

    print(f"Generating facets for {len(seeds)} positions in a "
          f"{width:.2f} deg bounding box ...")
    facet_points, facet_polys = tessellate(seeds, centre, (width, width))
    print(f"Generated {len(facet_points)} facets")

    # Keep the original facet centres so that the new centres introduced by
    # the splitting step can be identified afterwards.
    initial_points = np.asarray(facet_points, dtype=float)

    # Refine the tessellation by splitting any facets that are too large.
    print("Splitting large facets ...")
    facet_points, facet_polys = split_large_facets(
        facet_points, facet_polys, lsm, flux_threshold/10, area_factor_threshold=1.3,
    )
    print(f"Refined to {len(facet_points)} facets")

    facet_points = np.asarray(facet_points, dtype=float)
    split_centres = find_new_centres(facet_points, initial_points)
    if len(split_centres) > 0:
        print(f"Added {len(split_centres)} new facet centre(s) by splitting.")

    # Match each facet back to the nearest seed so that it gets the right
    # name (and therefore the same colour as its sources).
    facet_coords = SkyCoord(facet_points, unit="deg")
    nearest = facet_coords[:, None].separation(seeds).argmin(axis=1)
    facet_names = [str(seed_names[i]) for i in nearest]

    facets = [
        Facet(name, float(point[0]), float(point[1]), polygon)
        for name, point, polygon in zip(
            facet_names, facet_points, facet_polys, strict=True
        )
    ]
    print(f"Generated {len(facets)} facets.")
    return facets, split_centres


def load_background(image_path):
    """Load a FITS image and its celestial WCS."""
    with fits.open(image_path) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header

    while data.ndim > 2:
        data = data[0]

    if data.ndim != 2:
        raise ValueError(
            f"Could not interpret {image_path} as a 2-D image."
        )

    wcs = WCS(header).celestial
    return data, wcs


def make_blank_wcs(centre, width):
    """Make a simple WCS for the case where no background image is used."""
    from lsmtool.operations_lib import make_wcs

    size = 1000
    wcs = make_wcs(centre.ra.deg, centre.dec.deg, width / size)
    wcs.wcs.crpix = [size / 2.0, size / 2.0]
    wcs.pixel_shape = (size, size)
    return wcs


def plot_map(
    lsm,
    facets,
    split_centres,
    *,
    image_path,
    output,
    dpi,
    label_patches,
    no_image,
    show,
    model_name,
    seed,
):
    """Plot the background image, sources, patch centres and facets.

    Any facet centres introduced by splitting large facets are highlighted
    with a separate marker.
    """
    import matplotlib

    if not os.environ.get("DISPLAY") and not os.environ.get(
        "WAYLAND_DISPLAY"
    ):
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from astropy.visualization import (
        AsinhStretch,
        ImageNormalize,
        PercentileInterval,
    )
    from matplotlib.lines import Line2D

    # Load the background image if requested and available.
    background = None
    wcs = None
    if not no_image:
        if image_path.is_file():
            try:
                background, wcs = load_background(image_path)
                print(f"Loaded background image: {image_path}")
            except Exception as error:  # noqa: BLE001
                print(f"Could not load {image_path}: {error}")
        else:
            print(f"Background image {image_path} not found; plotting "
                  "without one.")

    if wcs is None:
        seeds, _ = get_seeds(lsm, seed if lsm.hasPatches else "source")
        centre = SkyCoord(
            float(np.mean(seeds.ra.deg)),
            float(np.mean(seeds.dec.deg)),
            unit="deg",
        )
        max_separation = float(np.max(centre.separation(seeds)).deg)
        wcs = make_blank_wcs(centre, max(2.0 * max_separation * 1.2, 1.0))

    # Patch colours, used for both sources and facets.
    patch_names = lsm.getPatchNames() if lsm.hasPatches else []
    if len(patch_names) > 0:
        colour_map = matplotlib.colormaps["tab20"].resampled(
            max(len(patch_names), 2)
        )
        patch_colours = {
            name: colour_map(index)
            for index, name in enumerate(patch_names)
        }
    else:
        patch_colours = {}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection=wcs)

    if background is not None:
        norm = ImageNormalize(
            background,
            interval=PercentileInterval(99.0),
            stretch=AsinhStretch(),
        )
        image = ax.imshow(
            background,
            origin="lower",
            cmap="Greys_r",
            norm=norm,
            interpolation="nearest",
        )
        colourbar = fig.colorbar(image, ax=ax, pad=0.01, shrink=0.9)
        colourbar.set_label("Flux density (Jy/beam)")

    ra = np.asarray(lsm.getColValues("Ra"), dtype=float)
    dec = np.asarray(lsm.getColValues("Dec"), dtype=float)
    flux = np.asarray(lsm.getColValues("I"), dtype=float)
    flux = np.nan_to_num(flux, nan=0.0)

    source_patches = (
        np.asarray(lsm.getColValues("Patch")) if lsm.hasPatches else None
    )
    if source_patches is not None:
        fallback_colour = matplotlib.colors.to_rgba("tab:blue")
        source_colours = np.array(
            [
                patch_colours.get(name, fallback_colour)
                for name in source_patches
            ]
        )
    else:
        source_colours = np.full(
            (len(ra), 4), matplotlib.colors.to_rgba("tab:blue")
        )

    max_flux = np.nanmax(flux) if len(flux) else 1.0
    if max_flux <= 0.0:
        max_flux = 1.0
    sizes = 10.0 + 80.0 * np.sqrt(np.clip(flux, 0.0, None) / max_flux)

    x_sources, y_sources = wcs.world_to_pixel(
        SkyCoord(ra, dec, unit="deg")
    )
    ax.scatter(
        x_sources,
        y_sources,
        s=sizes,
        c=source_colours,
        edgecolors="black",
        linewidths=0.3,
        alpha=0.9,
        zorder=5,
    )

    facet_x = []
    facet_y = []
    for facet in facets:
        vertices = facet.vertices
        x, y = wcs.world_to_pixel(
            SkyCoord(vertices[:, 0], vertices[:, 1], unit="deg")
        )
        facet_x.extend(x)
        facet_y.extend(y)
        colour = patch_colours.get(facet.name, "tab:orange")
        ax.fill(x, y, color=colour, alpha=0.12, zorder=2)
        ax.plot(x, y, color=colour, linewidth=1.2, zorder=4)

    x_patches = []
    y_patches = []
    if lsm.hasPatches:
        patch_ra, patch_dec = lsm.getPatchPositions(asArray=True)
        x_patches, y_patches = wcs.world_to_pixel(
            SkyCoord(patch_ra, patch_dec, unit="deg")
        )
        ax.scatter(
            x_patches,
            y_patches,
            marker="*",
            s=180,
            c="gold",
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
        )

        if label_patches:
            for name, x, y in zip(
                patch_names, x_patches, y_patches, strict=True
            ):
                ax.annotate(
                    name,
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    ha="left",
                    va="bottom",
                    zorder=7,
                )

    # Show the facet centres that were created when large facets were split.
    x_split = []
    y_split = []
    if len(split_centres) > 0:
        split_centres = np.asarray(split_centres, dtype=float)
        x_split, y_split = wcs.world_to_pixel(
            SkyCoord(split_centres[:, 0], split_centres[:, 1], unit="deg")
        )
        ax.scatter(
            x_split,
            y_split,
            marker="*",
            s=90,
            c="magenta",
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
        )

    # Zoom to the area containing sources, patch centres and facets.
    all_x = [*x_sources, *facet_x, *x_patches, *x_split]
    all_y = [*y_sources, *facet_y, *y_patches, *y_split]
    x_pad = 0.08 * (np.max(all_x) - np.min(all_x))
    y_pad = 0.08 * (np.max(all_y) - np.min(all_y))
    ax.set_xlim(np.min(all_x) - x_pad, np.max(all_x) + x_pad)
    ax.set_ylim(np.min(all_y) - y_pad, np.max(all_y) + y_pad)

    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.coords.grid(color="grey", alpha=0.3, linestyle=":")

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor="tab:blue",
            markeredgecolor="black",
            label="sources",
        ),
        Line2D(
            [],
            [],
            color="0.5",
            linewidth=1.2,
            label="facet boundaries",
        ),
    ]
    if lsm.hasPatches:
        handles.append(
            Line2D(
                [],
                [],
                marker="*",
                linestyle="",
                markersize=12,
                markerfacecolor="gold",
                markeredgecolor="black",
                label="patch centres",
            )
        )
    if len(x_split) > 0:
        handles.append(
            Line2D(
                [],
                [],
                marker="*",
                linestyle="",
                markersize=8,
                markerfacecolor="magenta",
                markeredgecolor="black",
                label="new facet centres (split)",
            )
        )
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    fig.suptitle(f"Voronoi facets from {model_name} ({len(facets)} facets)")

    plt.show()
    # output.parent.mkdir(parents=True, exist_ok=True)
    # fig.savefig(output, dpi=dpi, bbox_inches="tight")
    # print(f"Saved plot to {output}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    """Run the facet generation and plotting pipeline."""
    args = parse_args()

    lsm = load(args.skymodel)
    print(f"Loaded {len(lsm)} sources from {args.skymodel}")

    if args.seed == "patch":
        group_sources(lsm, args.target_flux, args.keep_patches)

    facets, split_centres = make_facets(
        lsm, args.seed, args.bbox_pad, args.flux_threshold
    )
    plot_map(
        lsm,
        facets,
        split_centres,
        image_path=args.image,
        output=args.output,
        dpi=args.dpi,
        label_patches=args.label_patches,
        no_image=args.no_image,
        show=args.show,
        model_name=args.skymodel.name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
