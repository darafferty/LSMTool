#!/usr/bin/env python
"""
Example script to run filter_skymodel with bdsf on custom paths.

This script demonstrates how to use the filter_skymodel function from lsmtool
to filter a sky model based on PyBDSF source detection.
"""

from pathlib import Path
from lsmtool.filter_skymodel import bdsf

# ============================================================================
# CONFIGURE THESE PATHS FOR YOUR DATA
# ============================================================================

# Input paths (update these to your actual file locations)
FLAT_NOISE_IMAGE = Path("tests/resources/test_image.fits")  # Input FITS image
TRUE_SKY_IMAGE = Path("tests/resources/test_image.fits")     # True sky image (can be same as flat_noise_image)
INPUT_TRUE_SKYMODEL = Path("tests/resources/sector_1-sources-pb.txt")  # Input sky model
VERTICES_FILE = Path("tests/resources/expected_sector_1_vertices.npy")  # Field vertices
BEAM_MS = Path("tests/resources/LOFAR_HBA_MOCK.ms")  # Measurement set for beam model

# Output paths (where filtered results will be saved)
OUTPUT_DIR = Path("./output_example")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_TRUE_SKY = OUTPUT_DIR / "filtered_true_sky.txt"
OUTPUT_APPARENT_SKY = OUTPUT_DIR / "filtered_apparent_sky.txt"
OUTPUT_CATALOG = OUTPUT_DIR / "catalog.fits"
OUTPUT_TRUE_RMS = OUTPUT_DIR / "true_sky_rms.fits"
OUTPUT_FLAT_NOISE_RMS = OUTPUT_DIR / "flat_noise_rms.fits"

# PyBDSF parameters
THRESH_ISL = 4.0  # Island threshold (sigma)
THRESH_PIX = 5.0  # Pixel threshold (sigma)

# ============================================================================
# RUN THE FILTERING
# ============================================================================

def main():
    print("=" * 70)
    print("Running filter_skymodel with PyBDSF")
    print("=" * 70)
    print(f"\nInput image: {FLAT_NOISE_IMAGE}")
    print(f"Input skymodel: {INPUT_TRUE_SKYMODEL}")
    print(f"Beam MS: {BEAM_MS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nThreshold island: {THRESH_ISL} sigma")
    print(f"Threshold pixel: {THRESH_PIX} sigma")
    print("\n" + "-" * 70)
    
    # Run the filter_skymodel function
    bdsf.filter_skymodel(
        flat_noise_image=FLAT_NOISE_IMAGE,
        true_sky_image=TRUE_SKY_IMAGE,
        input_true_skymodel=INPUT_TRUE_SKYMODEL,
        input_apparent_skymodel=None,  # Will be derived from true sky model
        output_apparent_sky=OUTPUT_APPARENT_SKY,
        output_true_sky=OUTPUT_TRUE_SKY,
        vertices_file=VERTICES_FILE,
        beam_ms=BEAM_MS,
        thresh_isl=THRESH_ISL,
        thresh_pix=THRESH_PIX,
        save_filtered_model_image=True,  # Save FITS image of filtered model
        # Optional diagnostic outputs:
        output_catalog=OUTPUT_CATALOG,
        output_true_rms=OUTPUT_TRUE_RMS,
        output_flat_noise_rms=OUTPUT_FLAT_NOISE_RMS,
    )
    
    print("\n" + "=" * 70)
    print("Filtering completed successfully!")
    print("=" * 70)
    print("\nOutput files created:")
    print(f"  - True sky model: {OUTPUT_TRUE_SKY}")
    print(f"  - Apparent sky model: {OUTPUT_APPARENT_SKY}")
    print(f"  - PyBDSF catalog: {OUTPUT_CATALOG}")
    print(f"  - True sky RMS map: {OUTPUT_TRUE_RMS}")
    print(f"  - Flat noise RMS map: {OUTPUT_FLAT_NOISE_RMS}")
    
    # Check for the filtered model image
    filtered_model = OUTPUT_DIR.parent / f"{OUTPUT_TRUE_SKY.stem}-filtered-model.fits"
    if filtered_model.exists():
        print(f"  - Filtered model image: {filtered_model}")
    
    print("\n")


if __name__ == "__main__":
    main()
