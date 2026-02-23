"""
run_pipeline.py — Complete AMD Mineral Mapping Pipeline
========================================================

Orchestrates the full three-stage workflow for EO-1 Hyperion AMD mapping.
Each stage builds on the previous one:

  Stage 1 — SAM Classification (multi_mineral_sam_fixed.py)
  ---------------------------------------------------------
  Computes NDVI/MNDWI-based soil mask, runs Spectral Angle Mapper against
  USGS endmember spectra, applies adaptive + null-model thresholds, runs
  post-classification validation (connected components, angular inertia,
  Moran's I), and saves all outputs including soil_mask.npy.
  Result: specific mineral-level detections (goethite, hematite, jarosite …)

  Stage 2 — Interactive Training Pixel Selection (training_pixel_selector.py)
  ---------------------------------------------------------------------------
  Opens an interactive matplotlib window where the user draws rectangular
  ROIs on three image composites (RGB, NIR, Fe-oxide ratio) to label
  training pixels for the binary AMD_FeOx / Background classification.
  Saves training_pixels.npz.  This stage is INTERACTIVE and requires a display.

  Stage 3 — Supervised SAM Classification (supervised_classification.py)
  -----------------------------------------------------------------------
  Loads training_pixels.npz and runs a supervised Spectral Angle Mapper
  using image-derived endmembers (mean spectrum per training class, L2-
  normalised).  This serves as the internal cross-validation of the primary
  library-driven SAM workflow — both methods use the identical SAM angular
  similarity logic, differing only in endmember source.

  A cross-method comparison between the supervised SAM AMD zone and the
  library SAM mineral map is computed automatically (Jaccard IoU, Cohen's
  Kappa, per-mineral containment), and divergences are interpreted as
  reflecting the spectral distance between USGS library references and
  actual Rio Tinto scene conditions.

  Result: broad AMD ferro-oxide lithology footprint (supervised SAM) with
  spatial agreement statistics against the primary library SAM output.

Usage
-----
    # Run all three stages in sequence
    python run_pipeline.py

    # Run individual stages
    python run_pipeline.py --stage sam
    python run_pipeline.py --stage select
    python run_pipeline.py --stage supervised

    # Skip the SAM stage if already computed (soil_mask.npy must exist)
    python run_pipeline.py --skip-sam

    # Run SAM + supervised classification only (skip interactive selection)
    python run_pipeline.py --skip-select

Dependencies
------------
Stage 1 (SAM):
    numpy, scipy, spectral, scikit-image, pandas, matplotlib
    esda, libpysal (for Moran's I)

Stage 2 (Training GUI):
    numpy, spectral, matplotlib, scipy

Stage 3 (Supervised SAM):
    numpy, spectral, scikit-learn, pandas, matplotlib
    esda, libpysal (for Moran's I)
    rasterio (optional, for GeoTIFF export)

Notes
-----
* Stage 2 is interactive.  When running in a headless environment, pass
  --skip-select and ensure training_pixels.npz already exists.
* Paths are inherited from the constants in each module.  Edit the HDR_FILE,
  OUTPUT_FOLDER, TRAINING_PIXELS_FILE, etc. in each module for custom paths.
* All outputs are written to amd_mapping/outputs/classifications/ (library SAM)
  and amd_mapping/outputs/supervised_sam/ (supervised SAM).
"""

import sys
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(title, width=70):
    """Print a bold section banner."""
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def _check_file(path, description):
    """Raise FileNotFoundError with a helpful message if *path* is absent."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{description} not found:\n  {p}\n"
            "Make sure the previous pipeline stage has completed successfully."
        )
    return p


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def run_sam_stage():
    """Run Stage 1: SAM classification and soil mask generation."""
    _section("STAGE 1 — SAM CLASSIFICATION  (multi_mineral_sam_fixed.py)")

    t0 = time.time()

    # Import and call the existing workflow's main() directly so that all
    # its configuration constants (HDR_FILE, OUTPUT_FOLDER, …) remain in
    # their original module and we do not duplicate them here.
    try:
        import multi_mineral_sam_fixed as sam
    except ImportError as exc:
        raise ImportError(
            "Could not import multi_mineral_sam_fixed.py. "
            "Ensure the file is in the same directory as run_pipeline.py."
        ) from exc

    sam.main()

    elapsed = time.time() - t0
    print(f"\n  Stage 1 completed in {elapsed / 60:.1f} min.")

    # Verify that soil_mask.npy was saved (written by sam.main() after the edit)
    soil_mask_path = Path(sam.HDR_FILE).parent / "soil_mask.npy"
    if not soil_mask_path.exists():
        print(
            f"\n  WARNING: soil_mask.npy not found at {soil_mask_path}.\n"
            "  Stages 2 and 3 will fall back to the valid-data mask.\n"
            "  Check that the SAM script completed without errors."
        )
    else:
        print(f"  soil_mask.npy confirmed at: {soil_mask_path}")


def run_selection_stage():
    """Run Stage 2: Interactive training pixel selection GUI."""
    _section("STAGE 2 — TRAINING PIXEL SELECTION  (training_pixel_selector.py)")

    print(
        "\n  This stage opens an interactive matplotlib window.\n"
        "  Binary classification: AMD_FeOx vs Background.\n\n"
        "  Instructions:\n"
        "    1. Switch to Fe-Oxide Ratio view (most diagnostic for AMD_FeOx).\n"
        "    2. Select 'AMD_FeOx' class and draw ROIs over iron-oxide zones\n"
        "       (rust-orange/yellow areas with high Fe-ratio values).\n"
        "    3. Select 'Background' class and draw ROIs over unaltered areas.\n"
        "    4. Click 'Save & Continue' to export training_pixels.npz and\n"
        "       close the window.\n"
        "\n  The soil mask overlay (yellow) shows which pixels are eligible.\n"
        "  The spectral profile panel updates live as you draw and hover.\n"
    )

    try:
        import training_pixel_selector as tps
    except ImportError as exc:
        raise ImportError(
            "Could not import training_pixel_selector.py. "
            "Ensure the file is in the same directory as run_pipeline.py."
        ) from exc

    # Verify the reflectance cube exists before opening the GUI
    hdr_path = Path(tps.HDR_FILE)
    _check_file(hdr_path, "Reflectance cube (.hdr)")

    selector = tps.TrainingPixelSelector(str(hdr_path))
    selector.show()   # blocks until the user clicks Save & Continue

    # Verify output was written
    npz_path = Path(tps.OUTPUT_FILE)
    if not npz_path.exists():
        print(
            f"\n  WARNING: training_pixels.npz was not saved "
            f"(expected at {npz_path}).\n"
            "  The window may have been closed without clicking "
            "'Save & Continue'.\n"
            "  Stage 3 will fail unless this file exists."
        )
    else:
        print(f"\n  training_pixels.npz saved at: {npz_path}")


def run_supervised_stage():
    """Run Stage 3: Supervised SAM classification and cross-method comparison."""
    _section(
        "STAGE 3 — SUPERVISED SAM CLASSIFICATION  (supervised_classification.py)"
    )

    t0 = time.time()

    try:
        import supervised_classification as sc
    except ImportError as exc:
        raise ImportError(
            "Could not import supervised_classification.py. "
            "Ensure the file is in the same directory as run_pipeline.py."
        ) from exc

    # Pre-flight checks before the heavy computation starts
    _check_file(sc.HDR_FILE,             "Reflectance cube (.hdr)")
    _check_file(sc.TRAINING_PIXELS_FILE, "Training pixel dictionary (.npz)")

    sc.run_supervised_classification()

    elapsed = time.time() - t0
    print(f"\n  Stage 3 completed in {elapsed / 60:.1f} min.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="AMD Mineral Mapping Pipeline — orchestrates library SAM + "
                    "training pixel selection + supervised SAM classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_pipeline.py                         # full pipeline (all 3 stages)
  python run_pipeline.py --stage sam             # Stage 1 only
  python run_pipeline.py --stage select          # Stage 2 only (interactive)
  python run_pipeline.py --stage supervised      # Stage 3 only
  python run_pipeline.py --skip-sam              # skip Stage 1 (soil mask must exist)
  python run_pipeline.py --skip-select           # skip Stage 2 (training_pixels.npz must exist)
        """,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--stage",
        choices=["sam", "select", "supervised"],
        help="Run only a single stage.",
    )
    mode_group.add_argument(
        "--skip-sam",
        action="store_true",
        help="Skip Stage 1 (SAM); soil_mask.npy must already exist.",
    )
    mode_group.add_argument(
        "--skip-select",
        action="store_true",
        help="Skip Stage 2 (interactive selection); training_pixels.npz must "
             "already exist.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  AMD MINERAL MAPPING PIPELINE")
    print("  EO-1 Hyperion Hyperspectral — Library SAM + Supervised SAM Workflow")
    print("=" * 70)
    print(f"  Working directory: {BASE_DIR}")

    pipeline_start = time.time()
    errors = []

    # Determine which stages to run
    run_sam        = True
    run_select     = True
    run_supervised = True

    if args.stage == "sam":
        run_select = run_supervised = False
    elif args.stage == "select":
        run_sam = run_supervised = False
    elif args.stage == "supervised":
        run_sam = run_select = False
    elif args.skip_sam:
        run_sam = False
    elif args.skip_select:
        run_select = False

    print(
        f"\n  Stages to run:"
        f"\n    Stage 1 — Library SAM classification :    "
        f"{'YES' if run_sam        else 'SKIPPED'}"
        f"\n    Stage 2 — Pixel selection            :    "
        f"{'YES' if run_select     else 'SKIPPED'}"
        f"\n    Stage 3 — Supervised SAM             :    "
        f"{'YES' if run_supervised else 'SKIPPED'}"
    )

    # ── Stage 1 ────────────────────────────────────────────────────────
    if run_sam:
        try:
            run_sam_stage()
        except Exception as exc:
            print(f"\n  [ERROR] Stage 1 failed: {exc}")
            errors.append(("Stage 1 (Library SAM)", exc))
            print("  Attempting to continue with remaining stages...")

    # ── Stage 2 ────────────────────────────────────────────────────────
    if run_select:
        try:
            run_selection_stage()
        except Exception as exc:
            print(f"\n  [ERROR] Stage 2 failed: {exc}")
            errors.append(("Stage 2 (Selection)", exc))
            print("  Attempting to continue with Stage 3...")

    # ── Stage 3 ────────────────────────────────────────────────────────
    if run_supervised:
        try:
            run_supervised_stage()
        except Exception as exc:
            print(f"\n  [ERROR] Stage 3 failed: {exc}")
            errors.append(("Stage 3 (Supervised SAM)", exc))

    # ── Summary ────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Total wall time: {total_elapsed / 60:.1f} min")

    if errors:
        print(f"\n  {len(errors)} stage(s) failed:")
        for stage_name, exc in errors:
            print(f"    ✗  {stage_name}: {type(exc).__name__}: {exc}")
        sys.exit(1)
    else:
        completed = sum([run_sam, run_select, run_supervised])
        print(f"\n  All {completed} requested stage(s) completed successfully.")
        print(
            "\n  Output locations:"
            "\n    Library SAM (multi-mineral) →  amd_mapping/outputs/classifications/"
            "\n    Supervised SAM              →  amd_mapping/outputs/supervised_sam/"
            "\n    Training pixels             →  amd_mapping/outputs/training_pixels.npz"
            "\n\n  Key outputs:"
            "\n    Library SAM  : mineral-specific map    classification_map.hdr"
            "\n    Supervised SAM: AMD_FeOx map           sam_classification_map.hdr"
            "\n    SAM cross-method: sam_cross_method_comparison.{csv,png}"
            "\n                      sam_per_mineral_containment.csv"
        )


if __name__ == "__main__":
    main()
