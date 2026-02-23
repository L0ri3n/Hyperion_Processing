# Documentation Index
## Hyperion AMD Mineral Mapping Project

**Repository:** PROCESSING_AND_POST
**Last Updated:** February 2026 (v1.4 — supervised SAM replaces Random Forest description)
**Project:** Remote Sensing - AMD Mineral Detection using Hyperion Hyperspectral Imagery

---

## Table of Contents
1. [Guides](#guides)
2. [Changelogs](#changelogs)
3. [Quick Navigation](#quick-navigation)

---

## Guides

Comprehensive documentation for setting up and using the Hyperion AMD mineral mapping workflow.

### [implementation_checklist.md](guides/implementation_checklist.md)
**Purpose:** Complete project implementation checklist and requirements
**Scope:** Software setup, data requirements, step-by-step workflow

**Summary:**
- **Software Requirements:** Python environment (conda), QGIS 3.28+, SNAP 9.0+, all required libraries
- **Data Requirements:** Hyperion imagery (atmospherically corrected), USGS Spectral Library, DEM, Sentinel-2 validation data
- **8-Step Workflow:**
  1. Preprocessing with SUREHYP
  2. Spectral library construction (USGS + publications)
  3. Spectral enhancement (Savitzky-Golay smoothing, continuum removal)
  4. SAM classification
  5. MTMF abundance mapping
  6. Postprocessing (morphological filtering)
  7. Validation (published maps, geochemistry, spatial consistency)
  8. Final products and documentation
- **Timeline:** 12-16 weeks part-time (15-20 hrs/week)
- **Troubleshooting Guide:** Common issues and solutions
- **Performance Optimization:** Vectorization, parallelization, memory management

**Key Minerals:** Jarosite, Goethite, Hematite, Schwertmannite, Kaolinite, Illite, Gypsum

---

### [qgis_snap_workflows.md](guides/qgis_snap_workflows.md)
**Purpose:** QGIS and SNAP software workflows for visualization and analysis
**Scope:** GUI-based operations, map production, spatial analysis

**Summary:**
- **QGIS Workflows:**
  - RGB composite creation (natural, false color, SWIR)
  - SAM classification using OTB
  - Mask creation (NDVI, water, combined)
  - Styling mineral maps with color schemes
  - Print layout creation for publication
  - Spatial analysis (area calculation, proximity analysis)
  - Spectral profile extraction

- **SNAP Workflows:**
  - Hyperion data import
  - Band math for masks
  - Linear spectral unmixing (FCLS)
  - RGB visualization
  - Export formats (GeoTIFF, BEAM-DIMAP)

- **Integrated Workflow Recommendations:** Optimal tool selection by task
- **Troubleshooting:** Common QGIS and SNAP issues with solutions

**Best Practices:** Use Python for preprocessing/classification, QGIS for visualization/maps, SNAP for alternative unmixing

---

### [quick_start_guide.md](guides/quick_start_guide.md)
**Purpose:** Copy-paste ready commands and code snippets
**Scope:** Quick reference for common operations

**Summary:**
- **Environment Setup:** One-command conda environment creation
- **Spectral Library:** Ready-to-use code for loading USGS spectra
- **Resampling:** Function to match Hyperion wavelengths
- **Smoothing:** Savitzky-Golay filter implementation
- **Masking:** NDVI-based vegetation mask creation
- **SAM Classification:** Single mineral and multi-mineral implementations
- **MTMF:** Matched filter code for abundance mapping
- **Postprocessing:** Combining SAM+MTMF, morphological cleaning
- **Statistics:** Area calculation and spectral plotting
- **QGIS Integration:** Python console commands
- **Validation:** Extract values at points for geochemical correlation

**Format:** All code blocks are complete and executable

---

### [WORKFLOW_DIAGRAM.md](guides/WORKFLOW_DIAGRAM.md)
**Purpose:** Visual representation of the complete workflow
**Scope:** Flowcharts, decision trees, timelines

**Summary:**
- **Complete Workflow Overview:** ASCII flowchart of all 8 steps
- **Decision Tree:** When to use which tool (Python, QGIS, SNAP)
- **Method Selection Guide:** Choosing classification algorithms
- **Validation Strategy:** Flowchart for validation approaches
- **Threshold Optimization:** Process for tuning SAM/MTMF parameters
- **Data Flow Diagram:** How data moves through pipeline
- **Timeline Visualization:** 12-week project timeline

**Visual Tools:** ASCII art diagrams for quick reference

---

### [SETUP_AND_USAGE_GUIDE.md](guides/SETUP_AND_USAGE_GUIDE.md)
**Purpose:** Detailed setup and execution instructions
**Scope:** Environment setup, data preparation, workflow execution

**Summary:**
- **Environment Status:** Conda environment `hyperion` with all packages installed
- **Required Data Structure:** Directory organization for project
- **Step-by-Step Execution:**
  - Step 0: Data preparation (Hyperion cube, spectral library)
  - Step 1: Verify setup
  - Step 2: Run workflow (automatic or step-by-step)
- **Expected Outputs:** Description of all generated files
- **Troubleshooting:** Common errors and solutions
- **Parameter Adjustment:** SAM threshold, MTMF thresholds, smoothing parameters
- **QGIS Visualization:** Converting outputs to GeoTIFF

**Includes:** Complete step-by-step script example

---

## Changelogs

Technical reports documenting fixes, improvements, and integrations.

### [SAM_Implementation_Report.md](changelogs/SAM_Implementation_Report.md)
**Date:** Recent implementation
**Issue:** SAM classification returning 0% classified pixels

**Summary:**
- **Problem Identified:**
  - Scale mismatch between Hyperion cube and endmember library
  - Band count mismatch (196 vs 224 bands)
  - Invalid/zero pixels not properly handled

- **Root Causes:**
  1. **Primary Issue:** Reflectance scale mismatch (0-10000 vs 0-1)
  2. Hyperion data in integer format (DN values)
  3. USGS library in float format (reflectance)
  4. Spectral angle calculations producing invalid cosine values

- **Solutions Implemented:**
  1. Scale checking and auto-correction code
  2. Band alignment validation
  3. Pre-SAM validation function
  4. Alternative PySptools implementation

- **Result:** Diagnostic code added to detect and fix scale issues automatically

**Key Fix:** Scale correction applied before SAM classification

---

### [INTEGRATION_SUMMARY.md](changelogs/INTEGRATION_SUMMARY.md)
**Date:** Library integration phase
**Feature:** USGS Spectral Library integration

**Summary:**
- **Created:** Standalone module `load_usgs_spectrum.py`
  - Loads USGS Hyperion-format spectral data
  - Automatic wavelength file loading
  - Configurable mineral list
  - No pandas dependency

- **Integrated:** Into `hyperion_workflow.py`
  - Modified `download_usgs_library()` function
  - Enhanced `create_endmember_library()` for automatic loading
  - Supports both automatic and manual loading

- **Outputs:**
  - CSV format (human-readable)
  - ENVI library format (.sli + .hdr)
  - 242 Hyperion wavelengths (355.6 - 2577.1 nm)
  - 8 minerals loaded (Jarosite, Goethite, Hematite, Kaolinite, Alunite, Illite, Smectite, Gypsum)

- **Benefits:** Modular design, flexible, ENVI-compatible, extensible

**Key Achievement:** Seamless USGS library integration with workflow

---

### [ENDMEMBER_FIX_SUMMARY.md](changelogs/ENDMEMBER_FIX_SUMMARY.md)
**Date:** SNAP compatibility fix
**Issue:** SNAP throwing `java.lang.RuntimeException: Waiting thread received a null tile`

**Summary:**
- **Original Issues:**
  1. Wrong file type: "ENVI Standard" instead of "ENVI Spectral Library"
  2. Incorrect dimensions: samples=242, lines=8, bands=1 (should be samples=242, lines=1, bands=8)
  3. Wrong interleave: bip instead of bil
  4. Missing wavelength units metadata
  5. Data corruption: Goethite and Gypsum had extreme negative values (-1.23e+34)

- **Solutions Implemented:**
  1. Updated `save_envi_library()` function with correct dimensions
  2. Created `validate_envi_library()` for validation
  3. Created `save_envi_library_alternative()` with data cleaning
  4. Automatic NaN/Inf replacement with 0.0
  5. Value clipping to [0, 1.5] range

- **Fixed Library:** `endmember_library_fixed_alt.hdr`
  - Correct ENVI Spectral Library format
  - BIL interleave
  - All 8 mineral spectra cleaned and validated
  - File size: 7744 bytes

- **Result:** SNAP-compatible spectral library ready for use

**Key Fix:** Proper ENVI Spectral Library format with BIL interleave

---

### [NULL_MODEL_THRESHOLD.md](changelogs/NULL_MODEL_THRESHOLD.md)
**Date:** February 2026
**Feature:** Statistically grounded SAM threshold derived from a background null model

**Summary:**
- **Problem:** The original adaptive threshold (`min_angle × 1.01`) is arbitrary and
  gives no indication of whether classified pixels are distinguishable from random
  background soil at any meaningful confidence level.
- **Solution:** For each mineral endmember, spectral angles are computed against a
  random sample of unclassified background soil pixels.  The **5th percentile** of
  this null distribution becomes the threshold, so only pixels more similar to the
  endmember than 95 % of random background pixels are classified.
- **New functions:** `derive_null_thresholds` (percentile computation),
  `compare_thresholds` (adaptive vs null side-by-side table with inertia ratios).
- **Outputs:** per-mineral null threshold table (step 7b), comparison table with
  inertia ratios (step 9b), null distribution overlay on validation histogram,
  `Null_thr_deg` / `Null_pixels` columns in `validation_summary.csv`.
- **Key parameters:** `NULL_MODEL_CONFIDENCE = 0.95`, `NULL_SAMPLE_SIZE = 5000`.

**Key Achievement:** Replaced arbitrary minimum-angle threshold with a
background-anchored statistical threshold that has a clear probabilistic interpretation.

---

### [ANGULAR_INERTIA_VALIDATION.md](changelogs/ANGULAR_INERTIA_VALIDATION.md)
**Date:** February 2026
**Feature:** Replacement of Euclidean KMeans inertia (metric 3 in `validate_sam_results`)
with an angular spectral inertia approach consistent with SAM's magnitude-invariant nature

**Summary:**
- **Problem:** The previous metric 3 used KMeans inertia (Euclidean distances in
  reflectance space) to measure the coherence of classified pixels. KMeans inertia
  penalises brightness variation that SAM explicitly ignores, making it inconsistent
  with the classifier being validated.
- **Solution:** For each mineral, the distribution of SAM angles at classified pixel
  locations (α_classified) is compared against SAM angles computed between a random
  background soil sample and the same endmember (α_null). If the classification is
  valid, α_classified should be stochastically lower than α_null.
- **Statistics reported:**
  - `Mean_angle_cls_deg` / `Mean_angle_null_deg` — mean angles of each distribution
  - `Angular_inertia_ratio` = mean(α_cls) / mean(α_null) — **< 1 = good signal**
  - `MannWhitney_p` — one-sided Mann-Whitney U p-value (H₁: α_cls < α_null)
  - `Effect_size` — rank-biserial r (> 0 = classified more similar to endmember than background)
- **Background pool:** shared across all minerals; excludes all pixels classified by
  any mineral. Pool size = max(n_classified_max, NULL_SAMPLE_SIZE), capped at
  available background pixels.
- **CSV change:** `Inertia_real`, `Inertia_null`, `Inertia_ratio` replaced by
  `Mean_angle_cls_deg`, `Mean_angle_null_deg`, `Angular_inertia_ratio`,
  `MannWhitney_p`, `Effect_size`. All other columns unchanged.
- **Figure change:** The gray histogram overlay in validation figures now shows the
  angular inertia null distribution (background angles to same endmember) rather
  than the `derive_null_thresholds` sample.
- **Scope:** Only `validate_sam_results` (step 10). The `compare_thresholds` table
  (step 9b) retains its own local KMeans inertia — that comparison is a separate
  diagnostic and is not affected.

**Key Achievement:** All four validation metrics now operate entirely in angle space,
making the post-classification quality assessment fully consistent with SAM's
magnitude-invariant design principle.

---

### [SUPERVISED_CLASSIFICATION.md](changelogs/SUPERVISED_CLASSIFICATION.md)
**Date:** February 2026
**Feature:** Supervised SAM classification extension — two new scripts
(`training_pixel_selector.py`, `supervised_classification.py`) and a pipeline
orchestration script (`run_pipeline.py`)

**Summary:**
- **Problem:** SAM classification relies on library endmembers that may not capture
  the specific atmospheric/illumination conditions of a given scene.  There is also
  no cross-method validation to confirm that SAM detections are geologically consistent.
- **Solution:** Added a three-stage supervised workflow that sits on top of the existing
  SAM pipeline.  Both methods use the identical SAM angular similarity logic; they differ
  only in endmember source:
  1. **`training_pixel_selector.py`** — dark-theme matplotlib GUI; the analyst draws
     rectangular ROIs on three switchable composites (RGB 660/550/480 nm; NIR false
     colour 850/660/550 nm; Fe³⁺ oxide ratio 900/660 nm).  Binary classification:
     `AMD_FeOx` vs `Background`.  A live spectral profile panel shows the mean ± 1σ
     of the current selection.  Outputs `training_pixels.npz` (one key per class;
     value is an int32 (n,2) array of soil-masked pixel indices).
  2. **`supervised_classification.py`** — computes mean L2-normalised spectrum per
     training class as the SAM endmember; applies nearest-endmember SAM to all soil
     pixels; filters noise components (< 4 px); computes Moran's I, min-angle
     statistics, and a cross-method comparison (Jaccard IoU, Cohen's Kappa,
     per-mineral containment) against the library SAM map.
  3. **`run_pipeline.py`** — thin orchestration wrapper; supports `--stage`,
     `--skip-sam`, and `--skip-select` flags for partial re-runs.
- **Soil mask fix:** `multi_mineral_sam_fixed.py` now saves `soil_mask.npy` next to
  the reflectance cube so downstream modules do not need to recompute it.
- **Outputs (supervised SAM):** `sam_classification_map.tif` (GeoTIFF), ENVI
  classification + angle maps, `sam_validation_metrics.csv`,
  `sam_cross_method_comparison.{csv,png}`, `sam_per_mineral_containment.csv`,
  `validation/sam_endmember_spectra.png`.

**Key Achievement:** Scene-specific supervised SAM classification that uses the same
algorithm as the library SAM stage but with image-derived endmembers, enabling a clean
method-controlled cross-validation via Jaccard IoU and per-mineral containment metrics.

---

### [SAM_OUTPUT_FIX_SUMMARY.md](changelogs/SAM_OUTPUT_FIX_SUMMARY.md)
**Date:** SAM output SNAP compatibility
**Issue:** `java.io.EOFException` when opening SAM classifications in SNAP

**Summary:**
- **Root Cause:** File extension mismatch
  - Python `spectral.envi.save_image()` creates `.img` files
  - SNAP expects data files without extension
  - Header (.hdr) points to file without extension
  - SNAP couldn't find data file → EOFException

- **Solution Applied:**
  1. Created `fix_sam_outputs.py` utility script
  2. Copied all `.img` files to files without extension
  3. Updated `hyperion_workflow.py` to auto-create both formats

- **Files Fixed:** 9 classification outputs
  - sam_multiclass (2.81 MB)
  - sam_angle_Jarosite through sam_angle_Gypsum (11.25 MB each)

- **File Structure:**
  - `.hdr` - Header file
  - `.img` - Binary data (Python format)
  - No extension - Binary data (SNAP format)

- **Result:** All SAM classification maps now open in SNAP without errors

**Key Fix:** Duplicate binary files with and without .img extension for compatibility

---

## Quick Navigation

### For New Users
1. Start with [SETUP_AND_USAGE_GUIDE.md](guides/SETUP_AND_USAGE_GUIDE.md)
2. Review [WORKFLOW_DIAGRAM.md](guides/WORKFLOW_DIAGRAM.md) for visual overview
3. Use [quick_start_guide.md](guides/quick_start_guide.md) for code snippets

### For Implementation
1. Follow [implementation_checklist.md](guides/implementation_checklist.md)
2. Reference [qgis_snap_workflows.md](guides/qgis_snap_workflows.md) for GUI operations

### For Troubleshooting
1. Check [SAM_Implementation_Report.md](changelogs/SAM_Implementation_Report.md) for classification issues
2. Review [ENDMEMBER_FIX_SUMMARY.md](changelogs/ENDMEMBER_FIX_SUMMARY.md) for library problems
3. See [SAM_OUTPUT_FIX_SUMMARY.md](changelogs/SAM_OUTPUT_FIX_SUMMARY.md) for SNAP compatibility

### For Running the Full Pipeline
1. Ensure `conda activate hyperion` (or equivalent)
2. Run `python run_pipeline.py` — executes SAM → training GUI → supervised SAM in sequence
3. For partial re-runs: `python run_pipeline.py --skip-sam --skip-select` (supervised SAM only)
4. See [SUPERVISED_CLASSIFICATION.md](changelogs/SUPERVISED_CLASSIFICATION.md) for
   details on the training GUI and supervised SAM validation metrics

### For Development
1. Review [INTEGRATION_SUMMARY.md](changelogs/INTEGRATION_SUMMARY.md) for architecture
2. Check changelogs for resolved issues before implementing fixes

---

## Document Statistics

### Guides
- **Total:** 5 documents
- **Total Pages:** ~140 pages equivalent
- **Coverage:**
  - Software setup and requirements
  - Complete 8-step workflow
  - QGIS and SNAP operations
  - Code snippets and examples
  - Visual diagrams and flowcharts

### Changelogs
- **Total:** 7 documents
- **Issues Resolved / Features Added:** 7 major items
- **Coverage:**
  - SAM classification debugging
  - USGS library integration
  - ENVI format compatibility
  - SNAP software compatibility
  - Null-model statistical thresholding
  - Angular spectral inertia validation metric
  - Supervised SAM classification extension (training GUI + pipeline)

---

## Project Overview

### Objective
Map acid mine drainage (AMD) minerals in the Rio Tinto area using Hyperion hyperspectral imagery through automated spectral classification techniques.

### Key Technologies
- **Python:** Core processing (spectral, pysptools, rasterio, numpy, scipy)
- **QGIS:** Visualization and map production
- **SNAP:** Alternative processing and validation
- **ENVI Format:** Standard hyperspectral data format

### Target Minerals
- **Primary AMD Minerals:** Jarosite, Goethite, Hematite, Schwertmannite
- **Secondary/Confusers:** Kaolinite, Illite, Smectite, Alunite, Gypsum

### Classification Methods
- **SAM (Spectral Angle Mapper):** Primary unsupervised classification (library endmembers)
- **Supervised SAM:** Scene-specific supervised classification (image-derived endmembers)
- **MTMF (Mixture Tuned Matched Filter):** Abundance and confidence
- **FCLS (Fully Constrained Least Squares):** Linear unmixing

### Project Status
- ✅ Environment setup complete
- ✅ USGS library integrated
- ✅ SAM implementation debugged
- ✅ SNAP compatibility fixed
- ✅ Null-model statistical thresholding implemented
- ✅ Angular spectral inertia validation metric implemented
- ✅ Supervised SAM classification pipeline added (training GUI + SAM + cross-validation)
- ✅ Cross-method library SAM ↔ supervised SAM Jaccard IoU comparison implemented
- 🔄 Ready for full workflow execution (`python run_pipeline.py`)

---

## File Organization

```
docs/
├── DOCUMENTATION_INDEX.md (this file)
│
├── guides/
│   ├── implementation_checklist.md
│   ├── qgis_snap_workflows.md
│   ├── quick_start_guide.md
│   ├── WORKFLOW_DIAGRAM.md
│   └── SETUP_AND_USAGE_GUIDE.md
│
└── changelogs/
    ├── SAM_Implementation_Report.md
    ├── INTEGRATION_SUMMARY.md
    ├── ENDMEMBER_FIX_SUMMARY.md
    ├── SAM_OUTPUT_FIX_SUMMARY.md
    ├── NULL_MODEL_THRESHOLD.md
    ├── ANGULAR_INERTIA_VALIDATION.md
    └── SUPERVISED_CLASSIFICATION.md   ← new (Feb 2026)
```

---

## Usage Recommendations

1. **First Time:** Read guides in order (Setup → Workflow Diagram → Implementation Checklist)
2. **Quick Reference:** Use Quick Start Guide for code snippets
3. **Troubleshooting:** Check changelogs for similar issues before debugging
4. **QGIS/SNAP:** Refer to qgis_snap_workflows.md for GUI operations
5. **Development:** Review changelogs to understand past fixes and avoid regressions

---

## Version History

- **v1.4** (2026-02-23): Corrected supervised classification description
  - `README_SAM_MultiMineral.md` — replaced Random Forest section with accurate
    Supervised SAM description; fixed `--stage rf` → `--stage supervised`; fixed
    class list (7 mineral classes → binary AMD_FeOx/Background); fixed output folder
    (`supervised_classification/` → `supervised_sam/`) and file names (`rf_*` → `sam_*`)
  - `docs/changelogs/SUPERVISED_CLASSIFICATION.md` — full rewrite to reflect actual
    Supervised SAM implementation (image-derived endmembers, nearest-endmember rule)
  - `docs/DOCUMENTATION_INDEX.md` — updated classifier description, project status,
    quick-navigation, and document statistics throughout

- **v1.3** (2026-02-21): Supervised classification extension added
  - Added `SUPERVISED_CLASSIFICATION.md` changelog
  - Updated `README_SAM_MultiMineral.md` — added supervised classification extension section
  - Updated `docs/README.md` and `DOCUMENTATION_INDEX.md`

- **v1.2** (2026-02-21): Angular spectral inertia validation metric
  - Added `ANGULAR_INERTIA_VALIDATION.md` changelog
  - Updated `README_SAM_MultiMineral.md` — added full "Post-Classification
    Validation Metrics" section covering all four metrics; added validation
    subsection and column table to "Output Files"
  - Updated `docs/README.md` — new changelog entry in folder tree and What's What
  - Updated `DOCUMENTATION_INDEX.md` — new index entry, project status, document counts, file tree

- **v1.1** (2026-02-21): Null-model threshold documentation
  - Added `NULL_MODEL_THRESHOLD.md` changelog
  - Updated project status checklist
  - Updated `README_SAM_MultiMineral.md` thresholding section

- **v1.0** (2026-01-13): Initial documentation organization
  - Created guide and changelog separation
  - Added comprehensive index
  - Consolidated all project documentation

---

## Contact & Support

For questions or issues:
1. Review relevant documentation section
2. Check changelogs for similar issues
3. Verify software versions match requirements
4. Consult troubleshooting sections in guides

**Project Repository:** PROCESSING_AND_POST
**Documentation Location:** `docs/`
