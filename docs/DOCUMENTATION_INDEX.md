# Documentation Index
## Hyperion AMD Mineral Mapping Project

**Repository:** PROCESSING_AND_POST
**Last Updated:** January 13, 2026
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
- **Total:** 4 documents
- **Issues Resolved:** 4 major compatibility/implementation issues
- **Coverage:**
  - SAM classification debugging
  - USGS library integration
  - ENVI format compatibility
  - SNAP software compatibility

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
- **SAM (Spectral Angle Mapper):** Primary classification
- **MTMF (Mixture Tuned Matched Filter):** Abundance and confidence
- **FCLS (Fully Constrained Least Squares):** Linear unmixing

### Project Status
- ✅ Environment setup complete
- ✅ USGS library integrated
- ✅ SAM implementation debugged
- ✅ SNAP compatibility fixed
- 🔄 Ready for full workflow execution
- 🔄 Awaiting Hyperion image cube

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
    └── SAM_OUTPUT_FIX_SUMMARY.md
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
