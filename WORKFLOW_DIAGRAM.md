# Visual Workflow Diagram - AMD Mineral Mapping with Hyperion

## Complete Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: PREPROCESSING                         │
│                  (Already completed with SUREHYP)                │
│                                                                   │
│  • Atmospheric correction                                        │
│  • Georeferencing                                                │
│  • Bad band removal                                              │
│  • Destriping                                                    │
│                                                                   │
│  Output: Clean Hyperion cube (rows × cols × bands)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: BUILD SPECTRAL LIBRARY                      │
│                                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │ USGS Library │   │ Publications │   │ Literature   │        │
│  │              │   │              │   │              │        │
│  │ • Jarosite   │   │ • Schwert-   │   │ • Rio Tinto  │        │
│  │ • Goethite   │   │   mannite    │   │   specific   │        │
│  │ • Hematite   │   │ • Goethite   │   │   spectra    │        │
│  │ • Clays      │   │   variants   │   │              │        │
│  └──────────────┘   └──────────────┘   └──────────────┘        │
│         │                   │                   │                │
│         └───────────────────┴───────────────────┘                │
│                             │                                     │
│                             ▼                                     │
│              Extract Hyperion wavelengths                        │
│                             │                                     │
│                             ▼                                     │
│              Resample all spectra to Hyperion                    │
│                             │                                     │
│                             ▼                                     │
│         Output: Endmember library (.csv + .sli)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           STEP 3: SPECTRAL ENHANCEMENT (Optional)                │
│                                                                   │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  Savitzky-Golay Smoothing                             │      │
│  │  • Window = 9, Polyorder = 2                          │      │
│  │  • Reduces noise, preserves features                  │      │
│  └───────────────────────────────────────────────────────┘      │
│                             │                                     │
│                             ▼                                     │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  Continuum Removal (Optional)                         │      │
│  │  • Emphasizes absorption features                     │      │
│  │  • Helps separate goethite/hematite                   │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                   │
│  Output: Enhanced Hyperion cube                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              CREATE MASKS (Part of Step 1 or here)               │
│                                                                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ Vegetation  │   │    Water    │   │Cloud/Shadow │           │
│  │  (NDVI>0.3) │   │ (NIR<thresh)│   │ (if needed) │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                  │
│                            │                                      │
│                            ▼                                      │
│                   Combined Mask Layer                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: SAM CLASSIFICATION                          │
│                                                                   │
│  For each endmember:                                             │
│                                                                   │
│  1. Compute spectral angle (radians) for every pixel            │
│     angle = arccos(pixel·endmember / ||pixel||·||endmember||)   │
│                                                                   │
│  2. Apply threshold (default: 0.10 radians)                     │
│     classified = 1 if angle < threshold, else 0                 │
│                                                                   │
│  3. Apply masks (remove vegetation, water)                      │
│                                                                   │
│  Outputs:                                                        │
│  • Classification map for each mineral                           │
│  • Angle map (quality indicator)                                 │
│  • Minimum angle map (best fit)                                  │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Jarosite │  │ Goethite │  │ Hematite │  │Schwert-  │       │
│  │   map    │  │   map    │  │   map    │  │mannite   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│       STEP 5: ABUNDANCE/CONFIDENCE MAPPING                       │
│                                                                   │
│  ┌────────────────────────────────────────────────────┐         │
│  │  Method 1: Matched Filter (MF)                     │         │
│  │  • Detects target in background                    │         │
│  │  • Output: MF score (higher = more likely)         │         │
│  └────────────────────────────────────────────────────┘         │
│                            │                                      │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────┐         │
│  │  Method 2: Mixture Tuned Matched Filter (MTMF)    │         │
│  │  • MF + infeasibility metric                       │         │
│  │  • Outputs: MF score + infeasibility               │         │
│  │  • Better false positive rejection                 │         │
│  └────────────────────────────────────────────────────┘         │
│                            │                                      │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────┐         │
│  │  Method 3: Linear Spectral Unmixing (FCLS)        │         │
│  │  • All endmembers simultaneously                   │         │
│  │  • Output: Abundance fractions (0-1)               │         │
│  │  • Constrained: sum=1, non-negative                │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                   │
│  For each mineral:                                               │
│  • Abundance map                                                 │
│  • Confidence/quality scores                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 6: POSTPROCESSING                              │
│                                                                   │
│  ┌────────────────────────────────────────────────────┐         │
│  │  Step 6.1: Combine SAM + MTMF                      │         │
│  │                                                     │         │
│  │  Pixel classified as mineral only if:              │         │
│  │    • SAM angle < threshold        AND              │         │
│  │    • MF score > threshold         AND              │         │
│  │    • Infeasibility < threshold                     │         │
│  │                                                     │         │
│  │  Result: High-confidence classification            │         │
│  └────────────────────────────────────────────────────┘         │
│                            │                                      │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────┐         │
│  │  Step 6.2: Morphological Filtering                 │         │
│  │                                                     │         │
│  │  • Remove small objects (< 5 pixels)               │         │
│  │  • Morphological closing (fill gaps)               │         │
│  │  • Majority filter (3×3) for smoothing             │         │
│  │                                                     │         │
│  │  Result: Cleaned classification maps               │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                   │
│  Output: Final refined mineral maps                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          STEP 7: VALIDATION (Without Ground Truth)               │
│                                                                   │
│  ┌────────────────────────────────────────────────────┐         │
│  │  7.1: Compare with Published Maps                  │         │
│  │  • Digitize mineral zones from literature          │         │
│  │  • Calculate spatial overlap                       │         │
│  │  • Compute precision/recall/F1                     │         │
│  └────────────────────────────────────────────────────┘         │
│                            │                                      │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────┐         │
│  │  7.2: Geochemical Correlation                      │         │
│  │  • Extract classification at sample points         │         │
│  │  • Test correlations:                              │         │
│  │    - Jarosite vs pH (expect < 3)                   │         │
│  │    - Jarosite vs Fe (expect high)                  │         │
│  │    - Jarosite vs SO4 (expect high)                 │         │
│  │  • Statistical tests (t-test, ANOVA)               │         │
│  └────────────────────────────────────────────────────┘         │
│                            │                                      │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────┐         │
│  │  7.3: Spatial Consistency                          │         │
│  │  • Jarosite near streams? (low elevation)          │         │
│  │  • Goethite on terraces? (higher elevation)        │         │
│  │  • Clustering patterns realistic?                  │         │
│  │  • Consistent with AMD process zones?              │         │
│  └────────────────────────────────────────────────────┘         │
│                            │                                      │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────┐         │
│  │  7.4: Cross-validation with Sentinel-2             │         │
│  │  • Calculate iron oxide indices                    │         │
│  │  • Visual comparison                               │         │
│  │  • Check high-confidence zones                     │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                   │
│  Output: Validation report with consistency metrics              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 8: FINAL PRODUCTS                              │
│                                                                   │
│  ┌────────────────────────────────────────────────────┐         │
│  │  8.1: Create QGIS Project                          │         │
│  │  • Import all layers                               │         │
│  │  • Style mineral maps (colors, opacity)            │         │
│  │  • Add basemaps (Sentinel-2, DEM hillshade)        │         │
│  │  • Create layouts (legend, scale, north arrow)     │         │
│  │  • Export high-res maps (300 dpi)                  │         │
│  └────────────────────────────────────────────────────┘         │
│                            │                                      │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────┐         │
│  │  8.2: Generate Statistics                          │         │
│  │  • Area calculations (hectares, km²)               │         │
│  │  • Pixel counts by mineral                         │         │
│  │  • Percentage coverage                             │         │
│  │  • Confidence metrics                              │         │
│  │  • Save as CSV/Excel tables                        │         │
│  └────────────────────────────────────────────────────┘         │
│                            │                                      │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────┐         │
│  │  8.3: Create Visualizations                        │         │
│  │  • Spectral profile plots                          │         │
│  │  • Abundance distribution histograms               │         │
│  │  • Validation plots (confusion matrix, etc.)       │         │
│  │  • Before/after comparisons                        │         │
│  │  • Save publication-ready figures                  │         │
│  └────────────────────────────────────────────────────┘         │
│                            │                                      │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────┐         │
│  │  8.4: Documentation                                │         │
│  │  • Methods description                             │         │
│  │  • Parameters used                                 │         │
│  │  • Data sources                                    │         │
│  │  • Results summary                                 │         │
│  │  • Limitations and uncertainties                   │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                   │
│  Final Deliverables:                                             │
│  ├── Classification GeoTIFFs (Jarosite, Goethite, etc.)         │
│  ├── Abundance maps (MF, MTMF, FCLS)                            │
│  ├── Confidence/quality layers                                   │
│  ├── QGIS project file with all layers                          │
│  ├── High-resolution map PDFs                                    │
│  ├── Statistics tables (CSV, Excel)                             │
│  ├── Publication-ready figures (PNG, 300 dpi)                   │
│  ├── Spectral library files                                     │
│  └── Complete documentation                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Decision Tree: When to Use Which Tool

```
                        ┌────────────────┐
                        │  Your Task     │
                        └────────┬───────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
            ┌───────▼──────┐         ┌───────▼──────┐
            │ Preprocessing│         │ Classification│
            │  & Masking   │         │   & Analysis  │
            └───────┬──────┘         └───────┬───────┘
                    │                        │
        ┌───────────┴───────────┐           │
        │                       │           │
┌───────▼──────┐      ┌────────▼───────┐   │
│ Python       │      │ QGIS           │   │
│ (spectral,   │      │ (Raster Calc,  │   │
│  rasterio)   │      │  OTB)          │   │
│              │      │                │   │
│ Use when:    │      │ Use when:      │   │
│ • Automating │      │ • Visual QC    │   │
│ • Batch      │      │ • Interactive  │   │
│ • Complex    │      │ • Quick masks  │   │
└──────────────┘      └────────────────┘   │
                                            │
                        ┌───────────────────┴───────────────┐
                        │                                   │
                ┌───────▼──────┐                   ┌───────▼──────┐
                │ Python       │                   │ QGIS         │
                │ (pysptools)  │                   │ (OTB SAM)    │
                │              │                   │              │
                │ Use when:    │                   │ Use when:    │
                │ • Full auto  │                   │ • Exploring  │
                │ • Custom     │                   │ • Visual     │
                │ • Research   │                   │ • Teaching   │
                └──────────────┘                   └──────────────┘
```

## Method Selection Guide

```
┌─────────────────────────────────────────────────────────────────┐
│              CLASSIFICATION METHOD SELECTION                     │
└─────────────────────────────────────────────────────────────────┘

    Do you have pure endmember spectra?
                 │
        ┌────────┴────────┐
        │                 │
       YES               NO
        │                 │
        ▼                 ▼
    Use SAM          Use N-FINDR or
    Use MTMF         PPI to extract
    Use FCLS         image endmembers
        │                 │
        └────────┬────────┘
                 │
                 ▼
    Are your endmembers spectrally similar?
    (spectral angle < 0.05 rad)
                 │
        ┌────────┴────────┐
        │                 │
       YES               NO
        │                 │
        ▼                 ▼
    Use MTMF         Use SAM
    Use MF           (faster, simpler)
    (better          
    discrimination)  
        │                 │
        └────────┬────────┘
                 │
                 ▼
    Do you need abundance estimates?
                 │
        ┌────────┴────────┐
        │                 │
       YES               NO
        │                 │
        ▼                 ▼
    Use FCLS         Use SAM only
    Use MTMF         (classification
    (provides        sufficient)
    fractions)
        │
        ▼
    Final product:
    • SAM classification (presence/absence)
    • MTMF scores (confidence)
    • FCLS abundances (percentages)
    • Combined high-confidence map
```

## Validation Strategy Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                     VALIDATION APPROACH                          │
└─────────────────────────────────────────────────────────────────┘

        Do you have field-collected ground truth?
                          │
                 ┌────────┴────────┐
                 │                 │
                YES               NO (Your case)
                 │                 │
                 ▼                 ▼
        Standard accuracy    Indirect validation:
        assessment:          
        • Confusion matrix   ┌─────────────────────┐
        • Kappa              │ 1. Published maps   │
        • Overall accuracy   │    • Spatial overlap│
        • Per-class metrics  │    • Precision/     │
                             │      recall         │
                             │                     │
                             │ 2. Geochemistry     │
                             │    • pH correlation │
                             │    • Fe correlation │
                             │    • SO4 correlation│
                             │                     │
                             │ 3. Spatial logic    │
                             │    • Topography     │
                             │    • Drainage       │
                             │    • Clustering     │
                             │                     │
                             │ 4. Multi-sensor     │
                             │    • Sentinel-2     │
                             │    • Drone imagery  │
                             │    • Historical     │
                             └─────────────────────┘
                                       │
                                       ▼
                             Compile evidence from
                             multiple sources
                                       │
                                       ▼
                             Report confidence level
                             and limitations
```

## Threshold Optimization Process

```
┌─────────────────────────────────────────────────────────────────┐
│               THRESHOLD OPTIMIZATION WORKFLOW                    │
└─────────────────────────────────────────────────────────────────┘

    Start with default thresholds:
    • SAM: 0.10 radians
    • MF: 0.5 (normalized)
    • MTMF infeasibility: 0.1
              │
              ▼
    Run classification with defaults
              │
              ▼
    Visual inspection in QGIS
              │
    ┌─────────┴─────────┐
    │                   │
Too strict          Too loose
(few pixels)      (too many pixels)
    │                   │
    ▼                   ▼
Increase           Decrease
threshold          threshold
    │                   │
    └─────────┬─────────┘
              │
              ▼
    Re-run classification
              │
              ▼
    Compare with:
    • Known mineral zones
    • Visual features (color)
    • Geochemical consistency
              │
    ┌─────────┴─────────┐
    │                   │
Good match         Poor match
    │                   │
    ▼                   ▼
Final          Adjust and
threshold      iterate
    │                   │
    └─────────┬─────────┘
              │
              ▼
    Document optimal thresholds
    for each mineral
```

## Data Flow Diagram

```
┌─────────────┐
│ Hyperion    │
│ L1T         │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ SUREHYP     │ ← You are here (completed)
│ Preprocessing│
└──────┬──────┘
       │
       ├─────────────────────────┐
       │                         │
       ▼                         ▼
┌─────────────┐          ┌─────────────┐
│ Clean cube  │          │ Bad band    │
│ (GeoTIFF/   │          │ list        │
│  ENVI)      │          │             │
└──────┬──────┘          └─────────────┘
       │
       ├──────────┬──────────┬──────────┐
       │          │          │          │
       ▼          ▼          ▼          ▼
┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Smoothing │ │ Masks  │ │  SAM   │ │ MTMF   │
└────┬─────┘ └───┬────┘ └───┬────┘ └───┬────┘
     │           │          │          │
     └───────┬───┴──────┬───┴──────────┘
             │          │
             ▼          ▼
      ┌──────────┐ ┌──────────┐
      │Combined  │ │Statistics│
      │maps      │ │& plots   │
      └────┬─────┘ └────┬─────┘
           │            │
           └──────┬─────┘
                  │
                  ▼
           ┌──────────┐
           │  QGIS    │
           │ Project  │
           └────┬─────┘
                │
                ▼
           ┌──────────┐
           │Publication│
           │ products │
           └──────────┘
```

## Timeline Visualization

```
Weeks 1-2:    [████████] Setup & Data Collection
              
Weeks 3-4:    [████████] Spectral Library + Enhancement
              
Week 5:       [████████] SAM Classification
              
Week 6:       [████████] MTMF/Abundance Mapping
              
Week 7:       [████████] Postprocessing
              
Weeks 8-9:    [████████████] Validation
              
Weeks 10-11:  [████████████] Final Products & Maps
              
Week 12:      [████████] Documentation

Total: 12 weeks (part-time, ~15-20 hrs/week)
```

---

*This workflow diagram provides a visual overview of the complete process. 
Refer to the detailed implementation documents for step-by-step instructions.*
