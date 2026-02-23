# Multi-Mineral SAM Analysis Scripts

## Overview

These scripts process multiple mineral spectra from a library folder and generate:
1. **Individual match score maps** for each mineral (0-1 scale)
2. **Composite map** showing the dominant mineral at each pixel
3. **Classification statistics** (CSV format)

Based on your working SAM example, extended for batch processing.

---

## Files Provided

### 1. `multi_mineral_sam.py` - **SEQUENTIAL VERSION**
- Processes minerals one at a time
- Simpler, easier to debug
- Good for small libraries (<10 minerals)

### 2. `multi_mineral_sam_parallel.py` - **PARALLEL VERSION**
- Processes multiple minerals simultaneously
- **Much faster** for large libraries
- Automatically uses all CPU cores
- Recommended for >5 minerals

---

## Quick Start

### 1. Update Paths in the Script

Edit these lines at the top of either script:

```python
# Paths
HDR_FILE = r"C:\path\to\your\reflectance.hdr"
LIBRARY_FOLDER = r"C:\path\to\mineral\library\folder"
OUTPUT_FOLDER = r"C:\path\to\output\results"

# SAM Parameters
SAM_THRESHOLD = 0.1  # Adjust as needed
```

### 2. Prepare Your Mineral Library

Your library folder should contain CSV files with:
- **Column 0**: Wavelength (nm)
- **Column 1**: Reflectance
- **Filename**: Will be used as mineral name (e.g., `Jarosite_K.csv` → "Jarosite_K")

Example structure:
```
Library_folder/
├── Jarosite_GDS98_K_90C_Syn_BECKa.csv
├── Goethite_WS968.csv
├── Hematite_HS9.csv
├── Kaolinite_CM9.csv
└── Gypsum_HS333.csv
```

### 3. Run the Script

```bash
python multi_mineral_sam.py
```

or for parallel processing:

```bash
python multi_mineral_sam_parallel.py
```

---

## Configuration Options

### Threshold Strategy

`multi_mineral_sam_fixed.py` derives **two complementary thresholds** for each mineral:

| Threshold | Config key | How it works |
|---|---|---|
| **Adaptive** | `SAM_THRESHOLD_MARGIN` | `min_angle × (1 + margin)` — anchored to the closest-matching soil pixel |
| **Null-model** | `NULL_MODEL_CONFIDENCE` | Nth-percentile of SAM angles measured on random background pixels |

```python
# Adaptive threshold: 1% above the minimum angle found in soil pixels
SAM_THRESHOLD_MARGIN = 0.01

# Null-model threshold
# A pixel must have a lower angle than NULL_MODEL_CONFIDENCE × 100% of
# random background soil pixels to be considered a statistically
# significant detection.  Default: 95% confidence → 5th percentile.
NULL_MODEL_CONFIDENCE = 0.95

# Number of background pixels sampled per mineral for the null model.
# Increase for larger or more heterogeneous scenes.
NULL_SAMPLE_SIZE = 5000
```

The **adaptive threshold** drives the match scores and output maps (unchanged
behaviour).  The **null-model threshold** is computed in step 7b of `main()` and
reported alongside the adaptive results — it provides a statistically grounded
reference point without replacing the primary classification.

#### Why two thresholds?

The adaptive threshold is intentionally tight (it tracks the single best-matching
pixel) so the match-score maps highlight only the most confident detections.  The
null-model threshold answers the complementary question: *how many pixels are
detectably different from random background at a given confidence level?*
Comparing the two helps identify whether the classification is capturing a small
cluster of extreme matches (adaptive) or a broader population of spectrally
similar pixels (null-model).

### Output Control

```python
SAVE_INDIVIDUAL_MAPS = True   # Save PNG for each mineral
SAVE_COMPOSITE_MAP = True     # Save composite classification map
SHOW_PLOTS = False            # Set True to display plots interactively
```

### Parallel Processing (parallel version only)

```python
USE_PARALLEL = True    # Enable/disable parallel processing
N_PROCESSES = None     # None = use all CPUs
                       # Or set specific number: N_PROCESSES = 4
```

---

## Output Files

After running, your `OUTPUT_FOLDER` will contain:

### 1. Individual Match Maps
- `{mineral_name}_SAM_match.png` - One per mineral
- Color scale: 0 (no match) → 1 (perfect match)
- Inferno colormap

### 2. Composite Map
- `Composite_SAM_map.png`
- Shows dominant mineral at each pixel
- Different color for each mineral
- Includes legend

### 3. Statistics
- `classification_statistics.csv`
- Contains:
  - Class name
  - Class code
  - Number of pixels
  - Percentage of image

### 4. Validation (`validation/` subfolder)
- `<mineral>_validation.png` — two-panel figure per mineral:
  - **Left:** SAM angle histogram with overlapping null/background distribution
  - **Right:** connected-component map coloured by log(component size), noise in red
- `validation_summary.csv` — one row per mineral with all four validation metrics:

| Column | Description |
|---|---|
| `Pixels` | Classified pixel count (adaptive threshold) |
| `Null_thr_deg` | Null-model threshold angle (degrees) |
| `Null_pixels` | Pixel count under null-model threshold |
| `Components` | Number of connected components |
| `Noise_px_frac` | Fraction of classified pixels in noise components (< `MIN_COMPONENT_SIZE` px) |
| `Mean_angle_deg` | Mean SAM angle of classified pixels (degrees) |
| `Std_angle_deg` | Standard deviation of classified SAM angles (degrees) |
| `Skewness` | Skewness of the classified angle distribution (negative = good signal) |
| `Mean_angle_cls_deg` | Mean α_classified — mean angle of classified pixels to endmember |
| `Mean_angle_null_deg` | Mean α_null — mean angle of background pixels to same endmember |
| `Angular_inertia_ratio` | `mean(α_cls) / mean(α_null)` — **< 1 = good** |
| `MannWhitney_p` | One-sided Mann-Whitney p-value (H₁: α_cls < α_null) |
| `Effect_size` | Rank-biserial r — **> 0 = classified more similar to endmember than background** |
| `Morans_I` | Moran's I spatial autocorrelation (Queen contiguity, soil domain) |
| `Morans_p` | Moran's I p-value (< 0.05 + positive I = spatially clustered detections) |

---

## Understanding the Results

### Match Score Maps (0-1 scale)

- **Score = 1.0**: Perfect spectral match
- **Score = 0.5**: Moderate match (threshold dependent)
- **Score = 0.0**: No match (angle ≥ threshold)

The match score is calculated as:
```
match_score = max(0, 1 - SAM_angle / threshold)
```

### Composite Map

Shows the **dominant mineral** at each pixel based on:
1. Which mineral has the highest match score
2. Only if that score > 0.5
3. Otherwise, pixel is "Unclassified"

### Post-Classification Validation Metrics

`validate_sam_results` (step 10) runs four independent checks per mineral:

#### Metric 1 — Connected Components
Labels the binary mask and counts component sizes.  Small isolated components
(< `MIN_COMPONENT_SIZE` pixels, default 4) are flagged as probable noise.
High `Noise_px_frac` values suggest the classification is scattered rather
than spatially coherent.

#### Metric 2 — SAM Angle Distribution
Builds a 30-bin histogram of classified-pixel angles over `[0, threshold]`
and computes mean, standard deviation, and skewness.

- Left-skewed distribution (skewness < −0.3) → most pixels cluster near angle
  zero, indicating confident matches.
- Flat or right-skewed → pixels are spread across the threshold window,
  suggesting marginal matches.

#### Metric 3 — Angular Spectral Inertia
Compares the SAM angle distributions of classified pixels (α_classified) and
a random background soil sample (α_null) against the **same endmember**.

```
Angular_inertia_ratio = mean(α_classified) / mean(α_null)
```

Because SAM is magnitude-invariant, this comparison is performed entirely in
angle space — consistent with how the classification was made.

| Angular_inertia_ratio | Interpretation |
|---|---|
| < 1.0 | Classified pixels are on average more similar to the endmember than random background soil — supports the classification |
| ≥ 1.0 | Classified pixels are no more similar to the endmember than background — classification is not distinguishable from chance |

The **Mann-Whitney U test** (one-sided, H₁: α_cls < α_null) provides a
p-value for this separation, and the **rank-biserial effect size r** quantifies
its magnitude:

- `r > 0` → classified pixels tend to have lower angles than background (good)
- `r ≈ 0` → no separation
- `r < 0` → classified pixels have *higher* angles than background (failure)

Background pixels are all soil pixels not classified by **any** mineral, giving
a clean unclassified baseline.  The null sample size is at least
`max(n_classified, NULL_SAMPLE_SIZE)` to ensure the test has sufficient power.

#### Metric 4 — Moran's I Spatial Autocorrelation
Tests whether the binary classification mask is spatially clustered within the
soil domain (Queen contiguity).  Significant positive Moran's I (p < 0.05)
indicates the detections form spatially coherent patches rather than being
randomly scattered — supporting geological plausibility.

---

## Troubleshooting

### Problem: "No CSV files found"
**Solution**: Check that:
- `LIBRARY_FOLDER` path is correct
- CSV files exist in that folder
- Files have `.csv` extension (not `.txt`)

### Problem: "Wavelength interpolation error"
**Solution**: Your mineral spectra might not cover the full Hyperion range
- Check your CSV wavelength range
- The script uses extrapolation but results may be unreliable outside the original range

### Problem: "Out of memory error" (large images)
**Solution**: 
- Process fewer minerals at once
- Reduce `N_PROCESSES` in parallel version
- Use sequential version instead

### Problem: All pixels classified as one mineral
**Solution**: The adaptive threshold may be too permissive for one endmember
- Inspect the `compare_thresholds` table printed in step 9b; an unusually large
  Δ px indicates that endmember is attracting many background-like pixels
- Raise `SAM_THRESHOLD_MARGIN` slightly (e.g., `0.05`) to tighten the adaptive window
- Or use `NULL_MODEL_CONFIDENCE` as the primary threshold by post-filtering with it
- Check if mineral spectra are properly normalised

---

## Advanced Customization

### Adjust the Null-Model Confidence Level

Change `NULL_MODEL_CONFIDENCE` at the top of the script:

```python
# More permissive: 90% confidence → 10th percentile of background angles
NULL_MODEL_CONFIDENCE = 0.90

# More restrictive: 99% confidence → 1st percentile of background angles
NULL_MODEL_CONFIDENCE = 0.99
```

Larger sample sizes give more stable percentile estimates for heterogeneous scenes:

```python
NULL_SAMPLE_SIZE = 10000   # double the default; slower but more stable
```

### Adjust the Adaptive Threshold Margin

```python
# Tighter (fewer pixels, higher confidence per pixel)
SAM_THRESHOLD_MARGIN = 0.05   # 5% above minimum

# Looser (more pixels, lower per-pixel confidence)
SAM_THRESHOLD_MARGIN = 0.20   # 20% above minimum
```

### Export as GeoTIFF (for GIS)

Add this function and call it in main():

```python
def save_as_geotiff(array, output_path, img_metadata):
    """Save array as GeoTIFF with georeferencing"""
    from osgeo import gdal, osr
    
    # Extract geotransform and projection from metadata
    # Implementation depends on your Hyperion metadata structure
    # See spectral.io documentation
    
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, 
                           array.shape[1], 
                           array.shape[0],
                           1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(array)
    # Set geotransform and projection here
    dataset.FlushCache()
```

---

## Performance Tips

### Sequential Version
- **Pro**: Simpler, uses less memory
- **Con**: Slower for many minerals
- **Best for**: <10 minerals, debugging

### Parallel Version
- **Pro**: Much faster (3-8x speedup typical)
- **Con**: Higher memory usage
- **Best for**: >5 minerals, production runs

**Typical processing times** (example: 256×3242 image, 8 minerals):
- Sequential: ~2-4 minutes
- Parallel (8 cores): ~30-60 seconds

---

## Integration with Your Workflow

### After SAM Classification

You can use the results for:

1. **Validation**: Compare with field data or other methods
2. **Abundance mapping**: Feed into MTMF or linear unmixing
3. **Change detection**: Compare scenes from different dates
4. **GIS analysis**: Import composite map into QGIS/ArcGIS

### Combining with MTMF

If you want to run MTMF after SAM, you can extract the endmember spectra:

```python
# After loading library in main()
endmember_dict = {}
for csv_file in csv_files:
    ref_spectrum, mineral_name = load_and_resample_spectrum(csv_file, wavelengths)
    endmember_dict[mineral_name] = ref_spectrum

# Now use endmember_dict for MTMF
# (See your hyperion_workflow.py for MTMF implementation)
```

---

## Key Differences from Original Script

### Your Working Script (Python_SAM_example.py):
- Single mineral
- Manual file paths
- Direct visualization

### These New Scripts:
- **Multiple minerals** processed automatically
- Library-based approach
- Batch output generation
- Optional parallel processing
- Statistical summaries
- Composite classification map

---

## Questions or Issues?

If you encounter problems:
1. Check file paths are correct
2. Verify CSV format matches expected structure
3. Try sequential version first (easier to debug)
4. Check console output for detailed error messages
5. Reduce threshold if no detections occur
6. Increase threshold if too many false positives

---

## Supervised Classification Extension (February 2026)

Two additional modules extend the workflow with a scene-specific supervised
SAM classification that runs after the library SAM stage and compares results
against it.  Both methods use the identical SAM angular similarity logic;
they differ only in endmember source (image-derived training means vs. USGS
library spectra).  Divergences therefore reflect the spectral distance between
USGS library references and actual scene conditions rather than any
algorithmic difference.

### Overview

```
Stage 1  multi_mineral_sam_fixed.py   → SAM classification + soil_mask.npy
Stage 2  training_pixel_selector.py   → Interactive training pixel labelling
Stage 3  supervised_classification.py → Supervised SAM classification + validation
```

Run all three stages with:

```bash
python run_pipeline.py
```

Or run individual stages:

```bash
python run_pipeline.py --stage sam          # Stage 1 only
python run_pipeline.py --stage select       # Stage 2 only (interactive)
python run_pipeline.py --stage supervised   # Stage 3 only
python run_pipeline.py --skip-sam           # Stages 2+3 (soil_mask.npy must exist)
```

---

### 4. `training_pixel_selector.py` — Interactive Training GUI

Opens a full-screen dark-themed matplotlib window.

**Image composites** (switchable via radio buttons):

| View | Bands | Use |
|------|-------|-----|
| RGB | 660 / 550 / 480 nm | True-colour context |
| NIR False Colour | 850 / 660 / 550 nm | Vegetation vs bare soil |
| Fe³⁺ Oxide Ratio | 900 / 660 nm (ratio) | Iron oxide intensity |

**Training classes:** `AMD_FeOx` (iron-oxide bearing pixels) and `Background`
(unaltered soil).  Binary classification is intentional: the supervised SAM
produces a broad ferro-oxide lithology footprint, which is then compared against
the per-mineral library SAM detections.

**How to use:**
1. Select a View and a Class using the radio buttons.
2. Left-click + drag to draw a rectangular ROI on the image.
3. The spectral profile panel (right) shows the mean ± 1σ spectrum of
   the selected soil pixels and reference lines at 430, 660, 875 nm.
4. Repeat for both classes.
5. Click **Save & Continue** to write `training_pixels.npz` and close.

**Soil mask overlay** (yellow semi-transparent fill, toggleable) shows
which pixels are eligible for inclusion in the training set.

**Output:** `amd_mapping/outputs/training_pixels.npz`

---

### 5. `supervised_classification.py` — Supervised SAM Classification

**Approach:** nearest-endmember Spectral Angle Mapper using image-derived
endmembers (mean L2-normalised training spectrum per class).

**Pipeline:**
1. Load training pixels from `.npz`; extract spectral feature vectors (all bands).
   NaN bands (water-vapour / detector-gap channels) are imputed with the
   per-band column mean.
2. Compute the mean L2-normalised spectrum per training class as the SAM
   endmember (`compute_class_endmembers`).
3. Apply nearest-endmember SAM to all soil-masked pixels.  If
   `SAM_ANGLE_THRESHOLD` is set, pixels whose minimum angle exceeds the
   threshold are labelled Unclassified; by default all soil pixels are
   classified.
4. Connected-component noise filter (< `MIN_COMPONENT_SIZE` px), matching
   the library SAM post-processing.
5. Validation metrics:
   - Noise fraction per class (pixels removed by size filter)
   - Moran's I spatial clustering (Queen contiguity within soil domain)
   - Min-angle statistics: mean and std of the minimum SAM angle per class
   - Endmember spectra figure saved to `validation/sam_endmember_spectra.png`

**Cross-method comparison (supervised SAM AMD zone vs library SAM mineral map):**

| Metric | Description |
|--------|-------------|
| Jaccard IoU | `(sup_SAM ∩ lib_SAM) / (sup_SAM ∪ lib_SAM)` — spatial overlap |
| Cohen's Kappa | Chance-corrected binary agreement within soil domain |
| Library SAM recall | Fraction of library SAM pixels inside supervised SAM zone |
| Supervised SAM efficiency | Fraction of supervised SAM pixels confirmed by ≥ 1 library mineral |
| Per-mineral containment | Fraction of each library mineral inside supervised SAM AMD zone |

**Key configuration constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `SAM_ANGLE_THRESHOLD` | `None` | Max SAM angle (rad) to accept; `None` = nearest-endmember (no rejection) |
| `MIN_COMPONENT_SIZE` | 4 | Noise component size threshold (pixels) |
| `EXCLUDE_BACKGROUND` | `True` | Relabel Background predictions to Unclassified |
| `REFLECTANCE_SCALE` | 10000.0 | Applied if `cube.max() > 2.0` |

**Outputs** (all written to `amd_mapping/outputs/supervised_sam/`):

| File | Description |
|------|-------------|
| `sam_classification_map.tif` | GeoTIFF classification map (int16) |
| `sam_classification_map.hdr/.img` | Same map in ENVI format |
| `sam_angle_maps.hdr/.img` | Per-class SAM angle maps (float32, radians) |
| `sam_classification_map.png` | Colour visualisation with legend |
| `sam_validation_metrics.csv` | Per-class: n_px, noise_frac, mean/std angle, Moran's I |
| `sam_cross_method_comparison.csv` | Jaccard IoU, Cohen's Kappa, lib-SAM recall, sup-SAM efficiency |
| `sam_per_mineral_containment.csv` | Per-library-mineral containment in supervised SAM AMD zone |
| `sam_cross_method_comparison.png` | Two-panel spatial agreement figure |
| `validation/sam_endmember_spectra.png` | L2-normalised endmember spectra per class |

---

## Citation

If you use these scripts in research, please cite the original SAM algorithm:

> Kruse, F. A., et al. (1993). The spectral image processing system (SIPS)—interactive visualization and analysis of imaging spectrometer data. Remote sensing of environment, 44(2-3), 145-163.
