# Supervised Classification Extension — Training Pixel Selection + Random Forest

**Date:** February 2026
**Scripts added:**
- `training_pixel_selector.py`
- `supervised_classification.py`
- `run_pipeline.py`

**Script modified:**
- `multi_mineral_sam_fixed.py` — soil mask persistence

---

## Background

The existing workflow performs unsupervised Spectral Angle Mapper (SAM)
classification against USGS library endmembers.  While robust and
physically interpretable, SAM classification is sensitive to the choice
of threshold and to the representativeness of library spectra relative to
the specific atmospheric and illumination conditions of a given scene.

The supervised extension addresses these limitations by:

1. Letting the analyst label scene-specific training pixels for each AMD
   mineral class using three complementary image composites.
2. Training a Random Forest classifier on those scene-specific spectra,
   which learns the local spectral signatures rather than relying on
   library reference shapes.
3. Producing a cross-method comparison (Jaccard IoU) between SAM and RF
   classifications so that agreement and disagreement zones can be
   identified and investigated.

---

## Changes to `multi_mineral_sam_fixed.py`

### Soil mask persistence (surgical one-block addition to `main()`)

**Location:** After `compute_soil_mask()` returns, immediately before
`save_soil_mask_plot()`.

**Change:**
```python
soil_mask_npy = Path(HDR_FILE).parent / "soil_mask.npy"
np.save(str(soil_mask_npy), soil_mask)
print(f"   Soil mask saved: {soil_mask_npy}")
```

**Rationale:**
Both new modules need the binary soil mask.  Previously it was computed
but never written to disk; both modules had to either recompute it (wasted
runtime) or use a valid-data fallback (potentially different from the
filtered mask used by SAM).  Saving it as `soil_mask.npy` in the same
directory as the `.hdr` file makes the mask available to any downstream
module without code coupling.

---

## New Script: `training_pixel_selector.py`

### Purpose

Interactive matplotlib GUI for labelling AMD mineral training pixels.  The
analyst draws rectangular ROIs on three switchable image composites and
assigns them to one of seven target classes.  Only pixels inside the soil
mask are retained.

### Image composites

| View | Bands | Diagnostic value |
|------|-------|-----------------|
| RGB | 660 / 550 / 480 nm | True-colour context |
| NIR False Colour | 850 / 660 / 550 nm | Vegetation / bare-soil contrast |
| Fe³⁺ Oxide Ratio | 900 / 660 nm (ratio) | Iron oxide intensity map |

All composites use 2nd–98th percentile linear stretching per channel.  The
Fe³⁺ oxide ratio (band_900 / band_660) highlights goethite, hematite, and
jarosite zones; the soil mask boundary is drawn as a thin white contour on
this view.

### GUI controls

| Widget | Type | Action |
|--------|------|--------|
| View | Radio buttons (3) | Switch composite |
| Class | Radio buttons (7) | Set active mineral class |
| Soil mask overlay | Toggle button | Show/hide yellow semi-transparent fill |
| ROI drawing | Left-click + drag | Draw rectangle; fires on release |
| Clear Class | Button | Remove all ROIs for current class |
| Clear All | Button | Remove all ROIs for all classes |
| Save & Continue | Button | Serialise to .npz, print summary, close window |

### Spectral profile panel

The right panel shows a live spectral profile that updates on:

* **Hover** — single-pixel spectrum (thin dotted grey line)
* **ROI completion** — mean ± 1 σ of soil-masked pixels in the rectangle
  (solid class-coloured line + shaded band)

Background class mean spectra from all previously drawn ROIs are shown as
faint coloured lines for cross-class comparison.  Vertical dashed
reference lines mark the three key AMD diagnostic wavelengths:

| Wavelength (nm) | Feature |
|-----------------|---------|
| 430 | Jarosite / hematite absorption |
| 660 | Fe charge-transfer band |
| 875 | Goethite / hematite crystal-field |

### Output format

`training_pixels.npz` — one key per AMD class (class name as string);
value is a de-duplicated int32 array of shape `(n_pixels, 2)` with
columns `[row, col]`.

Classes with zero labelled pixels are saved as `(0, 2)` empty arrays and
excluded from training automatically.

### Design assumptions

* **Soil mask pre-filter:** Only pixels inside the binary soil mask are
  stored.  This guarantees that all training samples come from the same
  domain as the prediction target.
* **De-duplication:** Overlapping ROIs are merged via a Python `set` so
  that no pixel is counted twice in the feature matrix.
* **Layered imshow:** Three stacked `imshow` artists (base image, mask
  boundary, overlay) are maintained as persistent matplotlib artists.
  Switching views only calls `set_data()` on the base artist rather than
  clearing and recreating axes, keeping state (drawn ROI patches) intact.

---

## New Script: `supervised_classification.py`

### Purpose

Loads the training pixel dictionary and runs the full supervised Random
Forest classification pipeline with validation metrics that mirror and
extend those of the existing SAM workflow.

### Classifier

```
RandomForestClassifier(
    n_estimators  = 200,
    class_weight  = 'balanced',
    random_state  = 42,
    n_jobs        = -1,
)
```

`class_weight='balanced'` compensates for the typically unequal class
sizes that arise from manual labelling (e.g., Goethite patches may be far
more extensive than Pyrite outcrops).

### Cross-validation

5-fold stratified cross-validation (fixed `random_state=42`).  Balanced
accuracy is reported per fold and as mean ± std:

```
Balanced Accuracy = mean(recall per class)
```

Using balanced accuracy (rather than overall accuracy) is appropriate
because class sizes in the soil domain are highly unequal.

### Prediction and rejection

The classifier is applied to all soil-masked pixels in a single batched
call to `predict_proba`.  Predictions where `max(class probabilities)
< 0.60` are set to Unclassified (code −1) to reject spectrally ambiguous
pixels.  The 0.60 threshold corresponds to a "confident majority" — the
winning class must account for at least 60 % of the posterior probability
mass.

### Post-processing

Connected components smaller than 4 pixels are removed (same
`MIN_COMPONENT_SIZE` constant as the SAM workflow).  This step operates
independently on each binary class mask; the cleaned masks are then
composited back into the integer class map.

### Validation metrics

| Metric | Description | Parity with SAM workflow |
|--------|-------------|--------------------------|
| Noise fraction | Fraction of classified pixels removed by size filter | Mirrors SAM connected-component metric |
| Moran's I | Spatial clustering of each binary class mask (Queen contiguity, row-standardised, within soil domain) | Identical implementation (`_compute_morans_i`) |
| MDI feature importances | Mean decrease in impurity per band, plotted against wavelength | New metric; complements SAM angle distributions |
| Max-probability statistics | Mean and std of `max(class proba)` for accepted pixels per class | New metric; low mean → uncertain classification |

### Cross-method comparison

For each RF class the script finds the best-matching SAM class (by
substring matching on the ENVI `class names` metadata), then computes
Jaccard IoU:

```
IoU = |RF_mask ∩ SAM_mask| / |RF_mask ∪ SAM_mask|
```

A high IoU (> 0.5) indicates that the two independent methods agree on
both the location and extent of a given mineral zone.  Low IoU may
indicate threshold sensitivity in SAM, insufficient training samples in
RF, or genuine spatial disagreement warranting field investigation.

### Outputs

| File | Format | Description |
|------|--------|-------------|
| `rf_classification_map.tif` | GeoTIFF (int16) | RF class codes; −1 = unclassified |
| `rf_classification_map.hdr/.img` | ENVI | Same map in ENVI format |
| `rf_probability_maps.hdr/.img` | ENVI (float32, n bands) | Per-class posterior probabilities |
| `rf_classification_map.png` | PNG | Colour visualisation with legend |
| `rf_cv_scores.csv` | CSV | Balanced accuracy per fold |
| `rf_validation_metrics.csv` | CSV | Per-class: n_px, noise_frac, mean_max_prob, Moran's I |
| `rf_cross_method_comparison.csv` | CSV | RF vs SAM Jaccard IoU per class |
| `validation/rf_feature_importances.csv` | CSV | MDI importance per band + wavelength |
| `validation/rf_feature_importances.png` | PNG | Bar chart of top-30 bands by MDI |

### Background class handling

`Background` pixels are included in training (so that the RF learns to
distinguish AMD minerals from generic bare soil), but predictions labelled
as Background in the output map are relabelled to Unclassified by default
(`EXCLUDE_BACKGROUND = True`).  This matches the SAM convention where
unclassified soil is code 0, not a positive mineral assignment.

---

## New Script: `run_pipeline.py`

Thin orchestration wrapper that imports and calls the `main()` / entry
functions of all three modules in sequence.  Provides `--stage`,
`--skip-sam`, and `--skip-select` flags for partial re-runs (e.g., when
the analyst wants to add more training pixels without re-running the
expensive SAM stage).

The script performs pre-flight file existence checks before each stage and
continues with the remaining stages if one fails, printing a summary at
the end.

---

## Dependency Summary

| Library | Required by | Purpose |
|---------|-------------|---------|
| `numpy` | all | array operations |
| `scipy` | sam, selector | morphological ops, ndimage |
| `spectral` | all | ENVI I/O |
| `scikit-learn` | rf | RandomForest, stratified CV |
| `pandas` | sam, rf | CSV tables |
| `matplotlib` | all | visualisation, GUI widgets |
| `scikit-image` | sam | binary morphology |
| `esda` + `libpysal` | sam, rf | Moran's I (optional) |
| `rasterio` | rf | GeoTIFF export (optional) |

---

## Workflow Summary

```
multi_mineral_sam_fixed.py
  └── computes + saves soil_mask.npy
  └── saves SAM classification_map.hdr/.img

training_pixel_selector.py
  ├── loads reflectance cube + soil_mask.npy
  └── saves training_pixels.npz

supervised_classification.py
  ├── loads cube + soil_mask.npy + training_pixels.npz
  ├── loads SAM classification_map.hdr (for Jaccard comparison)
  └── saves rf_classification_map.*, rf_probability_maps.*, validation CSVs
```
