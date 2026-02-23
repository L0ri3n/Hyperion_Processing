# Supervised Classification Extension — Training Pixel Selection + Supervised SAM

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
   class using three complementary image composites.
2. Computing an image-derived SAM endmember (mean L2-normalised spectrum)
   per training class and classifying the full scene with the identical
   SAM nearest-endmember rule — no algorithm change, only endmember source.
3. Producing a cross-method comparison (Jaccard IoU, Cohen's Kappa, per-mineral
   containment) between supervised SAM and library SAM classifications so that
   agreement and disagreement zones can be identified and investigated.

Divergences between the two SAM outputs reflect the spectral distance between
USGS library references and actual Rio Tinto scene conditions (mixed pixels,
atmospheric residuals at 30 m resolution), not algorithmic differences.

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
assigns them to one of two target classes (`AMD_FeOx` or `Background`).
Only pixels inside the soil mask are retained.

Binary classification is intentional: the supervised SAM produces a broad
ferro-oxide lithology footprint, which is then compared per-mineral against
the library SAM detections via per-mineral containment metrics.

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
| Class | Radio buttons (2) | Set active class (AMD_FeOx / Background) |
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

`training_pixels.npz` — one key per class (`AMD_FeOx`, `Background`);
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

Loads the training pixel dictionary and runs a supervised SAM classification
using image-derived endmembers (mean L2-normalised spectrum per training
class).  The identical SAM nearest-endmember rule is applied — the only
difference from the library SAM stage is the endmember source.  Validation
metrics mirror those of the library SAM workflow, plus a cross-method
comparison between the two SAM outputs.

### Classifier

```
Endmember = mean(training_spectra_for_class) / ||mean(training_spectra_for_class)||
Classification = argmin_class SAM_angle(pixel, endmember_class)
```

For each soil pixel the SAM angle to every class endmember is computed and
the pixel is assigned to the class with the minimum angle.  If
`SAM_ANGLE_THRESHOLD` is set (default `None`), pixels whose minimum angle
exceeds the threshold are labelled Unclassified; otherwise all soil pixels
receive a classification.

### Post-processing

Connected components smaller than 4 pixels are removed (same
`MIN_COMPONENT_SIZE` constant as the library SAM workflow).  This step
operates on each binary class mask; cleaned masks are then composited back
into the integer class map.

### Validation metrics

| Metric | Description | Parity with library SAM |
|--------|-------------|--------------------------|
| Noise fraction | Fraction of classified pixels removed by size filter | Mirrors SAM connected-component metric |
| Moran's I | Spatial clustering of AMD_FeOx binary mask (Queen contiguity, row-standardised, within soil domain) | Identical `_compute_morans_i` implementation |
| Min-angle statistics | Mean and std of the minimum SAM angle for pixels classified as each class | Directly comparable to SAM angle distributions |

Endmember spectra (L2-normalised mean training spectrum per class) are
saved to `validation/sam_endmember_spectra.png` for visual inspection.

### Cross-method comparison

The supervised SAM AMD_FeOx zone is compared against the library SAM
mineral map (union of all mineral detections):

| Metric | Formula / Description |
|--------|-----------------------|
| Jaccard IoU | `(sup_SAM ∩ lib_SAM) / (sup_SAM ∪ lib_SAM)` |
| Cohen's Kappa | Chance-corrected binary agreement within soil domain |
| Library SAM recall in sup SAM zone | Fraction of library SAM pixels inside supervised SAM AMD zone |
| Supervised SAM efficiency | Fraction of supervised SAM AMD pixels confirmed by ≥ 1 library mineral |
| Per-mineral containment | Fraction of each library SAM mineral inside the supervised SAM AMD zone |

A two-panel figure shows the spatial overlap map (colour-coded by agreement
category) and a per-mineral containment bar chart.

### Outputs

All files written to `amd_mapping/outputs/supervised_sam/`:

| File | Format | Description |
|------|--------|-------------|
| `sam_classification_map.tif` | GeoTIFF (int16) | Class codes; −1 = unclassified / non-soil |
| `sam_classification_map.hdr/.img` | ENVI | Same map in ENVI format |
| `sam_angle_maps.hdr/.img` | ENVI (float32, n bands) | Per-class SAM angle maps (radians) |
| `sam_classification_map.png` | PNG | Colour visualisation with legend |
| `sam_validation_metrics.csv` | CSV | Per-class: n_px, noise_frac, mean/std angle, Moran's I |
| `sam_cross_method_comparison.csv` | CSV | Jaccard IoU, Cohen's Kappa, lib-SAM recall, sup-SAM efficiency |
| `sam_per_mineral_containment.csv` | CSV | Per-library-mineral containment in supervised SAM AMD zone |
| `sam_cross_method_comparison.png` | PNG | Two-panel spatial agreement figure |
| `validation/sam_endmember_spectra.png` | PNG | L2-normalised endmember spectra per class |

### Background class handling

`Background` pixels are included during endmember computation (so the
classifier learns to distinguish AMD iron-oxide spectra from generic bare
soil), but pixels assigned to Background in the output map are relabelled
to Unclassified by default (`EXCLUDE_BACKGROUND = True`).  This matches
the library SAM convention where unclassified soil is code −1.

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
| `pandas` | sam, supervised | CSV tables |
| `matplotlib` | all | visualisation, GUI widgets |
| `scikit-image` | sam | binary morphology |
| `esda` + `libpysal` | sam, supervised | Moran's I (optional) |
| `sklearn.metrics` | supervised | Cohen's Kappa (optional) |
| `rasterio` | supervised | GeoTIFF export (optional) |

---

## Workflow Summary

```
multi_mineral_sam_fixed.py
  └── computes + saves soil_mask.npy
  └── saves SAM classification_map.hdr/.img

training_pixel_selector.py
  ├── loads reflectance cube + soil_mask.npy
  └── saves training_pixels.npz  (AMD_FeOx, Background)

supervised_classification.py
  ├── loads cube + soil_mask.npy + training_pixels.npz
  ├── loads SAM classification_map.hdr (for cross-method comparison)
  └── saves sam_classification_map.*, sam_angle_maps.*,
      sam_validation_metrics.csv, sam_cross_method_comparison.{csv,png},
      sam_per_mineral_containment.csv,
      validation/sam_endmember_spectra.png
```
