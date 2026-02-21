# Null-Model Statistical Threshold — Implementation Summary

**Date:** February 2026
**Script:** `multi_mineral_sam_fixed.py`
**Feature:** Statistically grounded per-mineral SAM threshold derived from a background null model

---

## Background

The original adaptive threshold was defined as:

```
threshold = min_angle × (1 + SAM_THRESHOLD_MARGIN)
```

where `SAM_THRESHOLD_MARGIN = 0.01` (1 % above the single best-matching soil pixel).
This is numerically convenient but statistically arbitrary — it depends entirely on
whichever pixel happens to have the smallest angle, and gives no indication of
whether the classified pixels are actually distinguishable from random background soil.

---

## Problem

A pixel classified as, say, Jarosite might have a SAM angle only marginally lower
than that of a typical bare-soil pixel against the same endmember.
Without knowing the background distribution, there is no principled way to claim
that the detection is meaningful at any confidence level.

---

## Solution: Spectral-Angle Null Model

For each mineral endmember a null distribution of spectral angles is built from
**background soil pixels** (soil pixels not classified by any adaptive threshold)
and used to derive a statistically anchored threshold.

### Algorithm (implemented in `derive_null_thresholds`)

1. **Define background:** union of all adaptive-threshold classification masks is
   excluded from the soil pixel pool.  The remaining soil pixels form the background.

2. **Sample:** up to `NULL_SAMPLE_SIZE` (default 5 000) background pixels are drawn
   at random (fixed seed 42 for reproducibility).

3. **Compute null angles:** SAM angles are computed between the sampled background
   pixels and each unit-normalised endmember via a batch dot-product:
   ```
   cos θ = (bg_spectra @ endmember) / ||bg_pixel||
   θ = arccos(clip(cos θ, -1, 1))
   ```
   Background spectra are extracted once and shared across all minerals, so the
   cost is one matrix multiplication per endmember.

4. **Derive threshold:** the threshold is the **(1 − confidence) × 100**-th percentile
   of the null angle distribution.  With the default `NULL_MODEL_CONFIDENCE = 0.95`
   this is the **5th percentile**.

### Interpretation

A pixel passes the null threshold only if its SAM angle is smaller than **95 %**
of all random background soil pixels against the same endmember.
This means the pixel is spectrally more similar to the mineral than would be
expected by chance at a 95 % confidence level.

---

## Key Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `NULL_MODEL_CONFIDENCE` | `0.95` | Fraction of background pixels that must have a *larger* angle than a classified pixel |
| `NULL_SAMPLE_SIZE` | `5000` | Max background pixels sampled per run (increase for larger scenes) |

Both are set in the `CONFIGURATION` block at the top of the script and apply to all minerals.

---

## Workflow Integration

The null model runs as **step 7b** in `main()`, immediately after the adaptive
thresholds are derived (step 7) and before match scores are computed (step 8).

```
Step 7   → Adaptive thresholds (min_angle × 1.01)
Step 7b  → Null-model thresholds (5th percentile of background angles)
Step 8   → Match scores (based on adaptive threshold, unchanged)
Step 9b  → compare_thresholds() — side-by-side table with inertia ratios
Step 10  → validate_sam_results() — per-mineral header shows both pixel counts;
            validation histogram overlays the null distribution in gray
```

---

## Outputs

### Console — step 7b
```
======================================================================
NULL-MODEL THRESHOLD DERIVATION  (confidence = 95%,  threshold = 5th percentile of null angles)
======================================================================
  Background pixels available : 12,450
  Sampled 5,000 background pixels for null distributions

  Mineral                   Null thr (rad)   Null thr (°)      Min null (°)      Max null (°)
  Goethite                          0.2314          13.25°             7.41°            87.33°
  Hematite                          0.2198          12.59°             6.82°            85.10°
  ...
```

### Console — step 9b (`compare_thresholds`)
```
======================================================================
THRESHOLD COMPARISON  (adaptive  vs  null-model @ 95% confidence)
======================================================================
  Mineral              Adapt thr (°)   Adapt px   Iner_A   Null thr (°)    Null px   Iner_N        Δ px
  Goethite                     0.31°          4   0.8721         13.25°      5,812   0.7340      +5,808
  ...
  Δ px     = null-threshold pixels − adaptive-threshold pixels
  Iner_A   = inertia ratio under adaptive threshold  (real / background)
  Iner_N   = inertia ratio under null-model threshold
  Inertia ratio < 1.0 → classified pixels more spectrally coherent than random soil
```

### Validation figures
Each `<mineral>_validation.png` histogram panel now shows:
- **Blue bars** — classified pixels (adaptive threshold)
- **Gray bars** — null/background angle distribution (scaled to 60 % of blue peak)
- **Purple dashed line** — adaptive threshold
- **Green dash-dot line** — null-model threshold
- x-axis extended to `max(adaptive_thr, null_thr) × 1.05`

### Validation summary CSV (`validation/validation_summary.csv`)
Two new columns added: `Null_thr_deg`, `Null_pixels`.

---

## Typical Behaviour

Because the adaptive threshold is anchored to the single lowest-angle pixel
(which may be an outlier), it is typically **much more restrictive** than the
null-model threshold.  The null threshold usually produces substantially more
classified pixels (positive Δ px).

If the inertia ratio under the null threshold is still < 1.0, the larger set
of classified pixels remains spectrally more coherent than random background,
which supports using the null threshold for the final map.

---

## New Functions

| Function | Location | Purpose |
|---|---|---|
| `derive_null_thresholds` | after `compute_soil_mask` | Compute per-mineral null distributions and thresholds |
| `compare_thresholds` | before `main` | Print adaptive vs null comparison table with inertia ratios |

### Modified functions

| Function | Change |
|---|---|
| `validate_sam_results` | Accepts `null_thresholds`, `null_distributions`; reports null pixel counts per mineral |
| `_save_validation_figure` | Accepts `null_thr`, `null_angles`; overlays null distribution on histogram |
| `main` | Calls steps 7b and 9b; passes null data to validation |
