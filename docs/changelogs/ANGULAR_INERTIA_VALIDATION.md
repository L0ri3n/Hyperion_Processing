# Angular Spectral Inertia — Validation Metric Replacement

**Date:** February 2026
**Script:** `multi_mineral_sam_fixed.py`
**Feature:** Replacement of the Euclidean KMeans inertia metric in
`validate_sam_results` with an angular spectral inertia approach that is
fully consistent with SAM's magnitude-invariant nature

---

## Background

The post-classification validation function `validate_sam_results` previously
computed metric 3 as a **KMeans inertia ratio**: it fit a single-cluster
K-Means model to the classified pixels' reflectance vectors, did the same for
an equal-sized random sample of non-classified soil pixels, and reported the
ratio of the two inertia values.

This approach has two fundamental problems for SAM outputs:

1. **Euclidean incoherence.** SAM is explicitly magnitude-invariant — two pixels
   with identical spectral shape but different brightness have angle zero. KMeans
   inertia is Euclidean (sum of squared distances from the centroid in reflectance
   space), so it penalises brightness variation that SAM purposefully ignores.
   A good SAM classification can produce a high inertia ratio if classified pixels
   vary in brightness while sharing the same spectral shape.

2. **Mismatched null model.** The KMeans null was sampled from pixels *not
   classified by that specific mineral*, which includes pixels classified as other
   minerals. The comparison was therefore not a clean classified-vs-background test.

---

## Solution: Angular Spectral Inertia

Metric 3 is replaced by a purely angular comparison that uses the same spectral
angle space as SAM itself.

### Concept

For each mineral endmember, two angle distributions are compared:

- **α_classified** — per-pixel SAM angles at classified pixel locations, read
  directly from the existing SAM rule image layer (`sam_angles_dict[name]`).
- **α_null** — SAM angles computed between a random background soil sample and
  the **same endmember**.

If the classification is valid, α_classified should be stochastically *lower*
than α_null: classified pixels are more similar to the endmember than
unclassified background soil.

### Background pool definition

```
background = soil ∩ ¬(any mineral classified under its adaptive threshold)
```

The pool excludes all pixels classified by **any** mineral, not just the one
being evaluated. This gives a clean unclassified soil baseline that is shared
across all minerals (extracted once before the per-mineral loop for efficiency).

### Pool size

```
pool_size = min(max(n_classified_max, NULL_SAMPLE_SIZE), n_background_available)
```

`n_classified_max` is the largest per-mineral classified count. This ensures
the null sample always matches or exceeds the number of classified pixels for
each mineral, giving the Mann-Whitney test sufficient power.

### Per-mineral angle computation

```python
cos_null      = (bg_spectra @ endmember) / bg_norms   # batch dot-product
cos_null      = clip(cos_null, -1, 1)
angles_null   = arccos(cos_null)
```

Background spectra are extracted once; only the dot-product with the endmember
changes per mineral. Endmembers are already unit-normalised from
`load_and_resample_spectrum`.

### Summary statistics

| Statistic | Formula | Interpretation |
|---|---|---|
| `Mean_angle_cls_deg` | mean(α_classified) in degrees | Average angle of classified pixels to endmember |
| `Mean_angle_null_deg` | mean(α_null) in degrees | Average angle of background to same endmember |
| `Angular_inertia_ratio` | mean(α_cls) / mean(α_null) | **< 1 → good** (classified closer to endmember than background) |
| `MannWhitney_p` | one-sided Mann-Whitney U p-value | Probability of observing this separation by chance |
| `Effect_size` | rank-biserial r = 1 − 2U / (n₁·n₂) | **> 0 → good** (+1 = all classified < all null; 0 = no difference) |

### Statistical test

```
H₀: α_classified and α_null follow the same distribution
H₁: α_classified is stochastically less than α_null
```

`scipy.stats.mannwhitneyu(angles_cls, angles_null, alternative='less')` is
used rather than a t-test because SAM angle distributions are generally not
normal (bounded below by zero, possibly skewed toward low angles for good
classifications).

#### Rank-biserial effect size r

```
r = 1 − 2U / (n₁ × n₂)
```

Where U is the Mann-Whitney statistic for the classified sample (counts
classified-angle < null-angle pairs):

- `r → +1` : all classified angles below all null angles (perfect separation)
- `r ≈ 0`  : no difference from background
- `r → −1` : classified angles are *larger* than background (classification failure)

---

## Scope of Change

This change affects only `validate_sam_results` (metric 3) and the downstream
CSV and figure it produces. The `compare_thresholds` function (step 9b) retains
its own local KMeans inertia computation for the adaptive vs null-threshold
comparison table — that table is explicitly comparing Euclidean coherence as a
secondary diagnostic and is not affected.

---

## Workflow Integration

```
Step 10  → validate_sam_results()
             ├── Metric 1: Connected components
             ├── Metric 2: SAM angle distribution (unchanged)
             ├── Metric 3: Angular spectral inertia  ← REPLACED
             │     ├── build background pool once (pre-loop)
             │     ├── per mineral: dot-product → arccos → α_null
             │     ├── Mann-Whitney U (one-sided)
             │     └── rank-biserial effect size r
             └── Metric 4: Moran's I (unchanged)
```

`mineral_spectra` (the unit-normalised endmember dict from `main()`) is now
passed as a keyword argument to `validate_sam_results` so the function can
compute per-endmember background angles without re-loading the library.

---

## Outputs

### Console — metric 3 block (per mineral)

```
     [3] Angular inertia:
         mean(classified) = 8.241°,  mean(null) = 19.873°,  ratio = 0.4147  (< null (good))
         Mann-Whitney p = 3.4521e-89,  effect size r = 0.8312
```

### Validation figures (`validation/<mineral>_validation.png`)

The left histogram panel now shows:

- **Blue bars** — α_classified (per-pixel SAM angles at classified locations)
- **Gray bars** — α_null (background angles to same endmember, scaled to 60 %
  of blue peak for visual overlay)
- **Purple dashed line** — adaptive threshold
- **Green dash-dot line** — null-model threshold (from `derive_null_thresholds`,
  unchanged)
- **Red dashed line** — mean(α_classified)
- **Orange dotted lines** — ±1σ of α_classified

The gray bars now represent the angular-inertia null distribution (freshly
sampled in `validate_sam_results`) rather than the `null_distributions` sample
from `derive_null_thresholds`. Both come from the same background population, so
the green null-threshold line remains consistent with the displayed distribution.

### Validation summary CSV (`validation/validation_summary.csv`)

Columns **replaced**:

| Removed | Added |
|---|---|
| `Inertia_real` | `Mean_angle_cls_deg` |
| `Inertia_null` | `Mean_angle_null_deg` |
| `Inertia_ratio` | `Angular_inertia_ratio` |
| *(n/a)* | `MannWhitney_p` |
| *(n/a)* | `Effect_size` |

All other columns (`Mineral`, `Pixels`, `Null_thr_deg`, `Null_pixels`,
`Components`, `Noise_px_frac`, `Mean_angle_deg`, `Std_angle_deg`, `Skewness`,
`Dist_quality`, `Morans_I`, `Morans_z`, `Morans_p`, `Morans_sig`) are
unchanged.

---

## Interpreting the New Columns

| Value | Meaning |
|---|---|
| `Angular_inertia_ratio` < 1.0 | Classified pixels are on average more similar to the endmember than random background soil — good signal |
| `Angular_inertia_ratio` ≥ 1.0 | Classification is not distinguishable from background in angle space |
| `MannWhitney_p` < 0.05 | The angular separation is statistically significant at the 5 % level |
| `Effect_size` > 0.5 | Large effect: substantial majority of classified pixels have lower angles than background |
| `Effect_size` < 0.1 | Negligible effect: classification barely distinguishable from background |

---

## Dependencies

| Library | Used for | Already in environment |
|---|---|---|
| `scipy.stats.mannwhitneyu` | One-sided Mann-Whitney U test | Yes (scipy already imported) |
| `numpy` | Angle computation, dot-product | Yes |

No new dependencies are introduced.

---

## Modified Functions

| Function | Change |
|---|---|
| `validate_sam_results` | New `mineral_spectra=None` parameter; metric 3 replaced; new CSV columns |
| `main` | Passes `mineral_spectra=mineral_spectra` to `validate_sam_results` |

### Removed dependency

`sklearn.cluster.KMeans` is no longer imported inside `validate_sam_results`.
It is still used by `compare_thresholds` (via its own local import) and is
therefore still required in the environment for the step-9b comparison table.
