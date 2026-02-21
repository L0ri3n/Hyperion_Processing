"""
supervised_classification.py — Random Forest AMD Ferro-oxide Lithology Classifier
==================================================================================

Binary supervised classification of AMD ferro-oxide lithology using a Random
Forest trained on manually selected training pixels (AMD_FeOx vs. Background).

The classifier identifies the broad AMD-altered zone — the lithological unit
that hosts all Fe³⁺ secondary minerals (goethite, hematite, jarosite, etc.)
collectively — rather than discriminating individual mineral species.  This
produces a more statistically robust, spatially coherent result than the SAM
mineral-specific approach, at the cost of spectral specificity.

Pipeline overview
-----------------
1. Load training pixels (.npz) and reflectance cube (ENVI .hdr).
2. Extract spectral feature vectors (all valid bands) and assemble a labelled
   dataset with classes [AMD_FeOx, Background].
3. Train RandomForestClassifier(n_estimators=200, class_weight='balanced',
   random_state=42) with 5-fold stratified cross-validation (balanced
   accuracy reported per fold).
4. Apply the trained classifier to all soil-masked pixels using predict_proba;
   predictions with max AMD_FeOx probability < 0.60 are rejected as uncertain.
5. Reconstruct the full-scene AMD_FeOx mask and probability map.
6. Apply a connected-component noise filter (components < 4 pixels removed),
   matching the post-processing used in the SAM workflow.
7. Validate results:
   (a) Noise fraction (pixels removed by size filter / pre-filter total).
   (b) Moran's I spatial clustering of the AMD_FeOx binary mask within the
       soil domain (Queen contiguity, same implementation as the SAM workflow).
   (c) MDI feature importances with corresponding band wavelengths.
   (d) Mean and std of AMD_FeOx posterior probability for accepted pixels.
8. Cross-method comparison (RF AMD lithology vs. SAM mineral map):
   The two results are complementary, not directly comparable class-to-class.
   The comparison aggregates all SAM mineral detections into a single binary
   'SAM AMD' mask and computes:
     - Jaccard IoU  (RF_AMD ∩ SAM_union) / (RF_AMD ∪ SAM_union)
     - Cohen's Kappa  (binary agreement within soil domain, chance-corrected)
     - SAM recall     fraction of SAM pixels captured by RF AMD zone
     - RF efficiency  fraction of RF AMD pixels confirmed by ≥1 SAM mineral
     - Per-mineral containment  fraction of each SAM mineral inside RF AMD zone
9. Save: classification map as GeoTIFF + ENVI, probability map as ENVI,
   all validation tables as CSV, feature importance and comparison plots as PNG.

Assumptions
-----------
* Training .npz contains keys 'AMD_FeOx' and 'Background'; values are int32
  arrays of shape (n_pixels, 2) with columns [row, col].
* Class 'Background' is excluded from the output map (set to Unclassified)
  but is used during training.  Set EXCLUDE_BACKGROUND = False to retain it.
* All 196 Hyperion bands are used as features.  NaN values are imputed with
  the per-band column mean before fitting.
* Moran's I requires esda and libpysal.  If not installed the metric is skipped
  with a warning, and the pipeline still completes successfully.
* GeoTIFF export requires rasterio.  If not installed the file is skipped and
  only the ENVI format is written.
* The SAM classification map is expected at SAM_CLASS_MAP_FILE (defined below).
  If absent the cross-method comparison section is skipped gracefully.
"""

import numpy as np
import pandas as pd
import spectral as sp
from pathlib import Path
from scipy.ndimage import label as nd_label
import matplotlib.pyplot as plt
import warnings

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Path to the training pixel dictionary produced by training_pixel_selector.py
TRAINING_PIXELS_FILE = str(
    BASE_DIR / "amd_mapping" / "outputs" / "training_pixels.npz"
)

# Reflectance cube (same file used by multi_mineral_sam_fixed.py)
HDR_FILE = str(
    BASE_DIR / "amd_mapping" / "data" / "hyperion"
    / "EO1H2020342013284110KF_reflectance.hdr"
)

# SAM classification map (ENVI .hdr) for cross-method comparison.
# Produced by multi_mineral_sam_fixed.py → save_envi_classification().
SAM_CLASS_MAP_FILE = str(
    BASE_DIR / "amd_mapping" / "outputs" / "classifications"
    / "classification_map.hdr"
)

# All output files go here
OUTPUT_FOLDER = str(
    BASE_DIR / "amd_mapping" / "outputs" / "supervised_classification"
)

# Random Forest hyper-parameters (fixed by specification)
RF_N_ESTIMATORS   = 200
RF_CLASS_WEIGHT   = "balanced"
RF_RANDOM_STATE   = 42

# Number of stratified k-fold cross-validation splits
CV_FOLDS = 5

# Minimum max-class probability to accept a prediction (else → Unclassified)
PROB_THRESHOLD = 0.60

# Connected components smaller than this are removed as noise.
# Matches MIN_COMPONENT_SIZE in multi_mineral_sam_fixed.py.
MIN_COMPONENT_SIZE = 4

# If True, pixels predicted as 'Background' are set to 0 (Unclassified)
# in the output map but are still used during training.
EXCLUDE_BACKGROUND = True

# Hyperion reflectance scale factor
REFLECTANCE_SCALE = 10000.0

# Colour palette (hex) — matches training_pixel_selector.py.
# AMD_FeOx uses a rust/burnt-orange colour evocative of iron-oxide mineralogy.
CLASS_COLORS = {
    "AMD_FeOx":   "#C05A0A",   # rust / burnt orange
    "Background": "#7F7F7F",   # neutral grey
}

# Supervised SAM: maximum SAM angle (radians) to accept a classification.
# If None, all soil pixels are classified using the nearest-endmember rule
# (i.e. every soil pixel is assigned to its closest class endmember).
# Set to e.g. np.pi / 4 (≈ 45°) to reject spectrally ambiguous pixels.
SAM_ANGLE_THRESHOLD = None

# Output folder for supervised SAM results (parallel to OUTPUT_FOLDER)
SAM_SUPERVISED_OUTPUT_FOLDER = str(
    BASE_DIR / "amd_mapping" / "outputs" / "supervised_sam"
)

# =============================================================================
# HELPERS
# =============================================================================


def _impute_nan_bands(spectra):
    """Replace per-band NaN values in-place with that band's column mean.

    If an entire band is NaN (e.g. a water-vapour or detector-gap band),
    the column is zeroed so that NaN does not propagate into model training
    or prediction.  This matches the implementation in multi_mineral_sam_fixed.py.

    Parameters
    ----------
    spectra : ndarray (n_pixels, n_bands), float64  — modified in-place
    """
    nan_bands = np.where(np.any(np.isnan(spectra), axis=0))[0]
    for b in nan_bands:
        col = spectra[:, b]
        mean_val = np.nanmean(col)
        col[np.isnan(col)] = 0.0 if np.isnan(mean_val) else mean_val


def _compute_morans_i(binary_mask, soil_r, soil_c, lps_weights, Moran):
    """Compute Moran's I for *binary_mask* restricted to the soil domain.

    Uses Queen (8-neighbour) contiguity weights within the soil-pixel set
    with row-standardisation.  The normal-approximation z-score and p-value
    are used (no permutations) for computational speed.

    This is a direct copy of the helper from multi_mineral_sam_fixed.py to
    maintain methodological consistency across SAM and RF validations.

    Parameters
    ----------
    binary_mask : ndarray (rows, cols), bool
    soil_r, soil_c : 1-D int arrays — row/col indices of all soil pixels
    lps_weights : libpysal.weights module
    Moran : esda.Moran class

    Returns
    -------
    I : float        — Moran's I statistic
    z : float        — normal-approximation z-score
    p : float        — two-tailed p-value
    sig_str : str    — human-readable significance description
    """
    # Build a fast (row, col) → soil-array index lookup
    idx_map = {
        (int(r), int(c)): i
        for i, (r, c) in enumerate(zip(soil_r, soil_c))
    }

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),           ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    neighbors    = {}
    weights_dict = {}
    for i, (r, c) in enumerate(zip(soil_r.tolist(), soil_c.tolist())):
        nbrs = [
            idx_map[(r + dr, c + dc)]
            for dr, dc in directions
            if (r + dr, c + dc) in idx_map
        ]
        neighbors[i]    = nbrs
        weights_dict[i] = [1.0] * len(nbrs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            w = lps_weights.W(neighbors, weights_dict, silence_warnings=True)
        except TypeError:
            w = lps_weights.W(neighbors, weights_dict)
    w.transform = "r"   # row-standardise

    y = binary_mask[soil_r, soil_c].astype(np.float64)
    if y.std() < 1e-12:
        return np.nan, np.nan, np.nan, "N/A (constant mask)"

    mi  = Moran(y, w, permutations=0)
    I, z, p = float(mi.I), float(mi.z_norm), float(mi.p_norm)

    if   p < 0.05 and I > 0:
        sig = "clustered (p<0.05)"
    elif p < 0.05 and I < 0:
        sig = "dispersed (p<0.05)"
    else:
        sig = f"random (p={p:.3f})"

    return I, z, p, sig


def _hex_to_rgba(hex_color, alpha=1.0):
    """Convert a hex colour string to an (r, g, b, a) float tuple."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)


def _save_geotiff(arr, out_path):
    """Write *arr* to a GeoTIFF file using rasterio.

    No CRS or geotransform metadata is attached because the Hyperion data in
    this workflow are not georeferenced at this processing stage.  The file is
    valid GeoTIFF and can accept georeferencing in a GIS application.

    Parameters
    ----------
    arr : ndarray (rows, cols), numeric
    out_path : Path or str
    """
    try:
        import rasterio
    except ImportError:
        print("  [GeoTIFF] rasterio not installed — skipping GeoTIFF output.")
        return

    from rasterio.transform import Affine

    rows, cols = arr.shape
    # Identity pixel-based transform (1 pixel = 1 unit, origin at top-left)
    transform = Affine(1, 0, 0, 0, -1, rows)

    with rasterio.open(
        str(out_path),
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=arr.dtype,
        compress="lzw",
        transform=transform,
    ) as dst:
        dst.write(arr, 1)

    print(f"  Saved GeoTIFF: {Path(out_path).name}")


# =============================================================================
# DATA LOADING
# =============================================================================


def load_training_data(npz_path, cube, soil_mask, wavelengths):
    """Load training pixels and extract spectral feature vectors.

    Reads the .npz produced by training_pixel_selector.py.  Each class key
    maps to an (n, 2) int32 array of [row, col] indices that already fall
    within the soil mask (guaranteed by the selector).  An additional
    soil-mask intersection check is applied here as a safety measure.

    NaN values in the spectra are imputed per-band before returning.

    Parameters
    ----------
    npz_path : str or Path
    cube : ndarray (rows, cols, bands), float32
    soil_mask : ndarray (rows, cols), bool
    wavelengths : ndarray (bands,), float

    Returns
    -------
    X : ndarray (n_samples, bands), float64   — feature matrix
    y : ndarray (n_samples,), int             — class labels (0-indexed)
    class_names : list of str                 — ordered class names
    label_to_idx : dict {str -> int}
    """
    data = np.load(str(npz_path), allow_pickle=True)
    class_names = [k for k in data.keys() if len(data[k]) > 0]

    if not class_names:
        raise RuntimeError(
            f"No non-empty classes found in {npz_path}.\n"
            "Please run training_pixel_selector.py first."
        )

    label_to_idx = {cls: i for i, cls in enumerate(class_names)}

    X_list, y_list = [], []
    print("\n  Training pixel summary (after soil-mask validation):")
    print(f"  {'Class':<22} {'Raw px':>8} {'Valid px':>9}")
    print("  " + "-" * 42)

    for cls in class_names:
        pixels = data[cls].astype(np.int32)           # (n, 2)
        n_raw = len(pixels)
        if n_raw == 0:
            continue

        rows_idx = pixels[:, 0]
        cols_idx = pixels[:, 1]

        # Re-validate soil mask membership (belt-and-braces)
        in_bounds = (
            (rows_idx >= 0) & (rows_idx < cube.shape[0]) &
            (cols_idx >= 0) & (cols_idx < cube.shape[1])
        )
        rows_idx = rows_idx[in_bounds]
        cols_idx = cols_idx[in_bounds]
        in_soil = soil_mask[rows_idx, cols_idx]
        rows_idx = rows_idx[in_soil]
        cols_idx = cols_idx[in_soil]
        n_valid = len(rows_idx)

        print(f"  {cls:<22} {n_raw:>8,} {n_valid:>9,}")

        if n_valid == 0:
            continue

        spectra = cube[rows_idx, cols_idx, :].astype(np.float64)
        X_list.append(spectra)
        y_list.extend([label_to_idx[cls]] * n_valid)

    print("  " + "-" * 42)

    if not X_list:
        raise RuntimeError(
            "No valid training pixels remain after soil-mask validation."
        )

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)

    # Impute NaN bands (water-vapour / detector-gap channels) in-place
    _impute_nan_bands(X)

    print(f"  Total training samples : {len(y):,}")
    print(f"  Features (bands)       : {X.shape[1]}")
    return X, y, class_names, label_to_idx


# =============================================================================
# TRAINING AND CROSS-VALIDATION
# =============================================================================


def train_with_cross_validation(X, y, class_names):
    """Train a RandomForestClassifier with 5-fold stratified cross-validation.

    Reports balanced accuracy per fold and the mean ± std across folds.
    The final classifier is retrained on the full training set.

    Parameters
    ----------
    X : ndarray (n_samples, n_bands), float64
    y : ndarray (n_samples,), int
    class_names : list of str

    Returns
    -------
    rf : fitted RandomForestClassifier
    cv_scores : ndarray (CV_FOLDS,), float — balanced accuracy per fold
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score

    rf_params = dict(
        n_estimators=RF_N_ESTIMATORS,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )

    print(f"\n  Random Forest hyper-parameters: {rf_params}")
    print(f"  Cross-validation: {CV_FOLDS}-fold stratified\n")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_scores = []

    print(f"  {'Fold':>6} {'Balanced Accuracy':>20}")
    print("  " + "-" * 28)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        rf_fold = RandomForestClassifier(**rf_params)
        rf_fold.fit(X[train_idx], y[train_idx])
        y_pred = rf_fold.predict(X[val_idx])
        ba = balanced_accuracy_score(y[val_idx], y_pred)
        cv_scores.append(ba)
        print(f"  {fold_idx:>6} {ba:>20.4f}")

    cv_scores = np.array(cv_scores)
    print("  " + "-" * 28)
    print(f"  {'Mean':>6} {cv_scores.mean():>20.4f}")
    print(f"  {'± Std':>6} {cv_scores.std():>20.4f}")

    # Retrain on the full set for final prediction
    print("\n  Retraining on full training set...")
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X, y)
    print("  Done.")

    return rf, cv_scores


# =============================================================================
# CLASSIFICATION AND POST-PROCESSING
# =============================================================================


def apply_classifier(rf, cube, soil_mask, class_names):
    """Apply the trained RF to all soil-masked pixels.

    Pixels where max class probability < PROB_THRESHOLD are labelled as
    Unclassified (code = -1) to reject spectrally ambiguous predictions.

    Parameters
    ----------
    rf : fitted RandomForestClassifier
    cube : ndarray (rows, cols, bands), float32
    soil_mask : ndarray (rows, cols), bool
    class_names : list of str

    Returns
    -------
    class_map : ndarray (rows, cols), int16
        Class codes: -1 = non-soil or rejected; 0..n-1 = class index.
    prob_maps : ndarray (rows, cols, n_classes), float32
        Per-class posterior probabilities (soil pixels only; others = 0).
    max_prob_map : ndarray (rows, cols), float32
        Maximum probability across all classes at each soil pixel.
    """
    rows, cols, bands = cube.shape
    n_classes = len(class_names)

    soil_r, soil_c = np.where(soil_mask)
    n_soil = len(soil_r)
    print(f"\n  Classifying {n_soil:,} soil pixels...")

    # Extract and impute soil spectra
    soil_spectra = cube[soil_r, soil_c, :].astype(np.float64)
    _impute_nan_bands(soil_spectra)

    # Predict in batches to control peak memory usage (1 000 000 px / batch)
    batch_size = 1_000_000
    probas_all  = np.zeros((n_soil, n_classes), dtype=np.float32)

    for start in range(0, n_soil, batch_size):
        end = min(start + batch_size, n_soil)
        probas_all[start:end] = rf.predict_proba(
            soil_spectra[start:end]
        ).astype(np.float32)

    max_proba  = probas_all.max(axis=1)
    pred_class = probas_all.argmax(axis=1).astype(np.int16)

    # Reject low-confidence predictions
    pred_class[max_proba < PROB_THRESHOLD] = -1

    n_rejected = int((max_proba < PROB_THRESHOLD).sum())
    n_accepted = int((max_proba >= PROB_THRESHOLD).sum())
    print(f"  Accepted predictions (prob ≥ {PROB_THRESHOLD:.2f}): {n_accepted:,} "
          f"({n_accepted / n_soil * 100:.1f}%)")
    print(f"  Rejected predictions (prob < {PROB_THRESHOLD:.2f}): {n_rejected:,} "
          f"({n_rejected / n_soil * 100:.1f}%)")

    # Reconstruct full-scene arrays
    class_map = np.full((rows, cols), -1, dtype=np.int16)
    class_map[soil_r, soil_c] = pred_class

    prob_maps = np.zeros((rows, cols, n_classes), dtype=np.float32)
    prob_maps[soil_r, soil_c] = probas_all

    max_prob_map = np.zeros((rows, cols), dtype=np.float32)
    max_prob_map[soil_r, soil_c] = max_proba

    # Optionally relabel Background predictions as Unclassified
    if EXCLUDE_BACKGROUND and "Background" in class_names:
        bg_idx = class_names.index("Background")
        class_map[class_map == bg_idx] = -1
        print(f"  Background class (idx {bg_idx}) relabelled to Unclassified "
              f"in output map.")

    return class_map, prob_maps, max_prob_map


def apply_noise_filter(class_map, class_names):
    """Remove connected components smaller than MIN_COMPONENT_SIZE pixels.

    Operates on each binary class mask independently, then composites the
    filtered masks back into a single integer class map.  This matches the
    connected-component post-processing applied in the SAM workflow.

    Parameters
    ----------
    class_map : ndarray (rows, cols), int16  — modified in-place
    class_names : list of str

    Returns
    -------
    noise_fracs : dict {class_name -> float}  — fraction of pixels removed
    pre_filter_counts : dict {class_name -> int}
    post_filter_counts : dict {class_name -> int}
    """
    noise_fracs        = {}
    pre_filter_counts  = {}
    post_filter_counts = {}

    print(f"\n  Connected-component noise filter (< {MIN_COMPONENT_SIZE} px):")
    print(f"  {'Class':<22} {'Pre-filter':>12} {'Removed':>10} {'Frac':>8}")
    print("  " + "-" * 56)

    for cls_idx, cls_name in enumerate(class_names):
        if EXCLUDE_BACKGROUND and cls_name == "Background":
            continue

        binary = (class_map == cls_idx)
        n_pre  = int(binary.sum())
        pre_filter_counts[cls_name] = n_pre

        if n_pre == 0:
            noise_fracs[cls_name]        = 0.0
            post_filter_counts[cls_name] = 0
            print(f"  {cls_name:<22} {n_pre:>12,} {0:>10,} {'N/A':>8}")
            continue

        labeled, n_comp = nd_label(binary)
        comp_sizes      = np.bincount(labeled.ravel())[1:]  # skip background (0)

        # Zero out noise components in the class map
        n_removed_px = 0
        for comp_id, size in enumerate(comp_sizes, start=1):
            if size < MIN_COMPONENT_SIZE:
                class_map[labeled == comp_id] = -1
                n_removed_px += int(size)

        n_post = n_pre - n_removed_px
        frac   = n_removed_px / n_pre if n_pre > 0 else 0.0
        noise_fracs[cls_name]        = frac
        post_filter_counts[cls_name] = n_post

        print(f"  {cls_name:<22} {n_pre:>12,} {n_removed_px:>10,} {frac:>7.1%}")

    return noise_fracs, pre_filter_counts, post_filter_counts


# =============================================================================
# VALIDATION
# =============================================================================


def validate_rf_results(class_map, prob_maps, max_prob_map,
                        soil_mask, cube, wavelengths, class_names,
                        rf, noise_fracs, output_dir):
    """Compute and save validation metrics for the RF classification.

    Metrics computed per class
    --------------------------
    (a) Noise fraction — fraction of pixels removed by the size filter.
        Mirrors the connected-component metric in validate_sam_results().
    (b) Moran's I — spatial clustering of each binary class mask within the
        soil domain, using Queen (8-neighbour) row-standardised weights.
        Requires esda + libpysal; skipped gracefully if absent.
    (c) MDI feature importances — mean decrease in impurity per band,
        plotted against wavelength and saved as PNG + CSV.
    (d) Max-class probability statistics — mean and std of max_proba per
        mineral class for accepted pixels; low mean → classifier uncertainty.

    Parameters
    ----------
    class_map : ndarray (rows, cols), int16
    prob_maps : ndarray (rows, cols, n_classes), float32
    max_prob_map : ndarray (rows, cols), float32
    soil_mask : ndarray (rows, cols), bool
    cube : ndarray (rows, cols, bands), float32  (unused here but kept for
           signature parity with validate_sam_results)
    wavelengths : ndarray (bands,), float
    class_names : list of str
    rf : fitted RandomForestClassifier
    noise_fracs : dict {class_name -> float}
    output_dir : Path

    Returns
    -------
    val_df : pd.DataFrame — validation metrics table
    """
    # ── optional spatial stats dependencies ──────────────────────────────
    try:
        import libpysal.weights as lps_weights
        from esda import Moran
        HAS_ESDA = True
    except ImportError:
        HAS_ESDA = False
        print("  [validation] WARNING: esda/libpysal not found — "
              "Moran's I skipped.")

    val_dir = output_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    soil_r, soil_c = np.where(soil_mask)

    print("\n" + "=" * 70)
    print("RF VALIDATION METRICS")
    print("=" * 70)

    # ── (c) MDI feature importances ──────────────────────────────────────
    importances = rf.feature_importances_            # (n_bands,)
    imp_df = pd.DataFrame({
        "band":       np.arange(len(importances)),
        "wavelength": wavelengths,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(val_dir / "rf_feature_importances.csv", index=False)

    # Feature importance figure
    n_top = min(30, len(importances))
    top_df = imp_df.head(n_top)

    fig, ax = plt.subplots(figsize=(14, 5))
    bar_colors = plt.cm.plasma(
        (top_df["importance"].values - top_df["importance"].min()) /
        (top_df["importance"].max() - top_df["importance"].min() + 1e-10)
    )
    ax.bar(range(n_top), top_df["importance"].values, color=bar_colors)
    ax.set_xticks(range(n_top))
    ax.set_xticklabels(
        [f"{w:.0f}" for w in top_df["wavelength"].values],
        rotation=55, ha="right", fontsize=8
    )
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("MDI Importance", fontsize=11)
    ax.set_title(
        f"Random Forest Feature Importances (MDI) — Top {n_top} Bands",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = val_dir / "rf_feature_importances.png"
    plt.savefig(str(fig_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Feature importance plot saved: {fig_path.name}")

    # ── Per-class metrics ─────────────────────────────────────────────────
    cw = (22, 10, 10, 10, 10, 10, 18)
    header = (
        f"  {'Class':<{cw[0]}} "
        f"{'n_px':>{cw[1]}} "
        f"{'NoiseFrac':>{cw[2]}} "
        f"{'MeanMaxP':>{cw[3]}} "
        f"{'StdMaxP':>{cw[4]}} "
        f"{'Moran_I':>{cw[5]}} "
        f"{'Significance':<{cw[6]}}"
    )
    print(f"\n{header}")
    print("  " + "-" * (sum(cw) + len(cw) - 1))

    records = []

    for cls_idx, cls_name in enumerate(class_names):
        if EXCLUDE_BACKGROUND and cls_name == "Background":
            continue

        binary = (class_map == cls_idx)
        n_px   = int(binary.sum())

        noise_frac = noise_fracs.get(cls_name, np.nan)

        # (d) Max-class probability stats for accepted pixels of this class
        if n_px > 0:
            accepted_proba = max_prob_map[binary]
            mean_mp = float(accepted_proba.mean())
            std_mp  = float(accepted_proba.std())
        else:
            mean_mp = np.nan
            std_mp  = np.nan

        # (b) Moran's I
        if HAS_ESDA and n_px >= 2:
            mi_I, mi_z, mi_p, mi_sig = _compute_morans_i(
                binary, soil_r, soil_c, lps_weights, Moran
            )
        else:
            mi_I, mi_z, mi_p = np.nan, np.nan, np.nan
            mi_sig = "skipped (esda missing or n=0)"

        nf_str = f"{noise_frac:.3f}" if not np.isnan(noise_frac) else "N/A"
        mm_str = f"{mean_mp:.3f}"    if not np.isnan(mean_mp)    else "N/A"
        ms_str = f"{std_mp:.3f}"     if not np.isnan(std_mp)     else "N/A"
        mi_str = f"{mi_I:.4f}"       if not np.isnan(mi_I)       else "N/A"

        print(
            f"  {cls_name:<{cw[0]}} "
            f"{n_px:>{cw[1]},} "
            f"{nf_str:>{cw[2]}} "
            f"{mm_str:>{cw[3]}} "
            f"{ms_str:>{cw[4]}} "
            f"{mi_str:>{cw[5]}} "
            f"{mi_sig:<{cw[6]}}"
        )

        records.append({
            "Class":          cls_name,
            "n_pixels":       n_px,
            "noise_frac":     noise_frac,
            "mean_max_prob":  mean_mp,
            "std_max_prob":   std_mp,
            "moran_I":        mi_I,
            "moran_z":        mi_z,
            "moran_p":        mi_p,
            "moran_sig":      mi_sig,
        })

    val_df = pd.DataFrame(records)
    val_csv = output_dir / "rf_validation_metrics.csv"
    val_df.to_csv(str(val_csv), index=False)
    print(f"\n  Validation metrics saved: {val_csv.name}")

    return val_df


# =============================================================================
# CROSS-METHOD COMPARISON (RF AMD lithology vs SAM mineral map)
# =============================================================================


def cross_method_comparison(class_map, class_names, soil_mask, output_dir):
    """Compare the RF AMD_FeOx lithology map against the aggregate SAM mineral map.

    The RF classifies a broad AMD ferro-oxide lithological unit; SAM classifies
    specific constituent minerals (goethite, hematite, jarosite, etc.) within
    that unit.  A direct class-to-class comparison is not meaningful because the
    two outputs operate at different levels of spectral specificity.

    Instead, all SAM mineral detections are aggregated into a single binary
    'SAM AMD' mask, and the following spatial agreement metrics are computed:

    1. **Jaccard IoU** — (RF_AMD ∩ SAM_union) / (RF_AMD ∪ SAM_union)
       Measures the spatial overlap between the two binary AMD footprints.
       Expected to be moderate (0.2–0.6): RF captures the full lithological
       unit while SAM only detects pixels with strong mineral signatures.

    2. **Cohen's Kappa** — chance-corrected binary agreement within the soil
       domain (both maps treated as binary AMD-present / not-present).
       Kappa > 0.4 suggests substantial agreement; the SAM map is used as
       reference since it provides mineral-level ground truth.

    3. **SAM recall within RF** — fraction of SAM-union pixels inside RF_AMD.
       High recall (→ 1.0) means the RF AMD zone successfully encompasses all
       SAM-detected mineral occurrences, as expected from the hypothesis that
       AMD_FeOx is the hosting lithological unit.

    4. **RF efficiency (SAM-confirmed)** — fraction of RF_AMD pixels that
       contain ≥ 1 SAM mineral detection.  Lower values are expected because
       RF captures the full mineralised zone, not only the spectral peaks.

    5. **Per-mineral SAM containment** — fraction of each SAM mineral inside
       the RF AMD zone.  All minerals should show high containment if RF
       correctly delineates the host lithology.

    A figure is saved alongside the CSVs showing the spatial overlap map and
    the per-mineral containment bar chart.

    Parameters
    ----------
    class_map   : ndarray (rows, cols), int16  — RF classification map
    class_names : list of str                  — ordered RF class names
    soil_mask   : ndarray (rows, cols), bool   — valid soil pixels
    output_dir  : Path

    Returns
    -------
    summary_df : pd.DataFrame or None
    """
    sam_hdr = Path(SAM_CLASS_MAP_FILE)
    if not sam_hdr.exists():
        print(
            f"\n  [cross-method] SAM classification map not found at:\n"
            f"    {sam_hdr}\n"
            "  Skipping cross-method comparison.  Run multi_mineral_sam_fixed.py "
            "first."
        )
        return None

    print("\n" + "=" * 70)
    print("CROSS-METHOD COMPARISON  (RF AMD Lithology vs SAM Mineral Map)")
    print("=" * 70)

    sam_img = sp.open_image(str(sam_hdr))
    sam_map = np.squeeze(sam_img.load()[:, :, 0]).astype(np.int32)

    # ENVI metadata: 'class names' = ['Unclassified', mineral_1, ...]
    sam_class_meta = sam_img.metadata.get("class names", [])
    sam_mineral_names = list(sam_class_meta[1:]) if len(sam_class_meta) > 1 else []
    print(f"\n  SAM mineral classes ({len(sam_mineral_names)}): {sam_mineral_names}")

    # ── RF AMD binary mask ────────────────────────────────────────────────
    if "AMD_FeOx" in class_names:
        amd_idx = class_names.index("AMD_FeOx")
    else:
        # Fallback: first non-Background class
        amd_idx = next(
            (i for i, n in enumerate(class_names) if n.lower() != "background"), 0
        )
        print(f"  WARNING: 'AMD_FeOx' not found in class_names — using "
              f"class index {amd_idx} ({class_names[amd_idx]}) as AMD mask.")

    rf_amd  = (class_map == amd_idx)
    sam_any = (sam_map > 0)          # union of all SAM-classified minerals

    n_rf  = int(rf_amd.sum())
    n_sam = int(sam_any.sum())
    n_intersection = int((rf_amd & sam_any).sum())
    n_union        = int((rf_amd | sam_any).sum())

    jaccard      = n_intersection / n_union        if n_union > 0 else 0.0
    sam_recall   = n_intersection / n_sam          if n_sam  > 0 else 0.0
    rf_efficiency = n_intersection / n_rf          if n_rf   > 0 else 0.0

    # ── Cohen's Kappa (soil domain only) ─────────────────────────────────
    # Restricting to soil avoids inflating kappa with the large
    # non-soil background (both maps trivially agree on 0 there).
    try:
        from sklearn.metrics import cohen_kappa_score
        soil_r, soil_c = np.where(soil_mask)
        y_rf  = rf_amd[soil_r, soil_c].astype(np.int32)
        y_sam = sam_any[soil_r, soil_c].astype(np.int32)
        kappa = float(cohen_kappa_score(y_sam, y_rf))
        kappa_str = f"{kappa:.4f}"
    except ImportError:
        kappa     = np.nan
        kappa_str = "N/A (sklearn missing)"
    except Exception as exc:
        kappa     = np.nan
        kappa_str = f"N/A ({exc})"

    # ── Print global summary ──────────────────────────────────────────────
    print(f"\n  {'Metric':<35} {'Value':>12}")
    print("  " + "-" * 49)
    print(f"  {'RF AMD_FeOx pixels':<35} {n_rf:>12,}")
    print(f"  {'SAM pixels (all minerals, union)':<35} {n_sam:>12,}")
    print(f"  {'Intersection':<35} {n_intersection:>12,}")
    print(f"  {'Union':<35} {n_union:>12,}")
    print(f"  {'Jaccard IoU':<35} {jaccard:>12.4f}")
    label_kappa = "Cohen's Kappa (soil domain)"
    print(f"  {label_kappa:<35} {kappa_str:>12}")
    print(f"  {'SAM recall within RF (%)':<35} {sam_recall * 100:>11.1f}%")
    print(f"  {'RF efficiency / SAM-confirmed (%)':<35} {rf_efficiency * 100:>11.1f}%")

    # ── Per-mineral containment ───────────────────────────────────────────
    per_mineral = []
    print(f"\n  Per-SAM-mineral containment within RF AMD_FeOx zone:")
    print(f"  {'SAM Mineral':<28} {'n_SAM':>8} {'n_in_RF':>8} {'% in RF':>8}")
    print("  " + "-" * 56)
    for i, mineral in enumerate(sam_mineral_names, start=1):
        mineral_mask = (sam_map == i)
        n_mineral    = int(mineral_mask.sum())
        n_in_rf      = int((mineral_mask & rf_amd).sum())
        pct          = n_in_rf / n_mineral * 100 if n_mineral > 0 else 0.0
        print(f"  {mineral:<28} {n_mineral:>8,} {n_in_rf:>8,} {pct:>7.1f}%")
        per_mineral.append({
            "SAM_mineral":   mineral,
            "n_SAM":         n_mineral,
            "n_in_RF_AMD":   n_in_rf,
            "pct_in_RF_AMD": pct,
        })

    # ── Save CSVs ─────────────────────────────────────────────────────────
    summary_records = [
        {"metric": "RF_AMD_pixels",        "value": n_rf},
        {"metric": "SAM_union_pixels",     "value": n_sam},
        {"metric": "Intersection",         "value": n_intersection},
        {"metric": "Union",                "value": n_union},
        {"metric": "Jaccard_IoU",          "value": jaccard},
        {"metric": "Cohens_Kappa",         "value": kappa},
        {"metric": "SAM_recall_in_RF",     "value": sam_recall},
        {"metric": "RF_efficiency_SAMconf","value": rf_efficiency},
    ]
    summary_df  = pd.DataFrame(summary_records)
    summary_csv = output_dir / "rf_cross_method_comparison.csv"
    summary_df.to_csv(str(summary_csv), index=False)

    per_min_df  = pd.DataFrame(per_mineral)
    per_min_csv = output_dir / "rf_per_mineral_containment.csv"
    per_min_df.to_csv(str(per_min_csv), index=False)

    print(f"\n  Summary metrics saved : {summary_csv.name}")
    print(f"  Per-mineral CSV saved : {per_min_csv.name}")

    # ── Comparison figure ─────────────────────────────────────────────────
    _save_comparison_figure(
        rf_amd, sam_any, per_mineral,
        jaccard, kappa, sam_recall, rf_efficiency,
        output_dir,
    )

    return summary_df


def _save_comparison_figure(rf_amd, sam_any, per_mineral,
                            jaccard, kappa, sam_recall, rf_efficiency,
                            output_dir):
    """Save a two-panel spatial agreement figure for the cross-method comparison.

    Left panel  — spatial overlap map coded by agreement category.
    Right panel — per-SAM-mineral containment bar chart + global metrics box.
    """
    from matplotlib.patches import Patch

    rows, cols = rf_amd.shape

    # ── Spatial overlap categories ────────────────────────────────────────
    rf_only  = rf_amd & ~sam_any
    sam_only = sam_any & ~rf_amd
    both     = rf_amd & sam_any

    # Build RGB image
    rgb_ov = np.ones((rows, cols, 3), dtype=np.float32) * 0.88  # neutral grey
    rgb_ov[rf_only]  = [0.25, 0.55, 0.85]   # blue  — RF AMD only
    rgb_ov[sam_only] = [0.90, 0.55, 0.10]   # orange — SAM only
    rgb_ov[both]     = [0.20, 0.72, 0.28]   # green  — agreement

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "RF AMD Ferro-oxide Lithology vs. SAM Mineral Map — Spatial Agreement",
        fontsize=13, fontweight="bold",
    )

    # ── Left: overlap map ─────────────────────────────────────────────────
    ax_map = axes[0]
    ax_map.imshow(rgb_ov, interpolation="nearest")
    ax_map.set_title("Spatial Overlap Map", fontsize=11, fontweight="bold")
    ax_map.axis("off")
    legend_elements = [
        Patch(fc=[0.25, 0.55, 0.85], ec="0.3", lw=0.5,
              label=f"RF AMD_FeOx only  ({int(rf_only.sum()):,} px)"),
        Patch(fc=[0.90, 0.55, 0.10], ec="0.3", lw=0.5,
              label=f"SAM minerals only  ({int(sam_only.sum()):,} px)"),
        Patch(fc=[0.20, 0.72, 0.28], ec="0.3", lw=0.5,
              label=f"Both agree  ({int(both.sum()):,} px)"),
        Patch(fc=[0.88, 0.88, 0.88], ec="0.4", lw=0.5,
              label="Neither / non-soil"),
    ]
    ax_map.legend(
        handles=legend_elements, loc="lower right", fontsize=8.5,
        frameon=True, fancybox=False, edgecolor="0.4",
    )

    # ── Right: per-mineral containment bar chart ──────────────────────────
    ax_bar = axes[1]
    minerals = [d["SAM_mineral"] for d in per_mineral]
    pcts     = [d["pct_in_RF_AMD"] for d in per_mineral]

    if pcts:
        bar_colors = plt.cm.RdYlGn(np.array(pcts) / 100.0)
        bars = ax_bar.barh(
            minerals, pcts,
            color=bar_colors, edgecolor="0.4", linewidth=0.6,
        )
        ax_bar.axvline(
            100, color="0.4", linestyle="--", linewidth=0.9, alpha=0.7,
            label="100 % containment",
        )
        ax_bar.set_xlim(0, 118)
        for bar, pct in zip(bars, pcts):
            ax_bar.text(
                min(pct + 1.2, 114),
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center", fontsize=8.5,
            )
    else:
        ax_bar.text(0.5, 0.5, "No SAM minerals found",
                    ha="center", va="center", transform=ax_bar.transAxes)

    ax_bar.set_xlabel("% of SAM mineral pixels inside RF AMD_FeOx zone", fontsize=9)
    ax_bar.set_title(
        "Per-mineral SAM containment in RF AMD zone\n"
        "(high % = RF AMD zone encloses the SAM mineral detections)",
        fontsize=10, fontweight="bold",
    )
    ax_bar.grid(True, axis="x", alpha=0.3)

    # Global metrics text box
    kappa_txt = f"{kappa:.3f}" if not (isinstance(kappa, float) and np.isnan(kappa)) else "N/A"
    metrics_txt = (
        f"Jaccard IoU       : {jaccard:.3f}\n"
        f"Cohen's Kappa     : {kappa_txt}\n"
        f"SAM recall in RF  : {sam_recall * 100:.1f}%\n"
        f"RF efficiency     : {rf_efficiency * 100:.1f}%"
    )
    ax_bar.text(
        0.98, 0.04, metrics_txt,
        transform=ax_bar.transAxes,
        fontsize=8.5, va="bottom", ha="right", family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="lightyellow", alpha=0.88, edgecolor="0.5",
        ),
    )

    plt.tight_layout()
    fig_path = output_dir / "rf_cross_method_comparison.png"
    plt.savefig(str(fig_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Comparison figure saved  : {fig_path.name}")


# =============================================================================
# SUPERVISED SAM CLASSIFICATION  (same training pixels as RF)
# =============================================================================


def compute_class_endmembers(X, y, class_names):
    """Compute the mean spectral endmember for each class from training data.

    The mean spectrum is computed over all valid training pixels for that class
    and then L2-normalised so that the SAM dot-product formula reduces to a
    simple pixel-norm division:  angle = arccos(dot(pixel, endmember) / ||pixel||).

    Parameters
    ----------
    X : ndarray (n_samples, n_bands), float64 — NaN-imputed training spectra
    y : ndarray (n_samples,), int             — class labels (0-indexed)
    class_names : list of str

    Returns
    -------
    endmembers : dict {class_name -> ndarray (n_bands,), float64}
        L2-normalised mean spectrum per class.
    """
    endmembers = {}
    print("\n  Class endmembers (mean training spectra, L2-normalised):")
    print(f"  {'Class':<22} {'n_train':>8} {'||mean||':>10}")
    print("  " + "-" * 44)

    for cls_idx, cls_name in enumerate(class_names):
        mask = (y == cls_idx)
        n = int(mask.sum())
        if n == 0:
            print(f"  {cls_name:<22} {0:>8} {'—':>10}")
            continue
        mean_spec = X[mask].mean(axis=0)
        norm = float(np.linalg.norm(mean_spec))
        endmembers[cls_name] = mean_spec / norm if norm > 1e-12 else mean_spec
        print(f"  {cls_name:<22} {n:>8,} {norm:>10.4f}")

    return endmembers


def apply_supervised_sam(cube, soil_mask, endmembers, class_names,
                         angle_threshold=None):
    """Classify all soil pixels using the SAM nearest-endmember rule.

    For each soil pixel the SAM angle to every class endmember is computed.
    The pixel is assigned to the class with the minimum angle.  If
    *angle_threshold* is set, pixels whose minimum angle exceeds the threshold
    are labelled Unclassified (code = -1).

    Parameters
    ----------
    cube : ndarray (rows, cols, bands), float32
    soil_mask : ndarray (rows, cols), bool
    endmembers : dict {class_name -> ndarray (n_bands,), float64}
        L2-normalised spectra from compute_class_endmembers().
    class_names : list of str
    angle_threshold : float or None
        Maximum SAM angle (radians) to accept a classification.

    Returns
    -------
    class_map : ndarray (rows, cols), int16
        Class codes: -1 = non-soil or unclassified; 0..n-1 = class index.
    angle_maps : dict {class_name -> ndarray (rows, cols), float32}
        SAM angle to each endmember (soil pixels only; others = π).
    min_angle_map : ndarray (rows, cols), float32
        Minimum SAM angle across all classes (soil pixels only; others = π).
    """
    rows, cols, bands = cube.shape
    n_classes = len(class_names)

    soil_r, soil_c = np.where(soil_mask)
    n_soil = len(soil_r)
    print(f"\n  Classifying {n_soil:,} soil pixels with SAM...")

    soil_spectra = cube[soil_r, soil_c, :].astype(np.float64)
    _impute_nan_bands(soil_spectra)

    # Pixel L2-norms (precomputed once; endmembers are already unit-normalised)
    pixel_norms = np.linalg.norm(soil_spectra, axis=1)
    pixel_norms = np.where(pixel_norms < 1e-12, 1e-12, pixel_norms)

    angle_all = np.full((n_soil, n_classes), np.pi, dtype=np.float64)

    for cls_idx, cls_name in enumerate(class_names):
        if cls_name not in endmembers:
            continue
        ref = endmembers[cls_name]          # L2-normalised → ||ref|| = 1
        dots = soil_spectra @ ref           # (n_soil,)
        cos_a = np.clip(dots / pixel_norms, -1.0, 1.0)
        angle_all[:, cls_idx] = np.arccos(cos_a)

    min_angle = angle_all.min(axis=1)
    pred_class = angle_all.argmin(axis=1).astype(np.int16)

    n_rejected = 0
    if angle_threshold is not None:
        reject_mask = min_angle > angle_threshold
        pred_class[reject_mask] = -1
        n_rejected = int(reject_mask.sum())

    n_accepted = n_soil - n_rejected
    print(f"  Accepted classifications : {n_accepted:,} "
          f"({n_accepted / n_soil * 100:.1f}%)")
    if angle_threshold is not None:
        thr_deg = np.degrees(angle_threshold)
        print(f"  Rejected (angle > {thr_deg:.1f}°)  : {n_rejected:,} "
              f"({n_rejected / n_soil * 100:.1f}%)")

    # ── Reconstruct full-scene arrays ─────────────────────────────────────
    class_map = np.full((rows, cols), -1, dtype=np.int16)
    class_map[soil_r, soil_c] = pred_class

    angle_maps = {}
    for cls_idx, cls_name in enumerate(class_names):
        amap = np.full((rows, cols), np.pi, dtype=np.float32)
        amap[soil_r, soil_c] = angle_all[:, cls_idx].astype(np.float32)
        angle_maps[cls_name] = amap

    min_angle_map = np.full((rows, cols), np.pi, dtype=np.float32)
    min_angle_map[soil_r, soil_c] = min_angle.astype(np.float32)

    # Optionally relabel Background predictions as Unclassified
    if EXCLUDE_BACKGROUND and "Background" in class_names:
        bg_idx = class_names.index("Background")
        class_map[class_map == bg_idx] = -1
        print(f"  Background class (idx {bg_idx}) relabelled to Unclassified.")

    return class_map, angle_maps, min_angle_map


def validate_sam_supervised_results(class_map, angle_maps, min_angle_map,
                                    soil_mask, class_names, noise_fracs,
                                    wavelengths, endmembers, output_dir):
    """Compute and save validation metrics for the supervised SAM classification.

    Metrics computed per class
    --------------------------
    (a) Noise fraction — fraction of pixels removed by the connected-component
        size filter (same method as RF validation).
    (b) Moran's I — spatial clustering of the binary class mask within the soil
        domain (Queen contiguity, row-standardised weights).
    (c) Min-angle statistics — mean and std of the minimum SAM angle for
        pixels classified as this class (lower = more spectrally distinct).

    Parameters
    ----------
    class_map     : ndarray (rows, cols), int16
    angle_maps    : dict {class_name -> ndarray (rows, cols), float32}
    min_angle_map : ndarray (rows, cols), float32
    soil_mask     : ndarray (rows, cols), bool
    class_names   : list of str
    noise_fracs   : dict {class_name -> float}
    wavelengths   : ndarray (bands,), float
    endmembers    : dict {class_name -> ndarray (n_bands,), float64}
    output_dir    : Path

    Returns
    -------
    val_df : pd.DataFrame
    """
    try:
        import libpysal.weights as lps_weights
        from esda import Moran
        HAS_ESDA = True
    except ImportError:
        HAS_ESDA = False
        print("  [validation] WARNING: esda/libpysal not found — "
              "Moran's I skipped.")

    val_dir = output_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    soil_r, soil_c = np.where(soil_mask)

    print("\n" + "=" * 70)
    print("SUPERVISED SAM VALIDATION METRICS")
    print("=" * 70)

    # ── Endmember spectra figure ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    for cls_name, spec in endmembers.items():
        color = CLASS_COLORS.get(cls_name, "#888888")
        ax.plot(wavelengths, spec, color=color, linewidth=1.3, label=cls_name)
    ax.set_xlabel("Wavelength (nm)", fontsize=10)
    ax.set_ylabel("Normalised Reflectance", fontsize=10)
    ax.set_title(
        "SAM Endmember Spectra — mean training spectrum per class (L2-normalised)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(val_dir / "sam_endmember_spectra.png"), dpi=200,
                bbox_inches="tight")
    plt.close()
    print(f"\n  Endmember spectra plot saved: sam_endmember_spectra.png")

    # ── Per-class metrics table ───────────────────────────────────────────
    cw = (22, 10, 10, 13, 12, 10, 18)
    header = (
        f"  {'Class':<{cw[0]}} "
        f"{'n_px':>{cw[1]}} "
        f"{'NoiseFrac':>{cw[2]}} "
        f"{'MeanAngle°':>{cw[3]}} "
        f"{'StdAngle°':>{cw[4]}} "
        f"{'Moran_I':>{cw[5]}} "
        f"{'Significance':<{cw[6]}}"
    )
    print(f"\n{header}")
    print("  " + "-" * (sum(cw) + len(cw) - 1))

    records = []
    for cls_idx, cls_name in enumerate(class_names):
        if EXCLUDE_BACKGROUND and cls_name == "Background":
            continue

        binary = (class_map == cls_idx)
        n_px   = int(binary.sum())
        noise_frac = noise_fracs.get(cls_name, np.nan)

        # (c) Min-angle statistics for pixels classified as this class
        if n_px > 0:
            cls_min_angles = min_angle_map[binary]
            mean_ang = float(np.degrees(cls_min_angles.mean()))
            std_ang  = float(np.degrees(cls_min_angles.std()))
        else:
            mean_ang = np.nan
            std_ang  = np.nan

        # (b) Moran's I
        if HAS_ESDA and n_px >= 2:
            mi_I, mi_z, mi_p, mi_sig = _compute_morans_i(
                binary, soil_r, soil_c, lps_weights, Moran
            )
        else:
            mi_I, mi_z, mi_p = np.nan, np.nan, np.nan
            mi_sig = "skipped (esda missing or n=0)"

        nf_str = f"{noise_frac:.3f}" if not np.isnan(noise_frac) else "N/A"
        ma_str = f"{mean_ang:.2f}°"  if not np.isnan(mean_ang)   else "N/A"
        sa_str = f"{std_ang:.2f}°"   if not np.isnan(std_ang)    else "N/A"
        mi_str = f"{mi_I:.4f}"       if not np.isnan(mi_I)       else "N/A"

        print(
            f"  {cls_name:<{cw[0]}} "
            f"{n_px:>{cw[1]},} "
            f"{nf_str:>{cw[2]}} "
            f"{ma_str:>{cw[3]}} "
            f"{sa_str:>{cw[4]}} "
            f"{mi_str:>{cw[5]}} "
            f"{mi_sig:<{cw[6]}}"
        )

        records.append({
            "Class":              cls_name,
            "n_pixels":           n_px,
            "noise_frac":         noise_frac,
            "mean_min_angle_deg": mean_ang,
            "std_min_angle_deg":  std_ang,
            "moran_I":            mi_I,
            "moran_z":            mi_z,
            "moran_p":            mi_p,
            "moran_sig":          mi_sig,
        })

    val_df = pd.DataFrame(records)
    val_csv = output_dir / "sam_validation_metrics.csv"
    val_df.to_csv(str(val_csv), index=False)
    print(f"\n  Validation metrics saved: {val_csv.name}")

    return val_df


def save_sam_supervised_outputs(class_map, angle_maps, class_names, output_dir):
    """Save all supervised SAM classification outputs.

    Files written
    -------------
    sam_classification_map.tif        GeoTIFF (class codes, int16)
    sam_classification_map.hdr/.img   ENVI classification image
    sam_angle_maps.hdr/.img           ENVI per-class SAM angle maps (float32)
    sam_classification_map.png        Colour visualisation with legend
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = class_map.shape

    # ── GeoTIFF ──────────────────────────────────────────────────────────
    _save_geotiff(class_map, output_dir / "sam_classification_map.tif")

    # ── ENVI classification map ───────────────────────────────────────────
    display_map = (class_map + 1).astype(np.uint16)
    envi_meta = {
        "lines":       rows,
        "samples":     cols,
        "bands":       1,
        "data type":   12,
        "interleave":  "bsq",
        "byte order":  0,
        "class names": ["Unclassified"] + class_names,
        "classes":     len(class_names) + 1,
        "description": "Supervised SAM classification map",
    }
    envi_hdr = output_dir / "sam_classification_map.hdr"
    sp.envi.save_image(str(envi_hdr), display_map, metadata=envi_meta, force=True)
    print(f"  ENVI classification map saved: sam_classification_map.hdr/.img")

    # ── ENVI per-class angle maps ─────────────────────────────────────────
    band_names = [cls for cls in class_names if cls in angle_maps]
    angle_cube = np.stack(
        [angle_maps[cls] for cls in band_names], axis=2
    ).astype(np.float32)
    angle_meta = {
        "lines":       rows,
        "samples":     cols,
        "bands":       angle_cube.shape[2],
        "data type":   4,
        "interleave":  "bip",
        "byte order":  0,
        "band names":  band_names,
        "description": "Supervised SAM per-class angle maps (radians)",
    }
    angle_hdr = output_dir / "sam_angle_maps.hdr"
    sp.envi.save_image(str(angle_hdr), angle_cube, metadata=angle_meta, force=True)
    print(f"  ENVI angle maps saved: sam_angle_maps.hdr/.img")

    # ── Colour visualisation ─────────────────────────────────────────────
    _save_sam_supervised_figure(class_map, class_names, output_dir)


def _save_sam_supervised_figure(class_map, class_names, output_dir):
    """Save a colour visualisation of the supervised SAM classification map."""
    from matplotlib.patches import Patch

    rows, cols = class_map.shape
    rgba = np.ones((rows, cols, 4), dtype=np.float32)
    rgba[class_map == -1] = [0.78, 0.78, 0.78, 1.0]

    for cls_idx, cls_name in enumerate(class_names):
        if EXCLUDE_BACKGROUND and cls_name == "Background":
            continue
        c = CLASS_COLORS.get(cls_name, "#888888")
        rgba[class_map == cls_idx] = _hex_to_rgba(c, alpha=1.0)

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(rgba, interpolation="nearest")
    thr_str = (f"{np.degrees(SAM_ANGLE_THRESHOLD):.1f}°"
               if SAM_ANGLE_THRESHOLD is not None else "none")
    ax.set_title(
        "Supervised SAM — AMD Ferro-oxide Lithology\n"
        f"(angle threshold = {thr_str}, "
        f"min component = {MIN_COMPONENT_SIZE} px)",
        fontsize=11, fontweight="bold",
    )
    ax.axis("off")

    legend_elements = [
        Patch(fc=(0.78, 0.78, 0.78), ec="0.5", lw=0.4,
              label="Unclassified / non-soil"),
    ]
    for cls_name in class_names:
        if EXCLUDE_BACKGROUND and cls_name == "Background":
            continue
        c = CLASS_COLORS.get(cls_name, "#888888")
        legend_elements.append(
            Patch(fc=c, ec="0.3", lw=0.4, label=cls_name.replace("_", " "))
        )
    ax.legend(
        handles=legend_elements, loc="lower right", fontsize=8,
        frameon=True, fancybox=False, edgecolor="0.4",
        handlelength=1.2, handleheight=0.9,
    )

    plt.tight_layout()
    fig_path = output_dir / "sam_classification_map.png"
    plt.savefig(str(fig_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Classification figure saved: {fig_path.name}")


def rf_vs_sam_supervised_comparison(rf_class_map, sam_class_map,
                                    class_names, soil_mask, output_dir):
    """Compare the RF and supervised SAM binary AMD_FeOx classifications.

    Both classifiers are trained on identical training pixels and applied to
    the same soil domain, but use fundamentally different decision rules:
    RF uses an ensemble of decision trees; supervised SAM uses the minimum
    spectral angle to a class mean endmember.  Agreement between the two
    methods is an independent consistency check: high agreement indicates
    that the spectral separability of the training data is robust across
    both distance-based and ensemble classification approaches.

    Metrics computed
    ----------------
    1. Jaccard IoU — (RF_AMD ∩ SAM_AMD) / (RF_AMD ∪ SAM_AMD)
    2. Cohen's Kappa — chance-corrected binary agreement in the soil domain
    3. Spatial agreement map — per-pixel agreement category saved as PNG

    Parameters
    ----------
    rf_class_map  : ndarray (rows, cols), int16  — RF output
    sam_class_map : ndarray (rows, cols), int16  — supervised SAM output
    class_names   : list of str
    soil_mask     : ndarray (rows, cols), bool
    output_dir    : Path  (RF output dir; figure saved here for cross-reference)
    """
    print("\n" + "=" * 70)
    print("RF vs SUPERVISED SAM COMPARISON")
    print("=" * 70)

    amd_idx = (class_names.index("AMD_FeOx")
               if "AMD_FeOx" in class_names else 0)
    rf_amd  = (rf_class_map  == amd_idx)
    sam_amd = (sam_class_map == amd_idx)

    n_rf  = int(rf_amd.sum())
    n_sam = int(sam_amd.sum())
    n_int = int((rf_amd & sam_amd).sum())
    n_uni = int((rf_amd | sam_amd).sum())
    rf_only  = n_rf  - n_int
    sam_only = n_sam - n_int

    jaccard = n_int / n_uni if n_uni > 0 else 0.0

    try:
        from sklearn.metrics import cohen_kappa_score
        soil_r, soil_c = np.where(soil_mask)
        y_rf  = rf_amd[soil_r,  soil_c].astype(np.int32)
        y_sam = sam_amd[soil_r, soil_c].astype(np.int32)
        kappa = float(cohen_kappa_score(y_rf, y_sam))
        kappa_str = f"{kappa:.4f}"
    except Exception as exc:
        kappa     = np.nan
        kappa_str = f"N/A ({exc})"

    print(f"\n  {'Metric':<38} {'Value':>12}")
    print("  " + "-" * 52)
    print(f"  {'RF AMD_FeOx pixels':<38} {n_rf:>12,}")
    print(f"  {'Supervised SAM AMD_FeOx pixels':<38} {n_sam:>12,}")
    print(f"  {'Both agree (intersection)':<38} {n_int:>12,}")
    print(f"  {'Either (union)':<38} {n_uni:>12,}")
    print(f"  {'RF only':<38} {rf_only:>12,}")
    print(f"  {'SAM supervised only':<38} {sam_only:>12,}")
    print(f"  {'Jaccard IoU':<38} {jaccard:>12.4f}")
    print(f"  {'Cohen Kappa (soil domain)':<38} {kappa_str:>12}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    summary_df = pd.DataFrame([
        {"metric": "RF_AMD_pixels",       "value": n_rf},
        {"metric": "SAM_sup_AMD_pixels",  "value": n_sam},
        {"metric": "Intersection",        "value": n_int},
        {"metric": "Union",               "value": n_uni},
        {"metric": "RF_only",             "value": rf_only},
        {"metric": "SAM_sup_only",        "value": sam_only},
        {"metric": "Jaccard_IoU",         "value": jaccard},
        {"metric": "Cohens_Kappa",        "value": kappa},
    ])
    csv_path = output_dir / "rf_vs_sam_supervised_comparison.csv"
    summary_df.to_csv(str(csv_path), index=False)
    print(f"\n  Comparison CSV saved: {csv_path.name}")

    _save_rf_vs_sam_figure(rf_amd, sam_amd, jaccard, kappa, output_dir)

    return summary_df


def _save_rf_vs_sam_figure(rf_amd, sam_amd, jaccard, kappa, output_dir):
    """Save a spatial agreement map for the RF vs supervised SAM comparison."""
    from matplotlib.patches import Patch

    rows, cols = rf_amd.shape
    rf_only  = rf_amd  & ~sam_amd
    sam_only = sam_amd & ~rf_amd
    both     = rf_amd  & sam_amd

    rgb_ov = np.ones((rows, cols, 3), dtype=np.float32) * 0.88
    rgb_ov[rf_only]  = [0.25, 0.55, 0.85]   # blue  — RF only
    rgb_ov[sam_only] = [0.85, 0.35, 0.20]   # red   — SAM supervised only
    rgb_ov[both]     = [0.20, 0.72, 0.28]   # green — both agree

    fig, ax = plt.subplots(figsize=(9, 12))
    fig.suptitle(
        "RF vs Supervised SAM — AMD_FeOx Spatial Agreement",
        fontsize=13, fontweight="bold",
    )
    ax.imshow(rgb_ov, interpolation="nearest")
    ax.axis("off")

    legend_elements = [
        Patch(fc=[0.25, 0.55, 0.85], ec="0.3", lw=0.5,
              label=f"RF only  ({int(rf_only.sum()):,} px)"),
        Patch(fc=[0.85, 0.35, 0.20], ec="0.3", lw=0.5,
              label=f"SAM supervised only  ({int(sam_only.sum()):,} px)"),
        Patch(fc=[0.20, 0.72, 0.28], ec="0.3", lw=0.5,
              label=f"Both agree  ({int(both.sum()):,} px)"),
        Patch(fc=[0.88, 0.88, 0.88], ec="0.4", lw=0.5,
              label="Neither / non-soil"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right", fontsize=9,
        frameon=True, fancybox=False, edgecolor="0.4",
    )

    kappa_txt = (f"{kappa:.3f}"
                 if not (isinstance(kappa, float) and np.isnan(kappa))
                 else "N/A")
    metrics_txt = (
        f"Jaccard IoU    : {jaccard:.3f}\n"
        f"Cohen's Kappa  : {kappa_txt}"
    )
    ax.text(
        0.98, 0.04, metrics_txt,
        transform=ax.transAxes,
        fontsize=9, va="bottom", ha="right", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="lightyellow", alpha=0.88, edgecolor="0.5"),
    )

    plt.tight_layout()
    fig_path = output_dir / "rf_vs_sam_supervised_comparison.png"
    plt.savefig(str(fig_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Comparison figure saved: {fig_path.name}")


def run_sam_supervised_classification(cube, soil_mask, X, y, class_names,
                                      wavelengths, output_dir):
    """Run the supervised SAM classification pipeline.

    Uses the same training pixel data already loaded for the RF classifier:
    the mean spectrum per class is the SAM endmember, and nearest-endmember
    assignment is applied to all soil pixels.

    Parameters
    ----------
    cube        : ndarray (rows, cols, bands), float32
    soil_mask   : ndarray (rows, cols), bool
    X           : ndarray (n_samples, bands), float64  — NaN-imputed spectra
    y           : ndarray (n_samples,), int            — 0-indexed class labels
    class_names : list of str
    wavelengths : ndarray (bands,), float
    output_dir  : Path

    Returns
    -------
    class_map : ndarray (rows, cols), int16
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SUPERVISED CLASSIFICATION — SAM (mean endmember per training class)")
    print("=" * 70)

    # ── 1. Compute class endmembers ───────────────────────────────────────
    print("\n1. Computing class endmembers from training pixels...")
    endmembers = compute_class_endmembers(X, y, class_names)

    # ── 2. Apply SAM nearest-endmember classification ─────────────────────
    thr_str = (f"{np.degrees(SAM_ANGLE_THRESHOLD):.1f}°"
               if SAM_ANGLE_THRESHOLD is not None else "none")
    print(f"\n2. Applying SAM classification (angle threshold = {thr_str})...")
    class_map, angle_maps, min_angle_map = apply_supervised_sam(
        cube, soil_mask, endmembers, class_names,
        angle_threshold=SAM_ANGLE_THRESHOLD,
    )

    # ── 3. Connected-component noise filter ───────────────────────────────
    print(f"\n3. Applying connected-component noise filter "
          f"(< {MIN_COMPONENT_SIZE} px)...")
    noise_fracs, pre_counts, post_counts = apply_noise_filter(
        class_map, class_names
    )

    print(f"\n   Post-filter class pixel counts:")
    for cls_name in class_names:
        if EXCLUDE_BACKGROUND and cls_name == "Background":
            continue
        pre  = pre_counts.get(cls_name, 0)
        post = post_counts.get(cls_name, 0)
        print(f"   {cls_name:<22} {post:>10,}  (was {pre:,})")

    # ── 4. Validation ─────────────────────────────────────────────────────
    print(f"\n4. Running validation metrics...")
    validate_sam_supervised_results(
        class_map, angle_maps, min_angle_map,
        soil_mask, class_names, noise_fracs,
        wavelengths, endmembers, output_dir,
    )

    # ── 5. Save outputs ───────────────────────────────────────────────────
    print(f"\n5. Saving classification outputs to:\n   {output_dir}")
    save_sam_supervised_outputs(class_map, angle_maps, class_names, output_dir)

    print("\n" + "=" * 70)
    print("SUPERVISED SAM CLASSIFICATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir.resolve()}")

    n_soil_total = int(soil_mask.sum())
    print("\nClass pixel counts (after noise filter):")
    print(f"  {'Class':<22} {'Pixels':>10} {'% of soil':>12}")
    print("  " + "-" * 48)
    for cls_idx, cls_name in enumerate(class_names):
        if EXCLUDE_BACKGROUND and cls_name == "Background":
            continue
        n = int((class_map == cls_idx).sum())
        pct = n / n_soil_total * 100 if n_soil_total > 0 else 0.0
        print(f"  {cls_name:<22} {n:>10,} {pct:>11.2f}%")
    n_unclass = max(int((class_map == -1).sum() - (~soil_mask).sum()), 0)
    print(f"  {'Unclassified (soil)':<22} {n_unclass:>10,}")
    print(f"  {'Total soil pixels':<22} {n_soil_total:>10,}")

    return class_map


# =============================================================================
# OUTPUT SAVING
# =============================================================================


def save_rf_outputs(class_map, prob_maps, max_prob_map,
                    class_names, wavelengths, output_dir):
    """Save all RF classification outputs.

    Files written
    -------------
    rf_classification_map.tif          GeoTIFF (class codes, int16)
    rf_classification_map.hdr/.img     ENVI classification image (int16)
    rf_probability_maps.hdr/.img       ENVI multi-band probability image (float32)
    rf_classification_map.png          Colour visualization with legend
    rf_cv_scores.csv                   (written separately by the caller)

    Parameters
    ----------
    class_map : ndarray (rows, cols), int16
        Class codes: -1 = unclassified; 0..n-1 = class index.
    prob_maps : ndarray (rows, cols, n_classes), float32
    max_prob_map : ndarray (rows, cols), float32  (unused here; available for
                  callers that want to save it separately)
    class_names : list of str
    wavelengths : ndarray (bands,), float  (unused; for API parity)
    output_dir : Path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = class_map.shape

    # ── GeoTIFF ──────────────────────────────────────────────────────────
    _save_geotiff(class_map, output_dir / "rf_classification_map.tif")

    # ── ENVI classification map ───────────────────────────────────────────
    # Class 0 = Unclassified (stored as 0 in uint16; -1 maps to 0)
    display_map = (class_map + 1).astype(np.uint16)   # -1→0, 0→1, ...
    envi_meta = {
        "lines":       rows,
        "samples":     cols,
        "bands":       1,
        "data type":   12,     # uint16
        "interleave":  "bsq",
        "byte order":  0,
        "class names": ["Unclassified"] + class_names,
        "classes":     len(class_names) + 1,
        "description": "Random Forest classification map",
    }
    envi_hdr = output_dir / "rf_classification_map.hdr"
    sp.envi.save_image(str(envi_hdr), display_map, metadata=envi_meta, force=True)
    print(f"  ENVI classification map saved: rf_classification_map.hdr/.img")

    # ── ENVI probability maps (n_classes bands) ──────────────────────────
    prob_meta = {
        "lines":       rows,
        "samples":     cols,
        "bands":       len(class_names),
        "data type":   4,      # float32
        "interleave":  "bip",
        "byte order":  0,
        "band names":  class_names,
        "description": "Random Forest per-class posterior probabilities",
    }
    prob_hdr = output_dir / "rf_probability_maps.hdr"
    sp.envi.save_image(str(prob_hdr), prob_maps, metadata=prob_meta, force=True)
    print(f"  ENVI probability maps saved: rf_probability_maps.hdr/.img")

    # ── Colour visualisation ─────────────────────────────────────────────
    _save_classification_figure(class_map, class_names, output_dir)


def _save_classification_figure(class_map, class_names, output_dir):
    """Save a publication-quality visualisation of the RF classification map.

    Mirrors the style of create_composite_map() in multi_mineral_sam_fixed.py:
    white background, light-grey non-soil domain, medium-grey unclassified
    soil pixels, and class-specific colours for mineral detections.
    """
    from matplotlib.patches import Patch

    rows, cols = class_map.shape
    rgba = np.ones((rows, cols, 4), dtype=np.float32)  # white background

    # Light grey for unclassified soil (-1 code mapped to white/grey)
    # Medium grey: unclassified = -1
    rgba[class_map == -1] = [0.78, 0.78, 0.78, 1.0]

    # Class colours
    for cls_idx, cls_name in enumerate(class_names):
        if EXCLUDE_BACKGROUND and cls_name == "Background":
            continue
        c = CLASS_COLORS.get(cls_name, "#888888")
        rgba[class_map == cls_idx] = _hex_to_rgba(c, alpha=1.0)

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(rgba, interpolation="nearest")
    ax.set_title(
        "Random Forest — AMD Ferro-oxide Lithology\n"
        f"(prob threshold = {PROB_THRESHOLD}, "
        f"min component = {MIN_COMPONENT_SIZE} px)",
        fontsize=11, fontweight="bold"
    )
    ax.axis("off")

    # Legend
    legend_elements = [
        Patch(fc=(0.78, 0.78, 0.78), ec="0.5", lw=0.4, label="Unclassified / non-soil"),
    ]
    for cls_name in class_names:
        if EXCLUDE_BACKGROUND and cls_name == "Background":
            continue
        c = CLASS_COLORS.get(cls_name, "#888888")
        legend_elements.append(
            Patch(fc=c, ec="0.3", lw=0.4, label=cls_name.replace("_", " "))
        )
    ax.legend(
        handles=legend_elements,
        loc="lower right", fontsize=8,
        frameon=True, fancybox=False, edgecolor="0.4",
        handlelength=1.2, handleheight=0.9,
    )

    plt.tight_layout()
    fig_path = output_dir / "rf_classification_map.png"
    plt.savefig(str(fig_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Classification figure saved: {fig_path.name}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_rf_classification(training_npz=None, hdr_file=None, output_dir=None):
    """Run the full supervised RF classification pipeline.

    Parameters
    ----------
    training_npz : str or Path, optional
        Path to the training pixel .npz file.  Defaults to TRAINING_PIXELS_FILE.
    hdr_file : str or Path, optional
        Path to the ENVI .hdr reflectance cube.  Defaults to HDR_FILE.
    output_dir : str or Path, optional
        Root output directory.  Defaults to OUTPUT_FOLDER.

    Returns
    -------
    class_map : ndarray (rows, cols), int16
    """
    training_npz = Path(training_npz or TRAINING_PIXELS_FILE)
    hdr_file     = Path(hdr_file     or HDR_FILE)
    output_dir   = Path(output_dir   or OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SUPERVISED CLASSIFICATION — RANDOM FOREST")
    print("=" * 70)

    # ── 1. Load image ─────────────────────────────────────────────────────
    print(f"\n1. Loading hyperspectral image: {hdr_file.name}")
    img  = sp.open_image(str(hdr_file))
    cube = img.load().astype(np.float32)
    wavelengths = np.array(img.metadata["wavelength"], dtype=float)
    rows, cols, bands = cube.shape
    print(f"   Shape: {rows} × {cols} × {bands} bands")
    print(f"   Wavelength range: {wavelengths.min():.1f}–{wavelengths.max():.1f} nm")

    if cube.max() > 2.0:
        cube /= REFLECTANCE_SCALE
        print(f"   Scaled by 1/{REFLECTANCE_SCALE:.0f} → reflectance [0, 1]")

    # ── 2. Load / compute soil mask ────────────────────────────────────────
    print("\n2. Loading soil mask...")
    mask_path = hdr_file.parent / "soil_mask.npy"
    if mask_path.exists():
        soil_mask = np.load(str(mask_path)).astype(bool)
        print(f"   Loaded soil_mask.npy ({soil_mask.sum():,} soil pixels)")
    else:
        # Fallback: valid-data mask (non-zero spectral norm)
        norms = np.linalg.norm(cube.reshape(-1, bands), axis=1)
        soil_mask = (norms > 0).reshape(rows, cols)
        print("   WARNING: soil_mask.npy not found — using valid-data mask.")

    # ── 3. Load training data ─────────────────────────────────────────────
    print(f"\n3. Loading training pixels: {training_npz.name}")
    if not training_npz.exists():
        raise FileNotFoundError(
            f"Training pixel file not found: {training_npz}\n"
            "Please run training_pixel_selector.py first."
        )
    X, y, class_names, label_to_idx = load_training_data(
        training_npz, cube, soil_mask, wavelengths
    )

    # ── 4. Cross-validation and training ──────────────────────────────────
    print(f"\n4. Training Random Forest ({CV_FOLDS}-fold stratified CV)...")
    rf, cv_scores = train_with_cross_validation(X, y, class_names)

    # Save CV scores
    cv_df = pd.DataFrame({
        "fold": np.arange(1, CV_FOLDS + 1),
        "balanced_accuracy": cv_scores,
    })
    cv_df.to_csv(output_dir / "rf_cv_scores.csv", index=False)
    print(f"\n   CV scores saved: rf_cv_scores.csv")

    # ── 5. Apply classifier ───────────────────────────────────────────────
    print(f"\n5. Applying classifier to all soil pixels "
          f"(prob threshold = {PROB_THRESHOLD})...")
    class_map, prob_maps, max_prob_map = apply_classifier(
        rf, cube, soil_mask, class_names
    )

    # ── 6. Connected-component noise filter ───────────────────────────────
    print(f"\n6. Applying connected-component noise filter "
          f"(< {MIN_COMPONENT_SIZE} px)...")
    noise_fracs, pre_counts, post_counts = apply_noise_filter(
        class_map, class_names
    )

    # Print post-filter pixel counts
    print(f"\n   Post-filter class pixel counts:")
    for cls_name in class_names:
        if EXCLUDE_BACKGROUND and cls_name == "Background":
            continue
        pre  = pre_counts.get(cls_name, 0)
        post = post_counts.get(cls_name, 0)
        print(f"   {cls_name:<22} {post:>10,}  (was {pre:,})")

    # ── 7. Validation ─────────────────────────────────────────────────────
    print(f"\n7. Running validation metrics...")
    val_df = validate_rf_results(
        class_map, prob_maps, max_prob_map,
        soil_mask, cube, wavelengths, class_names,
        rf, noise_fracs, output_dir
    )

    # ── 8. Cross-method comparison ────────────────────────────────────────
    print(f"\n8. Cross-method comparison (RF AMD lithology vs SAM mineral map)...")
    iou_df = cross_method_comparison(class_map, class_names, soil_mask, output_dir)

    # ── 9. Save outputs ───────────────────────────────────────────────────
    print(f"\n9. Saving classification outputs to:\n   {output_dir}")
    save_rf_outputs(
        class_map, prob_maps, max_prob_map,
        class_names, wavelengths, output_dir
    )

    # ── 10. Supervised SAM classification (same training pixels) ──────────
    print(f"\n10. Supervised SAM classification (same training pixels as RF)...")
    sam_output_dir = Path(SAM_SUPERVISED_OUTPUT_FOLDER)
    sam_class_map = run_sam_supervised_classification(
        cube, soil_mask, X, y, class_names, wavelengths, sam_output_dir
    )

    # ── 11. RF vs supervised SAM comparison ───────────────────────────────
    print(f"\n11. RF vs supervised SAM agreement comparison...")
    rf_vs_sam_supervised_comparison(
        class_map, sam_class_map, class_names, soil_mask, output_dir
    )

    print("\n" + "=" * 70)
    print("SUPERVISED CLASSIFICATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir.resolve()}")

    # Final summary
    print("\nClass pixel counts (after noise filter):")
    print(f"  {'Class':<22} {'Pixels':>10} {'% of soil':>12}")
    print("  " + "-" * 48)
    n_soil_total = int(soil_mask.sum())
    for cls_idx, cls_name in enumerate(class_names):
        if EXCLUDE_BACKGROUND and cls_name == "Background":
            continue
        n = int((class_map == cls_idx).sum())
        pct = n / n_soil_total * 100 if n_soil_total > 0 else 0.0
        print(f"  {cls_name:<22} {n:>10,} {pct:>11.2f}%")
    n_unclass = int((class_map == -1).sum() - (~soil_mask).sum())
    n_unclass = max(n_unclass, 0)
    print(f"  {'Unclassified (soil)':<22} {n_unclass:>10,}")
    print(f"  {'Total soil pixels':<22} {n_soil_total:>10,}")

    return class_map


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    """Command-line entry point for the supervised classification pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Random Forest supervised classification for EO-1 Hyperion AMD mapping."
    )
    parser.add_argument(
        "--training",
        default=TRAINING_PIXELS_FILE,
        help=f"Path to training_pixels.npz  [default: {TRAINING_PIXELS_FILE}]",
    )
    parser.add_argument(
        "--hdr",
        default=HDR_FILE,
        help=f"Path to reflectance cube .hdr  [default: {HDR_FILE}]",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FOLDER,
        help=f"Output directory  [default: {OUTPUT_FOLDER}]",
    )
    args = parser.parse_args()

    run_rf_classification(
        training_npz=args.training,
        hdr_file=args.hdr,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
