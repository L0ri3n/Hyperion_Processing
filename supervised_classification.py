"""
supervised_classification.py — Supervised SAM AMD Ferro-oxide Lithology Classifier
====================================================================================

Binary supervised classification of AMD ferro-oxide lithology using a Spectral
Angle Mapper (SAM) trained on manually selected training pixels (AMD_FeOx vs.
Background).  Image-derived endmembers (mean spectrum per training class,
L2-normalised) replace the USGS library references used in the primary
multi-mineral SAM workflow; the angular similarity decision rule is otherwise
identical.

Purpose
-------
Internal cross-validation of the primary library-driven SAM workflow.
Divergences between the two SAM outputs reflect the spectral distance between
USGS library references and actual Rio Tinto scene conditions (mixed pixels,
atmospheric residuals at 30 m resolution), not algorithmic differences — both
methods use the same SAM angular similarity logic.

Pipeline overview
-----------------
1. Load training pixels (.npz) and reflectance cube (ENVI .hdr).
2. Extract spectral feature vectors (all valid bands) and assemble a labelled
   dataset with classes [AMD_FeOx, Background].
3. Compute the mean L2-normalised spectrum per training class as the SAM
   endmember.
4. Apply nearest-endmember SAM classification to all soil-masked pixels.
5. Apply a connected-component noise filter (components < 4 pixels removed),
   matching the post-processing used in the library SAM workflow.
6. Validate results:
   (a) Noise fraction (pixels removed by size filter / pre-filter total).
   (b) Moran's I spatial clustering of the AMD_FeOx binary mask within the
       soil domain (Queen contiguity, same implementation as the library SAM).
   (c) Min-angle statistics (mean and std of minimum SAM angle per class).
7. Cross-method comparison (supervised SAM AMD zone vs. library SAM mineral map):
   - Jaccard IoU  (sup_SAM_AMD ∩ lib_SAM_union) / (sup_SAM_AMD ∪ lib_SAM_union)
   - Cohen's Kappa  (binary agreement within soil domain, chance-corrected)
   - Library SAM recall in supervised SAM zone
   - Supervised SAM efficiency (lib-SAM-confirmed fraction)
   - Per-mineral containment  fraction of each library SAM mineral inside
     the supervised SAM AMD zone
8. Save: classification map as GeoTIFF + ENVI, angle maps as ENVI,
   all validation tables as CSV, comparison figures as PNG.

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
* The library SAM classification map is expected at SAM_CLASS_MAP_FILE (defined
  below).  If absent the cross-method comparison section is skipped gracefully.
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

# Library SAM classification map (ENVI .hdr) for cross-method comparison.
# Produced by multi_mineral_sam_fixed.py → save_envi_classification().
SAM_CLASS_MAP_FILE = str(
    BASE_DIR / "amd_mapping" / "outputs" / "classifications"
    / "classification_map.hdr"
)

# All output files go here (supervised SAM results)
OUTPUT_FOLDER = str(
    BASE_DIR / "amd_mapping" / "outputs" / "supervised_sam"
)

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
SAM_ANGLE_THRESHOLD = 0.5

# Alias kept for any code that still references the old constant name.
SAM_SUPERVISED_OUTPUT_FOLDER = OUTPUT_FOLDER

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
    with row-standardisation.  Uses a permutation test (999 permutations) to
    obtain a pseudo p-value, which avoids the normality assumption that is
    inappropriate for binary data.

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
    p : float        — permutation-based pseudo p-value
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
        return np.nan, np.nan, "N/A (constant mask)"

    mi  = Moran(y, w, permutations=999)
    I, p = float(mi.I), float(mi.p_sim)

    if   p < 0.05 and I > 0:
        sig = "clustered (p<0.05)"
    elif p < 0.05 and I < 0:
        sig = "dispersed (p<0.05)"
    else:
        sig = f"random (p={p:.3f})"

    return I, p, sig


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
# POST-PROCESSING
# =============================================================================


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
# SUPERVISED SAM CLASSIFICATION  (image-derived endmembers)
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
            mi_I, mi_p, mi_sig = _compute_morans_i(
                binary, soil_r, soil_c, lps_weights, Moran
            )
        else:
            mi_I, mi_p = np.nan, np.nan
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


def _parse_envi_map_info(metadata):
    """Return a dict with UTM georef parameters from an ENVI map_info field.

    Returns None if the field is absent or cannot be parsed.
    """
    raw = metadata.get('map info', metadata.get('map_info', None))
    if raw is None:
        return None
    # spectral may return it as a list of stripped tokens or a single string
    if isinstance(raw, (list, tuple)):
        parts = [str(p).strip().strip('{}') for p in raw]
    else:
        parts = [p.strip().strip('{}') for p in str(raw).split(',')]
    if len(parts) < 9:
        return None
    try:
        return {
            'projection':  parts[0].strip(),
            'x_pixel':     float(parts[1]),   # reference pixel (1-based)
            'y_pixel':     float(parts[2]),
            'easting':     float(parts[3]),   # UTM easting of reference pixel centre
            'northing':    float(parts[4]),   # UTM northing of reference pixel centre
            'pixel_size_x': float(parts[5]),
            'pixel_size_y': float(parts[6]),
            'zone':        int(float(parts[7])),
            'hemisphere':  parts[8].strip().lower(),
        }
    except (ValueError, IndexError):
        return None


def _add_coordinate_ticks(ax, rows, cols, metadata=None, n_lon=5, n_lat=6):
    """Decorate *ax* borders with lat/lon ticks — no interior grid lines.

    Parses UTM georeferencing from the ENVI header (uses the module-level
    HDR_FILE if *metadata* is not supplied) and places formatted degree-minute
    labels on all four sides of the axes frame.

    Parameters
    ----------
    ax       : matplotlib Axes  (imshow already called; y-axis inverted)
    rows, cols : image dimensions in pixels
    metadata : ENVI header dict (spectral img.metadata).  If None the global
               HDR_FILE is opened automatically.
    n_lon, n_lat : approximate number of ticks for the lon / lat axes.
    """
    import math

    if metadata is None:
        metadata = sp.open_image(HDR_FILE).metadata

    mi = _parse_envi_map_info(metadata)
    if mi is None:
        ax.axis('off')
        return

    def _utm_to_latlon(easting, northing, zone, northern=True):
        """Pure-math UTM → WGS-84 lat/lon (degrees); no external library needed."""
        a   = 6378137.0
        f   = 1 / 298.257223563
        e2  = 2*f - f**2
        ep2 = e2 / (1 - e2)
        k0  = 0.9996
        E0  = 500000.0
        N0  = 0.0 if northern else 10_000_000.0
        lon0 = math.radians((zone - 1) * 6 - 180 + 3)

        x = easting  - E0
        y = northing - N0

        M   = y / k0
        mu  = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
        e1  = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
        ph1 = (mu
               + (3*e1/2 - 27*e1**3/32) * math.sin(2*mu)
               + (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu)
               + (151*e1**3/96) * math.sin(6*mu)
               + (1097*e1**4/512) * math.sin(8*mu))

        N1  = a / math.sqrt(1 - e2 * math.sin(ph1)**2)
        T1  = math.tan(ph1)**2
        C1  = ep2 * math.cos(ph1)**2
        R1  = a * (1 - e2) / (1 - e2 * math.sin(ph1)**2)**1.5
        D   = x / (N1 * k0)

        lat = ph1 - (N1 * math.tan(ph1) / R1) * (
            D**2/2
            - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*ep2) * D**4/24
            + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*ep2 - 3*C1**2) * D**6/720)
        lon = lon0 + (
            D
            - (1 + 2*T1 + C1) * D**3/6
            + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*ep2 + 24*T1**2) * D**5/120
        ) / math.cos(ph1)

        return math.degrees(lat), math.degrees(lon)

    northern = 'north' in mi['hemisphere']

    def _px_to_latlon(col, row):
        e = mi['easting']  + (col - (mi['x_pixel'] - 1)) * mi['pixel_size_x']
        n = mi['northing'] - (row - (mi['y_pixel'] - 1)) * mi['pixel_size_y']
        return _utm_to_latlon(e, n, mi['zone'], northern)

    # Lat/lon at image corners to determine tick range
    corners = [_px_to_latlon(c, r)
               for c, r in [(0, 0), (cols - 1, 0), (0, rows - 1), (cols - 1, rows - 1)]]
    lat_min = min(c[0] for c in corners)
    lat_max = max(c[0] for c in corners)
    lon_min = min(c[1] for c in corners)
    lon_max = max(c[1] for c in corners)

    def _nice_step(span, n):
        raw = span / n
        for step in [1/120, 1/60, 1/30, 1/12, 1/6, 0.5, 1.0, 2.0, 5.0, 10.0]:
            if step >= raw * 0.8:
                return step
        return 1.0

    lat_step = _nice_step(lat_max - lat_min, n_lat)
    lon_step = _nice_step(lon_max - lon_min, n_lon)
    lat_ticks = np.arange(math.ceil(lat_min / lat_step) * lat_step,
                          lat_max + lat_step * 0.01, lat_step)
    lon_ticks = np.arange(math.ceil(lon_min / lon_step) * lon_step,
                          lon_max + lon_step * 0.01, lon_step)

    # Convert lat/lon tick values → pixel positions using the mid-edge scan
    mid_col = (cols - 1) / 2
    mid_row = (rows - 1) / 2

    def _lat_to_row(lat_val):
        lat_top, _ = _px_to_latlon(mid_col, 0)
        lat_bot, _ = _px_to_latlon(mid_col, rows - 1)
        if lat_top == lat_bot:
            return mid_row
        return (lat_val - lat_top) / (lat_bot - lat_top) * (rows - 1)

    def _lon_to_col(lon_val):
        _, lon_left  = _px_to_latlon(0, mid_row)
        _, lon_right = _px_to_latlon(cols - 1, mid_row)
        if lon_left == lon_right:
            return mid_col
        return (lon_val - lon_left) / (lon_right - lon_left) * (cols - 1)

    row_pos = [_lat_to_row(v) for v in lat_ticks]
    col_pos = [_lon_to_col(v) for v in lon_ticks]

    valid_lat = [(r, v) for r, v in zip(row_pos, lat_ticks) if -0.5 <= r <= rows - 0.5]
    valid_lon = [(c, v) for c, v in zip(col_pos, lon_ticks) if -0.5 <= c <= cols - 0.5]

    def _fmt(deg, is_lat):
        hemi = ('N' if deg >= 0 else 'S') if is_lat else ('E' if deg >= 0 else 'W')
        d = abs(deg)
        deg_int = int(d)
        minutes = (d - deg_int) * 60
        if abs(minutes) < 0.05:
            return f"{deg_int}°{hemi}"
        return f"{deg_int}°{minutes:.0f}′{hemi}"

    if valid_lat:
        rp, lv = zip(*valid_lat)
        ax.set_yticks(list(rp))
        ax.set_yticklabels([_fmt(v, True) for v in lv], fontsize=7)

    if valid_lon:
        cp, lv = zip(*valid_lon)
        ax.set_xticks(list(cp))
        ax.set_xticklabels([_fmt(v, False) for v in lv], fontsize=7,
                           rotation=45, ha='right')

    # Ticks on all four sides, no grid
    ax.tick_params(axis='both', which='both', direction='out', length=3, width=0.6,
                   top=True, bottom=True, left=True, right=True,
                   labeltop=True, labelbottom=True, labelleft=True, labelright=True)
    ax.minorticks_off()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)


def _add_scale_north_arrow(ax, img_cols, pixel_m=30.0, scale_km=5.0, sb_y=0.04):
    """Overlay a north arrow and a scale bar onto *ax*, both INSIDE the axes.

    Parameters
    ----------
    ax       : matplotlib Axes – target axes
    img_cols : int   – image width in pixels, used to size the scale bar
    pixel_m  : float – ground-sampling distance in m/pixel (30 for Hyperion)
    scale_km : float – desired scale-bar length in km (default 5)
    sb_y     : float – axes-fraction y of the scale bar bottom edge.
                       Set just above the legend box top when a legend is present.
    """
    from matplotlib.patches import Rectangle

    _scale_xfrac = (scale_km * 1000.0 / pixel_m) / img_cols

    # North arrow — upper-right corner
    _ax, _ay0, _ay1 = 0.93, 0.87, 0.97
    ax.annotate('', xy=(_ax, _ay1), xycoords='axes fraction',
                xytext=(_ax, _ay0), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(_ax, _ay1 + 0.01, 'N', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')

    # Scale bar — lower-right corner, above the legend
    _sb_x0 = 0.97 - _scale_xfrac
    _sb_h  = 0.007
    ax.add_patch(Rectangle((_sb_x0, sb_y), _scale_xfrac, _sb_h,
                            transform=ax.transAxes,
                            color='black', zorder=5))
    ax.text(_sb_x0 + _scale_xfrac / 2, sb_y + _sb_h * 2.5,
            f'{scale_km:g} km',
            ha='center', va='bottom', transform=ax.transAxes,
            fontsize=6.5, color='black', zorder=5)


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
    _add_coordinate_ticks(ax, rows, cols)

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
        handles=legend_elements,
        loc="lower right",
        fontsize=8, ncol=2,
        frameon=True, fancybox=False, edgecolor="0.4",
        handlelength=1.2, handleheight=0.9,
    )

    _add_scale_north_arrow(ax, class_map.shape[1], sb_y=0.12)

    plt.tight_layout()
    fig_path = output_dir / "sam_classification_map.png"
    plt.savefig(str(fig_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Classification figure saved: {fig_path.name}")


def run_sam_supervised_classification(cube, soil_mask, X, y, class_names,
                                      wavelengths, output_dir):
    """Run the supervised SAM classification pipeline.

    Uses the same training pixel data already loaded for the pipeline:
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
# CROSS-METHOD COMPARISON  (Supervised SAM AMD zone vs Library SAM mineral map)
# =============================================================================


def sam_cross_method_comparison(sam_class_map, class_names, soil_mask,
                                output_dir):
    """Compare the supervised SAM AMD_FeOx zone against the library SAM mineral map.

    Both methods use the identical SAM angular similarity logic; they differ
    only in endmember source (image-derived training means vs. USGS library
    spectra).  Divergences therefore reflect the spectral distance between USGS
    library references and actual Rio Tinto scene conditions — mixed pixels,
    atmospheric residuals, and variable mineral assemblages at 30 m resolution
    — rather than any algorithmic difference.

    Metrics computed
    ----------------
    1. Jaccard IoU   (sup_SAM_AMD ∩ lib_SAM_union) / (sup_SAM_AMD ∪ lib_SAM_union)
       Spatial overlap between the two binary AMD footprints.

    2. Cohen's Kappa   chance-corrected binary agreement within the soil domain
       (both maps treated as AMD-present / not-present).

    3. Library SAM recall in supervised SAM zone   fraction of library SAM
       pixels (union of all minerals) that fall inside the supervised SAM AMD
       zone.  High recall indicates the supervised SAM zone encompasses the
       library-detected mineral occurrences.

    4. Supervised SAM efficiency   fraction of supervised SAM AMD pixels
       confirmed by ≥ 1 library SAM mineral detection.  Lower values are
       expected if the supervised SAM zone is broader than the library SAM
       mineral peaks.

    5. Per-mineral containment   fraction of each library SAM mineral inside
       the supervised SAM AMD zone.

    A figure is saved alongside the CSVs showing the spatial overlap map and
    the per-mineral containment bar chart.

    Parameters
    ----------
    sam_class_map : ndarray (rows, cols), int16  — supervised SAM output
    class_names   : list of str                  — supervised SAM class names
    soil_mask     : ndarray (rows, cols), bool
    output_dir    : Path

    Returns
    -------
    summary_df : pd.DataFrame or None
    """
    lib_sam_hdr = Path(SAM_CLASS_MAP_FILE)
    if not lib_sam_hdr.exists():
        print(
            f"\n  [sam-cross-method] Library SAM classification map not found at:\n"
            f"    {lib_sam_hdr}\n"
            "  Skipping cross-method comparison.  Run multi_mineral_sam_fixed.py "
            "first."
        )
        return None

    print("\n" + "=" * 70)
    print("CROSS-METHOD COMPARISON  "
          "(Supervised SAM AMD Zone vs Library SAM Mineral Map)")
    print("=" * 70)

    lib_img = sp.open_image(str(lib_sam_hdr))
    lib_map = np.squeeze(lib_img.load()[:, :, 0]).astype(np.int32)

    lib_class_meta = lib_img.metadata.get("class names", [])
    lib_mineral_names = list(lib_class_meta[1:]) if len(lib_class_meta) > 1 else []
    print(f"\n  Library SAM mineral classes ({len(lib_mineral_names)}): "
          f"{lib_mineral_names}")

    # ── Supervised SAM AMD_FeOx binary mask ───────────────────────────────
    if "AMD_FeOx" in class_names:
        amd_idx = class_names.index("AMD_FeOx")
    else:
        amd_idx = next(
            (i for i, n in enumerate(class_names) if n.lower() != "background"),
            0,
        )
        print(f"  WARNING: 'AMD_FeOx' not found in class_names — using "
              f"class index {amd_idx} ({class_names[amd_idx]}) as AMD mask.")

    sup_amd = (sam_class_map == amd_idx)
    lib_any = (lib_map > 0)   # union of all library SAM mineral detections

    n_sup          = int(sup_amd.sum())
    n_lib          = int(lib_any.sum())
    n_intersection = int((sup_amd & lib_any).sum())
    n_union        = int((sup_amd | lib_any).sum())

    jaccard        = n_intersection / n_union if n_union > 0 else 0.0
    lib_recall     = n_intersection / n_lib   if n_lib   > 0 else 0.0
    sup_efficiency = n_intersection / n_sup   if n_sup   > 0 else 0.0

    # ── Cohen's Kappa (soil domain only) ──────────────────────────────────
    try:
        from sklearn.metrics import cohen_kappa_score
        soil_r, soil_c = np.where(soil_mask)
        y_sup = sup_amd[soil_r, soil_c].astype(np.int32)
        y_lib = lib_any[soil_r, soil_c].astype(np.int32)
        kappa = float(cohen_kappa_score(y_lib, y_sup))
        kappa_str = f"{kappa:.4f}"
    except ImportError:
        kappa     = np.nan
        kappa_str = "N/A (sklearn missing)"
    except Exception as exc:
        kappa     = np.nan
        kappa_str = f"N/A ({exc})"

    # ── Print global summary ──────────────────────────────────────────────
    print(f"\n  {'Metric':<45} {'Value':>12}")
    print("  " + "-" * 59)
    print(f"  {'Supervised SAM AMD_FeOx pixels':<45} {n_sup:>12,}")
    print(f"  {'Library SAM pixels (all minerals, union)':<45} {n_lib:>12,}")
    print(f"  {'Intersection':<45} {n_intersection:>12,}")
    print(f"  {'Union':<45} {n_union:>12,}")
    print(f"  {'Jaccard IoU':<45} {jaccard:>12.4f}")
    lbl_kappa = "Cohen's Kappa (soil domain)"
    print(f"  {lbl_kappa:<45} {kappa_str:>12}")
    print(f"  {'Library SAM recall in sup SAM zone (%)':<45} "
          f"{lib_recall * 100:>11.1f}%")
    print(f"  {'Supervised SAM efficiency / lib-confirmed (%)':<45} "
          f"{sup_efficiency * 100:>11.1f}%")

    # ── Per-mineral containment ───────────────────────────────────────────
    per_mineral = []
    print(f"\n  Per-library-SAM-mineral containment within supervised SAM AMD zone:")
    print(f"  {'SAM Mineral':<28} {'n_lib':>8} {'n_in_sup':>10} {'% in sup':>9}")
    print("  " + "-" * 58)
    for i, mineral in enumerate(lib_mineral_names, start=1):
        mineral_mask = (lib_map == i)
        n_mineral    = int(mineral_mask.sum())
        n_in_sup     = int((mineral_mask & sup_amd).sum())
        pct          = n_in_sup / n_mineral * 100 if n_mineral > 0 else 0.0
        print(f"  {mineral:<28} {n_mineral:>8,} {n_in_sup:>10,} {pct:>8.1f}%")
        per_mineral.append({
            "SAM_mineral":    mineral,
            "n_lib_SAM":      n_mineral,
            "n_in_sup_SAM":   n_in_sup,
            "pct_in_sup_SAM": pct,
        })

    # ── Save CSVs ─────────────────────────────────────────────────────────
    summary_records = [
        {"metric": "SupSAM_AMD_pixels",        "value": n_sup},
        {"metric": "LibSAM_union_pixels",       "value": n_lib},
        {"metric": "Intersection",              "value": n_intersection},
        {"metric": "Union",                     "value": n_union},
        {"metric": "Jaccard_IoU",               "value": jaccard},
        {"metric": "Cohens_Kappa",              "value": kappa},
        {"metric": "LibSAM_recall_in_supSAM",   "value": lib_recall},
        {"metric": "SupSAM_efficiency",         "value": sup_efficiency},
    ]
    summary_df  = pd.DataFrame(summary_records)
    summary_csv = output_dir / "sam_cross_method_comparison.csv"
    summary_df.to_csv(str(summary_csv), index=False)

    per_min_df  = pd.DataFrame(per_mineral)
    per_min_csv = output_dir / "sam_per_mineral_containment.csv"
    per_min_df.to_csv(str(per_min_csv), index=False)

    print(f"\n  Summary metrics saved : {summary_csv.name}")
    print(f"  Per-mineral CSV saved : {per_min_csv.name}")

    # ── Comparison figure ─────────────────────────────────────────────────
    _save_sam_cross_method_figure(
        sup_amd, lib_any, per_mineral,
        jaccard, kappa, lib_recall, sup_efficiency,
        output_dir,
    )

    return summary_df


def _save_sam_cross_method_figure(sup_amd, lib_any, per_mineral,
                                   jaccard, kappa, lib_recall, sup_efficiency,
                                   output_dir):
    """Save a two-panel spatial agreement figure for the supervised vs library SAM.

    Left panel  — spatial overlap map coded by agreement category.
    Right panel — per-library-SAM-mineral containment bar chart + metrics box.

    Colour scheme
    -------------
    Blue   — supervised SAM AMD zone only
    Orange — library SAM minerals only
    Green  — both agree
    Grey   — neither / non-soil
    """
    from matplotlib.patches import Patch

    rows, cols = sup_amd.shape

    # ── Spatial overlap categories ────────────────────────────────────────
    sup_only = sup_amd & ~lib_any
    lib_only = lib_any & ~sup_amd
    both     = sup_amd & lib_any

    # Build RGB image
    rgb_ov = np.ones((rows, cols, 3), dtype=np.float32) * 0.88  # neutral grey
    rgb_ov[sup_only] = [0.25, 0.55, 0.85]   # blue   — supervised SAM only
    rgb_ov[lib_only] = [0.90, 0.55, 0.10]   # orange — library SAM only
    rgb_ov[both]     = [0.20, 0.72, 0.28]   # green  — agreement

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Supervised SAM AMD Zone vs. Library SAM Mineral Map — Spatial Agreement\n"
        "(divergences reflect USGS library–scene spectral distance, "
        "not algorithmic differences)",
        fontsize=12, fontweight="bold",
    )

    # ── Left: overlap map ─────────────────────────────────────────────────
    ax_map = axes[0]
    ax_map.imshow(rgb_ov, interpolation="nearest")
    ax_map.set_title("Spatial Overlap Map", fontsize=11, fontweight="bold")
    ax_map.axis("off")
    legend_elements = [
        Patch(fc=[0.25, 0.55, 0.85], ec="0.3", lw=0.5,
              label=f"Supervised SAM only  ({int(sup_only.sum()):,} px)"),
        Patch(fc=[0.90, 0.55, 0.10], ec="0.3", lw=0.5,
              label=f"Library SAM only  ({int(lib_only.sum()):,} px)"),
        Patch(fc=[0.20, 0.72, 0.28], ec="0.3", lw=0.5,
              label=f"Both agree  ({int(both.sum()):,} px)"),
        Patch(fc=[0.88, 0.88, 0.88], ec="0.4", lw=0.5,
              label="Neither / non-soil"),
    ]
    _add_scale_north_arrow(ax_map, cols, sb_y=0.06)

    fig.legend(
        handles=legend_elements,
        fontsize=8.5, ncol=1,
        frameon=True, fancybox=False, edgecolor="0.4",
        loc="lower center",
        bbox_to_anchor=(0.325, 0.01),
        bbox_transform=fig.transFigure,
    )
    fig.subplots_adjust(bottom=0.18, wspace=0.45)

    # ── Right: per-mineral containment bar chart ──────────────────────────
    ax_bar = axes[1]
    minerals = [d["SAM_mineral"] for d in per_mineral]
    pcts     = [d["pct_in_sup_SAM"] for d in per_mineral]

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
        ax_bar.text(0.5, 0.5, "No library SAM minerals found",
                    ha="center", va="center", transform=ax_bar.transAxes)

    ax_bar.set_xlabel(
        "% of library SAM mineral pixels inside supervised SAM AMD zone",
        fontsize=9,
    )
    ax_bar.set_title(
        "Per-mineral library SAM containment in supervised SAM AMD zone\n"
        "(high % = supervised SAM zone encloses the library SAM detections)",
        fontsize=10, fontweight="bold",
    )
    ax_bar.grid(True, axis="x", alpha=0.3)

    # Global metrics text box
    kappa_txt = (f"{kappa:.3f}"
                 if not (isinstance(kappa, float) and np.isnan(kappa))
                 else "N/A")
    metrics_txt = (
        f"Jaccard IoU              : {jaccard:.3f}\n"
        f"Cohen's Kappa            : {kappa_txt}\n"
        f"Lib SAM recall in sup SAM: {lib_recall * 100:.1f}%\n"
        f"Sup SAM efficiency       : {sup_efficiency * 100:.1f}%"
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
    fig_path = output_dir / "sam_cross_method_comparison.png"
    plt.savefig(str(fig_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Comparison figure saved  : {fig_path.name}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_supervised_classification(training_npz=None, hdr_file=None,
                                   output_dir=None):
    """Run the full supervised SAM classification pipeline.

    Loads training pixels, computes image-derived SAM endmembers, classifies
    all soil pixels, validates the result, and runs the cross-method comparison
    against the library SAM mineral map.

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
    print("SUPERVISED CLASSIFICATION — SAM (image-derived endmembers)")
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

    # ── 4. Supervised SAM classification ──────────────────────────────────
    print(f"\n4. Running supervised SAM classification...")
    sam_class_map = run_sam_supervised_classification(
        cube, soil_mask, X, y, class_names, wavelengths, output_dir
    )

    # ── 5. Cross-method comparison (supervised SAM vs library SAM) ─────────
    print(f"\n5. Cross-method comparison "
          f"(supervised SAM AMD zone vs library SAM mineral map)...")
    sam_cross_method_comparison(
        sam_class_map, class_names, soil_mask, output_dir
    )

    print("\n" + "=" * 70)
    print("SUPERVISED CLASSIFICATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir.resolve()}")

    print("\nClass pixel counts (after noise filter):")
    print(f"  {'Class':<22} {'Pixels':>10} {'% of soil':>12}")
    print("  " + "-" * 48)
    n_soil_total = int(soil_mask.sum())
    for cls_idx, cls_name in enumerate(class_names):
        if EXCLUDE_BACKGROUND and cls_name == "Background":
            continue
        n = int((sam_class_map == cls_idx).sum())
        pct = n / n_soil_total * 100 if n_soil_total > 0 else 0.0
        print(f"  {cls_name:<22} {n:>10,} {pct:>11.2f}%")
    n_unclass = max(int((sam_class_map == -1).sum() - (~soil_mask).sum()), 0)
    print(f"  {'Unclassified (soil)':<22} {n_unclass:>10,}")
    print(f"  {'Total soil pixels':<22} {n_soil_total:>10,}")

    return sam_class_map


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    """Command-line entry point for the supervised SAM classification pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Supervised SAM classification for EO-1 Hyperion AMD mapping.\n"
            "Computes image-derived endmembers from training pixels and runs\n"
            "SAM nearest-endmember classification, then compares the result\n"
            "against the library SAM mineral map."
        )
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

    run_supervised_classification(
        training_npz=args.training,
        hdr_file=args.hdr,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
