"""
Multi-Mineral SAM Analysis - EXPLICIT CLASSIFICATION VERSION
Fixed approach with proper handling of multi-class classification
"""

import numpy as np
import pandas as pd
import spectral as sp
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, label as nd_label
from scipy.stats import mannwhitneyu
from skimage.morphology import binary_opening, binary_closing, disk
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths (relative to repository root)
BASE_DIR = Path(__file__).resolve().parent
HDR_FILE = str(BASE_DIR / "amd_mapping" / "data" / "hyperion" / "EO1H2020342013284110KF_reflectance.hdr")
LIBRARY_FOLDER = str(BASE_DIR / "amd_mapping" / "data" / "spectral_library")
OUTPUT_FOLDER = str(BASE_DIR / "amd_mapping" / "outputs" / "classifications")

# Threshold margin: each mineral's threshold is set to its minimum SAM angle
# plus this percentage of that minimum angle.  e.g. 0.15 means +15%.
SAM_THRESHOLD_MARGIN = 0.01

# Pre-classification: index-based soil masking
# Pixels with NDVI >= this value are considered vegetation and excluded
NDVI_THRESHOLD = 0.50
# Pixels with MNDWI >= this value are considered water and excluded
MNDWI_THRESHOLD = 0.0
# Median filter kernel size (pixels) for cleaning the soil mask
SOIL_MASK_MEDIAN_SIZE = 3
# Morphological disk radius for opening/closing cleanup
SOIL_MASK_MORPH_RADIUS = 2

# Null-model threshold parameters
# Confidence level: the SAM threshold is set to the (1 - NULL_MODEL_CONFIDENCE)*100-th
# percentile of spectral angles measured on random background soil pixels.
# E.g. 0.95 → 5th percentile: a pixel must have a lower angle than 95% of all
# background pixels to be considered statistically distinguishable from background.
NULL_MODEL_CONFIDENCE = 0.95
# Maximum number of background pixels sampled per mineral for the null model.
# Larger values give more stable percentile estimates at the cost of runtime.
NULL_SAMPLE_SIZE = 5000

SAVE_INDIVIDUAL_MAPS = True
SAVE_COMPOSITE_MAP = True
SHOW_PLOTS = False

# Post-classification validation
# Connected components with fewer pixels than this are flagged as noise.
MIN_COMPONENT_SIZE = 4

# =============================================================================
# HELPERS
# =============================================================================

def short_mineral_name(full_name):
    """Derive a concise, unique display name from a spectral library filename.

    Handles duplicate base names (e.g. two Jarosite variants) by appending
    a distinguishing token such as '(Na)' or '(K)'.
    """
    parts = full_name.split('_')
    base = parts[0]                         # e.g. "Jarosite"
    rest = '_'.join(parts[1:]).upper()       # e.g. "GDS100_NA_90C_SYN_BECKA"
    # Disambiguate Na- vs K-jarosite (or similar duplicates)
    if 'JAROSITE' in base.upper():
        if '_NA_' in f'_{rest}_' or '_NA0' in rest:
            return f'{base} (Na)'
        if '_K_' in f'_{rest}_':
            return f'{base} (K)'
    return base


# Maximally distinct palette for up to 10 mineral classes.
# Based on Tableau-10 / colorbrewer qualitative schemes.
MINERAL_COLORS = [
    '#BCBD22',  # yellow-green  – Goethite
    '#5E3C99',  # purple        – Hematite
    '#D62728',  # red           – Jarosite (Na)
    '#2CA02C',  # green         – Jarosite (K)
    '#17BECF',  # cyan          – Schwertmannite
    '#E66101',  # orange        (spare)
    '#1F77B4',  # blue          (spare)
    '#E7298A',  # magenta       (spare)
    '#A6761D',  # brown         (spare)
    '#666666',  # grey          (spare)
]


# =============================================================================
# CORE SAM FUNCTIONS
# =============================================================================

def compute_sam_angles_manual(cube, ref_spectrum):
    """
    Manually compute SAM angles to ensure correct behavior
    
    SAM angle = arccos(dot(pixel, reference) / (||pixel|| * ||reference||))
    
    Parameters:
    -----------
    cube : array (rows, cols, bands)
        Image cube
    ref_spectrum : array (bands,)
        Reference spectrum (should be normalized)
    
    Returns:
    --------
    angles : array (rows, cols)
        SAM angles in radians
    """
    rows, cols, bands = cube.shape
    
    # Reshape cube to (n_pixels, bands)
    pixels = cube.reshape(-1, bands)
    
    # Compute dot product with reference
    dots = np.dot(pixels, ref_spectrum)
    
    # Compute norms of each pixel
    pixel_norms = np.linalg.norm(pixels, axis=1)
    
    # Avoid division by zero
    pixel_norms = np.where(pixel_norms == 0, 1e-10, pixel_norms)
    
    # Compute cosine similarity
    cos_angles = dots / pixel_norms
    
    # Clip to valid range [-1, 1] to avoid arccos errors
    cos_angles = np.clip(cos_angles, -1, 1)
    
    # Compute angles
    angles = np.arccos(cos_angles)
    
    # Reshape back to image
    angles = angles.reshape(rows, cols)
    
    return angles


def load_and_resample_spectrum(csv_path, wavelengths):
    """Load and resample mineral spectrum"""
    mineral_name = Path(csv_path).stem
    
    df = pd.read_csv(csv_path)
    spec_wl = df.iloc[:, 0].values
    spec_ref = df.iloc[:, 1].values
    
    # Interpolate
    f = interp1d(spec_wl, spec_ref,
                 bounds_error=False,
                 fill_value="extrapolate")
    
    ref_spectrum = f(wavelengths)
    
    # Normalize
    ref_spectrum /= np.linalg.norm(ref_spectrum)
    
    return ref_spectrum, mineral_name


# =============================================================================
# PRE-CLASSIFICATION: SOIL MASKING
# =============================================================================

def compute_soil_mask(cube, wavelengths):
    """
    Create a binary soil mask using NDVI and MNDWI index thresholding,
    followed by median filtering and morphological cleanup.

    Soil pixels = low NDVI (not vegetation) AND low MNDWI (not water)
    AND valid reflectance (not background/nodata).

    Parameters
    ----------
    cube : ndarray (rows, cols, bands)
        Reflectance image cube (0-1 scale).
    wavelengths : ndarray (bands,)
        Wavelengths in nm.

    Returns
    -------
    soil_mask : ndarray (rows, cols), bool
        True for soil pixels, False for vegetation/water/nodata.
    ndvi : ndarray (rows, cols)
        NDVI values (for diagnostics).
    mndwi : ndarray (rows, cols)
        MNDWI values (for diagnostics).
    """
    # --- find nearest bands ---
    red_idx = np.argmin(np.abs(wavelengths - 660))
    nir_idx = np.argmin(np.abs(wavelengths - 860))
    green_idx = np.argmin(np.abs(wavelengths - 550))
    swir_idx = np.argmin(np.abs(wavelengths - 1600))

    red = cube[:, :, red_idx].astype(np.float64)
    nir = cube[:, :, nir_idx].astype(np.float64)
    green = cube[:, :, green_idx].astype(np.float64)
    swir = cube[:, :, swir_idx].astype(np.float64)

    # --- NDVI: (NIR - Red) / (NIR + Red) ---
    ndvi_denom = nir + red
    ndvi = np.where(ndvi_denom > 0, (nir - red) / ndvi_denom, 0.0)

    # --- MNDWI (Xu 2006): (Green - SWIR) / (Green + SWIR) ---
    # Uses SWIR ~1600nm instead of NIR for much better water/land contrast
    mndwi_denom = green + swir
    mndwi = np.where(mndwi_denom > 0, (green - swir) / mndwi_denom, 0.0)

    # --- valid-data mask (exclude background / nodata pixels) ---
    pixel_norms = np.linalg.norm(cube.reshape(-1, cube.shape[2]), axis=1)
    valid_mask = (pixel_norms > 0).reshape(cube.shape[0], cube.shape[1])

    # --- thresholding ---
    not_vegetation = ndvi < NDVI_THRESHOLD
    not_water = mndwi < MNDWI_THRESHOLD
    soil_mask = valid_mask & not_vegetation & not_water

    # --- median filter to remove salt-and-pepper noise ---
    soil_mask = median_filter(soil_mask.astype(np.uint8),
                              size=SOIL_MASK_MEDIAN_SIZE).astype(bool)

    # --- morphological opening then closing for shape cleanup ---
    selem = disk(SOIL_MASK_MORPH_RADIUS)
    soil_mask = binary_opening(soil_mask, selem)
    soil_mask = binary_closing(soil_mask, selem)

    return soil_mask, ndvi, mndwi


# =============================================================================
# NULL-MODEL THRESHOLD DERIVATION
# =============================================================================

def derive_null_thresholds(sam_angles_dict, adaptive_thresholds, soil_mask,
                           cube, mineral_spectra, mineral_names,
                           confidence=NULL_MODEL_CONFIDENCE,
                           n_samples=NULL_SAMPLE_SIZE):
    """Derive per-mineral SAM thresholds from a spectral-angle null model.

    For each mineral endmember the function:
      1. Identifies 'background' soil pixels — soil pixels not classified by
         *any* mineral under the current adaptive threshold.
      2. Draws up to ``n_samples`` random background pixels (seed fixed to 42).
      3. Computes SAM angles between the endmember and each background pixel
         (efficient batch dot-product; endmembers are already unit-normalised).
      4. Returns the ``(1 - confidence) * 100``-th percentile of that angle
         distribution as the null-derived threshold.

    Interpretation: a pixel passes the null threshold only if its SAM angle is
    smaller than ``confidence * 100``% of all random background pixels, making
    it statistically distinguishable from background at the chosen confidence
    level.

    Parameters
    ----------
    sam_angles_dict : dict {name -> (rows, cols) ndarray}
        Per-mineral SAM angle images (non-soil pixels set to pi).
    adaptive_thresholds : dict {name -> float}
        Per-mineral adaptive thresholds already derived (min + margin).
    soil_mask : (rows, cols) bool ndarray
    cube : (rows, cols, bands) float32 ndarray
        Reflectance cube; used to extract background pixel spectra.
    mineral_spectra : dict {name -> (bands,) ndarray}
        Unit-normalised endmember spectra.
    mineral_names : list of str
    confidence : float, default ``NULL_MODEL_CONFIDENCE``
        Fraction in (0, 1).  Threshold = (1 - confidence) * 100-th percentile.
    n_samples : int, default ``NULL_SAMPLE_SIZE``
        Maximum number of background pixels to sample.

    Returns
    -------
    null_thresholds : dict {name -> float}  (radians; NaN if model skipped)
    null_distributions : dict {name -> 1-D ndarray}  (sampled null angles, radians)
    """
    percentile_pct = (1.0 - confidence) * 100.0   # e.g. 5.0 for 95% confidence

    # Build a unified "classified by any mineral under adaptive threshold" mask
    classified_any = np.zeros(soil_mask.shape, dtype=bool)
    for name in mineral_names:
        classified_any |= (sam_angles_dict[name] < adaptive_thresholds[name]) & soil_mask

    # Background = soil pixels not captured by any adaptive threshold
    bg_mask = soil_mask & ~classified_any
    bg_r, bg_c = np.where(bg_mask)
    n_bg = len(bg_r)

    print("\n" + "=" * 70)
    print(f"NULL-MODEL THRESHOLD DERIVATION  "
          f"(confidence = {confidence * 100:.0f}%,  "
          f"threshold = {percentile_pct:.0f}th percentile of null angles)")
    print("=" * 70)
    print(f"  Background pixels available : {n_bg:,}  "
          f"(soil pixels unclassified under any adaptive threshold)")

    null_thresholds    = {}
    null_distributions = {}

    if n_bg < 10:
        print("  WARNING: fewer than 10 background pixels — null model skipped.")
        for name in mineral_names:
            null_thresholds[name]    = np.nan
            null_distributions[name] = np.array([])
        return null_thresholds, null_distributions

    np.random.seed(42)
    sample_size = min(n_samples, n_bg)
    idx_s  = np.random.choice(n_bg, size=sample_size, replace=False)
    bg_r_s = bg_r[idx_s]
    bg_c_s = bg_c[idx_s]

    # Extract background spectra once — shared across all minerals
    bg_spectra = cube[bg_r_s, bg_c_s, :].astype(np.float64)  # (sample_size, bands)
    _impute_nan_bands(bg_spectra)

    bg_norms   = np.linalg.norm(bg_spectra, axis=1)           # (sample_size,)
    safe_norms = np.where(bg_norms == 0, 1e-10, bg_norms)

    print(f"  Sampled {sample_size:,} background pixels for null distributions\n")

    cw = (25, 16, 14, 18, 18)
    header = (f"  {'Mineral':<{cw[0]}} "
              f"{'Null thr (rad)':>{cw[1]}} "
              f"{'Null thr (°)':>{cw[2]}} "
              f"{'Min null (°)':>{cw[3]}} "
              f"{'Max null (°)':>{cw[4]}}")
    print(header)
    print("  " + "-" * (sum(cw) + len(cw)))

    for name in mineral_names:
        endmember = mineral_spectra[name]        # unit-normalised → its norm = 1
        cos_vals  = (bg_spectra @ endmember) / safe_norms
        cos_vals  = np.clip(cos_vals, -1.0, 1.0)
        null_angles = np.arccos(cos_vals)

        null_thr = float(np.percentile(null_angles, percentile_pct))
        null_thresholds[name]    = null_thr
        null_distributions[name] = null_angles

        short = short_mineral_name(name)
        print(f"  {short:<{cw[0]}} "
              f"{null_thr:>{cw[1]}.4f} "
              f"{np.degrees(null_thr):>{cw[2]}.2f}\u00b0 "
              f"{np.degrees(null_angles.min()):>{cw[3]}.2f}\u00b0 "
              f"{np.degrees(null_angles.max()):>{cw[4]}.2f}\u00b0")

    print()
    return null_thresholds, null_distributions


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def compare_thresholds(sam_angles_dict, soil_mask, adaptive_thresholds,
                       null_thresholds, mineral_names, cube,
                       confidence=NULL_MODEL_CONFIDENCE):
    """Print a side-by-side comparison of adaptive vs null-derived thresholds.

    For each mineral the function reports:
    * The threshold angle (degrees) under each method.
    * How many soil pixels survive each threshold.
    * The KMeans inertia ratio (real / null-background) under each threshold,
      recomputed independently so the comparison is self-contained.

    Parameters
    ----------
    sam_angles_dict : dict {name -> (rows, cols) ndarray}
    soil_mask : (rows, cols) bool ndarray
    adaptive_thresholds : dict {name -> float}
    null_thresholds : dict {name -> float}  (may contain NaN if model failed)
    mineral_names : list of str
    cube : (rows, cols, bands) float32 ndarray
    confidence : float  — displayed in the section header only.
    """
    try:
        from sklearn.cluster import KMeans as _KMeans
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False

    print("\n" + "=" * 70)
    print(f"THRESHOLD COMPARISON  "
          f"(adaptive  vs  null-model @ {confidence * 100:.0f}% confidence)")
    print("=" * 70)

    np.random.seed(42)

    # Background mask for inertia: soil pixels not claimed by any adaptive thr
    classified_any = np.zeros(soil_mask.shape, dtype=bool)
    for name in mineral_names:
        classified_any |= (sam_angles_dict[name] < adaptive_thresholds[name]) & soil_mask
    bg_mask_global = soil_mask & ~classified_any
    bg_r_g, bg_c_g = np.where(bg_mask_global)

    def _inertia_ratio(binary_mask):
        """KMeans inertia ratio (real classified pixels / random background)."""
        n_cls = int(binary_mask.sum())
        if n_cls < 2 or not HAS_SKLEARN:
            return np.nan
        r_idx, c_idx = np.where(binary_mask)
        sp_real = cube[r_idx, c_idx, :].astype(np.float64)
        _impute_nan_bands(sp_real)
        km = _KMeans(n_clusters=1, n_init=1, random_state=42)
        km.fit(sp_real)
        inertia_real = km.inertia_

        n_null = min(n_cls, len(bg_r_g))
        if n_null < 2:
            return np.nan
        idx_s  = np.random.choice(len(bg_r_g), size=n_null, replace=False)
        sp_null = cube[bg_r_g[idx_s], bg_c_g[idx_s], :].astype(np.float64)
        _impute_nan_bands(sp_null)
        km_null = _KMeans(n_clusters=1, n_init=1, random_state=42)
        km_null.fit(sp_null)
        return (inertia_real / km_null.inertia_
                if km_null.inertia_ > 0 else np.nan)

    hdr = (f"  {'Mineral':<20} "
           f"{'Adapt thr (°)':>14} {'Adapt px':>10} {'Iner_A':>8} "
           f"{'Null thr (°)':>14} {'Null px':>10} {'Iner_N':>8} "
           f"{'Δ px':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for name in mineral_names:
        short      = short_mineral_name(name)
        adap_thr   = adaptive_thresholds[name]
        null_thr   = null_thresholds[name]
        sam_angles = sam_angles_dict[name]

        adap_mask = (sam_angles < adap_thr) & soil_mask
        n_adap    = int(adap_mask.sum())
        ratio_adap = _inertia_ratio(adap_mask)

        if np.isnan(null_thr):
            n_null     = 0
            ratio_null = np.nan
            thr_null_s = f"{'N/A':>14}"
            n_null_s   = f"{'N/A':>10}"
            delta      = 0
        else:
            null_mask  = (sam_angles < null_thr) & soil_mask
            n_null     = int(null_mask.sum())
            ratio_null = _inertia_ratio(null_mask)
            thr_null_s = f"{np.degrees(null_thr):>13.2f}\u00b0"
            n_null_s   = f"{n_null:>10,}"
            delta      = n_null - n_adap

        r_a = f"{ratio_adap:.4f}" if not np.isnan(ratio_adap) else "   N/A"
        r_n = f"{ratio_null:.4f}" if not np.isnan(ratio_null) else "   N/A"

        print(f"  {short:<20} "
              f"{np.degrees(adap_thr):>13.2f}\u00b0 {n_adap:>10,} {r_a:>8} "
              f"{thr_null_s} {n_null_s} {r_n:>8} "
              f"{delta:>+9,}")

    print()
    if not HAS_SKLEARN:
        print("  NOTE: scikit-learn not found — inertia ratios skipped.")
    print("  \u0394 px     = null-threshold pixels \u2212 adaptive-threshold pixels")
    print("  Iner_A  = inertia ratio under adaptive threshold  (real / background)")
    print("  Iner_N  = inertia ratio under null-model threshold")
    print("  Inertia ratio < 1.0  \u2192  classified pixels more spectrally coherent "
          "than random soil")
    print("=" * 70)


def main():
    """Main processing workflow"""
    print("=" * 70)
    print("MULTI-MINERAL SAM ANALYSIS - EXPLICIT METHOD")
    print("=" * 70)
    
    # Create output folder
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # Load image
    print("\n1. Loading hyperspectral image...")
    img = sp.open_image(HDR_FILE)
    cube = img.load().astype(np.float32)
    rows, cols, bands = cube.shape
    wavelengths = np.array(img.metadata['wavelength'], dtype=float)

    print(f"   Image shape: {rows} x {cols} x {bands}")
    print(f"   Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")

    # Convert from scaled integers to reflectance (0-1)
    # Hyperion reflectance HDR stores values multiplied by 10000
    print(f"   Raw value range: {cube.min():.1f} - {cube.max():.1f}")
    cube /= 10000.0
    print(f"   After scaling (/10000): {cube.min():.4f} - {cube.max():.4f}")
    
    # Load library
    print(f"\n2. Loading mineral library...")
    csv_files = sorted(glob.glob(str(Path(LIBRARY_FOLDER) / "*.csv")))
    
    if not csv_files:
        print(f"   ERROR: No CSV files found in {LIBRARY_FOLDER}")
        return
    
    print(f"   Found {len(csv_files)} minerals")
    
    # Load all mineral spectra
    mineral_spectra = {}
    mineral_names = []
    
    for csv_file in csv_files:
        ref_spectrum, mineral_name = load_and_resample_spectrum(csv_file, wavelengths)
        mineral_spectra[mineral_name] = ref_spectrum
        mineral_names.append(mineral_name)
        print(f"   Loaded: {mineral_name}")
    
    # Create diagnostic plots
    print(f"\n3. Creating diagnostic plots...")
    create_diagnostic_plots(cube, wavelengths, mineral_names, OUTPUT_FOLDER)

    # Pre-classification: compute soil mask
    print(f"\n4. Pre-classification: computing soil mask...")
    print(f"   NDVI threshold: {NDVI_THRESHOLD}  (pixels >= this are vegetation)")
    print(f"   MNDWI threshold: {MNDWI_THRESHOLD}  (pixels >= this are water)")
    print(f"   Median filter size: {SOIL_MASK_MEDIAN_SIZE}")
    print(f"   Morphological disk radius: {SOIL_MASK_MORPH_RADIUS}")
    soil_mask, ndvi, mndwi = compute_soil_mask(cube, wavelengths)
    n_soil = np.sum(soil_mask)
    n_total = soil_mask.size
    print(f"   Soil pixels: {n_soil} / {n_total} ({n_soil / n_total * 100:.1f}%)")
    print(f"   Excluded pixels: {n_total - n_soil} ({(n_total - n_soil) / n_total * 100:.1f}%)")

    # Save soil mask diagnostic plot
    print(f"\n5. Saving pre-classification diagnostics...")
    save_soil_mask_plot(soil_mask, ndvi, mndwi, OUTPUT_FOLDER)

    # Compute SAM angles for all minerals (only on soil pixels)
    print(f"\n6. Computing SAM angles (soil pixels only)...")
    sam_angles_dict = {}

    for mineral_name in mineral_names:
        ref_spectrum = mineral_spectra[mineral_name]
        sam_angles = compute_sam_angles_manual(cube, ref_spectrum)
        # Set non-soil pixels to pi (maximum angle) so they are never classified
        sam_angles[~soil_mask] = np.pi
        sam_angles_dict[mineral_name] = sam_angles
        soil_min = np.min(sam_angles[soil_mask]) if n_soil > 0 else np.nan
        print(f"   {mineral_name}: min angle (soil only) = {soil_min:.4f} rad")

    # Derive per-mineral thresholds: min_angle * (1 + margin)
    print(f"\n7. Deriving adaptive thresholds (margin = {SAM_THRESHOLD_MARGIN:.0%} above minimum angle)...")
    thresholds = {}
    for name in mineral_names:
        soil_angles = sam_angles_dict[name][soil_mask]
        min_angle = np.min(soil_angles) if len(soil_angles) > 0 else np.pi
        thresholds[name] = min_angle * (1 + SAM_THRESHOLD_MARGIN)
        print(f"     {name}: {thresholds[name]:.4f} rad ({np.degrees(thresholds[name]):.1f}°)")

    # Derive null-model thresholds (statistical, background-anchored)
    print(f"\n7b. Deriving null-model thresholds "
          f"(confidence = {NULL_MODEL_CONFIDENCE * 100:.0f}%, "
          f"sample size = {NULL_SAMPLE_SIZE:,})...")
    null_thresholds, null_distributions = derive_null_thresholds(
        sam_angles_dict, thresholds, soil_mask,
        cube, mineral_spectra, mineral_names,
    )

    # Compute match scores and statistics
    print(f"\n8. Computing match scores...")
    match_scores_dict = {}

    for mineral_name in mineral_names:
        thr = thresholds[mineral_name]
        sam_angles = sam_angles_dict[mineral_name]

        # Compute match score
        match_score = np.clip(1 - sam_angles / thr, 0, 1)
        match_scores_dict[mineral_name] = match_score

        # Statistics
        pixels_below_threshold = np.sum(sam_angles < thr)
        percent_below = (pixels_below_threshold / sam_angles.size) * 100

        print(f"\n   {mineral_name} (threshold: {thr:.4f} rad)")
        print(f"     Min angle: {np.min(sam_angles):.4f} rad")
        print(f"     Mean angle: {np.mean(sam_angles):.4f} rad")
        print(f"     Pixels below threshold: {pixels_below_threshold} ({percent_below:.2f}%)")
        print(f"     Mean match score: {np.mean(match_score):.3f}")

        # Save individual map
        if SAVE_INDIVIDUAL_MAPS:
            save_individual_map(match_score, sam_angles, mineral_name,
                              thr, OUTPUT_FOLDER)

    # Create composite classification map
    if SAVE_COMPOSITE_MAP and len(mineral_names) > 0:
        print("\n9. Creating composite classification map...")
        create_composite_map(sam_angles_dict, match_scores_dict,
                           mineral_names, thresholds, soil_mask, OUTPUT_FOLDER)

    # Compare adaptive vs null-model thresholds
    print("\n9b. Comparing adaptive vs null-model thresholds...")
    compare_thresholds(
        sam_angles_dict, soil_mask, thresholds,
        null_thresholds, mineral_names, cube,
    )

    # Post-classification validation
    print("\n10. Running post-classification validation...")
    validate_sam_results(sam_angles_dict, thresholds, soil_mask, cube,
                         mineral_names, OUTPUT_FOLDER,
                         null_thresholds=null_thresholds,
                         null_distributions=null_distributions,
                         mineral_spectra=mineral_spectra)

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_FOLDER}")
    
    if SHOW_PLOTS:
        plt.show()


def create_diagnostic_plots(cube, wavelengths, mineral_names, output_folder):
    """
    Create comparative diagnostic plots for QA/QC of the SAM analysis.

    Generates:
    1. Hyperion RGB composite
    2. All mineral endmember spectra overlaid
    3. Individual mineral spectra (subplot grid)
    4. Mean image spectrum vs mineral endmembers
    5. Sample pixel spectra from the image

    Parameters:
    -----------
    cube : array (rows, cols, bands)
        Hyperspectral image cube
    wavelengths : array (bands,)
        Wavelengths in nm
    mineral_names : list
        Mineral name strings
    output_folder : str
        Base output folder (diagnostic_plots subfolder created inside)
    """
    diag_dir = Path(output_folder) / "diagnostic_plots"
    diag_dir.mkdir(parents=True, exist_ok=True)

    _, cols, bands = cube.shape

    # Reload un-normalized spectra for plotting (mineral_spectra are normalized for SAM)
    csv_files = sorted(glob.glob(str(Path(LIBRARY_FOLDER) / "*.csv")))
    raw_spectra = {}
    for csv_file in csv_files:
        name = Path(csv_file).stem
        df = pd.read_csv(csv_file)
        spec_wl = df.iloc[:, 0].values
        spec_ref = df.iloc[:, 1].values
        f = interp1d(spec_wl, spec_ref, bounds_error=False, fill_value="extrapolate")
        raw_spectra[name] = f(wavelengths)

    # ---- 1. Hyperion RGB composite ----
    print("     Creating Hyperion RGB composite...")
    fig, ax = plt.subplots(figsize=(8, 14))

    # Find RGB band indices (~660nm R, ~550nm G, ~480nm B)
    r_idx = np.argmin(np.abs(wavelengths - 660))
    g_idx = np.argmin(np.abs(wavelengths - 550))
    b_idx = np.argmin(np.abs(wavelengths - 480))

    rgb = np.stack([cube[:, :, r_idx],
                    cube[:, :, g_idx],
                    cube[:, :, b_idx]], axis=2)

    # Percentile stretch for visualization
    for i in range(3):
        band = rgb[:, :, i]
        valid = band[band > 0]
        if len(valid) > 0:
            lo, hi = np.percentile(valid, [2, 98])
            rgb[:, :, i] = np.clip((band - lo) / (hi - lo + 1e-10), 0, 1)

    ax.imshow(rgb)
    ax.set_title('Hyperion RGB Composite (Approximate True Color)', fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(diag_dir / "hyperion_rgb.png", dpi=150, bbox_inches='tight')
    if not SHOW_PLOTS:
        plt.close()

    # ---- 2. All mineral spectra overlaid ----
    print("     Creating mineral spectra comparison plot...")
    fig, ax = plt.subplots(figsize=(14, 7))

    # Highlight key absorption regions
    absorption_bands = [
        (830, 950, 'pink', 0.15),       # Fe3+ absorption
        (2150, 2250, 'lightyellow', 0.2) # Al-OH / clay absorption
    ]
    for lo, hi, color, alpha in absorption_bands:
        ax.axvspan(lo, hi, color=color, alpha=alpha)

    colors = plt.cm.tab10(np.linspace(0, 1, len(mineral_names)))
    for name, color in zip(mineral_names, colors):
        if name in raw_spectra:
            ax.plot(wavelengths, raw_spectra[name], label=name, color=color, linewidth=1.5)

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.set_title('AMD Mineral Endmember Spectra (Hyperion Wavelengths)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(diag_dir / "mineral_spectra.png", dpi=150, bbox_inches='tight')
    if not SHOW_PLOTS:
        plt.close()

    # ---- 3. Individual mineral spectra (subplot grid) ----
    print("     Creating individual mineral spectra grid...")
    n = len(mineral_names)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, name in enumerate(mineral_names):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if name in raw_spectra:
            ax.plot(wavelengths, raw_spectra[name], color=colors[idx], linewidth=1.2)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Wavelength (nm)', fontsize=9)
        ax.set_ylabel('Reflectance', fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    plt.tight_layout()
    plt.savefig(diag_dir / "mineral_spectra_individual.png", dpi=150, bbox_inches='tight')
    if not SHOW_PLOTS:
        plt.close()

    # ---- 4. Mean image spectrum vs mineral endmembers ----
    print("     Creating mean spectrum comparison plot...")
    fig, ax = plt.subplots(figsize=(14, 7))

    # Compute mean spectrum of the image (ignoring zero/nodata pixels)
    pixels = cube.reshape(-1, bands)
    pixel_norms = np.linalg.norm(pixels, axis=1)
    valid_mask = pixel_norms > 0
    mean_spectrum = np.mean(pixels[valid_mask], axis=0)

    ax.plot(wavelengths, mean_spectrum, 'k-', linewidth=2.5, label='Mean Image Spectrum', zorder=10)

    for name, color in zip(mineral_names, colors):
        if name in raw_spectra:
            ax.plot(wavelengths, raw_spectra[name], color=color, linewidth=1.2, alpha=0.7, label=name)

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.set_title('Mean Hyperion Spectrum vs Mineral Endmembers', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(diag_dir / "mean_spectrum_comparison.png", dpi=150, bbox_inches='tight')
    if not SHOW_PLOTS:
        plt.close()

    # ---- 5. Sample pixel spectra ----
    print("     Creating sample pixel spectra plot...")
    fig, ax = plt.subplots(figsize=(14, 7))

    np.random.seed(42)
    n_samples = 5
    sample_indices = np.random.choice(np.where(valid_mask)[0], size=min(n_samples, valid_mask.sum()), replace=False)

    sample_colors = plt.cm.Set1(np.linspace(0, 1, n_samples))
    for i, pix_idx in enumerate(sample_indices):
        r_pos, c_pos = divmod(int(pix_idx), cols)
        ax.plot(wavelengths, pixels[pix_idx], color=sample_colors[i],
                linewidth=1.2, label=f'Pixel ({r_pos}, {c_pos})')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.set_title('Sample Hyperion Pixel Spectra', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(diag_dir / "sample_pixel_spectra.png", dpi=150, bbox_inches='tight')
    if not SHOW_PLOTS:
        plt.close()

    print(f"     Diagnostic plots saved to: {diag_dir}")


def save_soil_mask_plot(soil_mask, ndvi, mndwi, output_folder):
    """Save publication-quality pre-classification soil mask figure."""
    from matplotlib.patches import Patch

    diag_dir = Path(output_folder) / "diagnostic_plots"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # --- build a valid-data mask (any pixel with nonzero index or soil flag) ---
    valid = (ndvi != 0) | (mndwi != 0) | soil_mask

    # Masked arrays so nodata renders as white (the axes facecolor)
    ndvi_m = np.ma.masked_where(~valid, ndvi)
    mndwi_m = np.ma.masked_where(~valid, mndwi)

    # --- figure setup ---
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 9.5),
                             gridspec_kw={'hspace': 0.28, 'wspace': 0.30})
    for ax in axes.flat:
        ax.set_facecolor('white')

    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    # ---- (a) NDVI ----
    ax = axes[0, 0]
    im0 = ax.imshow(ndvi_m, cmap='RdYlGn', vmin=-0.2, vmax=0.8,
                    interpolation='nearest')
    ax.axhline(y=0, color='none')  # force extent
    cbar0 = plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    cbar0.set_label('NDVI', fontsize=8)
    cbar0.ax.tick_params(labelsize=7)
    ax.set_title(f'NDVI  (threshold = {NDVI_THRESHOLD})', fontsize=9)
    ax.text(0.02, 0.97, panel_labels[0], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax.axis('off')

    # ---- (b) MNDWI ----
    ax = axes[0, 1]
    im1 = ax.imshow(mndwi_m, cmap='RdBu', vmin=-0.5, vmax=0.5,
                    interpolation='nearest')
    cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label('MNDWI', fontsize=8)
    cbar1.ax.tick_params(labelsize=7)
    ax.set_title(f'MNDWI  (threshold = {MNDWI_THRESHOLD})', fontsize=9)
    ax.text(0.02, 0.97, panel_labels[1], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax.axis('off')

    # ---- (c) Pre-classification overlay ----
    ax = axes[1, 0]
    # RGBA overlay: white background, then fill classes
    rgba_overlay = np.ones((*soil_mask.shape, 4))  # white + fully opaque
    # water takes priority over vegetation (MNDWI is the more specific discriminator)
    wat_mask = valid & (mndwi >= MNDWI_THRESHOLD)
    veg_mask = valid & (ndvi >= NDVI_THRESHOLD) & ~wat_mask
    nodata_mask = ~valid

    rgba_overlay[veg_mask]   = [0.20, 0.60, 0.20, 1.0]   # muted green
    rgba_overlay[wat_mask]   = [0.20, 0.40, 0.75, 1.0]   # muted blue
    rgba_overlay[soil_mask]  = [0.76, 0.56, 0.33, 1.0]   # earth brown
    rgba_overlay[nodata_mask] = [1.0, 1.0, 1.0, 1.0]     # white

    ax.imshow(rgba_overlay, interpolation='nearest')
    ax.set_title('Index-based pre-classification', fontsize=9)
    ax.text(0.02, 0.97, panel_labels[2], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax.axis('off')

    # Legend for overlay
    legend_patches = [
        Patch(facecolor=(0.76, 0.56, 0.33), edgecolor='k', linewidth=0.4, label='Bare soil'),
        Patch(facecolor=(0.20, 0.60, 0.20), edgecolor='k', linewidth=0.4, label='Vegetation'),
        Patch(facecolor=(0.20, 0.40, 0.75), edgecolor='k', linewidth=0.4, label='Water'),
    ]
    ax.legend(handles=legend_patches, loc='upper center',
              bbox_to_anchor=(0.5, -0.01), fontsize=7,
              frameon=True, fancybox=False, edgecolor='0.4',
              handlelength=1.2, handleheight=0.9, ncol=3)

    # ---- (d) Final soil mask (binary) ----
    ax = axes[1, 1]
    mask_rgba = np.where(valid[..., None],
                         np.where(soil_mask[..., None],
                                  np.array([0.55, 0.37, 0.24, 1.0]),   # dark brown
                                  np.array([0.85, 0.85, 0.85, 1.0])),  # light gray = excluded valid
                         np.array([1.0, 1.0, 1.0, 1.0]))               # white = nodata
    ax.imshow(mask_rgba, interpolation='nearest')
    n_soil = int(np.sum(soil_mask))
    n_valid = int(np.sum(valid))
    ax.set_title(
        f'Soil mask (filtered)\n'
        f'{n_soil:,} px / {n_valid:,} valid ({n_soil / max(n_valid, 1) * 100:.1f}%)',
        fontsize=9)
    ax.text(0.02, 0.97, panel_labels[3], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax.axis('off')

    legend_patches_d = [
        Patch(facecolor=(0.55, 0.37, 0.24), edgecolor='k', linewidth=0.4, label='Soil'),
        Patch(facecolor=(0.85, 0.85, 0.85), edgecolor='k', linewidth=0.4, label='Excluded'),
    ]
    ax.legend(handles=legend_patches_d, loc='upper center',
              bbox_to_anchor=(0.5, -0.01), fontsize=7,
              frameon=True, fancybox=False, edgecolor='0.4',
              handlelength=1.2, handleheight=0.9, ncol=2)

    out_path = diag_dir / "pre_classification_soil_mask.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"     Saved: {out_path}")
    if not SHOW_PLOTS:
        plt.close()


def save_individual_map(match_score, sam_angles, mineral_name, threshold, output_folder):
    """Save publication-quality individual mineral detection map."""
    short_name = short_mineral_name(mineral_name)

    # Mask nodata (sam_angles == pi means non-soil or nodata)
    is_nodata = sam_angles >= np.pi
    match_m = np.ma.masked_where(is_nodata, match_score)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 5.0))
    for ax in (ax1, ax2):
        ax.set_facecolor('white')

    # (a) Match score
    im1 = ax1.imshow(match_m, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
    ax1.set_title(
        f'{short_name} \u2014 match score\n'
        f'(threshold {threshold:.3f} rad / {np.degrees(threshold):.1f}\u00b0)',
        fontsize=9)
    ax1.text(0.02, 0.97, '(a)', transform=ax1.transAxes,
             fontsize=10, fontweight='bold', va='top',
             bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Match score', fontsize=8)
    cbar1.ax.tick_params(labelsize=7)

    # (b) Binary detection — RGBA for clean nodata handling
    detected = sam_angles < threshold
    rgba_det = np.ones((*sam_angles.shape, 4), dtype=np.float32)  # white bg
    rgba_det[~is_nodata & ~detected] = [0.85, 0.85, 0.85, 1.0]   # light gray = not detected
    rgba_det[detected] = [0.13, 0.55, 0.13, 1.0]                  # forest green = detected
    ax2.imshow(rgba_det, interpolation='nearest')
    n_det = int(np.sum(detected))
    ax2.set_title(
        f'{short_name} \u2014 binary detection\n'
        f'{n_det:,} pixels detected',
        fontsize=9)
    ax2.text(0.02, 0.97, '(b)', transform=ax2.transAxes,
             fontsize=10, fontweight='bold', va='top',
             bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax2.axis('off')

    from matplotlib.patches import Patch
    ax2.legend(
        handles=[
            Patch(fc=(0.13, 0.55, 0.13), ec='0.3', lw=0.4, label='Detected'),
            Patch(fc=(0.85, 0.85, 0.85), ec='0.3', lw=0.4, label='Not detected'),
        ],
        loc='lower right', fontsize=7,
        frameon=True, fancybox=False, edgecolor='0.4',
        handlelength=1.0, handleheight=0.8)

    plt.tight_layout()
    output_path = Path(output_folder) / f"{mineral_name}_SAM_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"     Saved: {output_path.name}")

    if not SHOW_PLOTS:
        plt.close()


def create_composite_map(sam_angles_dict, match_scores_dict, mineral_names, thresholds, soil_mask, output_folder):
    """
    Create composite classification map using proper SAM classification logic

    Classification rules:
    1. Only soil-masked pixels are candidates for classification
    2. For each pixel, mask out minerals whose angle >= their own threshold
    3. Among the remaining candidates, pick the mineral with the smallest angle
    4. If no mineral passes its threshold, the pixel is unclassified (0)

    Parameters:
    -----------
    thresholds : dict
        Mapping of mineral_name -> threshold (radians)
    soil_mask : ndarray (rows, cols), bool
        True for soil pixels eligible for SAM classification
    """

    # Get image dimensions
    rows, cols = list(sam_angles_dict.values())[0].shape
    n_minerals = len(mineral_names)

    # Stack all SAM angles into 3D array (rows, cols, n_minerals)
    all_angles = np.zeros((rows, cols, n_minerals))
    for idx, name in enumerate(mineral_names):
        all_angles[:, :, idx] = sam_angles_dict[name]

    # Build per-mineral threshold array and broadcast to (1, 1, n_minerals)
    thr_array = np.array([thresholds[name] for name in mineral_names])
    thr_broadcast = thr_array[np.newaxis, np.newaxis, :]

    # Mask: True where the mineral passes its own threshold
    passes = all_angles < thr_broadcast  # (rows, cols, n_minerals)

    # Replace angles that fail their threshold with inf so they can't win
    masked_angles = np.where(passes, all_angles, np.inf)

    # Among passing minerals, pick the one with the smallest angle
    min_angle_idx = np.argmin(masked_angles, axis=2)
    min_angle_value = np.min(masked_angles, axis=2)

    # Pixels where no mineral passed any threshold
    any_pass = np.any(passes, axis=2)

    # Create classification map
    # Class 0 = unclassified (no mineral passed its threshold or non-soil)
    # Class 1, 2, 3... = minerals
    class_map = np.where(any_pass & soil_mask,
                         min_angle_idx + 1,  # Mineral class (1-indexed)
                         0)                   # Unclassified

    # =====================================================================
    # Publication-quality visualization
    # =====================================================================
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec

    mineral_rgb = MINERAL_COLORS

    # Nodata / soil-but-unclassified / background distinction
    # has_data: any pixel with data (SAM angle < pi means it was not masked out)
    has_data = np.any(np.abs(
        np.stack([sam_angles_dict[n] for n in mineral_names], axis=-1)) < np.pi, axis=-1) | soil_mask

    # Build RGBA image manually for full control
    rgba = np.ones((rows, cols, 4), dtype=np.float32)  # white background

    # Layer 1: non-soil valid pixels -> very light gray
    rgba[has_data & ~soil_mask] = [0.92, 0.92, 0.92, 1.0]

    # Layer 2: soil but unclassified -> medium gray
    soil_unclass = soil_mask & (class_map == 0)
    rgba[soil_unclass] = [0.75, 0.75, 0.75, 1.0]

    # Layer 3: classified minerals
    for idx, name in enumerate(mineral_names):
        c = mineral_rgb[idx % len(mineral_rgb)]
        r_val = int(c[1:3], 16) / 255.0
        g_val = int(c[3:5], 16) / 255.0
        b_val = int(c[5:7], 16) / 255.0
        mask = class_map == (idx + 1)
        rgba[mask] = [r_val, g_val, b_val, 1.0]

    # --- Layout: classification map (large) + SAM angle map + threshold table ---
    fig = plt.figure(figsize=(7.5, 10.0))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1],
                           hspace=0.22, wspace=0.25)

    # (a) Classification map — spans left column
    ax_class = fig.add_subplot(gs[:, 0])
    ax_class.imshow(rgba, interpolation='nearest')
    ax_class.set_title('Mineral classification (SAM)', fontsize=9)
    ax_class.text(0.02, 0.98, '(a)', transform=ax_class.transAxes,
                  fontsize=10, fontweight='bold', va='top',
                  bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax_class.axis('off')

    # Legend
    legend_elements = [
        Patch(fc='white', ec='0.5', lw=0.4, label='No data'),
        Patch(fc=(0.92, 0.92, 0.92), ec='0.5', lw=0.4, label='Non-soil (masked)'),
        Patch(fc=(0.75, 0.75, 0.75), ec='0.5', lw=0.4, label='Soil (unclassified)'),
    ]
    for idx, name in enumerate(mineral_names):
        c = mineral_rgb[idx % len(mineral_rgb)]
        short_name = short_mineral_name(name)
        legend_elements.append(
            Patch(fc=c, ec='0.3', lw=0.4, label=short_name))

    ax_class.legend(handles=legend_elements,
                    loc='lower right', fontsize=6.5,
                    frameon=True, fancybox=False, edgecolor='0.4',
                    handlelength=1.0, handleheight=0.8,
                    borderpad=0.4, labelspacing=0.35)

    # (b) Minimum SAM angle — soil pixels only
    ax_sam = fig.add_subplot(gs[0, 1])
    sam_display = np.ma.masked_where(~soil_mask, min_angle_value)
    max_thr = max(thresholds.values())
    im_sam = ax_sam.imshow(sam_display, cmap='magma_r', vmin=0,
                           vmax=max_thr * 1.5, interpolation='nearest')
    ax_sam.set_title('Minimum SAM angle (soil pixels)', fontsize=9)
    ax_sam.text(0.02, 0.98, '(b)', transform=ax_sam.transAxes,
                fontsize=10, fontweight='bold', va='top',
                bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax_sam.axis('off')
    cbar = plt.colorbar(im_sam, ax=ax_sam, fraction=0.046, pad=0.04)
    cbar.set_label('Angle (rad)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # (c) Threshold summary table
    ax_tbl = fig.add_subplot(gs[1, 1])
    ax_tbl.axis('off')
    ax_tbl.set_title('SAM thresholds per mineral', fontsize=9)
    ax_tbl.text(0.02, 0.98, '(c)', transform=ax_tbl.transAxes,
                fontsize=10, fontweight='bold', va='top',
                bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.8))

    col_labels = ['Mineral', 'Thr (rad)', 'Thr (\u00b0)', 'Pixels', '%']
    table_data = []
    total_classified = 0
    for idx, name in enumerate(mineral_names):
        thr_val = thresholds[name]
        count = int(np.sum(class_map == idx + 1))
        total_classified += count
        short = short_mineral_name(name)
        table_data.append([
            short,
            f'{thr_val:.3f}',
            f'{np.degrees(thr_val):.1f}',
            f'{count:,}',
            f'{count / max(int(np.sum(soil_mask)), 1) * 100:.1f}'
        ])

    tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                       loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    tbl.scale(1.0, 1.2)

    # Style header row
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_text_props(fontweight='bold', fontsize=7)
        cell.set_facecolor('#e0e0e0')
        cell.set_edgecolor('0.4')

    # Color-code mineral rows
    for i, name in enumerate(mineral_names):
        c = mineral_rgb[i % len(mineral_rgb)]
        tbl[i + 1, 0].set_facecolor(c + '30')  # 30 = ~19% alpha in hex
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_edgecolor('0.6')

    # Save
    output_path = Path(output_folder) / "Composite_Classification_Map.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path.name}")

    if not SHOW_PLOTS:
        plt.close()
    
    # Save statistics
    save_statistics(class_map, mineral_names, output_folder)
    
    # Save as ENVI format for GIS
    save_envi_classification(class_map, mineral_names, output_folder)
    
    return class_map


def save_statistics(class_map, mineral_names, output_folder):
    """Save classification statistics"""
    stats_path = Path(output_folder) / "classification_statistics.csv"
    stats_data = []
    
    total_pixels = class_map.size
    
    # Unclassified
    unclassified = np.sum(class_map == 0)
    stats_data.append({
        'Class': 'Unclassified',
        'Code': 0,
        'Pixels': int(unclassified),
        'Percent': f"{(unclassified / total_pixels) * 100:.2f}",
        'Area_percentage': f"{(unclassified / total_pixels) * 100:.2f}"
    })
    
    # Each mineral
    for idx, name in enumerate(mineral_names):
        count = np.sum(class_map == idx + 1)
        stats_data.append({
            'Class': name,
            'Code': idx + 1,
            'Pixels': int(count),
            'Percent': f"{(count / total_pixels) * 100:.2f}",
            'Area_percentage': f"{(count / total_pixels) * 100:.2f}"
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_path, index=False)
    
    print(f"\n  Statistics saved: {stats_path.name}")
    print("\n  Classification Statistics:")
    print("  " + stats_df.to_string(index=False).replace('\n', '\n  '))


def save_envi_classification(class_map, mineral_names, output_folder):
    """Save classification map in ENVI format"""
    import spectral as sp
    
    # Create metadata
    metadata = {
        'lines': class_map.shape[0],
        'samples': class_map.shape[1],
        'bands': 1,
        'data type': 2,  # 16-bit integer
        'interleave': 'bsq',
        'byte order': 0,
        'class names': ['Unclassified'] + mineral_names,
        'classes': len(mineral_names) + 1
    }
    
    # Save
    output_path = Path(output_folder) / "classification_map.hdr"
    sp.envi.save_image(str(output_path), 
                       class_map.astype(np.uint16), 
                       metadata=metadata,
                       force=True)
    
    print(f"  ENVI format saved: classification_map.hdr/.img")


# =============================================================================
# POST-CLASSIFICATION VALIDATION
# =============================================================================

def _impute_nan_bands(spectra):
    """Replace per-band NaN values in-place with that band's column mean.

    If an entire band is NaN (e.g. a water-vapour band), the column is set
    to 0 rather than propagating NaN into downstream computations.

    Parameters
    ----------
    spectra : ndarray (n_pixels, n_bands), float64  – modified in place
    """
    nan_bands = np.where(np.any(np.isnan(spectra), axis=0))[0]
    for b in nan_bands:
        col = spectra[:, b]
        mean_val = np.nanmean(col)
        col[np.isnan(col)] = 0.0 if np.isnan(mean_val) else mean_val


def _compute_morans_i(binary_mask, soil_r, soil_c,
                      lps_weights, Moran):
    """Compute Moran's I for *binary_mask* restricted to the soil domain.

    Connectivity is Queen (8-neighbours) within the soil-pixel set; weights
    are row-standardised.  Uses the normal-approximation z-score and p-value
    (no permutations) for speed.

    Parameters
    ----------
    binary_mask : (rows, cols) bool
    soil_r, soil_c : 1-D int arrays – row/col indices of soil pixels
    lps_weights : libpysal.weights module
    Moran       : esda.Moran class

    Returns
    -------
    I, z, p : float
    sig_str : str  – human-readable significance description
    """
    import warnings

    # Build a fast lookup: (row, col) -> index within soil-pixel array
    idx_map = {(int(r), int(c)): i
               for i, (r, c) in enumerate(zip(soil_r, soil_c))}

    # Queen-contiguity neighbour dict restricted to soil pixels only
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0,  -1),          (0,  1),
                  (1,  -1), (1,  0), (1,  1)]
    neighbors    = {}
    weights_dict = {}
    for i, (r, c) in enumerate(zip(soil_r.tolist(), soil_c.tolist())):
        nbrs = [idx_map[(r + dr, c + dc)]
                for dr, dc in directions
                if (r + dr, c + dc) in idx_map]
        neighbors[i]    = nbrs
        weights_dict[i] = [1.0] * len(nbrs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            w = lps_weights.W(neighbors, weights_dict, silence_warnings=True)
        except TypeError:
            w = lps_weights.W(neighbors, weights_dict)
    w.transform = 'r'   # row-standardise

    y = binary_mask[soil_r, soil_c].astype(np.float64)
    if y.std() < 1e-12:
        return np.nan, np.nan, np.nan, "N/A (constant mask)"

    mi = Moran(y, w, permutations=0)
    I, z, p = mi.I, mi.z_norm, mi.p_norm

    if p < 0.05 and I > 0:
        sig = "clustered (p<0.05)"
    elif p < 0.05 and I < 0:
        sig = "dispersed (p<0.05)"
    else:
        sig = f"random (p={p:.3f})"

    return I, z, p, sig


def _save_validation_figure(name, short, soil_mask,
                             binary_mask, labeled, comp_sizes,
                             noise_px_mask, n_components, noise_px_frac,
                             n_noise_comps,
                             hist, bin_edges, bin_centers,
                             mean_a, std_a, skewness, dist_quality,
                             thr, val_dir,
                             null_thr=None, null_angles=None):
    """Save a two-panel validation figure per mineral.

    Left panel  – SAM angle histogram (30 bins, [0, threshold]) with mean
                  and ±1σ markers.
    Right panel – Labeled connected-component map coloured by log(size),
                  with noise components overlaid in red.
    """
    fig, (ax_hist, ax_map) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: SAM angle histogram with optional null-model overlay ────────
    _has_null = (null_thr is not None and null_angles is not None
                 and len(null_angles) > 0 and not np.isnan(null_thr))
    display_max = max(thr, null_thr) * 1.05 if _has_null else thr

    # Null-model background distribution (plotted first, sits behind)
    if _has_null:
        null_hist_vals, null_bin_edges = np.histogram(
            null_angles, bins=30, range=(0.0, float(display_max)))
        null_centers  = 0.5 * (null_bin_edges[:-1] + null_bin_edges[1:])
        null_bin_w_d  = np.degrees(null_bin_edges[1] - null_bin_edges[0])
        # Scale null histogram so its peak sits at 60% of classified histogram peak
        peak_cls  = float(hist.max())  if hist.max()           > 0 else 1.0
        peak_null = float(null_hist_vals.max()) if null_hist_vals.max() > 0 else 1.0
        scale     = peak_cls * 0.6 / peak_null
        ax_hist.bar(np.degrees(null_centers), null_hist_vals * scale,
                    width=null_bin_w_d * 0.88,
                    color='#AAAAAA', edgecolor='white', linewidth=0.3,
                    alpha=0.60, zorder=2,
                    label=f'Null/background (n={len(null_angles):,}, scaled)')

    # Classified pixel angle distribution
    bin_w_deg = np.degrees(bin_edges[1] - bin_edges[0])
    ax_hist.bar(np.degrees(bin_centers), hist,
                width=bin_w_deg * 0.88,
                color='#3A7ABF', edgecolor='white', linewidth=0.3,
                zorder=3,
                label=f'Classified pixels (n={int(hist.sum()):,})')

    # Threshold vertical lines
    ax_hist.axvline(np.degrees(thr),
                    color='#9467BD', linestyle='--', linewidth=1.5, zorder=4,
                    label=f'Adaptive thr = {np.degrees(thr):.2f}\u00b0')
    if _has_null:
        ax_hist.axvline(np.degrees(null_thr),
                        color='#2CA02C', linestyle='-.', linewidth=1.5, zorder=4,
                        label=f'Null thr = {np.degrees(null_thr):.2f}\u00b0')
    ax_hist.axvline(np.degrees(mean_a),
                    color='#D62728', linestyle='--', linewidth=1.5,
                    label=f'Mean = {np.degrees(mean_a):.2f}\u00b0')
    for sign in (-1, 1):
        ax_hist.axvline(np.degrees(mean_a + sign * std_a),
                        color='#FF7F0E', linestyle=':', linewidth=1.2,
                        label='\u00b11\u03c3' if sign == -1 else None)
    ax_hist.set_xlabel('SAM Angle (\u00b0)', fontsize=11)
    ax_hist.set_ylabel('Pixel Count', fontsize=11)
    ax_hist.set_title(
        f'{short} \u2014 SAM Angle Distribution\n'
        f'Skewness = {skewness:.3f}  \u2022  {dist_quality}',
        fontsize=10)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.25)
    ax_hist.set_xlim(0, np.degrees(display_max))

    # ── Right: component size map ─────────────────────────────────────────
    # Base layer: white = nodata, light-gray = soil domain
    base = np.ones((*binary_mask.shape, 3), dtype=np.float32)
    base[soil_mask] = [0.88, 0.88, 0.88]
    ax_map.imshow(base, interpolation='nearest')

    # Component size in log scale for wide dynamic range
    comp_size_map = np.zeros_like(binary_mask, dtype=np.float64)
    for comp_id, size in enumerate(comp_sizes, start=1):
        comp_size_map[labeled == comp_id] = float(size)

    display = np.ma.masked_where(~binary_mask, np.log1p(comp_size_map))
    max_log  = float(np.log1p(comp_sizes.max())) if len(comp_sizes) > 0 else 1.0
    im = ax_map.imshow(display, cmap='viridis', interpolation='nearest',
                       vmin=0, vmax=max_log)
    cbar = plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label('log(1 + component size)  [px]', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Noise overlay (semi-transparent red)
    if noise_px_mask.any():
        noise_rgba = np.zeros((*binary_mask.shape, 4), dtype=np.float32)
        noise_rgba[noise_px_mask] = [0.85, 0.10, 0.10, 0.70]
        ax_map.imshow(noise_rgba, interpolation='nearest')

    ax_map.set_title(
        f'{short} \u2014 Connected Components  (n = {n_components})\n'
        f'Noise (<{MIN_COMPONENT_SIZE} px): {n_noise_comps} comps, '
        f'{noise_px_frac * 100:.1f}% of classified pixels',
        fontsize=10)
    ax_map.axis('off')

    plt.tight_layout()
    safe = name.replace(' ', '_').replace('(', '').replace(')', '')
    fig_path = val_dir / f"{safe}_validation.png"
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    if not SHOW_PLOTS:
        plt.close()


def validate_sam_results(sam_angles_dict, thresholds, soil_mask, cube,
                         mineral_names, output_folder,
                         null_thresholds=None, null_distributions=None,
                         mineral_spectra=None):
    """Post-classification validation of SAM results.

    Four metrics are computed independently for each mineral class:

    1. Connected-component stats
       Labels the binary mask with scipy.ndimage.label and counts pixels per
       component (np.bincount).  Components with fewer than MIN_COMPONENT_SIZE
       pixels are flagged as probable noise.
       Reports: n_components, n_noise_comps, noise_px_frac.

    2. SAM angle distribution
       Extracts spectral-angle values at classified pixel locations from the
       rule image and builds a 30-bin histogram over [0, threshold_angle].
       Reports: mean_angle, std_angle, skewness (left-skewed toward zero =
       good signal; flat/right-skewed = noisy classification).

    3. Angular spectral inertia
       Compares the distribution of per-pixel SAM angles at classified
       locations (α_classified) against angles computed between a random
       background soil sample and the same endmember (α_null).
       Background pixels exclude ALL pixels classified by ANY mineral.
       Sample size is max(n_classified, NULL_SAMPLE_SIZE), capped at the
       number of available background pixels.
       Reports: mean_angle_cls_deg, mean_angle_null_deg,
       angular_inertia_ratio (mean_cls/mean_null; < 1 → good signal),
       mannwhitney_p (one-sided H1: classified angles < null), and
       effect_size r (rank-biserial; > 0 → classified more similar to
       endmember than background).

    4. Moran's I  (requires esda + libpysal)
       Builds Queen-contiguity spatial weights within the soil-mask domain and
       computes Moran's I on the binary mask.
       Reports: I, z-score, p-value.  Significant positive I (p < 0.05)
       supports spatially structured detections.

    Parameters
    ----------
    sam_angles_dict : dict  {mineral_name -> (rows, cols) ndarray}
        SAM rule images; non-soil pixels set to pi.
    thresholds : dict  {mineral_name -> float}
        Per-mineral SAM angle thresholds (radians).
    soil_mask : ndarray (rows, cols), bool
        True for soil pixels eligible for SAM classification.
    cube : ndarray (rows, cols, bands), float32
        Reflectance cube (0-1 scale); may contain NaN in bad bands.
    mineral_names : list of str
        Ordered mineral names matching sam_angles_dict keys.
    output_folder : str or Path
        Root output folder; figures and CSV go to <output_folder>/validation/.
    mineral_spectra : dict {name -> (bands,) ndarray}, optional
        Unit-normalised endmember spectra; required for the angular inertia
        metric.  Passed from main() as the mineral_spectra dict.
    """
    # ── optional dependencies ─────────────────────────────────────────────
    try:
        import libpysal.weights as lps_weights
        from esda import Moran
        HAS_ESDA = True
    except ImportError:
        HAS_ESDA = False
        print("  [validation] WARNING: esda/libpysal not found — "
              "Moran's I skipped.")

    HAS_SPECTRA = mineral_spectra is not None
    if not HAS_SPECTRA:
        print("  [validation] WARNING: mineral_spectra not provided — "
              "angular inertia metric skipped.")

    val_dir = Path(output_folder) / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Pre-compute soil-pixel row/col arrays once (reused for Moran's I)
    soil_r, soil_c = np.where(soil_mask)

    # ── angular inertia: build shared background pool ─────────────────────
    # Background = soil pixels NOT classified by ANY mineral under its
    # adaptive threshold.  Extracted once; per-mineral dot products computed
    # inside the loop against each endmember.
    _ai_bg_spectra = None   # (pool_sz, bands) float64  — None if unavailable
    _ai_bg_norms   = None   # (pool_sz,)        float64

    if HAS_SPECTRA:
        _classified_any_ai = np.zeros(soil_mask.shape, dtype=bool)
        for _nm in mineral_names:
            _classified_any_ai |= (sam_angles_dict[_nm] < thresholds[_nm]) & soil_mask

        _bg_r_ai, _bg_c_ai = np.where(soil_mask & ~_classified_any_ai)
        _n_bg_ai = len(_bg_r_ai)

        if _n_bg_ai >= 2:
            # Pool size: at least NULL_SAMPLE_SIZE or the largest per-mineral
            # classified count, capped at available background pixels so that
            # the null sample always matches or exceeds n_classified.
            _n_cls_per = {
                _nm: int(((sam_angles_dict[_nm] < thresholds[_nm]) & soil_mask).sum())
                for _nm in mineral_names
            }
            _n_max_cls  = max(_n_cls_per.values(), default=0)
            _pool_sz    = min(max(_n_max_cls, NULL_SAMPLE_SIZE), _n_bg_ai)
            np.random.seed(42)
            _pool_idx   = np.random.choice(_n_bg_ai, size=_pool_sz, replace=False)
            _ai_bg_spectra = cube[_bg_r_ai[_pool_idx],
                                  _bg_c_ai[_pool_idx], :].astype(np.float64)
            _impute_nan_bands(_ai_bg_spectra)
            _ai_bg_norms = np.linalg.norm(_ai_bg_spectra, axis=1)
            _ai_bg_norms = np.where(_ai_bg_norms == 0, 1e-10, _ai_bg_norms)
            print(f"  Angular inertia background pool: "
                  f"{_n_bg_ai:,} pixels available, {_pool_sz:,} sampled")
        else:
            print("  [validation] Angular inertia: fewer than 2 background "
                  "pixels available — metric skipped.")

    summary_rows = []

    print("\n" + "=" * 70)
    print("POST-CLASSIFICATION VALIDATION")
    print("=" * 70)

    for name in mineral_names:
        short      = short_mineral_name(name)
        thr        = thresholds[name]
        sam_angles = sam_angles_dict[name]

        # Binary mask: classified AND within soil domain
        binary_mask  = (sam_angles < thr) & soil_mask
        n_classified = int(binary_mask.sum())

        # Null-model threshold info for this mineral (may be None/NaN if skipped)
        null_thr        = (null_thresholds.get(name, np.nan)
                           if null_thresholds else np.nan)
        null_angles_dist = (null_distributions.get(name, np.array([]))
                            if null_distributions else np.array([]))
        _null_valid = null_thresholds is not None and not np.isnan(null_thr)
        if _null_valid:
            null_binary_mask  = (sam_angles < null_thr) & soil_mask
            n_null_classified = int(null_binary_mask.sum())
        else:
            n_null_classified = 0

        if _null_valid:
            print(f"\n  \u2500\u2500 {short} \u2500\u2500  "
                  f"adaptive: {n_classified:,} px  |  "
                  f"null-model ({NULL_MODEL_CONFIDENCE * 100:.0f}%): "
                  f"{n_null_classified:,} px")
        else:
            print(f"\n  \u2500\u2500 {short} \u2500\u2500 ({n_classified} classified pixels)")
        row = {"Mineral": short, "Pixels": n_classified}

        if n_classified == 0:
            print("     No classified pixels \u2014 all metrics skipped.")
            row.update({
                "Components": 0, "Noise_comps": 0, "Noise_px_frac": np.nan,
                "Mean_angle_deg": np.nan, "Std_angle_deg": np.nan,
                "Skewness": np.nan, "Dist_quality": "N/A",
                "Mean_angle_cls_deg": np.nan, "Mean_angle_null_deg": np.nan,
                "Angular_inertia_ratio": np.nan,
                "MannWhitney_p": np.nan, "Effect_size": np.nan,
                "Morans_I": np.nan, "Morans_z": np.nan,
                "Morans_p": np.nan, "Morans_sig": "N/A",
            })
            summary_rows.append(row)
            continue

        # ── 1. Connected-component statistics ─────────────────────────────
        labeled, n_components = nd_label(binary_mask)
        # Component pixel counts; bincount index 0 = background, skip it
        comp_sizes    = np.bincount(labeled.ravel())[1:]
        noise_ids     = np.where(comp_sizes < MIN_COMPONENT_SIZE)[0] + 1
        n_noise_comps = int(len(noise_ids))
        n_noise_px    = int(comp_sizes[comp_sizes < MIN_COMPONENT_SIZE].sum())
        noise_px_frac = n_noise_px / n_classified
        noise_px_mask = np.isin(labeled, noise_ids)

        print(f"     [1] Connected components: {n_components} total")
        print(f"         Noise (<{MIN_COMPONENT_SIZE} px): "
              f"{n_noise_comps} comps, {n_noise_px} px "
              f"({noise_px_frac * 100:.1f}% of classified pixels)")

        row.update({
            "Components":    n_components,
            "Noise_comps":   n_noise_comps,
            "Noise_px_frac": round(noise_px_frac, 4),
        })

        # ── 2. SAM angle distribution ─────────────────────────────────────
        angles_cls = sam_angles[binary_mask]   # all in [0, thr)
        hist, bin_edges = np.histogram(angles_cls, bins=30, range=(0.0, thr))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        mean_a = float(np.mean(angles_cls))
        std_a  = float(np.std(angles_cls))
        skewness = (float(np.mean(((angles_cls - mean_a) / std_a) ** 3))
                    if std_a > 1e-12 else 0.0)

        if skewness < -0.3:
            dist_quality = "left-skewed \u2192 good signal"
        elif skewness > 0.3:
            dist_quality = "right-skewed \u2192 noisy"
        else:
            dist_quality = "symmetric"

        print(f"     [2] SAM angle distribution:")
        print(f"         Mean = {np.degrees(mean_a):.2f}\u00b0,  "
              f"Std = {np.degrees(std_a):.2f}\u00b0,  "
              f"Skewness = {skewness:.3f}  ({dist_quality})")

        row.update({
            "Mean_angle_deg": round(np.degrees(mean_a), 3),
            "Std_angle_deg":  round(np.degrees(std_a),  3),
            "Skewness":       round(skewness, 4),
            "Dist_quality":   dist_quality,
        })

        # ── 3. Angular spectral inertia ───────────────────────────────────
        # Compare α_classified (SAM angles at classified pixels) against
        # α_null (SAM angles of random background soil vs same endmember).
        # Ratio = mean(α_cls) / mean(α_null): < 1 → classified pixels are
        # on average more similar to the endmember than background soil.
        # Mann-Whitney U (one-sided, H1: α_cls < α_null) quantifies whether
        # this separation is statistically significant.
        # Effect size r = 1 − 2U/(n1·n2): +1 → classified << null (good).
        mean_cls_deg = mean_null_deg = ang_ratio = mwu_p = effect_r = np.nan
        _angles_null_ai = None

        if _ai_bg_spectra is not None and n_classified >= 2:
            endmember = mineral_spectra[name]          # unit-normalised
            _cos_null  = (_ai_bg_spectra @ endmember) / _ai_bg_norms
            _cos_null  = np.clip(_cos_null, -1.0, 1.0)
            _angles_null_ai = np.arccos(_cos_null)

            _mean_cls_rad  = float(np.mean(angles_cls))
            _mean_null_rad = float(np.mean(_angles_null_ai))
            mean_cls_deg   = float(np.degrees(_mean_cls_rad))
            mean_null_deg  = float(np.degrees(_mean_null_rad))
            ang_ratio      = (_mean_cls_rad / _mean_null_rad
                              if _mean_null_rad > 0 else np.nan)

            # One-sided Mann-Whitney: H1 = classified angles stochastically
            # less than null angles.  U counts (cls_i < null_j) pairs.
            # Small U → large p; large U → small p (supports H1).
            _U, mwu_p = mannwhitneyu(angles_cls, _angles_null_ai,
                                     alternative='less')
            n1, n2   = len(angles_cls), len(_angles_null_ai)
            # Rank-biserial r: +1 = all classified < all null (perfect)
            #                   0 = no difference
            #                  -1 = all classified > all null (inverted)
            effect_r = 1.0 - (2.0 * _U) / (n1 * n2)

            _coherence = "< null (good)" if ang_ratio < 1.0 else ">= null (poor)"
            print(f"     [3] Angular inertia:")
            print(f"         mean(classified) = {mean_cls_deg:.3f}\u00b0,  "
                  f"mean(null) = {mean_null_deg:.3f}\u00b0,  "
                  f"ratio = {ang_ratio:.4f}  ({_coherence})")
            print(f"         Mann-Whitney p = {mwu_p:.4e},  "
                  f"effect size r = {effect_r:.4f}")
        elif _ai_bg_spectra is not None:
            print("     [3] Angular inertia: < 2 classified pixels — skipped.")
        else:
            print("     [3] Angular inertia: skipped "
                  "(mineral_spectra not provided or no background pixels).")

        row.update({
            "Mean_angle_cls_deg":    (round(mean_cls_deg,  3)
                                      if not np.isnan(mean_cls_deg)  else np.nan),
            "Mean_angle_null_deg":   (round(mean_null_deg, 3)
                                      if not np.isnan(mean_null_deg) else np.nan),
            "Angular_inertia_ratio": (round(ang_ratio,     4)
                                      if not np.isnan(ang_ratio)     else np.nan),
            "MannWhitney_p":         (round(mwu_p,         6)
                                      if not np.isnan(mwu_p)         else np.nan),
            "Effect_size":           (round(effect_r,      4)
                                      if not np.isnan(effect_r)      else np.nan),
        })

        # ── 4. Moran's I spatial autocorrelation ─────────────────────────
        moran_I = moran_z = moran_p = np.nan
        moran_sig = "N/A"

        if HAS_ESDA and len(soil_r) >= 4:
            try:
                moran_I, moran_z, moran_p, moran_sig = _compute_morans_i(
                    binary_mask, soil_r, soil_c,
                    lps_weights, Moran)
                print(f"     [4] Moran's I = {moran_I:.4f},  "
                      f"z = {moran_z:.2f},  p = {moran_p:.4f}"
                      f"  ({moran_sig})")
            except Exception as exc:
                print(f"     [4] Moran's I: failed ({exc})")
        elif not HAS_ESDA:
            print("     [4] Moran's I: skipped (esda/libpysal unavailable).")
        else:
            print("     [4] Moran's I: insufficient soil pixels.")

        row.update({
            "Morans_I":   (round(moran_I,   4)
                           if not np.isnan(moran_I)   else np.nan),
            "Morans_z":   (round(moran_z,   4)
                           if not np.isnan(moran_z)   else np.nan),
            "Morans_p":   (round(moran_p,   4)
                           if not np.isnan(moran_p)   else np.nan),
            "Morans_sig": moran_sig,
        })

        # Null-model threshold columns
        row.update({
            "Null_thr_deg": (round(np.degrees(null_thr), 3)
                             if _null_valid else np.nan),
            "Null_pixels":  n_null_classified,
        })

        summary_rows.append(row)

        # ── Validation figure ─────────────────────────────────────────────
        # Prefer the angular-inertia null distribution for the histogram
        # overlay (same background population, correct endmember angles).
        # Fall back to null_angles_dist from derive_null_thresholds when
        # mineral_spectra was not provided.
        _fig_null_angles = (_angles_null_ai
                            if _angles_null_ai is not None
                            else (null_angles_dist if _null_valid else None))
        _save_validation_figure(
            name, short, soil_mask,
            binary_mask, labeled, comp_sizes,
            noise_px_mask, n_components, noise_px_frac, n_noise_comps,
            hist, bin_edges, bin_centers,
            mean_a, std_a, skewness, dist_quality,
            thr, val_dir,
            null_thr=null_thr if _null_valid else None,
            null_angles=_fig_null_angles,
        )
        safe = name.replace(' ', '_').replace('(', '').replace(')', '')
        print(f"     Figure saved: {safe}_validation.png")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        display_cols = [
            "Mineral", "Pixels", "Null_thr_deg", "Null_pixels",
            "Components", "Noise_px_frac",
            "Mean_angle_deg", "Std_angle_deg", "Skewness",
            "Mean_angle_cls_deg", "Mean_angle_null_deg",
            "Angular_inertia_ratio", "MannWhitney_p", "Effect_size",
            "Morans_I", "Morans_p", "Morans_sig",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        print(df[display_cols].to_string(index=False))

        csv_path = val_dir / "validation_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Full table saved: {csv_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
