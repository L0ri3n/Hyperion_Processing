"""
Multi-Mineral SAM Analysis - EXPLICIT CLASSIFICATION VERSION
Fixed approach with proper handling of multi-class classification
"""

import numpy as np
import pandas as pd
import spectral as sp
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
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

SAVE_INDIVIDUAL_MAPS = True
SAVE_COMPOSITE_MAP = True
SHOW_PLOTS = False

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
    '#E66101',  # orange        – Alunite
    '#5E3C99',  # purple        – Goethite
    '#D62728',  # red           – Hematite
    '#1F77B4',  # blue          – Jarosite (Na)
    '#17BECF',  # cyan          – Jarosite (K)
    '#BCBD22',  # yellow-green  – Pyrite
    '#2CA02C',  # green         – Schwertmannite
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
# MAIN PROCESSING
# =============================================================================

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
    print(f"\n7. Deriving thresholds (margin = {SAM_THRESHOLD_MARGIN:.0%} above minimum angle)...")
    thresholds = {}
    for name in mineral_names:
        soil_angles = sam_angles_dict[name][soil_mask]
        min_angle = np.min(soil_angles) if len(soil_angles) > 0 else np.pi
        thresholds[name] = min_angle * (1 + SAM_THRESHOLD_MARGIN)
        print(f"     {name}: {thresholds[name]:.4f} rad ({np.degrees(thresholds[name]):.1f}°)")

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
    ax.legend(handles=legend_patches, loc='lower right', fontsize=7,
              frameon=True, fancybox=False, edgecolor='0.4',
              handlelength=1.2, handleheight=0.9)

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
    ax.legend(handles=legend_patches_d, loc='lower right', fontsize=7,
              frameon=True, fancybox=False, edgecolor='0.4',
              handlelength=1.2, handleheight=0.9)

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


if __name__ == "__main__":
    main()
