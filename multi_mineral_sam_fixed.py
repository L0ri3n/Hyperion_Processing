"""
Multi-Mineral SAM Analysis - EXPLICIT CLASSIFICATION VERSION
Fixed approach with proper handling of multi-class classification
"""

import numpy as np
import pandas as pd
import spectral as sp
from scipy.interpolate import interp1d
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

SAM_DEFAULT_THRESHOLD = 0.85   # Radians (~5.7°) — used when no per-mineral value is set

# Per-mineral thresholds (radians). Keys must match CSV filenames (without .csv).
# Any mineral not listed here will use SAM_DEFAULT_THRESHOLD.
SAM_THRESHOLDS = {
     "Goethite_HS36.3_BECKb":                  0.90,
     "Hematite_GDS27_BECKa":                   1.20,
     "Jarosite_GDS100_Na_90C_Syn_BECKa":       0.80,
     "Pyrite_HS35.3_BECKb":                    0.70,
     "Schwertmannite_BZ93-1_BECKb":            0.95,
     "Alunite_GDS84_Na03_BECKa":               0.70,
}

SAVE_INDIVIDUAL_MAPS = True
SAVE_COMPOSITE_MAP = True
SHOW_PLOTS = False

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

    # Resolve per-mineral thresholds
    thresholds = {}
    for name in mineral_names:
        thresholds[name] = SAM_THRESHOLDS.get(name, SAM_DEFAULT_THRESHOLD)

    # Compute SAM for all minerals
    print(f"\n4. Computing SAM angles...")
    for name in mineral_names:
        print(f"     {name}: threshold = {thresholds[name]:.3f} rad ({np.degrees(thresholds[name]):.1f}°)")

    # Store all SAM results
    sam_angles_dict = {}
    match_scores_dict = {}

    for mineral_name in mineral_names:
        thr = thresholds[mineral_name]
        print(f"\n   Processing: {mineral_name} (threshold: {thr:.3f} rad)")

        # Compute SAM angles using manual method
        ref_spectrum = mineral_spectra[mineral_name]
        sam_angles = compute_sam_angles_manual(cube, ref_spectrum)

        # Compute match score
        match_score = np.clip(1 - sam_angles / thr, 0, 1)

        # Store
        sam_angles_dict[mineral_name] = sam_angles
        match_scores_dict[mineral_name] = match_score

        # Statistics
        pixels_below_threshold = np.sum(sam_angles < thr)
        percent_below = (pixels_below_threshold / sam_angles.size) * 100

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
        print("\n5. Creating composite classification map...")
        create_composite_map(sam_angles_dict, match_scores_dict,
                           mineral_names, thresholds, OUTPUT_FOLDER)
    
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


def save_individual_map(match_score, sam_angles, mineral_name, threshold, output_folder):
    """Save individual mineral detection map"""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Match score (like your working script)
    im1 = ax1.imshow(match_score, cmap='inferno', vmin=0, vmax=1)
    ax1.set_title(f'{mineral_name} - Match Score\nThreshold: {threshold:.3f} rad ({np.degrees(threshold):.1f}°)',
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Match Score (0=no match, 1=perfect)', fontsize=10)
    
    # Plot 2: Binary detection
    binary_detection = (sam_angles < threshold).astype(float)
    im2 = ax2.imshow(binary_detection, cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_title(f'{mineral_name} - Binary Detection\n(Green = Detected, Red = Not Detected)',
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Detection (0=absent, 1=present)', fontsize=10)
    
    plt.tight_layout()
    
    output_path = Path(output_folder) / f"{mineral_name}_SAM_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"     Saved: {output_path.name}")
    
    if not SHOW_PLOTS:
        plt.close()


def create_composite_map(sam_angles_dict, match_scores_dict, mineral_names, thresholds, output_folder):
    """
    Create composite classification map using proper SAM classification logic

    Classification rules:
    1. For each pixel, mask out minerals whose angle >= their own threshold
    2. Among the remaining candidates, pick the mineral with the smallest angle
    3. If no mineral passes its threshold, the pixel is unclassified (0)

    Parameters:
    -----------
    thresholds : dict
        Mapping of mineral_name -> threshold (radians)
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
    # Class 0 = unclassified (no mineral passed its threshold)
    # Class 1, 2, 3... = minerals
    class_map = np.where(any_pass,
                         min_angle_idx + 1,  # Mineral class (1-indexed)
                         0)                   # Unclassified

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot 1: Classification map
    n_classes = n_minerals + 1
    cmap = plt.cm.get_cmap('tab20', n_classes)

    im1 = ax1.imshow(class_map, cmap=cmap, vmin=0, vmax=n_classes-1)
    ax1.set_title('Composite Mineral Classification Map', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Legend with thresholds embedded in labels
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cmap(0), label='Unclassified')]
    for idx, name in enumerate(mineral_names):
        thr_val = thresholds[name]
        legend_elements.append(
            Patch(facecolor=cmap(idx+1),
                  label=f'{name}  (thr={thr_val:.2f} rad / {np.degrees(thr_val):.1f}\u00b0)'))

    ax1.legend(handles=legend_elements,
              loc='upper left',
              bbox_to_anchor=(0, -0.02),
              fontsize=9, ncol=2, frameon=True)

    # Plot 2: Minimum SAM angle map
    max_thr = max(thresholds.values())
    im2 = ax2.imshow(min_angle_value, cmap='viridis', vmin=0, vmax=max_thr*2)
    ax2.set_title('Minimum SAM Angle Map\n(Lower = Better Match)',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('SAM Angle (radians)', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_folder) / "Composite_Classification_Map.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
