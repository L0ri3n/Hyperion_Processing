"""
Multi-Mineral SAM Analysis
Processes a library of mineral spectra and generates:
1. Individual match score maps for each mineral
2. Composite map showing all minerals present
"""

import numpy as np
import pandas as pd
import spectral as sp
from spectral.algorithms import spectral_angles
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
HDR_FILE = r"C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT\EO1H2020342013284110KF_reflectance.hdr"
LIBRARY_FOLDER = r"C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\QGIS_TEST\Library_folder"
OUTPUT_FOLDER = r"C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SAM_Results"

# SAM Parameters
SAM_THRESHOLD = 0.1  # Radians (~5.7°)

# Visualization
SAVE_INDIVIDUAL_MAPS = True
SAVE_COMPOSITE_MAP = True
SHOW_PLOTS = False  # Set to True if you want to see plots during processing

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_and_resample_spectrum(csv_path, wavelengths):
    """
    Load a mineral spectrum from CSV and resample to image wavelengths
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file (column 0: wavelength, column 1: reflectance)
    wavelengths : array
        Target wavelengths from the hyperspectral image
    
    Returns:
    --------
    ref_spectrum : array
        Normalized reflectance spectrum at target wavelengths
    mineral_name : str
        Name extracted from filename
    """
    # Extract mineral name from filename
    mineral_name = Path(csv_path).stem
    
    # Load spectrum
    df = pd.read_csv(csv_path)
    spec_wl = df.iloc[:, 0].values
    spec_ref = df.iloc[:, 1].values
    
    # Interpolate to image wavelengths
    f = interp1d(spec_wl, spec_ref,
                 bounds_error=False,
                 fill_value="extrapolate")
    
    ref_spectrum = f(wavelengths)
    
    # Normalize (critical for SAM!)
    ref_spectrum /= np.linalg.norm(ref_spectrum)
    
    return ref_spectrum, mineral_name


def compute_sam_match_score(cube, ref_spectrum, threshold):
    """
    Compute SAM and convert to match score
    
    Parameters:
    -----------
    cube : array (rows, cols, bands)
        Hyperspectral image cube
    ref_spectrum : array (bands,)
        Reference spectrum (normalized)
    threshold : float
        SAM threshold in radians
    
    Returns:
    --------
    match_score : array (rows, cols)
        Match score: 1 = perfect match, 0 = at/beyond threshold
    sam_angles : array (rows, cols)
        Raw SAM angles in radians
    """
    # Compute SAM angles
    sam_angles = spectral_angles(cube, ref_spectrum.reshape(1, -1)).squeeze()
    
    # Convert to match score (0-1 scale)
    match_score = np.clip(1 - sam_angles / threshold, 0, 1)
    
    return match_score, sam_angles


def save_match_map(match_score, mineral_name, threshold, output_folder):
    """
    Save individual mineral match map as PNG
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(match_score, cmap='inferno', vmin=0, vmax=1)
    ax.set_title(f'{mineral_name} Presence (SAM)\nThreshold: {threshold:.3f} rad ({np.degrees(threshold):.1f}°)',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Match Score\n(0 = no match, 1 = perfect match)', 
                   fontsize=11)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_folder) / f"{mineral_name}_SAM_match.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    
    if not SHOW_PLOTS:
        plt.close()
    
    return output_path


def create_composite_map(match_scores_dict, mineral_names, threshold, output_folder):
    """
    Create a composite map showing all minerals detected
    
    Parameters:
    -----------
    match_scores_dict : dict
        Dictionary of {mineral_name: match_score_array}
    mineral_names : list
        Ordered list of mineral names
    threshold : float
        SAM threshold used
    output_folder : str
        Output directory
    
    Returns:
    --------
    composite_map : array
        Map where each pixel shows which mineral (if any) has highest match score
    """
    # Stack all match scores into 3D array
    rows, cols = list(match_scores_dict.values())[0].shape
    n_minerals = len(mineral_names)
    
    all_scores = np.zeros((rows, cols, n_minerals))
    for idx, name in enumerate(mineral_names):
        all_scores[:, :, idx] = match_scores_dict[name]
    
    # Find mineral with maximum match score at each pixel
    max_idx = np.argmax(all_scores, axis=2)
    max_score = np.max(all_scores, axis=2)
    
    # Create classification map
    # 0 = no mineral detected (all scores below threshold)
    # 1, 2, 3... = mineral classes
    composite_map = np.where(max_score > 0.5, max_idx + 1, 0)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create custom colormap
    n_classes = n_minerals + 1  # +1 for background
    cmap = plt.cm.get_cmap('tab20', n_classes)
    
    im = ax.imshow(composite_map, cmap=cmap, vmin=0, vmax=n_classes-1)
    ax.set_title(f'Composite Mineral Map (SAM)\n'
                 f'Threshold: {threshold:.3f} rad ({np.degrees(threshold):.1f}°)',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cmap(0), label='No detection')]
    for idx, name in enumerate(mineral_names):
        legend_elements.append(Patch(facecolor=cmap(idx+1), label=name))
    
    ax.legend(handles=legend_elements, 
             loc='center left', 
             bbox_to_anchor=(1, 0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_folder) / f"Composite_SAM_map.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Composite map saved: {output_path.name}")
    
    if not SHOW_PLOTS:
        plt.close()
    
    # Save statistics
    stats_path = Path(output_folder) / "classification_statistics.csv"
    stats_data = []
    
    total_pixels = composite_map.size
    unclassified = np.sum(composite_map == 0)
    stats_data.append({
        'Class': 'Unclassified',
        'Code': 0,
        'Pixels': int(unclassified),
        'Percent': f"{(unclassified / total_pixels) * 100:.2f}"
    })
    
    for idx, name in enumerate(mineral_names):
        count = np.sum(composite_map == idx + 1)
        stats_data.append({
            'Class': name,
            'Code': idx + 1,
            'Pixels': int(count),
            'Percent': f"{(count / total_pixels) * 100:.2f}"
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_path, index=False)
    print(f"  Statistics saved: {stats_path.name}")
    print("\nClassification Statistics:")
    print(stats_df.to_string(index=False))
    
    return composite_map


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """
    Main processing workflow
    """
    print("=" * 70)
    print("MULTI-MINERAL SAM ANALYSIS")
    print("=" * 70)
    
    # Create output folder
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # Load hyperspectral image
    print("\n1. Loading hyperspectral image...")
    img = sp.open_image(HDR_FILE)
    cube = img.load().astype(np.float32)
    rows, cols, bands = cube.shape
    wavelengths = np.array(img.metadata['wavelength'], dtype=float)
    
    print(f"   Image shape: {rows} x {cols} x {bands}")
    print(f"   Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
    
    # Find all CSV files in library folder
    print(f"\n2. Loading mineral library from: {LIBRARY_FOLDER}")
    csv_files = glob.glob(str(Path(LIBRARY_FOLDER) / "*.csv"))
    
    if not csv_files:
        print(f"   ERROR: No CSV files found in {LIBRARY_FOLDER}")
        return
    
    print(f"   Found {len(csv_files)} mineral spectra")
    
    # Process each mineral
    print(f"\n3. Computing SAM for each mineral (threshold: {SAM_THRESHOLD:.3f} rad)...")
    
    match_scores_dict = {}
    mineral_names = []
    
    for csv_file in sorted(csv_files):
        # Load and resample spectrum
        ref_spectrum, mineral_name = load_and_resample_spectrum(csv_file, wavelengths)
        mineral_names.append(mineral_name)
        
        print(f"\n   Processing: {mineral_name}")
        
        # Compute SAM
        match_score, sam_angles = compute_sam_match_score(cube, ref_spectrum, SAM_THRESHOLD)
        match_scores_dict[mineral_name] = match_score
        
        # Report statistics
        pixels_detected = np.sum(match_score > 0.5)
        percent_detected = (pixels_detected / match_score.size) * 100
        print(f"     Pixels detected (score > 0.5): {pixels_detected} ({percent_detected:.2f}%)")
        print(f"     Mean match score: {np.mean(match_score):.3f}")
        print(f"     Max match score: {np.max(match_score):.3f}")
        
        # Save individual map
        if SAVE_INDIVIDUAL_MAPS:
            save_match_map(match_score, mineral_name, SAM_THRESHOLD, OUTPUT_FOLDER)
    
    # Create composite map
    if SAVE_COMPOSITE_MAP:
        print("\n4. Creating composite map...")
        composite_map = create_composite_map(match_scores_dict, 
                                             mineral_names, 
                                             SAM_THRESHOLD, 
                                             OUTPUT_FOLDER)
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_FOLDER}")
    
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
