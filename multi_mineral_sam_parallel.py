"""
Multi-Mineral SAM Analysis - PARALLEL VERSION
Processes multiple minerals simultaneously for faster execution
"""

import numpy as np
import pandas as pd
import spectral as sp
from spectral.algorithms import spectral_angles
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from multiprocessing import Pool, cpu_count
from functools import partial

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
HDR_FILE = r"C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT\EO1H2020342013284110KF_reflectance.hdr"
LIBRARY_FOLDER = r"C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\QGIS_TEST\Library_folder"
OUTPUT_FOLDER = r"C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SAM_Results"

# SAM Parameters
SAM_THRESHOLD = 0.1  # Radians (~5.7°)

# Parallel Processing
USE_PARALLEL = True  # Set to False to disable parallel processing
N_PROCESSES = None   # None = use all available CPUs, or set specific number

# Visualization
SAVE_INDIVIDUAL_MAPS = True
SAVE_COMPOSITE_MAP = True
SHOW_PLOTS = False

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_and_resample_spectrum(csv_path, wavelengths):
    """Load and resample mineral spectrum to image wavelengths"""
    mineral_name = Path(csv_path).stem
    
    df = pd.read_csv(csv_path)
    spec_wl = df.iloc[:, 0].values
    spec_ref = df.iloc[:, 1].values
    
    f = interp1d(spec_wl, spec_ref,
                 bounds_error=False,
                 fill_value="extrapolate")
    
    ref_spectrum = f(wavelengths)
    ref_spectrum /= np.linalg.norm(ref_spectrum)
    
    return ref_spectrum, mineral_name


def process_single_mineral(args):
    """
    Process a single mineral spectrum (for parallel processing)
    
    Parameters:
    -----------
    args : tuple
        (csv_file, cube, wavelengths, threshold, output_folder)
    
    Returns:
    --------
    result : dict
        Contains mineral_name, match_score, statistics
    """
    csv_file, cube, wavelengths, threshold, output_folder, save_maps = args
    
    # Load spectrum
    ref_spectrum, mineral_name = load_and_resample_spectrum(csv_file, wavelengths)
    
    # Compute SAM
    sam_angles = spectral_angles(cube, ref_spectrum.reshape(1, -1)).squeeze()
    match_score = np.clip(1 - sam_angles / threshold, 0, 1)
    
    # Statistics
    pixels_detected = np.sum(match_score > 0.5)
    percent_detected = (pixels_detected / match_score.size) * 100
    mean_score = np.mean(match_score)
    max_score = np.max(match_score)
    
    # Save individual map
    output_path = None
    if save_maps:
        output_path = save_match_map(match_score, mineral_name, threshold, output_folder)
    
    return {
        'mineral_name': mineral_name,
        'match_score': match_score,
        'csv_file': csv_file,
        'pixels_detected': pixels_detected,
        'percent_detected': percent_detected,
        'mean_score': mean_score,
        'max_score': max_score,
        'output_path': output_path
    }


def save_match_map(match_score, mineral_name, threshold, output_folder):
    """Save individual mineral match map as PNG"""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(match_score, cmap='inferno', vmin=0, vmax=1)
    ax.set_title(f'{mineral_name} Presence (SAM)\nThreshold: {threshold:.3f} rad ({np.degrees(threshold):.1f}°)',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Match Score\n(0 = no match, 1 = perfect match)', fontsize=11)
    
    plt.tight_layout()
    
    output_path = Path(output_folder) / f"{mineral_name}_SAM_match.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if not SHOW_PLOTS:
        plt.close()
    
    return output_path


def create_composite_map(results_list, threshold, output_folder):
    """
    Create composite map from processing results
    
    Parameters:
    -----------
    results_list : list of dict
        Results from process_single_mineral
    threshold : float
        SAM threshold
    output_folder : str
        Output directory
    """
    # Extract data
    mineral_names = [r['mineral_name'] for r in results_list]
    match_scores_dict = {r['mineral_name']: r['match_score'] for r in results_list}
    
    # Stack scores
    rows, cols = list(match_scores_dict.values())[0].shape
    n_minerals = len(mineral_names)
    
    all_scores = np.zeros((rows, cols, n_minerals))
    for idx, name in enumerate(mineral_names):
        all_scores[:, :, idx] = match_scores_dict[name]
    
    # Find dominant mineral
    max_idx = np.argmax(all_scores, axis=2)
    max_score = np.max(all_scores, axis=2)
    
    # Classification map (0 = background, 1+ = minerals)
    composite_map = np.where(max_score > 0.5, max_idx + 1, 0)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    n_classes = n_minerals + 1
    cmap = plt.cm.get_cmap('tab20', n_classes)
    
    im = ax.imshow(composite_map, cmap=cmap, vmin=0, vmax=n_classes-1)
    ax.set_title(f'Composite Mineral Map (SAM)\n'
                 f'Threshold: {threshold:.3f} rad ({np.degrees(threshold):.1f}°)',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cmap(0), label='No detection')]
    for idx, name in enumerate(mineral_names):
        legend_elements.append(Patch(facecolor=cmap(idx+1), label=name))
    
    ax.legend(handles=legend_elements, 
             loc='center left', 
             bbox_to_anchor=(1, 0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    output_path = Path(output_folder) / f"Composite_SAM_map.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Composite map saved: {output_path.name}")
    
    if not SHOW_PLOTS:
        plt.close()
    
    # Statistics
    save_classification_statistics(composite_map, mineral_names, output_folder)
    
    return composite_map


def save_classification_statistics(composite_map, mineral_names, output_folder):
    """Save classification statistics to CSV"""
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


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """Main processing workflow"""
    print("=" * 70)
    print("MULTI-MINERAL SAM ANALYSIS (PARALLEL)")
    print("=" * 70)
    
    # Setup
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # Load image
    print("\n1. Loading hyperspectral image...")
    img = sp.open_image(HDR_FILE)
    cube = img.load().astype(np.float32)
    rows, cols, bands = cube.shape
    wavelengths = np.array(img.metadata['wavelength'], dtype=float)
    
    print(f"   Image shape: {rows} x {cols} x {bands}")
    print(f"   Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
    
    # Find mineral library
    print(f"\n2. Loading mineral library from: {LIBRARY_FOLDER}")
    csv_files = sorted(glob.glob(str(Path(LIBRARY_FOLDER) / "*.csv")))
    
    if not csv_files:
        print(f"   ERROR: No CSV files found in {LIBRARY_FOLDER}")
        return
    
    print(f"   Found {len(csv_files)} mineral spectra")
    
    # Determine number of processes
    n_cpus = cpu_count()
    n_proc = N_PROCESSES if N_PROCESSES else n_cpus
    
    if USE_PARALLEL and len(csv_files) > 1:
        print(f"\n3. Computing SAM in PARALLEL ({n_proc} processes)...")
        print(f"   Threshold: {SAM_THRESHOLD:.3f} rad ({np.degrees(SAM_THRESHOLD):.1f}°)")
        
        # Prepare arguments for parallel processing
        args_list = [(csv_file, cube, wavelengths, SAM_THRESHOLD, 
                     OUTPUT_FOLDER, SAVE_INDIVIDUAL_MAPS) 
                     for csv_file in csv_files]
        
        # Process in parallel
        with Pool(processes=n_proc) as pool:
            results = pool.map(process_single_mineral, args_list)
        
        print(f"\n   Processed {len(results)} minerals")
        
    else:
        print(f"\n3. Computing SAM SEQUENTIALLY...")
        print(f"   Threshold: {SAM_THRESHOLD:.3f} rad ({np.degrees(SAM_THRESHOLD):.1f}°)")
        
        results = []
        for csv_file in csv_files:
            args = (csv_file, cube, wavelengths, SAM_THRESHOLD, 
                   OUTPUT_FOLDER, SAVE_INDIVIDUAL_MAPS)
            result = process_single_mineral(args)
            results.append(result)
            print(f"   Processed: {result['mineral_name']}")
    
    # Print individual results
    print("\n   Individual mineral results:")
    for result in results:
        print(f"\n   {result['mineral_name']}:")
        print(f"     Pixels detected (score > 0.5): {result['pixels_detected']} "
              f"({result['percent_detected']:.2f}%)")
        print(f"     Mean match score: {result['mean_score']:.3f}")
        print(f"     Max match score: {result['max_score']:.3f}")
    
    # Create composite map
    if SAVE_COMPOSITE_MAP and len(results) > 0:
        print("\n4. Creating composite map...")
        composite_map = create_composite_map(results, SAM_THRESHOLD, OUTPUT_FOLDER)
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_FOLDER}")
    
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
