"""
AMD Mineral Mapping with Hyperion: Implementation Plan
Step-by-step workflow using Python, QGIS, and SNAP
Starting from Step 2 (Step 1 completed with SUREHYP)
"""

# ============================================================================
# STEP 2: BUILD MINERAL SPECTRAL LIBRARY
# ============================================================================

"""
GOAL: Create a Hyperion-compatible spectral library without field data
TOOLS: Python (spectral, numpy, pandas)
OUTPUT: Resampled endmember spectra at Hyperion wavelengths
"""

# Step 2.1: Download USGS Spectral Library
# ------------------------------------------
# Source: https://www.usgs.gov/labs/spec-lab/capabilities/spectral-library (Kokaly, R.F., Clark, R.N., Swayze, G.A., Livo, K.E., Hoefen, T.M., Pearson, N.C., Wise, R.A., Benzel, W.M., Lowers, H.A., Driscoll, R.L., and Klein, A.J., 2017, USGS Spectral Library Version 7 Data: U.S. Geological Survey data release, https://doi.org/10.5066/F7RR1WDJ.)
# Required minerals:
# - Jarosite (various types: K-jarosite, Na-jarosite, H3O-jarosite)
# - Goethite
# - Hematite
# - Kaolinite, illite, smectite (as confusers)
# - Gypsum (common in AMD environments)

# Download files (USGS format: .txt or .asd)
# Place in: ./spectral_library/usgs/

import numpy as np
import pandas as pd
from spectral import *
import matplotlib.pyplot as plt
import sys
import os

# Add the code directory to path to import our USGS loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'amd_mapping', 'code'))

# Import USGS spectral library functions
try:
    from load_usgs_spectrum import load_minerals_spectra
    USGS_LOADER_AVAILABLE = True
except ImportError:
    USGS_LOADER_AVAILABLE = False
    print("[WARNING] USGS loader not available. Using fallback methods.")

def download_usgs_library():
    """
    Load USGS spectral library using the custom loader

    The loader automatically:
    - Loads Hyperion wavelengths from the USGS library
    - Searches for mineral files by name (case-insensitive)
    - Returns dict of {mineral_name: (wavelength, reflectance)}

    To customize minerals, edit MINERALS_TO_LOAD in load_usgs_spectrum.py
    """
    if not USGS_LOADER_AVAILABLE:
        print("[ERROR] USGS loader module not found")
        return {}

    # Load minerals using the custom loader
    spectra = load_minerals_spectra()

    return spectra

# Step 2.2: Collect AMD-Specific Spectra from Publications
# ----------------------------------------------------------
# Key papers for Rio Tinto AMD minerals:
# - Schwertmannite: Bigham et al. (1996), Acero et al. (2006)
# - Jarosite variants in AMD: Alpers et al. (various)
# - Spectral discrimination: Crowley et al. (2003)
#
# Extract spectra from:
# 1. Supplementary data files
# 2. Digitize from figures (if needed, use WebPlotDigitizer)
# Place in: ./spectral_library/publications/

# Step 2.3: Load Hyperion Wavelength Information
# -----------------------------------------------
def load_hyperion_wavelengths(metadata_file):
    """
    Extract wavelengths from Hyperion metadata
    
    Parameters:
    -----------
    metadata_file : str
        Path to Hyperion .hdr or metadata file
    
    Returns:
    --------
    wavelengths : array
        Wavelengths in nanometers for usable bands
    """
    # After SUREHYP processing, wavelengths should be in metadata
    img = envi.open(metadata_file)
    wavelengths = np.array(img.metadata['wavelength'], dtype=float)
    
    # Identify usable bands (excluding bad bands)
    usable_indices = get_usable_bands()
    wavelengths_clean = wavelengths[usable_indices]
    
    return wavelengths_clean

def get_usable_bands():
    """
    Return indices of usable Hyperion bands
    Excludes:
    - Bands 1-7 (VNIR overlap/bad)
    - Bands 58-76 (water vapor ~1360-1400 nm)
    - Bands 123-135 (water vapor ~1800-1950 nm)  
    - Bands 225-242 (SWIR edge/bad)
    """
    all_bands = np.arange(242)
    bad_bands = np.concatenate([
        np.arange(0, 7),      # Bad VNIR
        np.arange(57, 76),    # Water vapor 1
        np.arange(122, 135),  # Water vapor 2
        np.arange(224, 242)   # Bad SWIR
    ])
    usable = np.setdiff1d(all_bands, bad_bands)
    return usable

# Step 2.4: Resample Library Spectra to Hyperion Wavelengths
# -----------------------------------------------------------
def resample_spectrum_to_hyperion(lib_wavelengths, lib_reflectance, 
                                   hyperion_wavelengths):
    """
    Resample library spectrum to Hyperion band centers
    
    Parameters:
    -----------
    lib_wavelengths : array
        Wavelengths from library (nm)
    lib_reflectance : array
        Reflectance values from library
    hyperion_wavelengths : array
        Target Hyperion wavelengths (nm)
    
    Returns:
    --------
    resampled_reflectance : array
        Reflectance at Hyperion wavelengths
    """
    resampled = np.interp(hyperion_wavelengths, lib_wavelengths, lib_reflectance)
    return resampled

def create_endmember_library(library_dir=None, hyperion_wavelengths=None, output_file=None):
    """
    Process all library spectra and save as Hyperion-compatible library

    Parameters:
    -----------
    library_dir : str, optional
        Directory containing library spectra files (ignored if using USGS loader)
    hyperion_wavelengths : array, optional
        Hyperion wavelengths for resampling (USGS loader provides its own)
    output_file : str
        Output file for endmember library (.sli or .csv)

    Returns:
    --------
    endmembers : dict
        Dictionary with {mineral_name: reflectance_array}
    wavelengths : array
        Corresponding wavelengths
    """

    # Method 1: Use USGS loader (preferred)
    if USGS_LOADER_AVAILABLE and library_dir is None:
        print("Loading USGS spectral library with built-in loader...")
        spectra_dict = load_minerals_spectra()

        if not spectra_dict:
            print("[ERROR] No spectra loaded")
            return {}, np.array([])

        # Extract wavelengths (same for all minerals)
        wavelengths = list(spectra_dict.values())[0][0]

        # Create endmember dict with just reflectance values
        endmembers = {}
        for mineral_name, (wvl, ref) in spectra_dict.items():
            endmembers[mineral_name] = ref

        print("Loaded {} minerals with {} bands".format(len(endmembers), len(wavelengths)))

    # Method 2: Fallback to manual loading
    else:
        print("Using manual spectral library loading...")
        import glob

        if library_dir is None or hyperion_wavelengths is None:
            print("[ERROR] library_dir and hyperion_wavelengths required for manual loading")
            return {}, np.array([])

        endmembers = {}
        wavelengths = hyperion_wavelengths

        # Process each spectrum file
        for filepath in glob.glob(os.path.join(library_dir, '*.txt')):
            mineral_name = os.path.basename(filepath).replace('.txt', '')

            # Load spectrum (adjust format as needed)
            data = np.loadtxt(filepath, skiprows=1)
            lib_wvl = data[:, 0]
            lib_ref = data[:, 1]

            # Resample to Hyperion
            resampled = resample_spectrum_to_hyperion(lib_wvl, lib_ref,
                                                       hyperion_wavelengths)
            endmembers[mineral_name] = resampled

    # Save library if output file specified
    if output_file is not None:
        # Format 1: CSV for easy viewing
        df = pd.DataFrame(endmembers, index=wavelengths)
        df.to_csv(output_file.replace('.sli', '.csv'))
        print("Saved CSV library to: {}".format(output_file.replace('.sli', '.csv')))

        # Format 2: ENVI spectral library format
        save_envi_library(endmembers, wavelengths, output_file)
        print("Saved ENVI library to: {}".format(output_file))

    return endmembers, wavelengths

def save_envi_library(endmembers, wavelengths, output_file):
    """
    Save endmembers in ENVI spectral library format
    """
    n_endmembers = len(endmembers)
    n_bands = len(wavelengths)
    
    # Stack endmembers as columns
    library_array = np.column_stack([endmembers[name] for name in endmembers.keys()])
    
    # Save with ENVI format
    metadata = {
        'samples': n_endmembers,
        'lines': 1,
        'bands': n_bands,
        'header offset': 0,
        'file type': 'ENVI Spectral Library',
        'data type': 4,  # 32-bit float
        'interleave': 'bsq',
        'byte order': 0,
        'wavelength': wavelengths.tolist(),
        'spectra names': list(endmembers.keys())
    }
    
    envi.save_image(output_file + '.hdr', library_array.T, metadata=metadata, 
                    force=True, ext='')


# ============================================================================
# STEP 3: ENHANCE SPECTRAL FEATURES (OPTIONAL BUT RECOMMENDED)
# ============================================================================

"""
GOAL: Improve signal-to-noise and emphasize diagnostic features
TOOLS: Python (scipy, pysptools)
OUTPUT: Enhanced Hyperion cube and endmember library
"""

from scipy.signal import savgol_filter

# Step 3.1: Savitzky-Golay Smoothing
# -----------------------------------
def apply_savgol_smoothing(cube, window_length=9, polyorder=2):
    """
    Apply Savitzky-Golay filter to reduce noise while preserving features
    
    Parameters:
    -----------
    cube : array (rows, cols, bands)
        Hyperion image cube
    window_length : int (default=9)
        Size of smoothing window (must be odd)
    polyorder : int (default=2)
        Order of polynomial fit
    
    Returns:
    --------
    smoothed_cube : array
        Smoothed image cube
    """
    rows, cols, bands = cube.shape
    smoothed = np.zeros_like(cube)
    
    for i in range(rows):
        for j in range(cols):
            smoothed[i, j, :] = savgol_filter(cube[i, j, :], 
                                               window_length, polyorder)
    
    return smoothed

# Step 3.2: Continuum Removal
# ----------------------------
def continuum_removal(spectrum):
    """
    Remove spectral continuum to emphasize absorption features
    
    Parameters:
    -----------
    spectrum : array
        Input spectrum (reflectance)
    
    Returns:
    --------
    cr_spectrum : array
        Continuum-removed spectrum
    """
    from pysptools.spectro import continuum_removal as cr
    
    # Apply continuum removal
    cr_spectrum = cr(spectrum)
    return cr_spectrum

def apply_continuum_removal_cube(cube):
    """
    Apply continuum removal to entire cube
    """
    from pysptools.spectro import continuum_removal as cr
    
    rows, cols, bands = cube.shape
    cr_cube = np.zeros_like(cube)
    
    for i in range(rows):
        for j in range(cols):
            if np.all(cube[i, j, :] > 0):  # Only process valid spectra
                cr_cube[i, j, :] = cr(cube[i, j, :])
    
    return cr_cube


# ============================================================================
# STEP 4: RUN SPECTRAL ANGLE MAPPER (SAM)
# ============================================================================

"""
GOAL: Classify pixels based on spectral similarity to endmembers
TOOLS: Python (spectral, numpy)
OUTPUT: SAM classification maps for each mineral
"""

# Method 1: Python implementation (recommended for automation)
# -------------------------------------------------------------

def spectral_angle(pixel, endmember):
    """
    Compute spectral angle in radians

    Parameters:
    -----------
    pixel : array
        Pixel spectrum
    endmember : array
        Endmember spectrum

    Returns:
    --------
    angle : float
        Spectral angle in radians
    """
    dot_product = np.dot(pixel, endmember)
    norm_product = np.linalg.norm(pixel) * np.linalg.norm(endmember)

    # Avoid division by zero and numerical errors
    if norm_product == 0:
        return np.pi  # Maximum angle

    cos_angle = np.clip(dot_product / norm_product, -1, 1)
    return np.arccos(cos_angle)

def run_sam_single_mineral(cube, endmember_spectrum, threshold=0.10):
    """
    Run SAM for single mineral

    Parameters:
    -----------
    cube : array (rows, cols, bands)
        Preprocessed Hyperion cube
    endmember_spectrum : array (bands,)
        Target mineral spectrum
    threshold : float (default=0.10)
        Maximum angle in radians for classification

    Returns:
    --------
    class_map : array (rows, cols)
        1 = classified as mineral, 0 = not classified
    angle_map : array (rows, cols)
        Spectral angle in radians
    """
    rows, cols, bands = cube.shape
    angle_map = np.zeros((rows, cols), dtype=np.float32)

    print(f"Processing {rows} x {cols} pixels...")
    for i in range(rows):
        if i % 50 == 0:
            print(f"  Row {i}/{rows}")
        for j in range(cols):
            pixel = cube[i, j, :]
            if np.any(pixel > 0):  # Valid pixel
                angle_map[i, j] = spectral_angle(pixel, endmember_spectrum)
            else:
                angle_map[i, j] = np.pi  # Invalid

    # Classify based on threshold
    class_map = (angle_map < threshold).astype(np.uint8)

    print(f"Classified pixels: {class_map.sum()} ({class_map.sum()/class_map.size*100:.2f}%)")

    # Only print angle statistics if there are valid pixels
    valid_angles = angle_map[angle_map < np.pi]
    if len(valid_angles) > 0:
        print(f"Mean angle: {valid_angles.mean():.3f} rad")
        print(f"Min angle: {valid_angles.min():.3f} rad")
    else:
        print("Warning: No valid pixels found (all pixels are zero/invalid)")

    return class_map, angle_map

def run_sam_all_minerals(cube, endmember_dict, threshold=0.10):
    """
    Run SAM for all minerals and create multi-class map

    Parameters:
    -----------
    cube : array (rows, cols, bands)
        Preprocessed Hyperion cube
    endmember_dict : dict
        Dictionary with {mineral_name: endmember_spectrum}
    threshold : float (default=0.10)
        Maximum angle in radians for classification

    Returns:
    --------
    class_map : array (rows, cols)
        0 = unclassified, 1-N = mineral classes
    angle_maps : dict
        Dictionary of angle maps for each mineral
    """
    rows, cols, bands = cube.shape
    n_minerals = len(endmember_dict)

    # Compute angle maps for all minerals
    angle_maps = {}
    print("Computing spectral angles...")

    for mineral_name, endmember in endmember_dict.items():
        print(f"  Processing {mineral_name}...")
        angle_map = np.zeros((rows, cols), dtype=np.float32)

        for i in range(rows):
            if i % 50 == 0:
                print(f"    Row {i}/{rows}")
            for j in range(cols):
                pixel = cube[i, j, :]
                if np.any(pixel > 0):
                    angle_map[i, j] = spectral_angle(pixel, endmember)
                else:
                    angle_map[i, j] = np.pi

        angle_maps[mineral_name] = angle_map

    # Create classification map (assign to closest mineral if < threshold)
    print("Creating classification map...")
    class_map = np.zeros((rows, cols), dtype=np.uint8)
    min_angles = np.full((rows, cols), np.pi)

    for idx, (mineral_name, angle_map) in enumerate(angle_maps.items(), 1):
        # Update classification where this mineral has minimum angle
        mask = (angle_map < min_angles) & (angle_map < threshold)
        class_map[mask] = idx
        min_angles[mask] = angle_map[mask]

    # Print statistics
    print("\nClassification results:")
    print(f"Unclassified: {np.sum(class_map == 0)} pixels")
    for idx, mineral_name in enumerate(endmember_dict.keys(), 1):
        count = np.sum(class_map == idx)
        pct = count / class_map.size * 100
        print(f"{mineral_name}: {count} pixels ({pct:.2f}%)")

    return class_map, angle_maps

def run_sam_classification(cube, endmembers, threshold=0.10):
    """
    Run Spectral Angle Mapper classification (wrapper function for compatibility)

    Parameters:
    -----------
    cube : array (rows, cols, bands)
        Preprocessed Hyperion cube
    endmembers : dict or array
        Endmember spectra (minerals)
    threshold : float (default=0.10)
        Maximum angle in radians for classification

    Returns:
    --------
    class_map : array (rows, cols)
        Classification map (0=unclassified, 1-N=mineral classes)
    angle_maps : dict or array
        Spectral angle maps for each endmember
    """
    # Use improved implementation based on input type
    if isinstance(endmembers, dict):
        return run_sam_all_minerals(cube, endmembers, threshold)
    else:
        # Convert array to dict if needed
        endmember_dict = {f"mineral_{i+1}": endmembers[i]
                          for i in range(endmembers.shape[0])}
        return run_sam_all_minerals(cube, endmember_dict, threshold)

# Method 2: QGIS with OTB (Orfeo ToolBox)
# ----------------------------------------
"""
1. Open QGIS
2. Processing Toolbox → OTB → Spectral Angle Mapper
   - Input image: preprocessed_hyperion.tif
   - Reference spectra: endmember_library.tif (stack of endmembers)
   - Threshold: 0.10 radians
3. Output: SAM classification raster

Alternative: Use QGIS Python console to automate OTB
"""

# Method 3: SNAP (ESA Sentinel Application Platform)
# ---------------------------------------------------
"""
1. Open SNAP
2. Load Hyperion product
3. Optical → Thematic Land Processing → Unsupervised Classification → SAM
4. Load endmember library
5. Set threshold: 0.08-0.12 radians
6. Export results as GeoTIFF

Note: SNAP may require plugins for hyperspectral support
"""

def save_sam_results(class_map, angle_map, endmember_names, output_dir):
    """
    Save SAM classification results with proper georeferencing
    """
    import rasterio
    from rasterio.transform import from_bounds
    
    # Save classification map
    # (Copy georeferencing from original Hyperion image)
    
    # Save individual angle maps for each mineral
    for i, name in enumerate(endmember_names):
        output_file = f"{output_dir}/sam_angle_{name}.tif"
        # Save angle_map[:,:,i] with georeferencing
    
    # Save minimum angle map (for quality assessment)
    min_angle = np.min(angle_map, axis=2)
    # Save min_angle with georeferencing


# ============================================================================
# STEP 5: ADD ABUNDANCE / CONFIDENCE MAPS (MTMF OR MF)
# ============================================================================

"""
GOAL: Handle mixed pixels and get abundance estimates
TOOLS: Python (pysptools)
OUTPUT: Abundance maps showing mineral likelihood
"""

from pysptools.abundance_maps import FCLS, NNLS
from pysptools.detection import MatchedFilter
# Note: MTMF is implemented manually below (not in pysptools 0.15.0)

# Step 5.1: Matched Filter (MF)
# ------------------------------
def run_matched_filter(cube, endmember_spectrum):
    """
    Run Matched Filter for single endmember detection
    
    Parameters:
    -----------
    cube : array (rows, cols, bands)
        Hyperion image
    endmember_spectrum : array (bands,)
        Target mineral spectrum
    
    Returns:
    --------
    mf_score : array (rows, cols)
        Matched filter score (higher = more likely)
    """
    mf = MatchedFilter()
    mf_score = mf.map(cube, endmember_spectrum)
    return mf_score

# Step 5.2: Mixture Tuned Matched Filter (MTMF)
# ----------------------------------------------
def run_mtmf(cube, endmember_spectrum):
    """
    Run MTMF for more robust detection with infeasibility metric
    Uses Matched Filter + infeasibility calculation

    Returns:
    --------
    mf_score : array (rows, cols)
        Matched filter component
    infeasibility : array (rows, cols)
        Infeasibility metric (lower = more feasible)
    """
    # Run matched filter first
    mf = MatchedFilter()
    mf_score = mf.map(cube, endmember_spectrum)

    # Calculate infeasibility (simplified implementation)
    # Infeasibility measures how well pixel fits as mixture of target + background
    rows, cols, bands = cube.shape
    infeasibility = np.zeros((rows, cols))

    # Normalize endmember
    target = endmember_spectrum / np.linalg.norm(endmember_spectrum)

    for i in range(rows):
        for j in range(cols):
            pixel = cube[i, j, :]
            if np.linalg.norm(pixel) > 0:
                pixel_norm = pixel / np.linalg.norm(pixel)
                # Infeasibility as orthogonal distance
                projection = np.dot(pixel_norm, target) * target
                infeasibility[i, j] = np.linalg.norm(pixel_norm - projection)

    return mf_score, infeasibility

# Step 5.3: Linear Spectral Unmixing
# -----------------------------------
def run_linear_unmixing(cube, endmembers):
    """
    Fully constrained least squares unmixing
    
    Parameters:
    -----------
    cube : array (rows, cols, bands)
        Hyperion image
    endmembers : array (n_endmembers, bands)
        All endmember spectra
    
    Returns:
    --------
    abundances : array (rows, cols, n_endmembers)
        Abundance fraction for each endmember (0-1)
    """
    fcls = FCLS()
    abundances = fcls.map(cube, endmembers)
    return abundances

def run_unmixing_for_all_minerals(cube, endmember_dict):
    """
    Run unmixing and save abundance maps for each mineral
    """
    # Convert endmembers to array
    names = list(endmember_dict.keys())
    E = np.array([endmember_dict[name] for name in names])
    
    # Run FCLS
    abundances = run_linear_unmixing(cube, E)
    
    # Save individual abundance maps
    results = {}
    for i, name in enumerate(names):
        results[name] = abundances[:, :, i]
    
    return results


# ============================================================================
# STEP 6: POSTPROCESSING
# ============================================================================

"""
GOAL: Combine results and clean up spurious classifications
TOOLS: Python (scipy, scikit-image), QGIS
OUTPUT: Refined mineral maps
"""

from scipy import ndimage

# Step 6.1: Combine SAM + MF/MTMF
# --------------------------------
def combine_sam_mtmf(sam_class, sam_angle, mtmf_score, mtmf_infeasibility,
                     angle_threshold=0.10, mf_threshold=0.5, 
                     infeas_threshold=0.1):
    """
    Combine SAM and MTMF for more reliable classification
    
    Pixel classified as target mineral only if:
    - SAM angle < angle_threshold
    - MTMF score > mf_threshold  
    - MTMF infeasibility < infeas_threshold
    
    Returns:
    --------
    combined_mask : array (rows, cols)
        Binary mask of detected mineral
    """
    sam_mask = sam_angle < angle_threshold
    mtmf_mask = (mtmf_score > mf_threshold) & (mtmf_infeasibility < infeas_threshold)
    combined_mask = sam_mask & mtmf_mask
    
    return combined_mask

# Step 6.2: Morphological Filtering
# ----------------------------------
def clean_classification_map(class_map, min_pixels=5):
    """
    Remove small isolated pixels and smooth boundaries
    
    Parameters:
    -----------
    class_map : array (rows, cols)
        Classification map or binary mask
    min_pixels : int
        Minimum cluster size to keep
    
    Returns:
    --------
    cleaned_map : array
        Cleaned classification
    """
    from skimage.morphology import remove_small_objects, closing, disk
    
    # Convert to boolean if needed
    if class_map.dtype != bool:
        binary_map = class_map > 0
    else:
        binary_map = class_map
    
    # Remove small objects
    cleaned = remove_small_objects(binary_map, min_size=min_pixels)
    
    # Morphological closing (fill small holes)
    selem = disk(1)
    cleaned = closing(cleaned, selem)
    
    return cleaned

# Step 6.3: Majority Filter (3×3)
# --------------------------------
def majority_filter(class_map):
    """
    Apply 3×3 majority filter to smooth classification
    """
    from scipy.stats import mode
    
    filtered = ndimage.generic_filter(class_map, 
                                      lambda x: mode(x, keepdims=True)[0][0],
                                      size=3)
    return filtered


# ============================================================================
# STEP 7: VALIDATION WITHOUT GROUND TRUTH
# ============================================================================

"""
GOAL: Assess mapping quality using external datasets
TOOLS: Python, QGIS for spatial analysis
OUTPUT: Validation report with consistency checks
"""

# Step 7.1: Compare with Published Rio Tinto Mineral Maps
# ---------------------------------------------------------
"""
Key publications with Rio Tinto mineral/geochemical maps:

1. Sánchez-España et al. (2005) - Iron terraces and AMD minerals
2. Amils et al. (2007) - Geomicrobiology and mineralogy
3. Fernández-Remolar et al. (2005) - Sediment mineralogy
4. Recent Sentinel-2 iron oxide mapping studies

Action:
- Digitize mineral occurrence zones from publications
- Import as vector layers in QGIS
- Calculate spatial overlap with your SAM/MTMF results
"""

def calculate_spatial_overlap(your_mineral_map, reference_polygons):
    """
    Calculate overlap between your classification and published zones
    
    Returns:
    --------
    overlap_stats : dict
        Precision, recall, F1-score by mineral type
    """
    pass  # Implement using geopandas and rasterio

# Step 7.2: Compare with Geochemical Data
# ----------------------------------------
"""
Rio Tinto geochemistry datasets available from:
- Spanish Geological Survey (IGME)
- Published papers with sample locations
- Water chemistry databases

Expected correlations:
- Jarosite zones → pH < 3, high Fe, high SO4
- Schwertmannite → pH 3-4.5, moderate Fe
- Goethite/Hematite → older oxidized zones, pH < 6

Action:
- Obtain geochemical point data (lat/lon + chemistry)
- Extract mineral classification at each point
- Test correlations
"""

def validate_with_geochemistry(mineral_maps, geochem_points):
    """
    Test mineralogical-geochemical consistency
    
    Parameters:
    -----------
    mineral_maps : dict
        {mineral_name: classification_array}
    geochem_points : GeoDataFrame
        Point data with pH, Fe, SO4, etc.
    
    Returns:
    --------
    validation_report : dict
        Statistical tests of consistency
    """
    results = {}
    
    # For each geochemical point:
    # 1. Extract predicted mineral
    # 2. Check if chemistry matches expected range
    # 3. Calculate agreement rate
    
    return results

# Step 7.3: Spatial Consistency Checks
# -------------------------------------
def check_spatial_consistency(mineral_maps, dem=None):
    """
    Test spatial logic of mineral distribution
    
    Checks:
    - Jarosite near stream channels (low topography)
    - Goethite on terraces (higher elevation)
    - Clustering patterns (minerals shouldn't be random)
    - Consistency with known AMD process zones
    
    Returns:
    --------
    consistency_report : dict
    """
    pass

# Step 7.4: Cross-Validation with High-Res Imagery
# -------------------------------------------------
"""
Sources:
- Sentinel-2 bands 11, 12 (SWIR) for iron oxides
- PlanetScope 3m resolution
- Drone orthomosaics from publications
- Google Earth historical imagery

Action:
- Calculate iron oxide indices from Sentinel-2
- Visual comparison with your Hyperion maps
- Check if high-confidence zones match visible features
"""


# ============================================================================
# STEP 8: FINAL OUTPUTS AND VISUALIZATION
# ============================================================================

"""
GOAL: Create publication-ready maps and analysis results
TOOLS: QGIS for cartography, Python for automation
OUTPUT: Maps, statistics, project files
"""

# Step 8.1: Create QGIS Project
# ------------------------------
"""
Layer structure:
1. Base layers:
   - Hillshade (from DEM)
   - Sentinel-2 RGB composite
   - OpenStreetMap/satellite basemap

2. Mineral classification layers:
   - Jarosite (SAM + MTMF combined)
   - Goethite
   - Hematite
   - Schwertmannite (if available)

3. Confidence/quality layers:
   - SAM minimum angle
   - MTMF infeasibility
   - Abundance maps

4. Validation layers:
   - Published mineral zones (digitized)
   - Geochemical sample points
   - Masked areas (vegetation, water)

5. Annotations:
   - Scale bar, north arrow
   - Legend with mineral colors
   - Coordinate grid

Styling tips:
- Use perceptually uniform colormaps (viridis, plasma)
- Semi-transparent overlays (60-80% opacity)
- Clear mineral color scheme (e.g., jarosite=yellow, goethite=brown)
"""

# Step 8.2: Generate Statistics Table
# ------------------------------------
def generate_mineral_statistics(mineral_maps, masks):
    """
    Calculate area statistics for each mineral
    
    Returns:
    --------
    stats_table : DataFrame
        Columns: Mineral, Area_ha, Area_pct, Pixel_count, Confidence_mean
    """
    import pandas as pd
    
    stats = []
    total_valid_pixels = np.sum(~masks['vegetation'] & ~masks['water'])
    
    for mineral_name, classification in mineral_maps.items():
        pixel_count = np.sum(classification > 0)
        area_ha = pixel_count * (30 * 30) / 10000  # Hyperion 30m pixels
        area_pct = (pixel_count / total_valid_pixels) * 100
        
        stats.append({
            'Mineral': mineral_name,
            'Pixels': pixel_count,
            'Area (ha)': area_ha,
            'Percent': area_pct
        })
    
    df = pd.DataFrame(stats)
    return df

# Step 8.3: Create Spectral Profile Plots
# ----------------------------------------
def plot_mineral_spectra(endmember_dict, wavelengths, output_file):
    """
    Create publication-quality spectral signature plots
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, spectrum in endmember_dict.items():
        ax.plot(wavelengths, spectrum, label=name, linewidth=2)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.set_title('AMD Mineral Endmember Spectra (Hyperion)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Highlight diagnostic regions
    ax.axvspan(2200, 2260, alpha=0.2, color='yellow', label='Jarosite region')
    ax.axvspan(850, 950, alpha=0.2, color='red', label='Hematite region')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

# Step 8.4: Export Final Products
# --------------------------------
def export_final_products(results_dict, output_dir):
    """
    Export all products in multiple formats
    
    Outputs:
    - GeoTIFF rasters (for GIS use)
    - PNG images (for reports)
    - KML files (for Google Earth)
    - CSV statistics
    - QGIS project file
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Export each mineral map as GeoTIFF
    # Export statistics as CSV
    # Create README with methods summary
    pass


# ============================================================================
# MAIN WORKFLOW EXECUTION
# ============================================================================

def main_workflow():
    """
    Complete workflow from Step 2 onwards
    (Assuming Step 1 done with SUREHYP)

    Configure which steps to run by setting True/False below
    """

    # ========================================================================
    # CONFIGURATION - Set True/False to enable/disable each step
    # ========================================================================

    RUN_STEP_2_BUILD_LIBRARY = False      # Build spectral library (if not done)
    RUN_STEP_3_ENHANCEMENT = False        # Spectral enhancement (smoothing, etc.)
    RUN_STEP_4_SAM = True                 # SAM classification
    RUN_STEP_5_MTMF = False               # MTMF abundance mapping
    RUN_STEP_6_POSTPROCESSING = False     # Combine and clean results
    RUN_STEP_7_VALIDATION = False         # Validation checks
    RUN_STEP_8_EXPORT = True              # Export final products

    # ========================================================================
    # FILE PATHS - Update these paths for your data
    # ========================================================================

    # Input files
    HYPERION_CUBE_PATH = './amd_mapping/data/hyperion/EO1H2020342016359110KF_reflectance.hdr'
    ENDMEMBER_LIBRARY_PATH = './amd_mapping/data/outputs/endmember_library_matched.csv'

    # Output directory
    OUTPUT_DIR = './amd_mapping/outputs/classifications'

    # SAM parameters
    SAM_THRESHOLD = 1.40  # radians (smaller = more strict)

    # Subset for testing (set to None to process full image)
    # Format: ((row_start, row_end), (col_start, col_end))
    PROCESS_SUBSET = None  # None = full image
    # PROCESS_SUBSET = ((1000, 1500), (200, 700))  # Example subset

    # ========================================================================
    # WORKFLOW EXECUTION
    # ========================================================================

    print("=" * 70)
    print("AMD MINERAL MAPPING WORKFLOW")
    print("=" * 70)
    print("\nConfiguration:")
    print("  - Build Library: {}".format(RUN_STEP_2_BUILD_LIBRARY))
    print("  - Enhancement: {}".format(RUN_STEP_3_ENHANCEMENT))
    print("  - SAM Classification: {}".format(RUN_STEP_4_SAM))
    print("  - MTMF: {}".format(RUN_STEP_5_MTMF))
    print("  - Postprocessing: {}".format(RUN_STEP_6_POSTPROCESSING))
    print("  - Validation: {}".format(RUN_STEP_7_VALIDATION))
    print("  - Export: {}".format(RUN_STEP_8_EXPORT))
    print("=" * 70)

    # ========================================================================
    # STEP 2: Build Spectral Library (Optional)
    # ========================================================================

    if RUN_STEP_2_BUILD_LIBRARY:
        print("\n=== STEP 2: Building Spectral Library ===")

        if USGS_LOADER_AVAILABLE:
            endmembers, hyperion_wvl = create_endmember_library(
                output_file='./amd_mapping/data/outputs/endmember_library.sli'
            )
        else:
            hyperion_wvl = load_hyperion_wavelengths(HYPERION_CUBE_PATH)
            endmembers, hyperion_wvl = create_endmember_library(
                library_dir='./spectral_library/',
                hyperion_wavelengths=hyperion_wvl,
                output_file='./amd_mapping/data/outputs/endmember_library.sli'
            )

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print("\n=== Loading Data ===")

    # Load Hyperion cube
    print("Loading Hyperion cube: {}".format(HYPERION_CUBE_PATH))
    img = envi.open(HYPERION_CUBE_PATH)

    if PROCESS_SUBSET is not None:
        rows, cols = PROCESS_SUBSET
        print("  Processing subset: rows {}-{}, cols {}-{}".format(
            rows[0], rows[1], cols[0], cols[1]))
        cube = img.read_subregion(rows, cols)
    else:
        print("  Processing full image: {} x {} x {} bands".format(
            img.nrows, img.ncols, img.nbands))
        cube = img.load()
        # Only filter bands if we have the full 242-band dataset
        if img.nbands == 242:
            usable_bands = get_usable_bands()
            cube = cube[:, :, usable_bands]
            print("  Filtered to {} usable bands".format(len(usable_bands)))
        else:
            print("  Data already preprocessed with {} bands".format(img.nbands))

    # Convert to numpy array - img.load() returns a memmap-like object
    print("  Converting spectral image to numpy array...")
    cube = np.asarray(cube, dtype=np.float32)

    print("  Cube shape: {}".format(cube.shape))
    print("  Cube type: {}".format(type(cube)))

    # Load endmember library
    print("Loading endmember library: {}".format(ENDMEMBER_LIBRARY_PATH))
    library_df = pd.read_csv(ENDMEMBER_LIBRARY_PATH, index_col=0)
    endmembers = {name: library_df[name].values for name in library_df.columns}
    print("  Loaded {} minerals: {}".format(len(endmembers), list(endmembers.keys())))

    # ========================================================================
    # CRITICAL DIAGNOSTIC: Check band counts and scales
    # ========================================================================
    print("\n" + "=" * 70)
    print("CRITICAL DIAGNOSTIC CHECKS")
    print("=" * 70)

    # Check 1: Band counts
    cube_bands = cube.shape[2]
    endmember_bands = len(list(endmembers.values())[0])
    print("\n1. BAND COUNT CHECK:")
    print("  Cube bands: {}".format(cube_bands))
    print("  Endmember bands: {}".format(endmember_bands))

    if cube_bands != endmember_bands:
        print("  *** CRITICAL ERROR: Band mismatch!")
        print("  SAM CANNOT WORK with different band counts!")
        print("\n  Need to match the bands...")

        # Try to match by trimming cube to endmember length
        if cube_bands > endmember_bands:
            print("  Trimming cube from {} to {} bands".format(cube_bands, endmember_bands))
            cube = cube[:, :, :endmember_bands]
            print("  New cube shape: {}".format(cube.shape))
        else:
            print("  ERROR: Cube has fewer bands than endmembers!")
            print("  Cannot proceed - check your endmember library file")
    else:
        print("  [OK] Band counts match!")

    # Check 2: Scale validation
    cube_max = np.max(cube[cube > 0]) if np.any(cube > 0) else 0
    cube_min = np.min(cube[cube > 0]) if np.any(cube > 0) else 0
    endmember_max = max(np.max(spec) for spec in endmembers.values())
    endmember_min = min(np.min(spec) for spec in endmembers.values())

    print("\n2. SCALE CHECK:")
    print("  Cube range: {:.6f} to {:.6f}".format(cube_min, cube_max))
    print("  Endmember range: {:.6f} to {:.6f}".format(endmember_min, endmember_max))

    if cube_max > 2.0 or endmember_max > 2.0:
        print("  [WARNING] Values > 2.0 detected (should be 0-1 reflectance)")
        if cube_max > 2.0:
            print("  Applying scale correction to cube...")
            # Check if values are in 0-65535 range (16-bit) or 0-10000 range
            if cube_max > 10000:
                print("  Detected 16-bit scale (0-65535), dividing by 65535...")
                cube = cube / 65535.0
            else:
                print("  Detected 10000 scale, dividing by 10000...")
                cube = cube / 10000.0
            print("  Cube range after scaling: {:.6f} to {:.6f}".format(np.min(cube), np.max(cube)))
        if endmember_max > 2.0:
            print("  Applying scale correction to endmembers...")
            endmembers = {name: spec / 10000.0 for name, spec in endmembers.items()}
            endmember_max = max(np.max(spec) for spec in endmembers.values())
            print("  Endmember range after scaling: {:.6f} to {:.6f}".format(endmember_min, endmember_max))
    else:
        print("  [OK] Scales look correct (0-1 range)")

    # Check 3: Valid pixels
    valid_pixels = np.sum(np.any(cube > 0, axis=2))
    total_pixels = cube.shape[0] * cube.shape[1]
    print("\n3. DATA VALIDITY CHECK:")
    print("  Valid pixels: {:,} / {:,} ({:.1f}%)".format(
        valid_pixels, total_pixels, valid_pixels/total_pixels*100))

    if valid_pixels == 0:
        print("  *** CRITICAL ERROR: No valid pixels!")
        print("  Cannot proceed - check your Hyperion data file")
    else:
        print("  [OK] Data contains valid pixels")

    # Check 4: Manual spectral angle test
    print("\n4. MANUAL SPECTRAL ANGLE TEST:")
    test_row, test_col = cube.shape[0]//2, cube.shape[1]//2
    test_pixel = cube[test_row, test_col, :]

    if np.any(test_pixel > 0):
        print("  Testing pixel at ({}, {})".format(test_row, test_col))
        print("  Pixel range: {:.6f} to {:.6f}".format(np.min(test_pixel), np.max(test_pixel)))

        for mineral_name in list(endmembers.keys())[:3]:  # Test first 3 minerals
            test_endmember = endmembers[mineral_name]
            dot = np.dot(test_pixel, test_endmember)
            norm_p = np.linalg.norm(test_pixel)
            norm_e = np.linalg.norm(test_endmember)

            if norm_p > 0 and norm_e > 0:
                cos_angle = dot / (norm_p * norm_e)
                print("\n  {} vs {}:".format("Test pixel", mineral_name))
                print("    cos(angle) = {:.6f}".format(cos_angle))

                if cos_angle > 1.0 or cos_angle < -1.0:
                    print("    *** INVALID! Scale mismatch detected!")
                else:
                    angle = np.arccos(cos_angle)
                    print("    angle = {:.4f} rad ({:.1f} deg)".format(angle, np.degrees(angle)))
                    if angle < SAM_THRESHOLD:
                        print("    [OK] Would classify at threshold {:.2f}".format(SAM_THRESHOLD))
                    else:
                        print("    [NO] Would not classify (threshold {:.2f})".format(SAM_THRESHOLD))
    else:
        print("  Test pixel is invalid (all zeros)")

    print("=" * 70 + "\n")

    # ========================================================================
    # VISUALIZATION: Plot mineral spectra and Hyperion data
    # ========================================================================
    print("\n=== GENERATING DIAGNOSTIC PLOTS ===")

    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving files
    import matplotlib.pyplot as plt
    import os

    # Create output directory for plots
    plot_dir = os.path.join(OUTPUT_DIR, 'diagnostic_plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: All mineral spectra
    print("Creating mineral spectra plot...")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get wavelengths from the library index
    wavelengths = library_df.index.values.astype(float)

    colors = plt.cm.tab10(np.linspace(0, 1, len(endmembers)))
    for idx, (mineral_name, spectrum) in enumerate(endmembers.items()):
        ax.plot(wavelengths[:len(spectrum)], spectrum, label=mineral_name,
                linewidth=2, color=colors[idx])

    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
    ax.set_title('AMD Mineral Endmember Spectra (Hyperion Wavelengths)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Highlight diagnostic regions
    ax.axvspan(2200, 2260, alpha=0.15, color='yellow', label='Jarosite (2200-2260nm)')
    ax.axvspan(850, 950, alpha=0.15, color='red', label='Fe oxides (850-950nm)')

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, 'mineral_spectra.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: {}".format(plot_path))

    # Plot 2: Individual mineral spectra (separate subplots)
    print("Creating individual mineral spectra plots...")
    n_minerals = len(endmembers)
    n_cols = 3
    n_rows = int(np.ceil(n_minerals / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*3))
    axes = axes.flatten() if n_minerals > 1 else [axes]

    for idx, (mineral_name, spectrum) in enumerate(endmembers.items()):
        ax = axes[idx]
        ax.plot(wavelengths[:len(spectrum)], spectrum, linewidth=2, color=colors[idx])
        ax.set_title(mineral_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Wavelength (nm)', fontsize=9)
        ax.set_ylabel('Reflectance', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(1.0, np.max(spectrum) * 1.1)])

    # Hide unused subplots
    for idx in range(n_minerals, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, 'mineral_spectra_individual.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: {}".format(plot_path))

    # Plot 3: Hyperion RGB composite (bands as RGB)
    print("Creating Hyperion RGB composite...")
    # Select bands for RGB visualization (approximate true color)
    # Red: ~650nm, Green: ~550nm, Blue: ~450nm
    # For Hyperion, we'll use bands that approximate these

    if cube.shape[2] >= 50:
        # Use bands from the visible range
        red_band = cube[:, :, min(30, cube.shape[2]-1)]
        green_band = cube[:, :, min(20, cube.shape[2]-1)]
        blue_band = cube[:, :, min(10, cube.shape[2]-1)]

        # Normalize for display
        def normalize_band(band):
            valid = band[band > 0]
            if len(valid) > 0:
                p2, p98 = np.percentile(valid, [2, 98])
                normalized = np.clip((band - p2) / (p98 - p2), 0, 1)
                return normalized
            return band

        rgb = np.dstack([
            normalize_band(red_band),
            normalize_band(green_band),
            normalize_band(blue_band)
        ])

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(rgb)
        ax.set_title('Hyperion RGB Composite (Approximate True Color)',
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, 'hyperion_rgb.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print("  Saved: {}".format(plot_path))

    # Plot 4: Sample pixel spectra from different locations
    print("Creating sample pixel spectra plot...")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Sample 5 random valid pixels
    valid_mask = np.any(cube > 0, axis=2)
    valid_coords = np.argwhere(valid_mask)

    if len(valid_coords) > 0:
        sample_indices = np.random.choice(len(valid_coords),
                                         min(5, len(valid_coords)),
                                         replace=False)

        for i, idx in enumerate(sample_indices):
            row, col = valid_coords[idx]
            pixel_spectrum = cube[row, col, :]
            ax.plot(wavelengths[:len(pixel_spectrum)], pixel_spectrum,
                   label='Pixel ({}, {})'.format(row, col),
                   linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
        ax.set_title('Sample Hyperion Pixel Spectra', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, 'sample_pixel_spectra.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: {}".format(plot_path))

    # Plot 5: Mean spectrum of entire image vs minerals
    print("Creating mean image spectrum comparison...")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate mean spectrum of all valid pixels
    valid_pixels_array = cube[valid_mask]
    mean_spectrum = np.mean(valid_pixels_array, axis=0)

    ax.plot(wavelengths[:len(mean_spectrum)], mean_spectrum,
           label='Mean Image Spectrum', linewidth=3, color='black', alpha=0.8)

    # Overlay mineral spectra (lighter)
    for idx, (mineral_name, spectrum) in enumerate(endmembers.items()):
        ax.plot(wavelengths[:len(spectrum)], spectrum,
               label=mineral_name, linewidth=1.5, alpha=0.5, color=colors[idx])

    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
    ax.set_title('Mean Hyperion Spectrum vs Mineral Endmembers',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, 'mean_spectrum_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: {}".format(plot_path))

    print("\nAll diagnostic plots saved to: {}".format(plot_dir))
    print("=" * 70 + "\n")

    # ========================================================================
    # STEP 3: Spectral Enhancement (Optional)
    # ========================================================================

    if RUN_STEP_3_ENHANCEMENT:
        print("\n=== STEP 3: Enhancing Spectral Features ===")
        print("Applying Savitzky-Golay smoothing...")
        cube_smooth = apply_savgol_smoothing(cube)
        print("  Done!")
        # Use smoothed cube for classification
        cube_for_sam = cube_smooth
    else:
        print("\n=== STEP 3: Spectral Enhancement SKIPPED ===")
        # Use original cube
        cube_for_sam = cube

    # ========================================================================
    # STEP 4: SAM Classification
    # ========================================================================

    class_map = None
    angle_maps = None

    if RUN_STEP_4_SAM:
        print("\n=== STEP 4: Running SAM Classification ===")
        print("Threshold: {} radians".format(SAM_THRESHOLD))

        class_map, angle_maps = run_sam_all_minerals(
            cube_for_sam,
            endmembers,
            threshold=SAM_THRESHOLD
        )

        print("\nSAM Classification completed!")
    else:
        print("\n=== STEP 4: SAM Classification SKIPPED ===")

    # ========================================================================
    # STEP 5: MTMF for Abundances (Optional)
    # ========================================================================

    abundance_maps = {}

    if RUN_STEP_5_MTMF:
        print("\n=== STEP 5: Running MTMF for Abundances ===")
        for mineral_name, spectrum in endmembers.items():
            print("  Processing {}...".format(mineral_name))
            mf_score, infeas = run_mtmf(cube_for_sam, spectrum)
            abundance_maps[mineral_name] = {
                'mf_score': mf_score,
                'infeasibility': infeas
            }
        print("MTMF completed!")
    else:
        print("\n=== STEP 5: MTMF SKIPPED ===")

    # ========================================================================
    # STEP 6: Postprocessing (Optional)
    # ========================================================================

    refined_maps = {}

    if RUN_STEP_6_POSTPROCESSING:
        print("\n=== STEP 6: Postprocessing ===")

        if class_map is not None and abundance_maps:
            for idx, mineral_name in enumerate(endmembers.keys(), 1):
                print("  Cleaning {}...".format(mineral_name))
                combined = combine_sam_mtmf(
                    sam_class=class_map,
                    sam_angle=angle_maps[mineral_name],
                    mtmf_score=abundance_maps[mineral_name]['mf_score'],
                    mtmf_infeasibility=abundance_maps[mineral_name]['infeasibility']
                )
                cleaned = clean_classification_map(combined)
                refined_maps[mineral_name] = cleaned
            print("Postprocessing completed!")
        else:
            print("  Skipping - need both SAM and MTMF results")
    else:
        print("\n=== STEP 6: Postprocessing SKIPPED ===")

    # ========================================================================
    # STEP 7: Validation (Optional)
    # ========================================================================

    if RUN_STEP_7_VALIDATION:
        print("\n=== STEP 7: Validation ===")
        print("  Load validation data here...")
        print("  Run consistency checks here...")
    else:
        print("\n=== STEP 7: Validation SKIPPED ===")

    # ========================================================================
    # STEP 8: Export Results
    # ========================================================================

    if RUN_STEP_8_EXPORT:
        print("\n=== STEP 8: Exporting Results ===")

        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Prepare metadata from original image
        metadata = img.metadata.copy()

        # Export SAM results
        if class_map is not None:
            print("Exporting SAM classification map...")
            output_path = os.path.join(OUTPUT_DIR, 'sam_multiclass.hdr')
            envi.save_image(output_path, class_map, metadata=metadata, force=True)
            print("  Saved: {}".format(output_path))

            # Export individual angle maps
            print("Exporting individual angle maps...")
            for mineral_name, angle_map in angle_maps.items():
                output_path = os.path.join(OUTPUT_DIR, 'sam_angle_{}.hdr'.format(mineral_name))
                envi.save_image(output_path, angle_map, metadata=metadata, force=True)
                print("  Saved: {}".format(output_path))

        # Export MTMF results
        if abundance_maps:
            print("Exporting MTMF abundance maps...")
            for mineral_name, maps in abundance_maps.items():
                output_path = os.path.join(OUTPUT_DIR, 'mtmf_mf_{}.hdr'.format(mineral_name))
                envi.save_image(output_path, maps['mf_score'], metadata=metadata, force=True)
                print("  Saved: {}".format(output_path))

        # Export refined maps
        if refined_maps:
            print("Exporting refined classification maps...")
            for mineral_name, refined_map in refined_maps.items():
                output_path = os.path.join(OUTPUT_DIR, 'refined_{}.hdr'.format(mineral_name))
                envi.save_image(output_path, refined_map, metadata=metadata, force=True)
                print("  Saved: {}".format(output_path))

        # Generate and save statistics
        if class_map is not None:
            print("\nGenerating classification statistics...")
            stats_data = []
            total_pixels = class_map.size

            # Unclassified pixels
            unclassified = np.sum(class_map == 0)
            stats_data.append({
                'Class': 'Unclassified',
                'Code': 0,
                'Pixels': unclassified,
                'Percent': (unclassified / total_pixels) * 100
            })

            # Each mineral class
            for idx, mineral_name in enumerate(endmembers.keys(), 1):
                count = np.sum(class_map == idx)
                stats_data.append({
                    'Class': mineral_name,
                    'Code': idx,
                    'Pixels': count,
                    'Percent': (count / total_pixels) * 100
                })

            stats_df = pd.DataFrame(stats_data)
            stats_path = os.path.join(OUTPUT_DIR, 'classification_statistics.csv')
            stats_df.to_csv(stats_path, index=False)
            print("  Saved statistics: {}".format(stats_path))
            print("\n{}".format(stats_df.to_string(index=False)))

        print("\nAll results exported to: {}".format(OUTPUT_DIR))
    else:
        print("\n=== STEP 8: Export SKIPPED ===")

    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main_workflow()
