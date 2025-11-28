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

    print(f"Processing {rows} × {cols} pixels...")
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
    print(f"Mean angle: {angle_map[angle_map < np.pi].mean():.3f} rad")
    print(f"Min angle: {angle_map[angle_map < np.pi].min():.3f} rad")

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
    """

    print("=== STEP 2: Building Spectral Library ===")

    # Option 1: Use USGS loader (automatic - recommended)
    if USGS_LOADER_AVAILABLE:
        endmembers, hyperion_wvl = create_endmember_library(
            output_file='./amd_mapping/data/outputs/endmember_library.sli'
        )

    # Option 2: Manual loading (if USGS loader not available)
    else:
        # Load Hyperion wavelengths from your image
        hyperion_wvl = load_hyperion_wavelengths('hyperion_cube.hdr')

        # Create endmember library from custom directory
        endmembers, hyperion_wvl = create_endmember_library(
            library_dir='./spectral_library/',
            hyperion_wavelengths=hyperion_wvl,
            output_file='./amd_mapping/data/outputs/endmember_library.sli'
        )
    
    print("=== STEP 3: Enhancing Spectral Features ===")
    # Load preprocessed cube from Step 1
    cube = envi.open('hyperion_cube.hdr').load()
    
    # Apply Savitzky-Golay smoothing
    cube_smooth = apply_savgol_smoothing(cube)
    
    # Optional: Continuum removal
    # cube_cr = apply_continuum_removal_cube(cube_smooth)
    
    print("=== STEP 4: Running SAM Classification ===")
    class_map, angle_map = run_sam_classification(
        cube_smooth, 
        endmembers, 
        threshold=0.10
    )
    
    print("=== STEP 5: Running MTMF for Abundances ===")
    abundance_maps = {}
    for mineral_name, spectrum in endmembers.items():
        mf_score, infeas = run_mtmf(cube_smooth, spectrum)
        abundance_maps[mineral_name] = {
            'mf_score': mf_score,
            'infeasibility': infeas
        }
    
    print("=== STEP 6: Postprocessing ===")
    refined_maps = {}
    for mineral_name in endmembers.keys():
        combined = combine_sam_mtmf(
            sam_class=class_map,
            sam_angle=angle_map[:,:,i],  # Index for this mineral
            mtmf_score=abundance_maps[mineral_name]['mf_score'],
            mtmf_infeasibility=abundance_maps[mineral_name]['infeasibility']
        )
        cleaned = clean_classification_map(combined)
        refined_maps[mineral_name] = cleaned
    
    print("=== STEP 7: Validation ===")
    # Load validation data
    # Run consistency checks
    
    print("=== STEP 8: Final Outputs ===")
    stats = generate_mineral_statistics(refined_maps, masks={})
    print(stats)
    
    # Export all products
    export_final_products(refined_maps, './final_outputs/')
    
    print("=== WORKFLOW COMPLETE ===")

if __name__ == "__main__":
    main_workflow()
