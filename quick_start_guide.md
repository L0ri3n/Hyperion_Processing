# Quick Start Guide - Hyperion AMD Mineral Mapping
# Copy-paste ready commands and code snippets

# =============================================================================
# INITIAL SETUP - RUN ONCE
# =============================================================================

## Create Project Environment
```bash
# Create conda environment
conda create -n hyperion python=3.9 -y
conda activate hyperion

# Install core packages
conda install -c conda-forge numpy scipy pandas matplotlib gdal rasterio geopandas -y
pip install spectral pysptools scikit-image

# Verify installation
python -c "import spectral; import pysptools; print('Success!')"
```

## Create Project Directory
```bash
# Create directory structure
mkdir -p amd_mapping/{data,code,outputs,qgis_projects,docs}
mkdir -p amd_mapping/data/{hyperion,spectral_library,ancillary,validation}
mkdir -p amd_mapping/data/spectral_library/{usgs,publications,resampled}
mkdir -p amd_mapping/outputs/{classifications,abundances,masks,figures,stats}

cd amd_mapping
```

# =============================================================================
# STEP 2: SPECTRAL LIBRARY CONSTRUCTION
# =============================================================================

## Example: Load USGS Spectrum File
```python
# save as: code/load_usgs_spectrum.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_usgs_spectrum(filepath):
    """
    Load USGS spectral library file
    
    USGS format (after header):
    Column 1: Wavelength (micrometers or nanometers)
    Column 2: Reflectance
    """
    # Read the file, skip header lines
    data = pd.read_csv(filepath, sep='\s+', skiprows=18, header=None)
    
    wavelength = data.iloc[:, 0].values
    reflectance = data.iloc[:, 1].values
    
    # Convert to nm if in micrometers (check if max < 10)
    if wavelength.max() < 10:
        wavelength = wavelength * 1000  # Convert µm to nm
    
    return wavelength, reflectance

# Example usage
wvl, ref = load_usgs_spectrum('data/spectral_library/usgs/jarosite.txt')
print(f"Wavelength range: {wvl.min():.1f} - {wvl.max():.1f} nm")
print(f"Number of points: {len(wvl)}")

# Quick plot
plt.figure(figsize=(10, 5))
plt.plot(wvl, ref)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Jarosite Spectrum')
plt.grid(True)
plt.savefig('outputs/figures/jarosite_spectrum.png', dpi=150)
plt.show()
```

## Example: Resample to Hyperion Wavelengths
```python
# save as: code/resample_library.py

import numpy as np
from spectral import *
import glob

def get_hyperion_wavelengths(hyperion_file):
    """Extract wavelengths from Hyperion metadata"""
    img = envi.open(hyperion_file)
    wavelengths = np.array(img.metadata['wavelength'], dtype=float)
    return wavelengths

def get_usable_hyperion_bands():
    """Return indices of good Hyperion bands"""
    all_bands = np.arange(242)
    bad_bands = np.concatenate([
        np.arange(0, 7),      # Bad VNIR
        np.arange(57, 76),    # Water vapor 1360-1400 nm
        np.arange(122, 135),  # Water vapor 1800-1950 nm
        np.arange(224, 242)   # Bad SWIR
    ])
    usable = np.setdiff1d(all_bands, bad_bands)
    return usable

def resample_spectrum(lib_wvl, lib_ref, target_wvl):
    """Linear interpolation to target wavelengths"""
    return np.interp(target_wvl, lib_wvl, lib_ref)

# Main execution
hyperion_file = 'data/hyperion/preprocessed/hyperion_clean.hdr'
hyperion_wvl = get_hyperion_wavelengths(hyperion_file)
usable_idx = get_usable_hyperion_bands()
hyperion_wvl_clean = hyperion_wvl[usable_idx]

print(f"Hyperion bands: {len(hyperion_wvl_clean)}")
print(f"Wavelength range: {hyperion_wvl_clean.min():.1f} - {hyperion_wvl_clean.max():.1f} nm")

# Process all USGS spectra
library_dir = 'data/spectral_library/usgs/'
resampled_spectra = {}

for filepath in glob.glob(f"{library_dir}/*.txt"):
    mineral_name = filepath.split('/')[-1].replace('.txt', '')
    print(f"Processing: {mineral_name}")
    
    # Load original spectrum
    lib_wvl, lib_ref = load_usgs_spectrum(filepath)
    
    # Resample to Hyperion
    resampled_ref = resample_spectrum(lib_wvl, lib_ref, hyperion_wvl_clean)
    resampled_spectra[mineral_name] = resampled_ref

# Save as CSV
import pandas as pd
df = pd.DataFrame(resampled_spectra, index=hyperion_wvl_clean)
df.index.name = 'Wavelength_nm'
df.to_csv('data/spectral_library/resampled/endmember_library.csv')
print("Saved: endmember_library.csv")

# Plot all endmembers
plt.figure(figsize=(12, 6))
for name, spectrum in resampled_spectra.items():
    plt.plot(hyperion_wvl_clean, spectrum, label=name, linewidth=2)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Resampled Endmember Library (Hyperion wavelengths)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/figures/endmember_library.png', dpi=200)
plt.show()
```

# =============================================================================
# STEP 3: SPECTRAL ENHANCEMENT
# =============================================================================

## Example: Apply Savitzky-Golay Smoothing
```python
# save as: code/smooth_cube.py

import numpy as np
from spectral import *
from scipy.signal import savgol_filter
import time

def smooth_hyperion_cube(input_file, output_file, window=9, polyorder=2):
    """
    Apply Savitzky-Golay smoothing to Hyperion cube
    
    Parameters:
    -----------
    input_file : str
        Path to input Hyperion .hdr file
    output_file : str  
        Path to output smoothed cube
    window : int (default=9)
        Window length (must be odd)
    polyorder : int (default=2)
        Polynomial order
    """
    print("Loading Hyperion cube...")
    img = envi.open(input_file)
    cube = img.load()
    rows, cols, bands = cube.shape
    print(f"Dimensions: {rows} × {cols} × {bands}")
    
    print(f"Applying Savitzky-Golay filter (window={window}, order={polyorder})...")
    smoothed = np.zeros_like(cube, dtype=np.float32)
    
    start = time.time()
    for i in range(rows):
        if i % 50 == 0:
            print(f"  Row {i}/{rows} ({i/rows*100:.1f}%)")
        for j in range(cols):
            spectrum = cube[i, j, :].astype(np.float32)
            # Only smooth valid pixels (non-zero)
            if np.any(spectrum > 0):
                smoothed[i, j, :] = savgol_filter(spectrum, window, polyorder)
    
    elapsed = time.time() - start
    print(f"Smoothing complete in {elapsed:.1f} seconds")
    
    # Save output
    print(f"Saving to: {output_file}")
    metadata = img.metadata.copy()
    metadata['description'] = f'Savitzky-Golay smoothed (window={window}, order={polyorder})'
    
    envi.save_image(output_file + '.hdr', smoothed, metadata=metadata, force=True)
    print("Done!")

# Run
input_file = 'data/hyperion/preprocessed/hyperion_clean.hdr'
output_file = 'data/hyperion/preprocessed/hyperion_smooth'
smooth_hyperion_cube(input_file, output_file, window=9, polyorder=2)
```

## Example: Create Vegetation Mask
```python
# save as: code/create_masks.py

import numpy as np
from spectral import *
import rasterio
from rasterio.transform import from_bounds

def create_ndvi_mask(input_file, output_file, threshold=0.3):
    """
    Create NDVI-based vegetation mask
    
    Hyperion bands (0-indexed):
    Red: Band 28 (~650 nm)
    NIR: Band 49 (~830 nm)
    """
    print("Loading Hyperion cube...")
    img = envi.open(input_file)
    cube = img.load()
    
    # Extract Red and NIR bands
    red = cube[:, :, 28].astype(np.float32)
    nir = cube[:, :, 49].astype(np.float32)
    
    print("Computing NDVI...")
    # Avoid division by zero
    denominator = nir + red
    ndvi = np.zeros_like(red)
    valid = denominator > 0
    ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]
    
    print(f"NDVI range: {ndvi.min():.3f} to {ndvi.max():.3f}")
    
    # Create mask (1 = vegetation, 0 = not vegetation)
    veg_mask = (ndvi > threshold).astype(np.uint8)
    print(f"Vegetation pixels: {veg_mask.sum()} ({veg_mask.sum()/veg_mask.size*100:.1f}%)")
    
    # Save as GeoTIFF
    print(f"Saving mask to: {output_file}")
    metadata = img.metadata
    
    # Save NDVI
    envi.save_image(output_file.replace('.tif', '_ndvi.hdr'), 
                    ndvi, metadata=metadata, force=True)
    
    # Save mask
    envi.save_image(output_file.replace('.tif', '_mask.hdr'),
                    veg_mask, metadata=metadata, force=True)
    print("Done!")
    
    return veg_mask

# Run
input_file = 'data/hyperion/preprocessed/hyperion_smooth.hdr'
output_file = 'outputs/masks/vegetation_mask.tif'
veg_mask = create_ndvi_mask(input_file, output_file, threshold=0.3)
```

# =============================================================================
# STEP 4: SAM CLASSIFICATION
# =============================================================================

## Example: Run SAM for Single Mineral
```python
# save as: code/run_sam_single.py

import numpy as np
from spectral import *
import pandas as pd

def spectral_angle(pixel, endmember):
    """Compute spectral angle in radians"""
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

# Load data
print("Loading Hyperion cube...")
img = envi.open('data/hyperion/preprocessed/hyperion_smooth.hdr')
cube = img.load()

print("Loading endmember library...")
library_df = pd.read_csv('data/spectral_library/resampled/endmember_library.csv', 
                          index_col=0)

# Run SAM for Jarosite
mineral_name = 'jarosite'  # Change this for different minerals
print(f"\nRunning SAM for {mineral_name}...")
endmember = library_df[mineral_name].values

class_map, angle_map = run_sam_single_mineral(cube, endmember, threshold=0.10)

# Save results
print("Saving results...")
metadata = img.metadata.copy()
envi.save_image(f'outputs/classifications/sam_{mineral_name}_class.hdr',
                class_map, metadata=metadata, force=True)
envi.save_image(f'outputs/classifications/sam_{mineral_name}_angle.hdr',
                angle_map, metadata=metadata, force=True)
print("Done!")
```

## Example: Run SAM for All Minerals
```python
# save as: code/run_sam_all.py

import numpy as np
from spectral import *
import pandas as pd

def run_sam_all_minerals(cube, endmember_dict, threshold=0.10):
    """
    Run SAM for all minerals and create multi-class map
    
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
            for j in range(cols):
                pixel = cube[i, j, :]
                if np.any(pixel > 0):
                    dot = np.dot(pixel, endmember)
                    norm = np.linalg.norm(pixel) * np.linalg.norm(endmember)
                    if norm > 0:
                        angle_map[i, j] = np.arccos(np.clip(dot/norm, -1, 1))
                    else:
                        angle_map[i, j] = np.pi
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

# Load data
img = envi.open('data/hyperion/preprocessed/hyperion_smooth.hdr')
cube = img.load()

library_df = pd.read_csv('data/spectral_library/resampled/endmember_library.csv',
                          index_col=0)

# Convert to dict
endmember_dict = {col: library_df[col].values for col in library_df.columns}

# Run SAM
class_map, angle_maps = run_sam_all_minerals(cube, endmember_dict, threshold=0.10)

# Save
metadata = img.metadata.copy()
envi.save_image('outputs/classifications/sam_multiclass.hdr',
                class_map, metadata=metadata, force=True)

# Save individual angle maps
for name, angle_map in angle_maps.items():
    envi.save_image(f'outputs/classifications/sam_angle_{name}.hdr',
                    angle_map, metadata=metadata, force=True)

print("All done!")
```

# =============================================================================
# STEP 5: ABUNDANCE MAPPING WITH MTMF
# =============================================================================

## Example: Run MTMF (Simplified Version)
```python
# save as: code/run_mtmf_simple.py

import numpy as np
from spectral import *
import pandas as pd

def matched_filter(cube, target_spectrum):
    """
    Compute Matched Filter score
    
    Returns:
    --------
    mf_score : array (rows, cols)
        Higher values = more likely to be target
    """
    rows, cols, bands = cube.shape
    
    # Reshape for matrix operations
    X = cube.reshape(-1, bands).T  # bands × pixels
    
    # Compute background statistics
    print("Computing background covariance...")
    mean = np.mean(X, axis=1)
    X_centered = X - mean[:, np.newaxis]
    cov = np.cov(X_centered)
    
    # Add small regularization to avoid singularity
    cov += np.eye(bands) * 1e-6
    
    # Compute matched filter
    print("Computing matched filter...")
    cov_inv = np.linalg.inv(cov)
    t = target_spectrum - mean
    
    numerator = np.dot(cov_inv, t)
    denominator = np.sqrt(np.dot(t.T, np.dot(cov_inv, t)))
    
    mf_score = np.dot(numerator.T, X) / denominator
    mf_score = mf_score.reshape(rows, cols)
    
    return mf_score

# Load data
print("Loading cube...")
img = envi.open('data/hyperion/preprocessed/hyperion_smooth.hdr')
cube = img.load()

print("Loading endmembers...")
library_df = pd.read_csv('data/spectral_library/resampled/endmember_library.csv',
                          index_col=0)

# Run for each mineral
for mineral_name in library_df.columns:
    print(f"\n=== {mineral_name} ===")
    endmember = library_df[mineral_name].values
    
    mf_score = matched_filter(cube, endmember)
    
    print(f"MF score range: {mf_score.min():.3f} to {mf_score.max():.3f}")
    
    # Save
    metadata = img.metadata.copy()
    envi.save_image(f'outputs/abundances/mf_{mineral_name}.hdr',
                    mf_score, metadata=metadata, force=True)

print("\nAll done!")
```

# =============================================================================
# STEP 6: POSTPROCESSING
# =============================================================================

## Example: Combine SAM + MTMF
```python
# save as: code/combine_sam_mtmf.py

import numpy as np
from spectral import *

def combine_detections(sam_angle, mf_score, 
                       angle_thresh=0.10, mf_thresh=0.5):
    """
    Combine SAM and MF for robust detection
    
    Pixel classified as target only if:
    - SAM angle < angle_thresh AND
    - MF score > mf_thresh
    """
    combined = (sam_angle < angle_thresh) & (mf_score > mf_thresh)
    return combined.astype(np.uint8)

# Example for jarosite
sam_angle = envi.open('outputs/classifications/sam_angle_jarosite.hdr').load()
mf_score = envi.open('outputs/abundances/mf_jarosite.hdr').load()

combined = combine_detections(sam_angle, mf_score, 
                              angle_thresh=0.10, mf_thresh=0.5)

print(f"SAM only: {np.sum(sam_angle < 0.10)} pixels")
print(f"MF only: {np.sum(mf_score > 0.5)} pixels")
print(f"Combined: {np.sum(combined)} pixels")

# Save
img = envi.open('data/hyperion/preprocessed/hyperion_smooth.hdr')
metadata = img.metadata.copy()
envi.save_image('outputs/classifications/combined_jarosite.hdr',
                combined, metadata=metadata, force=True)
```

## Example: Morphological Cleaning
```python
# save as: code/clean_classification.py

import numpy as np
from spectral import *
from scipy import ndimage
from skimage.morphology import remove_small_objects, closing, disk

def clean_classification(class_map, min_size=5, close_radius=1):
    """
    Clean classification map
    - Remove small isolated objects
    - Close small gaps
    """
    # Remove small objects
    cleaned = remove_small_objects(class_map.astype(bool), 
                                    min_size=min_size)
    
    # Morphological closing
    selem = disk(close_radius)
    cleaned = closing(cleaned, selem)
    
    return cleaned.astype(np.uint8)

# Example
class_map = envi.open('outputs/classifications/combined_jarosite.hdr').load()

print(f"Before cleaning: {class_map.sum()} pixels")
cleaned = clean_classification(class_map, min_size=5, close_radius=1)
print(f"After cleaning: {cleaned.sum()} pixels")

# Save
img = envi.open('data/hyperion/preprocessed/hyperion_smooth.hdr')
metadata = img.metadata.copy()
envi.save_image('outputs/classifications/jarosite_final.hdr',
                cleaned, metadata=metadata, force=True)
```

# =============================================================================
# STEP 8: VISUALIZATION AND STATISTICS
# =============================================================================

## Example: Calculate Area Statistics
```python
# save as: code/calculate_stats.py

import numpy as np
from spectral import *
import pandas as pd

def calculate_mineral_areas(classification_files, pixel_size_m=30):
    """
    Calculate area for each classified mineral
    
    Parameters:
    -----------
    classification_files : dict
        {mineral_name: filepath}
    pixel_size_m : float
        Hyperion pixel size in meters (30m)
    """
    stats = []
    
    for mineral_name, filepath in classification_files.items():
        # Load classification
        class_map = envi.open(filepath).load()
        
        # Count pixels
        pixel_count = np.sum(class_map > 0)
        
        # Calculate area
        area_m2 = pixel_count * (pixel_size_m ** 2)
        area_ha = area_m2 / 10000
        area_km2 = area_m2 / 1e6
        
        stats.append({
            'Mineral': mineral_name,
            'Pixels': pixel_count,
            'Area_m2': area_m2,
            'Area_ha': area_ha,
            'Area_km2': area_km2
        })
    
    df = pd.DataFrame(stats)
    return df

# Run
classification_files = {
    'Jarosite': 'outputs/classifications/jarosite_final.hdr',
    'Goethite': 'outputs/classifications/goethite_final.hdr',
    'Hematite': 'outputs/classifications/hematite_final.hdr'
}

stats_df = calculate_mineral_areas(classification_files)
print(stats_df)

# Save
stats_df.to_csv('outputs/stats/mineral_areas.csv', index=False)
print("Saved: outputs/stats/mineral_areas.csv")
```

## Example: Create Spectral Profile Plots
```python
# save as: code/plot_endmembers.py

import pandas as pd
import matplotlib.pyplot as plt

# Load endmember library
df = pd.read_csv('data/spectral_library/resampled/endmember_library.csv',
                 index_col=0)

# Plot
fig, ax = plt.subplots(figsize=(14, 7))

colors = ['#FFD700', '#8B4513', '#DC143C', '#FF8C00', '#87CEEB']
for i, col in enumerate(df.columns):
    ax.plot(df.index, df[col], label=col, linewidth=2.5, 
            color=colors[i % len(colors)])

ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Reflectance', fontsize=14, fontweight='bold')
ax.set_title('AMD Mineral Endmember Spectra\nResampled to Hyperion Wavelengths', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')

# Highlight diagnostic regions
ax.axvspan(2200, 2260, alpha=0.15, color='yellow', label='Jarosite')
ax.axvspan(850, 950, alpha=0.15, color='red', label='Hematite')

plt.tight_layout()
plt.savefig('outputs/figures/endmember_spectra_publication.png', dpi=300)
print("Saved: endmember_spectra_publication.png")
plt.show()
```

# =============================================================================
# QGIS QUICK COMMANDS
# =============================================================================

## Load Results in QGIS (Python Console)
```python
# In QGIS Python Console

# Load classification
layer = iface.addRasterLayer(
    '/path/to/outputs/classifications/jarosite_final.tif',
    'Jarosite Classification'
)

# Style as binary (0/1)
from qgis.core import QgsPalettedRasterRenderer, QgsColorRampShader
renderer = layer.renderer()
# Set colors: 0=transparent, 1=yellow for jarosite
# (GUI easier for this)

# Refresh
layer.triggerRepaint()
```

## Calculate Raster Statistics (QGIS Python Console)
```python
# Get raster statistics
layer = iface.activeLayer()
provider = layer.dataProvider()
stats = provider.bandStatistics(1)  # Band 1

print(f"Min: {stats.minimumValue}")
print(f"Max: {stats.maximumValue}")
print(f"Mean: {stats.mean}")
print(f"Pixels: {stats.elementCount}")
```

# =============================================================================
# VALIDATION EXAMPLES
# =============================================================================

## Example: Extract Values at Points
```python
# save as: code/extract_at_points.py

import numpy as np
from spectral import *
import geopandas as gpd
import rasterio

def extract_raster_at_points(raster_file, points_file):
    """
    Extract raster values at point locations
    
    Parameters:
    -----------
    raster_file : str
        Path to classification raster
    points_file : str
        Path to points shapefile with geochemistry
    
    Returns:
    --------
    GeoDataFrame with extracted values
    """
    # Load points
    points = gpd.read_file(points_file)
    
    # Load raster
    with rasterio.open(raster_file) as src:
        # Extract values at each point
        coords = [(x, y) for x, y in zip(points.geometry.x, points.geometry.y)]
        values = [x[0] for x in src.sample(coords)]
    
    points['classification'] = values
    return points

# Example usage
points = extract_raster_at_points(
    'outputs/classifications/jarosite_final.tif',
    'data/validation/geochem_samples.shp'
)

# Analyze
jarosite_detected = points[points['classification'] == 1]
print(f"Jarosite detected at {len(jarosite_detected)} points")
print(f"Mean pH: {jarosite_detected['pH'].mean():.2f}")
```

# =============================================================================
# TROUBLESHOOTING SNIPPETS
# =============================================================================

## Check Hyperion Cube Info
```python
from spectral import *
img = envi.open('data/hyperion/preprocessed/hyperion_clean.hdr')
print(img.metadata)
print(f"Shape: {img.nrows} × {img.ncols} × {img.nbands}")
print(f"Wavelengths: {len(img.metadata['wavelength'])}")
```

## Convert ENVI to GeoTIFF
```python
import rasterio
from spectral import *

img = envi.open('file.hdr')
cube = img.load()

# Save as GeoTIFF
with rasterio.open('output.tif', 'w',
                   driver='GTiff',
                   height=cube.shape[0],
                   width=cube.shape[1],
                   count=cube.shape[2],
                   dtype=cube.dtype) as dst:
    for b in range(cube.shape[2]):
        dst.write(cube[:, :, b], b+1)
```

## Quick Visual Check
```python
import matplotlib.pyplot as plt
from spectral import *

img = envi.open('file.hdr')
cube = img.load()

# Show RGB composite
plt.figure(figsize=(10, 10))
plt.imshow(cube[:, :, [29, 23, 13]])  # R, G, B bands
plt.title('Hyperion RGB Composite')
plt.axis('off')
plt.show()
```

# =============================================================================
# END OF QUICK START GUIDE
# =============================================================================
