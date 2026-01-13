# Hyperion AMD Mineral Mapping - Setup and Usage Guide

## Environment Setup (COMPLETED)

### 1. Conda Environment: `hyperion`
- **Status**: ✓ Ready
- **Python Version**: 3.9.25
- **Location**: `C:\Users\lorie\anaconda3\envs\hyperion`

### 2. Installed Packages
All required packages have been installed:
- ✓ numpy (2.0.2)
- ✓ pandas (2.3.3)
- ✓ matplotlib (3.9.4)
- ✓ scipy (1.13.1)
- ✓ scikit-image (0.24.0)
- ✓ scikit-learn (1.6.1)
- ✓ spectral (0.24) - for ENVI file handling
- ✓ pysptools (0.15.0) - for hyperspectral classification
- ✓ rasterio (1.4.3) - for geospatial raster I/O
- ✓ geopandas (1.0.1) - for geospatial vector data

### 3. Activation Command
Always activate the environment before running the workflow:
```bash
conda activate hyperion
```

---

## Data Preparation (BEFORE RUNNING)

### Required Input Data Structure

Create the following directory structure in your project folder:

```
PROCESSING_AND_POST/
├── hyperion_workflow.py          # Main workflow script (ready)
├── SETUP_AND_USAGE_GUIDE.md      # This guide
│
├── data/                          # Your Hyperion data
│   ├── hyperion_cube.hdr          # ENVI header file from SUREHYP
│   ├── hyperion_cube              # ENVI data file (binary)
│   └── hyperion_cube.tif          # Optional: GeoTIFF version
│
├── spectral_library/              # Mineral spectral signatures
│   ├── jarosite.txt               # Format: wavelength(nm), reflectance
│   ├── goethite.txt
│   ├── hematite.txt
│   ├── kaolinite.txt
│   ├── gypsum.txt
│   └── schwertmannite.txt         # If available
│
├── outputs/                       # Intermediate results
│   └── (will be created)
│
└── final_outputs/                 # Final products
    └── (will be created)
```

---

## STEP-BY-STEP WORKFLOW EXECUTION

### STEP 0: Prepare Your Data

#### A. Hyperion Image Cube (from SUREHYP preprocessing)
You should have completed atmospheric correction using SUREHYP (Step 1 mentioned in code).
Expected output: ENVI format files (.hdr + binary)

**Place your files in**: `./data/`
- `hyperion_cube.hdr` - ENVI header with wavelength metadata
- `hyperion_cube` - binary data file

#### B. Download Spectral Library

**Source**: USGS Spectral Library Version 7
- **URL**: https://doi.org/10.5066/F7RR1WDJ
- **Reference**: Kokaly et al., 2017

**Required minerals for AMD mapping**:
1. **Jarosite** (K-jarosite, Na-jarosite, H3O-jarosite)
2. **Goethite**
3. **Hematite**
4. **Kaolinite** (as confuser)
5. **Illite** (as confuser)
6. **Gypsum** (common in AMD)
7. **Schwertmannite** (if available - check publications)

**File format**: Each mineral should be a `.txt` file with two columns:
```
# Header (optional)
wavelength_nm  reflectance
356.0          0.0523
367.0          0.0541
...
```

**Save to**: `./spectral_library/`

---

### STEP 1: Verify Your Setup

Activate environment and test imports:
```bash
conda activate hyperion
cd "c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\PROCESSING_AND_POST"
python -c "from hyperion_workflow import *; print('All modules loaded successfully!')"
```

---

### STEP 2: Run the Complete Workflow

#### Option A: Run All Steps Automatically
```bash
conda activate hyperion
python hyperion_workflow.py
```

This will execute:
- STEP 2: Build spectral library
- STEP 3: Apply spectral smoothing
- STEP 4: SAM classification
- STEP 5: MTMF abundance mapping
- STEP 6: Post-processing
- STEP 7: Validation
- STEP 8: Export results

#### Option B: Run Steps Individually (Recommended for First Time)

Create a new script `run_workflow_step_by_step.py`:

```python
"""
Run Hyperion workflow step by step with checkpoints
"""
import numpy as np
from hyperion_workflow import *

# =============================================================================
# STEP 2: Build Spectral Library
# =============================================================================
print("=" * 60)
print("STEP 2: Building Spectral Library")
print("=" * 60)

# Update these paths to your actual data locations
HYPERION_HDR = './data/hyperion_cube.hdr'
LIBRARY_DIR = './spectral_library/'
OUTPUT_DIR = './outputs/'

# Load Hyperion wavelengths from your preprocessed data
print("Loading Hyperion wavelengths...")
hyperion_wvl = load_hyperion_wavelengths(HYPERION_HDR)
print(f"  → Found {len(hyperion_wvl)} usable bands")
print(f"  → Wavelength range: {hyperion_wvl.min():.1f} - {hyperion_wvl.max():.1f} nm")

# Create endmember library
print("\nResampling library spectra to Hyperion wavelengths...")
endmembers = create_endmember_library(
    library_dir=LIBRARY_DIR,
    hyperion_wavelengths=hyperion_wvl,
    output_file=OUTPUT_DIR + 'endmember_library.sli'
)
print(f"  → Created library with {len(endmembers)} minerals:")
for name in endmembers.keys():
    print(f"     - {name}")

print("\n✓ STEP 2 COMPLETE\n")

# =============================================================================
# STEP 3: Enhance Spectral Features
# =============================================================================
print("=" * 60)
print("STEP 3: Enhancing Spectral Features")
print("=" * 60)

# Load preprocessed cube
print("Loading Hyperion image cube...")
cube = envi.open(HYPERION_HDR).load()
print(f"  → Cube shape: {cube.shape} (rows, cols, bands)")

# Apply Savitzky-Golay smoothing
print("Applying Savitzky-Golay smoothing...")
print("  (This may take a few minutes...)")
cube_smooth = apply_savgol_smoothing(cube, window_length=9, polyorder=2)
print("  → Smoothing complete")

# Optional: Save smoothed cube for later use
print("Saving smoothed cube...")
envi.save_image(OUTPUT_DIR + 'cube_smoothed.hdr', cube_smooth,
                metadata=envi.open(HYPERION_HDR).metadata, force=True)

print("\n✓ STEP 3 COMPLETE\n")

# =============================================================================
# STEP 4: Run SAM Classification
# =============================================================================
print("=" * 60)
print("STEP 4: Running SAM Classification")
print("=" * 60)

print("Running Spectral Angle Mapper...")
class_map, angle_map = run_sam_classification(
    cube_smooth,
    endmembers,
    threshold=0.10  # 0.10 radians ≈ 5.7 degrees
)
print(f"  → Classification map shape: {class_map.shape}")
print(f"  → Angle map shape: {angle_map.shape}")

# Save results
print("Saving SAM results...")
np.save(OUTPUT_DIR + 'sam_class_map.npy', class_map)
np.save(OUTPUT_DIR + 'sam_angle_map.npy', angle_map)

print("\n✓ STEP 4 COMPLETE\n")

# =============================================================================
# STEP 5: Run MTMF for Abundance Maps
# =============================================================================
print("=" * 60)
print("STEP 5: Running MTMF for Abundances")
print("=" * 60)

abundance_maps = {}
mineral_names = list(endmembers.keys())

for i, (mineral_name, spectrum) in enumerate(endmembers.items()):
    print(f"  [{i+1}/{len(endmembers)}] Processing {mineral_name}...")
    mf_score, infeas = run_mtmf(cube_smooth, spectrum)
    abundance_maps[mineral_name] = {
        'mf_score': mf_score,
        'infeasibility': infeas
    }

    # Save individual results
    np.save(OUTPUT_DIR + f'mtmf_{mineral_name}_score.npy', mf_score)
    np.save(OUTPUT_DIR + f'mtmf_{mineral_name}_infeas.npy', infeas)

print("\n✓ STEP 5 COMPLETE\n")

# =============================================================================
# STEP 6: Post-processing
# =============================================================================
print("=" * 60)
print("STEP 6: Post-processing Classifications")
print("=" * 60)

refined_maps = {}

for i, mineral_name in enumerate(mineral_names):
    print(f"  Processing {mineral_name}...")

    # Combine SAM and MTMF
    combined = combine_sam_mtmf(
        sam_class=class_map,
        sam_angle=angle_map[:, :, i],
        mtmf_score=abundance_maps[mineral_name]['mf_score'],
        mtmf_infeasibility=abundance_maps[mineral_name]['infeasibility'],
        angle_threshold=0.10,
        mf_threshold=0.5,
        infeas_threshold=0.1
    )

    # Clean up classification
    cleaned = clean_classification_map(combined, min_pixels=5)
    refined_maps[mineral_name] = cleaned

    # Save refined map
    np.save(OUTPUT_DIR + f'refined_{mineral_name}.npy', cleaned)

    # Print statistics
    pixel_count = np.sum(cleaned)
    area_ha = pixel_count * (30 * 30) / 10000  # Hyperion 30m pixels
    print(f"     → Detected: {pixel_count} pixels ({area_ha:.2f} ha)")

print("\n✓ STEP 6 COMPLETE\n")

# =============================================================================
# STEP 7 & 8: Statistics and Visualization
# =============================================================================
print("=" * 60)
print("STEP 7-8: Generating Statistics and Plots")
print("=" * 60)

# Generate statistics
stats = generate_mineral_statistics(refined_maps, masks={'vegetation': np.zeros(cube.shape[:2], dtype=bool),
                                                          'water': np.zeros(cube.shape[:2], dtype=bool)})
print("\nMineral Distribution Statistics:")
print(stats.to_string(index=False))

# Save statistics
stats.to_csv(OUTPUT_DIR + 'mineral_statistics.csv', index=False)

# Plot spectral signatures
print("\nGenerating spectral plots...")
plot_mineral_spectra(endmembers, hyperion_wvl, OUTPUT_DIR + 'mineral_spectra.png')

print("\n" + "=" * 60)
print("✓ ALL STEPS COMPLETE!")
print("=" * 60)
print(f"\nResults saved to: {OUTPUT_DIR}")
print("\nNext steps:")
print("1. Review output maps in QGIS")
print("2. Adjust thresholds if needed")
print("3. Export final products as GeoTIFF for publication")
```

**Run this script**:
```bash
conda activate hyperion
python run_workflow_step_by_step.py
```

---

## Expected Outputs

### In `./outputs/` directory:

1. **Spectral Library**:
   - `endmember_library.sli.hdr` - ENVI spectral library header
   - `endmember_library.sli` - Binary spectral library
   - `endmember_library.csv` - Human-readable CSV version

2. **Processed Cube**:
   - `cube_smoothed.hdr` - Smoothed Hyperion cube

3. **Classification Results**:
   - `sam_class_map.npy` - SAM classification map
   - `sam_angle_map.npy` - SAM angle maps
   - `mtmf_[mineral]_score.npy` - MTMF scores per mineral
   - `mtmf_[mineral]_infeas.npy` - MTMF infeasibility per mineral
   - `refined_[mineral].npy` - Final refined maps

4. **Statistics & Plots**:
   - `mineral_statistics.csv` - Area statistics table
   - `mineral_spectra.png` - Spectral signature plot

---

## Troubleshooting

### Issue 1: "File not found" errors
**Solution**: Check that all file paths in your script match your actual data locations.

### Issue 2: Memory errors with large images
**Solution**: Process image in tiles. Add this function to workflow:
```python
def process_in_tiles(cube, function, tile_size=500):
    # Process cube in tiles to reduce memory usage
    pass
```

### Issue 3: No minerals detected
**Solution**:
1. Check if spectral library wavelengths match Hyperion range
2. Adjust SAM threshold (try 0.15-0.20 radians)
3. Verify preprocessing quality

### Issue 4: Import errors
**Solution**: Make sure you activated the conda environment:
```bash
conda activate hyperion
```

---

## Adjusting Parameters

### SAM Threshold
- **Default**: 0.10 radians (≈ 5.7°)
- **Liberal**: 0.15-0.20 radians (more detections, less accurate)
- **Conservative**: 0.05-0.08 radians (fewer detections, more accurate)

### MTMF Thresholds
- **MF score**: Higher = stronger match (default: 0.5)
- **Infeasibility**: Lower = more feasible (default: 0.1)

### Smoothing Parameters
- **Window length**: 7-11 bands (must be odd)
- **Polynomial order**: 2-3 (higher = less smoothing)

---

## Visualizing Results in QGIS

1. Convert numpy arrays to GeoTIFF using rasterio
2. Import GeoTIFF files into QGIS
3. Apply color ramps for each mineral
4. Create composite map with transparency

Example conversion script:
```python
import rasterio
from rasterio.transform import from_origin

# Get georeference from original Hyperion file
with rasterio.open('data/hyperion_cube.tif') as src:
    transform = src.transform
    crs = src.crs

# Export refined map
with rasterio.open(
    'final_outputs/jarosite_map.tif',
    'w',
    driver='GTiff',
    height=refined_maps['jarosite'].shape[0],
    width=refined_maps['jarosite'].shape[1],
    count=1,
    dtype=refined_maps['jarosite'].dtype,
    crs=crs,
    transform=transform
) as dst:
    dst.write(refined_maps['jarosite'], 1)
```

---

## Citation

If you use this workflow in your research, please cite:

**USGS Spectral Library**:
Kokaly, R.F., Clark, R.N., Swayze, G.A., Livo, K.E., Hoefen, T.M., Pearson, N.C., Wise, R.A., Benzel, W.M., Lowers, H.A., Driscoll, R.L., and Klein, A.J., 2017, USGS Spectral Library Version 7 Data: U.S. Geological Survey data release, https://doi.org/10.5066/F7RR1WDJ.

**Python Packages**:
- Spectral Python: http://www.spectralpython.net
- pysptools: https://pysptools.sourceforge.io

---

## Support

For issues with:
- **This workflow**: Check your data paths and parameters
- **Python packages**: Consult package documentation
- **SUREHYP preprocessing**: Refer to SUREHYP manual

Good luck with your AMD mineral mapping!