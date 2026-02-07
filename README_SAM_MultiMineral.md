# Multi-Mineral SAM Analysis Scripts

## Overview

These scripts process multiple mineral spectra from a library folder and generate:
1. **Individual match score maps** for each mineral (0-1 scale)
2. **Composite map** showing the dominant mineral at each pixel
3. **Classification statistics** (CSV format)

Based on your working SAM example, extended for batch processing.

---

## Files Provided

### 1. `multi_mineral_sam.py` - **SEQUENTIAL VERSION**
- Processes minerals one at a time
- Simpler, easier to debug
- Good for small libraries (<10 minerals)

### 2. `multi_mineral_sam_parallel.py` - **PARALLEL VERSION**
- Processes multiple minerals simultaneously
- **Much faster** for large libraries
- Automatically uses all CPU cores
- Recommended for >5 minerals

---

## Quick Start

### 1. Update Paths in the Script

Edit these lines at the top of either script:

```python
# Paths
HDR_FILE = r"C:\path\to\your\reflectance.hdr"
LIBRARY_FOLDER = r"C:\path\to\mineral\library\folder"
OUTPUT_FOLDER = r"C:\path\to\output\results"

# SAM Parameters
SAM_THRESHOLD = 0.1  # Adjust as needed
```

### 2. Prepare Your Mineral Library

Your library folder should contain CSV files with:
- **Column 0**: Wavelength (nm)
- **Column 1**: Reflectance
- **Filename**: Will be used as mineral name (e.g., `Jarosite_K.csv` → "Jarosite_K")

Example structure:
```
Library_folder/
├── Jarosite_GDS98_K_90C_Syn_BECKa.csv
├── Goethite_WS968.csv
├── Hematite_HS9.csv
├── Kaolinite_CM9.csv
└── Gypsum_HS333.csv
```

### 3. Run the Script

```bash
python multi_mineral_sam.py
```

or for parallel processing:

```bash
python multi_mineral_sam_parallel.py
```

---

## Configuration Options

### Basic Settings

```python
SAM_THRESHOLD = 0.1  # Radians (~5.7°)
                     # Lower = stricter matching
                     # Higher = more detections
```

### Output Control

```python
SAVE_INDIVIDUAL_MAPS = True   # Save PNG for each mineral
SAVE_COMPOSITE_MAP = True     # Save composite classification map
SHOW_PLOTS = False            # Set True to display plots interactively
```

### Parallel Processing (parallel version only)

```python
USE_PARALLEL = True    # Enable/disable parallel processing
N_PROCESSES = None     # None = use all CPUs
                       # Or set specific number: N_PROCESSES = 4
```

---

## Output Files

After running, your `OUTPUT_FOLDER` will contain:

### 1. Individual Match Maps
- `{mineral_name}_SAM_match.png` - One per mineral
- Color scale: 0 (no match) → 1 (perfect match)
- Inferno colormap

### 2. Composite Map
- `Composite_SAM_map.png`
- Shows dominant mineral at each pixel
- Different color for each mineral
- Includes legend

### 3. Statistics
- `classification_statistics.csv`
- Contains:
  - Class name
  - Class code
  - Number of pixels
  - Percentage of image

---

## Understanding the Results

### Match Score Maps (0-1 scale)

- **Score = 1.0**: Perfect spectral match
- **Score = 0.5**: Moderate match (threshold dependent)
- **Score = 0.0**: No match (angle ≥ threshold)

The match score is calculated as:
```
match_score = max(0, 1 - SAM_angle / threshold)
```

### Composite Map

Shows the **dominant mineral** at each pixel based on:
1. Which mineral has the highest match score
2. Only if that score > 0.5
3. Otherwise, pixel is "Unclassified"

---

## Troubleshooting

### Problem: "No CSV files found"
**Solution**: Check that:
- `LIBRARY_FOLDER` path is correct
- CSV files exist in that folder
- Files have `.csv` extension (not `.txt`)

### Problem: "Wavelength interpolation error"
**Solution**: Your mineral spectra might not cover the full Hyperion range
- Check your CSV wavelength range
- The script uses extrapolation but results may be unreliable outside the original range

### Problem: "Out of memory error" (large images)
**Solution**: 
- Process fewer minerals at once
- Reduce `N_PROCESSES` in parallel version
- Use sequential version instead

### Problem: All pixels classified as one mineral
**Solution**: Your threshold might be too high
- Try reducing `SAM_THRESHOLD` (e.g., from 0.1 to 0.08)
- Check if mineral spectra are properly normalized

---

## Advanced Customization

### Adjust Detection Threshold for Composite Map

In the `create_composite_map` function, change this line:
```python
# Current: pixels with score > 0.5 are classified
composite_map = np.where(max_score > 0.5, max_idx + 1, 0)

# More strict: only pixels with score > 0.7
composite_map = np.where(max_score > 0.7, max_idx + 1, 0)
```

### Change Color Scheme

For individual maps:
```python
# Current: 'inferno' colormap
im = ax.imshow(match_score, cmap='inferno', vmin=0, vmax=1)

# Try: 'viridis', 'plasma', 'hot', 'jet', etc.
im = ax.imshow(match_score, cmap='viridis', vmin=0, vmax=1)
```

For composite map:
```python
# Current: 'tab20' for up to 20 minerals
cmap = plt.cm.get_cmap('tab20', n_classes)

# Try: 'Set1', 'Set3', 'Paired', etc.
cmap = plt.cm.get_cmap('Set3', n_classes)
```

### Export as GeoTIFF (for GIS)

Add this function and call it in main():

```python
def save_as_geotiff(array, output_path, img_metadata):
    """Save array as GeoTIFF with georeferencing"""
    from osgeo import gdal, osr
    
    # Extract geotransform and projection from metadata
    # Implementation depends on your Hyperion metadata structure
    # See spectral.io documentation
    
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, 
                           array.shape[1], 
                           array.shape[0],
                           1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(array)
    # Set geotransform and projection here
    dataset.FlushCache()
```

---

## Performance Tips

### Sequential Version
- **Pro**: Simpler, uses less memory
- **Con**: Slower for many minerals
- **Best for**: <10 minerals, debugging

### Parallel Version
- **Pro**: Much faster (3-8x speedup typical)
- **Con**: Higher memory usage
- **Best for**: >5 minerals, production runs

**Typical processing times** (example: 256×3242 image, 8 minerals):
- Sequential: ~2-4 minutes
- Parallel (8 cores): ~30-60 seconds

---

## Integration with Your Workflow

### After SAM Classification

You can use the results for:

1. **Validation**: Compare with field data or other methods
2. **Abundance mapping**: Feed into MTMF or linear unmixing
3. **Change detection**: Compare scenes from different dates
4. **GIS analysis**: Import composite map into QGIS/ArcGIS

### Combining with MTMF

If you want to run MTMF after SAM, you can extract the endmember spectra:

```python
# After loading library in main()
endmember_dict = {}
for csv_file in csv_files:
    ref_spectrum, mineral_name = load_and_resample_spectrum(csv_file, wavelengths)
    endmember_dict[mineral_name] = ref_spectrum

# Now use endmember_dict for MTMF
# (See your hyperion_workflow.py for MTMF implementation)
```

---

## Key Differences from Original Script

### Your Working Script (Python_SAM_example.py):
- Single mineral
- Manual file paths
- Direct visualization

### These New Scripts:
- **Multiple minerals** processed automatically
- Library-based approach
- Batch output generation
- Optional parallel processing
- Statistical summaries
- Composite classification map

---

## Questions or Issues?

If you encounter problems:
1. Check file paths are correct
2. Verify CSV format matches expected structure
3. Try sequential version first (easier to debug)
4. Check console output for detailed error messages
5. Reduce threshold if no detections occur
6. Increase threshold if too many false positives

## Citation

If you use these scripts in research, please cite the original SAM algorithm:

> Kruse, F. A., et al. (1993). The spectral image processing system (SIPS)—interactive visualization and analysis of imaging spectrometer data. Remote sensing of environment, 44(2-3), 145-163.
