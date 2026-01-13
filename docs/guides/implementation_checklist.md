# Implementation Checklist and Software Requirements
# AMD Mineral Mapping with Hyperion - Rio Tinto Project

# =============================================================================
# SOFTWARE REQUIREMENTS
# =============================================================================

## Python Environment (Recommended: Conda)

### Create dedicated environment:
```bash
conda create -n hyperion python=3.9
conda activate hyperion
```

### Core scientific libraries:
```bash
# NumPy, SciPy, Pandas
conda install numpy scipy pandas matplotlib

# Geospatial
conda install -c conda-forge rasterio gdal
conda install -c conda-forge geopandas

# Image processing
conda install scikit-image
pip install opencv-python
```

### Hyperspectral-specific libraries:
```bash
# Spectral Python
pip install spectral

# PySptools (for SAM, MTMF, unmixing)
pip install pysptools

# SUREHYP (you mentioned you used this)
# If not already installed:
pip install surehyp
```

### Optional but useful:
```bash
# Seaborn for better plots
pip install seaborn

# Jupyter for interactive work
conda install jupyter

# h5py if working with HDF5 Hyperion files
conda install h5py
```

## QGIS

### Version: 
- QGIS 3.28+ (Long Term Release) recommended
- Download: https://qgis.org/en/site/forusers/download.html

### Required Plugins:
1. **Semi-Automatic Classification Plugin (SCP)**
   - For: Band management, spectral signatures
   - Install: Plugins → Manage and Install Plugins → Search "Semi-Automatic"

2. **Orfeo ToolBox (OTB)**
   - For: SAM classification, advanced processing
   - Usually included with QGIS, verify in Processing Toolbox

3. **Profile Tool**
   - For: Spectral profile extraction
   - Install from plugin repository

4. **QuickMapServices** (optional)
   - For: Easy basemap addition
   - Useful for context

### System Requirements:
- RAM: 16 GB minimum, 32 GB recommended
- Storage: 100 GB free (for imagery and processing)
- GPU: Not required, but helps with rendering

## SNAP (Sentinel Application Platform)

### Version:
- SNAP 9.0+ recommended
- Download: https://step.esa.int/main/download/snap-download/

### Configuration:
```
After installation:
1. Help → Check for Updates
2. Tools → Plugins → Available Plugins
   - Install any hyperspectral-related plugins
3. Tools → Options → Performance
   - Set cache: 4-8 GB
   - VM parameters: -Xmx8G (or more)
```

### System Requirements:
- RAM: 16 GB minimum
- Java: Included with SNAP
- Note: SNAP can be memory-intensive

## Supporting Tools

### LibreOffice (for DOCX → PDF conversion in QGIS workflow):
```bash
sudo apt-get install libreoffice  # Linux
# Or download from: https://www.libreoffice.org/
```

### Pandoc (for document format conversion):
```bash
conda install -c conda-forge pandoc
```

### Git (for version control):
```bash
conda install git
```

# =============================================================================
# DATA REQUIREMENTS
# =============================================================================

## Input Data (You should have from Step 1)

### Hyperion imagery:
- ✓ Atmospherically corrected
- ✓ Georeferenced
- ✓ Bad bands removed (via SUREHYP)
- Format: GeoTIFF or ENVI format (.hdr + .img)
- Size: Expect ~1-2 GB per scene

### Ancillary data needed:

1. **Digital Elevation Model (DEM)**
   - For topographic context
   - Source: SRTM 30m or similar
   - Download: https://earthexplorer.usgs.gov/

2. **Sentinel-2 imagery (for validation)**
   - Same area, similar date
   - Download: https://scihub.copernicus.eu/

3. **Stream network / drainage (if available)**
   - For spatial analysis
   - Can derive from DEM

## Reference Data to Collect

### 1. USGS Spectral Library
```
Download location: https://crustal.usgs.gov/speclab/QueryAll07a.php

Required minerals:
- Jarosite (multiple types):
  * Jarosite-K GDS84 Na
  * Jarosite GDS99 K1
  * Jarosite HS342.3B

- Goethite:
  * Goethite WS222
  * Goethite GDS27

- Hematite:
  * Hematite GDS27
  * Hematite HS102.3B

- Clays (confusers):
  * Kaolinite CM9
  * Illite GDS4
  * Smectite SWa-1

- Additional AMD minerals:
  * Gypsum HS333.3B
  * Schwertmannite (if available)

File format: .txt or .asd
Columns: Wavelength (µm or nm), Reflectance
```

### 2. Literature-Based Spectra

Key papers to find:

```
1. Bigham et al. (1996) - Schwertmannite spectra
   "A poorly crystallized oxyhydroxysulfate of iron..."
   Clays and Clay Minerals

2. Crowley et al. (2003) - Spectral reflectance
   "Spectral reflectance properties (0.4-2.5 µm) of secondary..."
   Geochemistry: Exploration, Environment, Analysis

3. Acero et al. (2006) - Rio Tinto specific
   "The behavior of trace elements during schwertmannite..."
   Geochimica et Cosmochimica Acta

4. Sánchez-España et al. (2016) - Recent Rio Tinto mineralogy
   "Mineralogy and geochemistry of the Tinto and Odiel..."
   Journal of Geochemical Exploration

Action: Download supplementary materials, extract spectra
```

### 3. Published Rio Tinto Maps (for validation)

```
Sources:
1. Geological maps from Spanish Geological Survey (IGME)
2. Geochemical sample maps from publications
3. Previous remote sensing studies:
   - Sentinel-2 iron oxide indices
   - ASTER mineral maps
   - Published Hyperion/AVIRIS studies if available

File formats: 
- Shapefiles (.shp)
- GeoTIFF
- KML for Google Earth
- Or digitize from published figures
```

### 4. Geochemical Data (if accessible)

```
Ideal data points:
- Location (lat/lon)
- pH
- Fe concentration
- SO4 concentration
- Al concentration
- Sample date

Sources:
- Open databases
- Published papers with supplementary data
- Spanish environmental agencies
- Research collaborations

Format: CSV or Excel
Minimum: 30-50 points for meaningful validation
```

# =============================================================================
# STEP-BY-STEP IMPLEMENTATION CHECKLIST
# =============================================================================

## Phase 1: Setup and Preparation

### Week 1: Environment Setup
- [ ] Install Python (conda environment)
- [ ] Install all Python libraries (test imports)
- [ ] Install QGIS 3.28+
- [ ] Install required QGIS plugins
- [ ] Install SNAP Desktop
- [ ] Verify all software works (test with sample data)

### Week 1-2: Data Collection
- [ ] Organize Hyperion data from Step 1
- [ ] Download USGS spectral library minerals
- [ ] Collect AMD-specific spectra from publications
- [ ] Download DEM for study area
- [ ] Download Sentinel-2 imagery (validation)
- [ ] Gather published Rio Tinto maps
- [ ] Organize geochemical data (if available)

### Week 2: Data Organization
```
Create directory structure:
project/
├── data/
│   ├── hyperion/
│   │   ├── raw/
│   │   ├── preprocessed/  # Output from SUREHYP
│   │   └── metadata/
│   ├── spectral_library/
│   │   ├── usgs/
│   │   ├── publications/
│   │   └── resampled/
│   ├── ancillary/
│   │   ├── dem/
│   │   ├── sentinel2/
│   │   └── streams/
│   └── validation/
│       ├── maps/
│       ├── geochemistry/
│       └── digitized/
├── code/
│   ├── hyperion_workflow.py
│   ├── spectral_library.py
│   ├── sam_classification.py
│   ├── validation.py
│   └── utils.py
├── outputs/
│   ├── classifications/
│   ├── abundances/
│   ├── masks/
│   ├── figures/
│   └── stats/
├── qgis_projects/
├── docs/
└── README.md
```

- [ ] Create project directory structure
- [ ] Copy Hyperion preprocessed data to project
- [ ] Organize all reference data

## Phase 2: Spectral Library Construction (Step 2)

### Week 3: Build Library
- [ ] Load USGS library files
- [ ] Load publication-based spectra
- [ ] Extract Hyperion wavelengths from metadata
- [ ] Verify usable bands list (compare with SUREHYP output)
- [ ] Resample all spectra to Hyperion wavelengths
- [ ] Create endmember library file (.sli and .csv)
- [ ] Visualize all endmember spectra (QC check)
- [ ] Save library in multiple formats

### Validation checkpoint:
- [ ] Plot all endmembers on same graph
- [ ] Verify diagnostic features visible:
  * Jarosite: 2.20-2.26 µm absorption
  * Goethite: ~900 nm shoulder
  * Hematite: 850-950 nm crystal field
- [ ] Check for unrealistic values (negative, >1)

## Phase 3: Spectral Enhancement (Step 3)

### Week 3-4: Preprocessing
- [ ] Load Hyperion cube from Step 1
- [ ] Verify dimensions (rows × cols × bands)
- [ ] Check for remaining artifacts (stripes, noise)
- [ ] Apply Savitzky-Golay smoothing
  * Test window lengths: 7, 9, 11
  * Use polyorder = 2 or 3
  * Compare smoothed vs original spectra
- [ ] (Optional) Apply continuum removal
  * Test on sample pixels first
  * May help with goethite/hematite separation
- [ ] Save enhanced cube
- [ ] Create before/after comparison plots

### Quality checks:
- [ ] Visual inspection in QGIS (RGB composites)
- [ ] Extract random pixel spectra, verify smoothing
- [ ] Check that diagnostic features preserved

## Phase 4: Create Masks (Part of Step 1, but if not done)

### Week 4: Masking
- [ ] Create NDVI layer
- [ ] Threshold NDVI for vegetation mask (> 0.3)
- [ ] Create water mask (NIR < threshold)
- [ ] Create cloud/shadow mask (if needed)
- [ ] Combine all masks
- [ ] Visual QC in QGIS
- [ ] Save mask layers as GeoTIFF

## Phase 5: SAM Classification (Step 4)

### Week 5: Run SAM
- [ ] Load enhanced Hyperion cube
- [ ] Load endmember library
- [ ] Verify dimensions match
- [ ] Run SAM for each mineral:
  * Start with threshold = 0.10 radians
  * Generate classification map
  * Generate angle map
- [ ] Create minimum angle map (quality indicator)
- [ ] Apply masks to remove vegetation/water
- [ ] Save all classification results

### Parameter optimization:
- [ ] Test threshold range: 0.08, 0.10, 0.12, 0.15
- [ ] Visual comparison in QGIS
- [ ] Look for:
  * Too strict (0.08): Small, fragmented patches
  * Too loose (0.15): Large, unlikely areas
- [ ] Select optimal threshold per mineral

### Week 5: Visualization in QGIS
- [ ] Import SAM results into QGIS
- [ ] Create styled layers for each mineral
- [ ] Overlay on Sentinel-2 RGB
- [ ] Visual validation:
  * Jarosite: Yellow sediments/terraces
  * Goethite: Brown oxidized zones
  * Hematite: Red ancient terraces
- [ ] Take notes on suspicious areas

## Phase 6: Abundance Mapping (Step 5)

### Week 6: MTMF and Unmixing
- [ ] Run Matched Filter for each mineral
- [ ] Run MTMF for each mineral
  * MF score (abundance indicator)
  * Infeasibility (false positive indicator)
- [ ] Run Linear Spectral Unmixing (FCLS)
  * All endmembers together
  * Get abundance fractions
- [ ] Save all abundance/confidence maps

### Threshold tuning:
- [ ] For MTMF: Test MF score thresholds
- [ ] For MTMF: Test infeasibility thresholds
- [ ] Compare MTMF vs SAM results
- [ ] Document optimal thresholds

## Phase 7: Postprocessing (Step 6)

### Week 7: Refine Classifications
- [ ] Combine SAM + MTMF criteria:
  * SAM angle < threshold AND
  * MF score > threshold AND
  * Infeasibility < threshold
- [ ] Apply morphological filters:
  * Remove isolated pixels (< 5 pixels)
  * Close small gaps
  * Smooth boundaries
- [ ] Apply majority filter (3×3)
- [ ] Final QC in QGIS
- [ ] Save refined classification maps

### Statistical summary:
- [ ] Count pixels per mineral class
- [ ] Calculate areas (ha)
- [ ] Calculate percentages
- [ ] Create summary table

## Phase 8: Validation (Step 7)

### Week 8-9: Indirect Validation

#### With published maps:
- [ ] Import published mineral distribution maps
- [ ] Digitize if needed (in QGIS)
- [ ] Calculate spatial overlap
  * Precision (user's accuracy)
  * Recall (producer's accuracy)
  * F1-score
- [ ] Create confusion matrix if possible

#### With geochemical data:
- [ ] Import geochemical sample points
- [ ] Extract classification at each point
- [ ] Test correlations:
  * Jarosite vs pH (expect pH < 3)
  * Jarosite vs Fe (expect high)
  * Jarosite vs SO4 (expect high)
- [ ] Statistical tests (t-test, ANOVA)
- [ ] Create validation plots

#### With Sentinel-2:
- [ ] Calculate iron oxide index from S2
- [ ] Compare spatial patterns
- [ ] Visual correlation assessment

#### Spatial consistency:
- [ ] Overlay with DEM
- [ ] Test elevation associations
- [ ] Test proximity to streams
- [ ] Cluster analysis (K-means on locations)

### Week 9: Validation Report
- [ ] Compile all validation results
- [ ] Create validation plots
- [ ] Write validation summary
- [ ] Identify limitations
- [ ] Suggest improvements

## Phase 9: Final Products (Step 8)

### Week 10: Map Production in QGIS
- [ ] Create master QGIS project
- [ ] Import all final layers
- [ ] Style all layers professionally
- [ ] Create multiple map layouts:
  * Individual mineral maps
  * Combined mineral map
  * Confidence/quality maps
  * Validation comparison maps
- [ ] Add all map elements:
  * Legend, scale bar, north arrow
  * Title, date, credits
  * Coordinate grid
  * Study area inset map
- [ ] Export high-resolution images (300 dpi)
- [ ] Export PDFs for publication

### Week 10: Statistical Outputs
- [ ] Generate area statistics table
- [ ] Create spectral profile plots
- [ ] Create abundance distribution histograms
- [ ] Generate validation plots:
  * Confusion matrices
  * Correlation plots
  * Spatial agreement maps

### Week 10-11: Documentation
- [ ] Write methods description
- [ ] Document all parameters used
- [ ] Create README for project
- [ ] Document data sources
- [ ] List all software versions
- [ ] Create workflow diagram
- [ ] Write results summary

### Week 11: Final Deliverables
- [ ] Organize all outputs
- [ ] Create deliverables package:
  * Final classification GeoTIFFs
  * Abundance maps
  * Confidence/quality layers
  * Masks
  * QGIS project file
  * High-res map PDFs
  * Statistics tables (CSV, Excel)
  * Spectral library files
  * Documentation (methods, results)
  * Validation report
- [ ] Archive raw and processed data
- [ ] Create DOI/Zenodo archive (optional)

# =============================================================================
# TROUBLESHOOTING GUIDE
# =============================================================================

## Problem: Python library installation fails

### Issue: pysptools won't install
Solution:
```bash
# Try installing dependencies first
pip install numpy scipy matplotlib
pip install scikit-learn cvxopt
# Then pysptools
pip install pysptools

# If still fails, install from GitHub:
pip install git+https://github.com/ctherien/pysptools.git
```

### Issue: GDAL/Rasterio conflicts
Solution:
```bash
# Use conda for geospatial
conda install -c conda-forge gdal rasterio
# Don't mix pip and conda for GDAL
```

## Problem: Hyperion cube too large for memory

### Solution 1: Process in chunks
```python
# Process by tiles
from itertools import product

def process_in_chunks(cube, chunk_size=500):
    rows, cols, bands = cube.shape
    results = np.zeros((rows, cols))
    
    for i in range(0, rows, chunk_size):
        for j in range(0, cols, chunk_size):
            chunk = cube[i:i+chunk_size, j:j+chunk_size, :]
            # Process chunk
            results[i:i+chunk_size, j:j+chunk_size] = process(chunk)
    
    return results
```

### Solution 2: Use memory-mapped arrays
```python
import numpy as np

# Create memory-mapped file
mm_cube = np.memmap('hyperion_cube.dat', 
                     dtype='float32', 
                     mode='r',
                     shape=(rows, cols, bands))
```

### Solution 3: Subset to smaller area
- Use QGIS to clip to study area only
- Reduce spatial extent in preprocessing

## Problem: SAM produces unrealistic results

### Diagnostic checklist:
- [ ] Verify wavelengths match between cube and library
- [ ] Check for scale differences (reflectance 0-1 vs 0-100)
- [ ] Inspect endmember spectra (plot them)
- [ ] Check cube for nodata values (set to NaN)
- [ ] Verify masks are applied correctly

### Common fixes:
```python
# Ensure reflectance scale matches
if cube.max() > 10:
    cube = cube / 10000.0  # Scale to 0-1

# Remove invalid pixels
cube[cube < 0] = np.nan
cube[cube > 1] = np.nan

# Normalize endmembers if needed
endmembers = endmembers / np.linalg.norm(endmembers, axis=1)[:, None]
```

## Problem: QGIS crashes with large rasters

### Solutions:
1. Build pyramids:
   ```
   Right-click layer → Properties → Pyramids
   Build pyramids at multiple resolutions
   ```

2. Use virtual rasters (VRT):
   ```
   Raster → Miscellaneous → Build Virtual Raster
   Reduces memory use
   ```

3. Adjust QGIS settings:
   ```
   Settings → Options → Rendering
   - Enable "Render layers in parallel"
   - Adjust cache size
   ```

## Problem: Validation shows poor accuracy

### Diagnosis steps:
1. Visual inspection first
   - Do results make sense spatially?
   - Are there obvious errors?

2. Check spectral separability
   ```python
   # Calculate spectral angles between endmembers
   from scipy.spatial.distance import cosine
   
   for i, name1 in enumerate(names):
       for j, name2 in enumerate(names[i+1:], i+1):
           angle = np.arccos(1 - cosine(E[i], E[j]))
           print(f"{name1} vs {name2}: {angle:.3f} rad")
   
   # If angles < 0.05 rad, minerals too similar
   ```

3. Try alternative methods
   - If SAM poor, try SFF (Spectral Feature Fitting)
   - If unmixing poor, try MESMA (Multiple Endmember Spectral Mixture Analysis)

4. Refine endmember library
   - Use image-derived endmembers (purest pixels)
   - Mix laboratory + field spectra
   
5. Consider environmental factors
   - Weathering alters spectra
   - Surface conditions (wet vs dry)
   - Mixing with other minerals

# =============================================================================
# PERFORMANCE OPTIMIZATION TIPS
# =============================================================================

## Python Processing

### Use vectorization (avoid loops):
```python
# BAD: Slow loop
for i in range(rows):
    for j in range(cols):
        result[i,j] = some_function(cube[i,j,:])

# GOOD: Vectorized
result = some_function(cube.reshape(-1, bands)).reshape(rows, cols)
```

### Use numba for speed:
```python
from numba import jit

@jit(nopython=True)
def compute_sam_angles(cube, endmember):
    rows, cols, bands = cube.shape
    angles = np.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            pixel = cube[i, j, :]
            angles[i, j] = np.arccos(
                np.dot(pixel, endmember) / 
                (np.linalg.norm(pixel) * np.linalg.norm(endmember))
            )
    return angles
```

### Parallelize with multiprocessing:
```python
from multiprocessing import Pool

def process_row(args):
    row_idx, cube_row, endmembers = args
    # Process single row
    return row_result

with Pool(8) as pool:
    results = pool.map(process_row, [(i, cube[i], E) for i in range(rows)])
```

## QGIS Processing

### Use processing algorithms (faster than manual):
```python
# In QGIS Python console
import processing

processing.run("otb:SpectralAngleMapper", {
    'in': 'input.tif',
    'ref': 'endmembers.tif',
    'out': 'sam_result.tif'
})
```

### Batch processing script:
```python
# Process multiple scenes
scenes = ['scene1.tif', 'scene2.tif', 'scene3.tif']

for scene in scenes:
    processing.run("algorithm", {'in': scene, ...})
```

# =============================================================================
# PROJECT TIMELINE SUMMARY
# =============================================================================

## Minimum Timeline: 8-10 weeks
- Assumes working part-time (15-20 hrs/week)
- Hyperion preprocessing already complete
- Reference data readily available

## Realistic Timeline: 12-16 weeks
- Includes time for:
  * Learning curve with tools
  * Data collection/organization
  * Troubleshooting
  * Validation challenges
  * Iteration on methods
  * Documentation

## Detailed Timeline:

Week 1-2:   Setup + Data Collection
Week 3:     Spectral Library
Week 3-4:   Enhancement + Masking
Week 5:     SAM Classification
Week 6:     MTMF/Abundance
Week 7:     Postprocessing
Week 8-9:   Validation
Week 10-11: Final Products
Week 12:    Documentation + Delivery

# =============================================================================
# SUCCESS CRITERIA
# =============================================================================

## Minimum Viable Product (MVP):
- [ ] SAM classification maps for 3+ minerals
- [ ] Abundance/confidence maps
- [ ] Basic validation (visual + 1 quantitative)
- [ ] Styled maps in QGIS
- [ ] Methods documentation

## Full Product:
- [ ] All minerals mapped with confidence
- [ ] Multiple validation approaches
- [ ] Publication-quality maps
- [ ] Statistical analysis complete
- [ ] Comprehensive documentation
- [ ] Reproducible workflow (scripts + parameters)
- [ ] Comparison with published studies

## Stretch Goals:
- [ ] Temporal analysis (if multiple dates available)
- [ ] Integration with geochemical models
- [ ] Machine learning classification comparison
- [ ] Web map (e.g., Leaflet) for interactive viewing
- [ ] Manuscript draft for publication

# =============================================================================
# NEXT STEPS - START HERE
# =============================================================================

1. [ ] Print this checklist
2. [ ] Review software requirements, install what's missing
3. [ ] Set up directory structure
4. [ ] Review your Hyperion data from Step 1:
   - [ ] What format? (GeoTIFF, ENVI, HDF5?)
   - [ ] How many bands after cleaning?
   - [ ] What is the spatial extent?
   - [ ] What is the acquisition date?
5. [ ] Start with spectral library (easiest first step)
6. [ ] Test Python workflow on small subset first
7. [ ] Gradually expand to full scene
8. [ ] Iterate and refine

Good luck with your project!
