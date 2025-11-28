# QGIS and SNAP Workflows for Hyperion AMD Mineral Mapping
# Companion to hyperion_workflow.py

# =============================================================================
# QGIS WORKFLOWS
# =============================================================================

## WORKFLOW 1: Visualization and Map Production
## --------------------------------------------

### Initial Setup
1. Install required plugins:
   - Semi-Automatic Classification Plugin (SCP)
   - Profile Tool (for spectral plots)
   - Orfeo ToolBox (OTB) provider
   
   Installation: Plugins → Manage and Install Plugins

2. Project setup:
   - CRS: Use UTM zone appropriate for Rio Tinto (likely UTM 29N or 30N)
   - Enable "on-the-fly" CRS transformation
   - Project → Properties → CRS

### Loading Hyperion Data in QGIS

```
Method 1: Direct import
1. Layer → Add Layer → Add Raster Layer
2. Select: hyperion_cube.tif (or .img)
3. QGIS will ask about bands - select "all bands"
4. Result: Single multiband raster

Method 2: Using SCP plugin
1. SCP → Band set → Add files
2. Select all Hyperion bands (after bad band removal)
3. SCP will organize them properly
```

### Creating RGB Composites for Visualization

```
Natural Color (True Color):
- Red: Band 29 (~650 nm)
- Green: Band 23 (~550 nm)  
- Blue: Band 13 (~480 nm)

False Color (Vegetation):
- Red: Band 50 (~830 nm, NIR)
- Green: Band 29 (~650 nm)
- Blue: Band 23 (~550 nm)

False Color (Iron Oxides):
- Red: Band 29 (~650 nm)
- Green: Band 23 (~550 nm)
- Blue: Band 13 (~480 nm)
Then adjust min/max to 2-98% stretch

SWIR Composite (Minerals):
- Red: Band 180 (~2200 nm)
- Green: Band 50 (~830 nm)
- Blue: Band 29 (~650 nm)
```

### Running SAM in QGIS (Using OTB)

```
1. Processing Toolbox → search "Spectral Angle Mapper"
2. OTB → Spectral Angle Mapper (SAM)

Parameters:
- Input Image: hyperion_smoothed.tif
- Spectral library: endmember_library.tif
  (Stack endmembers as separate bands)
- Output classification image: sam_classification.tif
- Threshold: 0.10 (radians)

3. Run and wait for completion

Output interpretation:
- Each pixel assigned to closest endmember
- Value 0 = unclassified (angle > threshold)
- Values 1-N = mineral classes
```

### Creating Masks (Vegetation, Water, Clouds)

```
NDVI Mask (Vegetation):
1. Raster → Raster Calculator
2. Formula: (NIR - Red) / (NIR + Red)
   - NIR = Band 50 (~830 nm)
   - Red = Band 29 (~650 nm)
3. Expression: ("hyperion@50" - "hyperion@29") / ("hyperion@50" + "hyperion@29")
4. Save as: ndvi_mask.tif
5. Reclassify: Values > 0.3 = vegetation (mask out)

Water Mask:
1. Use NIR band (Band 50)
2. Water has very low NIR reflectance
3. Raster Calculator: "hyperion@50" < 0.05
4. Save as: water_mask.tif

Combined Mask:
1. Raster Calculator:
   ("ndvi_mask" = 1) OR ("water_mask" = 1)
2. Save as: combined_mask.tif
3. Use to exclude pixels from classification
```

### Styling Mineral Classification Maps

```
1. Right-click layer → Properties → Symbology
2. Render type: Paletted/Unique values
3. Click "Classify" to auto-generate classes

Recommended color scheme for AMD minerals:
- Unclassified (0): Transparent or light gray
- Jarosite: Yellow (#FFFF00)
- Goethite: Brown (#8B4513)
- Hematite: Red (#DC143C)
- Schwertmannite: Orange (#FF8C00)
- Kaolinite: Light blue (#87CEEB)

4. Adjust opacity: 60-70% to see basemap underneath
5. Apply → OK
```

### Creating Map Layouts for Publication

```
1. Project → New Print Layout
2. Name: "Rio_Tinto_AMD_Minerals"

3. Add map:
   - Add Item → Add Map
   - Draw rectangle on canvas
   - Item Properties → Set scale (e.g., 1:50000)

4. Add legend:
   - Add Item → Add Legend
   - Remove unwanted layers
   - Edit mineral names for clarity
   - Set font size (10-12 pt)

5. Add scale bar:
   - Add Item → Add Scale Bar
   - Style: Numeric or Single Box
   - Units: Meters or Kilometers

6. Add north arrow:
   - Add Item → Add Picture
   - Search directories → Select north arrow SVG
   - Resize appropriately

7. Add grid/graticule:
   - Select map item
   - Item Properties → Grids
   - Add grid with appropriate interval
   - Frame style: Zebra

8. Add text labels:
   - Title: "AMD Mineral Distribution - Rio Tinto"
   - Date, projection, data source
   - Author information

9. Export:
   - Layout → Export as Image (PNG, 300 dpi)
   - Or Export as PDF for publication
```

### Spatial Analysis in QGIS

```
Calculate mineral occurrence areas:
1. Raster → Raster Calculator
2. For each mineral class:
   - Create binary: "sam_classification@1" = 1 (for jarosite)
   - Count pixels: Raster → Zonal Statistics
   - Multiply pixel count × 900 (30m × 30m) = area in m²

Proximity analysis:
1. Raster → Proximity → Proximity (Raster Distance)
2. Input: Jarosite classification
3. Output: Distance map to nearest jarosite pixel
4. Use to analyze spatial patterns

Buffer analysis around streams:
1. Import stream network (vector)
2. Vector → Geoprocessing Tools → Buffer
3. Distance: 50m, 100m, 200m
4. Extract mineral stats within buffers
5. Test: Is jarosite concentrated near streams?
```

### Spectral Profile Tool

```
1. Install "Profile Tool" plugin
2. Plugins → Profile Tool → Terrain profile
3. Select Hyperion multiband layer
4. Click on pixels of interest
5. Spectral curve appears in profile window
6. Compare with endmember library spectra
7. Export profile as CSV for analysis
```

## WORKFLOW 2: Accuracy Assessment (with reference data)

```
If you digitize reference polygons from publications:

1. Create vector layer:
   - Layer → Create Layer → New Shapefile Layer
   - Geometry: Polygon
   - Add field: "mineral" (Text)
   - Digitize known jarosite/goethite zones

2. Convert to raster:
   - Raster → Conversion → Rasterize
   - Match extent/resolution to Hyperion

3. Accuracy assessment:
   - SCP → Postprocessing → Accuracy
   - Reference: Digitized ground truth
   - Classification: Your SAM output
   - Output: Confusion matrix, Kappa, Overall accuracy

4. Interpret results:
   - Producer's accuracy (recall)
   - User's accuracy (precision)
   - F1-score for each mineral
```

# =============================================================================
# SNAP (Sentinel Application Platform) WORKFLOWS  
# =============================================================================

## Initial Setup

### Installation
1. Download SNAP from: https://step.esa.int/main/download/snap-download/
2. Install SNAP Desktop (includes Sentinel-1, -2, -3 toolboxes)
3. Optional: Install additional modules:
   - Help → Install Plugins → Available Plugins
   - Install "Spectral Unmixing" if available

### Configure SNAP for Hyperion
```
1. Open SNAP Desktop
2. Tools → Manage External Products
3. If Hyperion not listed, you may need to:
   - Import as generic GeoTIFF
   - Or use BEAM-DIMAP format from Python export
```

## WORKFLOW 1: Loading and Preprocessing Hyperion in SNAP

### Import Hyperion Data
```
1. File → Import → Generic Formats → GeoTIFF
2. Select: hyperion_cube.tif
3. Product Explorer: Shows band structure
4. Verify:
   - Bands are properly labeled
   - Georeferencing is correct (check World Map)
```

### Band Math for Creating Masks
```
1. Raster → Band Maths
2. For NDVI mask:
   - Expression: (B50 - B29) / (B50 + B29)
   - Name: NDVI
   - Virtual: No (save to product)
3. For water mask:
   - Expression: B50 < 0.05
   - Name: Water_mask
4. For vegetation mask:
   - Expression: NDVI > 0.3
   - Name: Veg_mask
5. Combined mask:
   - Expression: Veg_mask OR Water_mask
   - Name: Combined_mask
```

### Spectral Angle Mapper in SNAP

```
Unfortunately, SNAP doesn't have native SAM for custom libraries.

Workaround options:

Option A: Use SNAP for preprocessing, export for Python SAM
1. Do masking and smoothing in SNAP
2. File → Export → GeoTIFF
3. Run SAM in Python (see hyperion_workflow.py)

Option B: Use SNAP Spectral Unmixing
1. Optical → Thematic Land Processing → Spectral Unmixing
2. Load endmember library (may need specific format)
3. Run constrained unmixing
4. Output: Abundance maps (similar to Step 5 in workflow)

Option C: Use SNAP with external SAM plugin
1. Check SNAP plugin repository
2. Community-developed hyperspectral tools
3. May require BEAM/SNAP 9.0+
```

### Linear Spectral Unmixing in SNAP

```
1. Prepare endmember file:
   - Format: Text file with wavelengths and reflectances
   - One endmember per column
   - Header with names

2. Optical → Thematic Land Processing → Spectral Unmixing
   - Algorithm: Fully Constrained Least Squares (FCLS)
   - Endmembers: Load file or select from product
   - Target product: Specify output name

3. Output bands:
   - One abundance band per endmember
   - Values: 0-1 (fraction)
   - RMSE band (reconstruction error)

4. Visualize:
   - Color Manipulation → RGB Image
   - Assign abundance bands to RGB channels
```

### Visualization and RGB Composites

```
1. Window → RGB Image View
2. Select bands:
   - Red channel: Band number or name
   - Green channel: Band number or name
   - Blue channel: Band number or name
3. Adjust histogram stretch:
   - Right-click on histogram
   - Options: 95%, 98%, or custom percentiles
4. Save as image:
   - File → Export View as Image
   - Format: PNG, JPEG, GeoTIFF
```

## WORKFLOW 2: Exporting Results from SNAP

### Export Individual Bands
```
1. Right-click band in Product Explorer
2. Export Band → GeoTIFF
3. Specify output file
4. Settings:
   - Compression: LZW (recommended)
   - Tiling: 256×256
5. Use in QGIS or Python
```

### Export Entire Product
```
1. File → Export → BEAM-DIMAP
   - Native SNAP format
   - Preserves all metadata
   - Use for archiving

2. File → Export → GeoTIFF-BigTIFF
   - Standard GIS format
   - All bands in one file
   - Compatible with QGIS, ArcGIS
```

### Export Subset (Region of Interest)
```
1. Raster → Subset
2. Define subset:
   - Spatial: Draw rectangle or enter coordinates
   - Band: Select bands to keep
3. Creates new product with subset
4. Export as above
```

# =============================================================================
# INTEGRATED WORKFLOW RECOMMENDATIONS
# =============================================================================

## Optimal Tool Selection by Task

### Task: Preprocessing (Steps 1-3)
**Recommended: Python (SUREHYP, spectral library)**
- Reason: Automated, reproducible, batch processing
- Output: Clean cube ready for classification

### Task: Spectral library creation (Step 2)
**Recommended: Python**
- Reason: Precise wavelength resampling needed
- Libraries: spectral, pysptools

### Task: SAM Classification (Step 4)
**Recommended: Python (pysptools)**
- Reason: Full control, custom thresholds, automation
- Alternative: QGIS-OTB (good for visual exploration)

### Task: Abundance mapping (Step 5)
**Recommended: Python or SNAP**
- Python: More algorithms (MTMF, MF, FCLS, NNLS)
- SNAP: Visual interface, good for exploration

### Task: Postprocessing (Step 6)
**Recommended: Python**
- Reason: Precise control over filters, thresholds
- Easy to combine multiple criteria

### Task: Validation (Step 7)
**Recommended: QGIS + Python**
- QGIS: Visual comparison, digitizing reference zones
- Python: Statistical analysis, correlation tests

### Task: Map production (Step 8)
**Recommended: QGIS**
- Reason: Best cartographic tools
- Professional-quality layout and styling

## Suggested Workflow Integration

```
Step 1: Python (SUREHYP) → Clean cube
        ↓
Step 2-3: Python → Spectral library, smoothing
        ↓
Step 4-5: Python → SAM + MTMF classification
        ↓
        → Export GeoTIFFs
        ↓
Step 6-7: QGIS → Visual validation, spatial analysis
        ↓
        → Digitize reference zones
        ↓
        → Python accuracy assessment
        ↓
Step 8: QGIS → Final maps, layout, export

Parallel workflow in SNAP (optional):
- Import Python outputs
- Create alternative composites
- Export for comparison
```

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

## Common QGIS Issues

**Issue: Hyperion appears all black**
Solution:
- Properties → Symbology → Min/Max values
- Set to "Cumulative count cut" 2%-98%
- Or "Actual (slower)" and adjust manually

**Issue: OTB tools not available**
Solution:
- Processing → Options → Providers
- Enable Orfeo ToolBox
- Set OTB folder (usually auto-detected)
- Restart QGIS

**Issue: Out of memory errors**
Solution:
- Processing → Options → General
- Increase "Maximum threads"
- Or reduce to 1 thread to save memory
- Process smaller subsets

## Common SNAP Issues

**Issue: SNAP very slow**
Solution:
- Tools → Options → Performance
- Increase cache size (e.g., 8 GB)
- Increase parallelism (# of threads)
- Disable "load bands on demand" if slow

**Issue: Can't import Hyperion**
Solution:
- Ensure GeoTIFF has proper metadata
- Or convert to BEAM-DIMAP format first:
  ```python
  # In Python with rasterio
  import rasterio
  from rasterio.envi import ENVI
  # Export with ENVI format
  ```

**Issue: Unmixing produces negative values**
Solution:
- Use Fully Constrained Least Squares (FCLS)
- Ensures sum-to-one and non-negativity
- Or post-process: clip negatives to 0

# =============================================================================
# ADDITIONAL RESOURCES
# =============================================================================

## QGIS Resources
- QGIS User Guide: https://docs.qgis.org/
- SCP Documentation: https://fromgistors.blogspot.com/
- OTB Applications: https://www.orfeo-toolbox.org/

## SNAP Resources
- SNAP Tutorials: https://step.esa.int/main/doc/tutorials/
- Forum: https://forum.step.esa.int/
- Spectral Unmixing: https://step.esa.int/docs/tutorials/su_tutorial.pdf

## Python Libraries
- PySptools: https://pysptools.sourceforge.io/
- Spectral Python: http://www.spectralpython.net/
- Rasterio: https://rasterio.readthedocs.io/

## AMD Mineralogy References
- USGS Spectral Library: https://crustal.usgs.gov/speclab/
- Jarosite spectra: Cloutis et al. (2006) Icarus
- Schwertmannite: Bigham et al. (1996) Clays & Clay Minerals
- Rio Tinto mineralogy: Fernández-Remolar et al. (2005) EPSL
