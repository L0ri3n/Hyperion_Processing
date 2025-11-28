# AMD Mineral Mapping with Hyperion - Complete Implementation Package

## Overview

This package contains a complete, step-by-step implementation plan for mapping acid mine drainage (AMD) minerals in the Rio Tinto area using Hyperion hyperspectral imagery. Since you've already completed Step 1 (preprocessing with SUREHYP), this starts from Step 2 onwards.

## What's Included

### 1. **hyperion_workflow.py** (Main Implementation)
   - Complete Python workflow from Steps 2-8
   - Fully documented functions for:
     * Spectral library construction
     * Hyperion wavelength extraction
     * Spectral resampling
     * Savitzky-Golay smoothing
     * Continuum removal
     * SAM classification
     * MTMF/Matched Filter abundance mapping
     * Linear spectral unmixing
     * Postprocessing and morphological filtering
     * Validation functions
     * Final output generation
   - Main execution workflow
   - Ready to customize and run

### 2. **qgis_snap_workflows.md** (GUI Tool Instructions)
   - Comprehensive QGIS workflows:
     * Loading Hyperion data
     * Creating RGB composites
     * Running SAM with OTB
     * Creating masks (vegetation, water)
     * Styling and map layout
     * Spatial analysis
     * Accuracy assessment
   - SNAP workflows:
     * Importing Hyperion
     * Spectral unmixing
     * Band math for masks
     * Exporting results
   - Tool selection recommendations
   - Integration strategies

### 3. **implementation_checklist.md** (Project Management)
   - Complete software requirements:
     * Python environment setup
     * QGIS plugins needed
     * SNAP configuration
     * System requirements
   - Data requirements:
     * USGS spectral library links
     * Literature references for AMD spectra
     * Validation data sources
   - Phase-by-phase checklist (Weeks 1-12):
     * Setup and preparation
     * Spectral library construction
     * Preprocessing and masking
     * Classification and unmixing
     * Validation
     * Final products
   - Quality control checkpoints
   - Troubleshooting guide
   - Performance optimization tips
   - Timeline estimates

### 4. **quick_start_guide.md** (Copy-Paste Commands)
   - Ready-to-use code snippets:
     * Environment setup commands
     * Loading USGS spectra
     * Resampling to Hyperion
     * Running SAM classification
     * MTMF abundance mapping
     * Postprocessing
     * Statistics calculation
     * Visualization
   - QGIS Python console commands
   - Validation examples
   - Troubleshooting snippets
   - Quick visual checks

## Target Minerals

This workflow focuses on AMD minerals characteristic of Rio Tinto:

1. **Jarosite** (KFe₃(SO₄)₂(OH)₆)
   - Diagnostic: 2.20-2.26 µm absorption
   - Associated with pH < 3

2. **Goethite** (α-FeOOH)
   - Diagnostic: VNIR ferric shoulder (~900 nm)
   - Long-term oxidation product

3. **Hematite** (Fe₂O₃)
   - Diagnostic: 850-950 nm crystal field
   - Ancient oxidized terraces

4. **Schwertmannite** (Fe₈O₈(OH)₆SO₄)
   - Diagnostic: Poorly crystalline, broad SWIR features
   - pH 3-4.5 zones

## Methodology Overview

### Classification Approach
- **SAM (Spectral Angle Mapper)**: Illumination-invariant similarity measure
- **MTMF (Mixture Tuned Matched Filter)**: Abundance estimation with infeasibility check
- **Linear Unmixing (FCLS)**: Constrained abundance fractions
- **Combined SAM+MTMF**: High-confidence detection

### Validation Strategy (No Ground Truth)
1. Comparison with published Rio Tinto mineral maps
2. Correlation with geochemical data (pH, Fe, SO₄)
3. Spatial consistency checks (topography, drainage)
4. Cross-validation with Sentinel-2 iron oxide indices

### Key Features
- Uses published laboratory spectra (USGS + literature)
- No field data required
- Handles mixed pixels (30m resolution)
- Morphological postprocessing
- Multiple validation approaches
- Publication-ready outputs

## Getting Started

### Quick Start (Recommended for Beginners)
1. Start with **quick_start_guide.md**
2. Follow the environment setup
3. Use the copy-paste code snippets
4. Test on a small subset first

### Comprehensive Approach (For Full Project)
1. Read **implementation_checklist.md** completely
2. Set up software and data (Weeks 1-2)
3. Follow the weekly checklist
4. Use **hyperion_workflow.py** as your main codebase
5. Use **qgis_snap_workflows.md** for visualization and validation

### Code-First Approach (For Experienced Users)
1. Review **hyperion_workflow.py**
2. Customize functions as needed
3. Run step-by-step or use `main_workflow()`
4. Refer to other documents as needed

## Software Requirements

### Python (Core Environment)
```bash
conda create -n hyperion python=3.9
conda activate hyperion
conda install numpy scipy pandas matplotlib gdal rasterio geopandas
pip install spectral pysptools scikit-image
```

### QGIS
- Version: 3.28+ (LTR recommended)
- Plugins: SCP, OTB, Profile Tool

### SNAP
- Version: 9.0+
- Optional for alternative workflows

## Directory Structure

Recommended project organization:
```
amd_mapping/
├── data/
│   ├── hyperion/preprocessed/    # Your SUREHYP output
│   ├── spectral_library/         # USGS + publication spectra
│   ├── ancillary/                # DEM, Sentinel-2
│   └── validation/               # Published maps, geochemistry
├── code/                         # Python scripts
├── outputs/
│   ├── classifications/          # SAM results
│   ├── abundances/              # MTMF results
│   ├── masks/                   # Vegetation, water
│   ├── figures/                 # Plots
│   └── stats/                   # Statistics tables
├── qgis_projects/               # QGIS files
└── docs/                        # Documentation
```

## Expected Outputs

### Classification Maps
- Individual mineral maps (Jarosite, Goethite, Hematite)
- Multi-class classification
- Spectral angle maps (quality indicators)

### Abundance Maps
- Matched Filter scores
- MTMF infeasibility maps
- Linear unmixing fractions

### Validation Results
- Spatial overlap statistics
- Geochemical correlations
- Accuracy metrics (if reference data available)

### Visualizations
- Styled QGIS maps
- Publication-ready figures
- Spectral profile plots

### Statistics
- Area calculations (hectares, km²)
- Pixel counts by mineral
- Confidence/quality metrics

## Timeline

- **Minimum**: 8-10 weeks (part-time, straightforward implementation)
- **Realistic**: 12-16 weeks (includes learning, troubleshooting, validation)
- **With complications**: 16-20 weeks (data collection issues, method refinement)

## Key References

### Spectral Libraries
- USGS Spectral Library: https://crustal.usgs.gov/speclab/

### AMD Mineralogy Literature
- Bigham et al. (1996) - Schwertmannite
- Crowley et al. (2003) - Spectral reflectance of secondary minerals
- Acero et al. (2006) - Rio Tinto schwertmannite
- Sánchez-España et al. (2016) - Tinto-Odiel mineralogy

### Methods Papers
- Kruse et al. (1993) - Spectral Angle Mapper
- Boardman et al. (1998) - Mixture Tuned Matched Filter
- Heinz & Chang (2001) - Fully Constrained Least Squares

## Support and Troubleshooting

### Common Issues and Solutions

1. **Memory errors with large cubes**
   - Process in chunks
   - Use memory-mapped arrays
   - Subset to smaller area

2. **Poor classification results**
   - Check wavelength alignment
   - Verify reflectance scale (0-1 vs 0-100)
   - Test spectral separability
   - Refine thresholds

3. **QGIS crashes**
   - Build pyramids
   - Use VRT (virtual rasters)
   - Adjust cache settings

4. **Low validation accuracy**
   - Visual inspection first
   - Check endmember quality
   - Consider image-derived endmembers
   - Account for weathering effects

See **implementation_checklist.md** for detailed troubleshooting.

## Customization

This workflow is designed to be modular and customizable:

- **Add minerals**: Expand endmember library
- **Change thresholds**: Adjust SAM/MTMF parameters
- **Alternative methods**: Swap SAM for SFF or other classifiers
- **Additional validation**: Integrate field data if available
- **Temporal analysis**: Compare multiple dates
- **Machine learning**: Benchmark against RF/SVM classifiers

## Best Practices

1. **Start small**: Test on subsets before full scene
2. **Document everything**: Parameters, thresholds, decisions
3. **Version control**: Use Git for code
4. **Validate often**: Check intermediate results
5. **Visualize frequently**: Use QGIS for QC
6. **Save checkpoints**: Don't overwrite intermediate products
7. **Compare methods**: Run multiple approaches
8. **Iterate**: Refine based on results

## Contributing and Feedback

This workflow represents best practices as of 2025. Suggestions for improvement:

- Alternative preprocessing methods
- Additional validation approaches
- Optimized implementations
- New tools or libraries
- Rio Tinto-specific insights

## Citation

If you use this workflow in publications, please cite:

- Hyperion sensor (NASA EO-1)
- USGS Spectral Library
- Relevant algorithm papers (SAM, MTMF, etc.)
- Software packages (Python, QGIS, SNAP)
- Rio Tinto literature sources

## License

This implementation guide is provided for educational and research purposes. 
Please respect licenses for:
- Software packages (check individual licenses)
- Hyperion data (NASA/USGS terms)
- USGS spectral library (public domain)
- Published spectra (check paper copyright)

## Contact and Collaboration

For questions about:
- Rio Tinto geology: Consult published literature
- Hyperion processing: USGS EarthExplorer support
- Software issues: Respective package forums
- Spectral analysis: Remote sensing community forums

## Version History

- v1.0 (2025): Initial comprehensive package
  * Steps 2-8 implementation
  * Python, QGIS, SNAP workflows
  * Validation without ground truth
  * AMD mineral focus

---

## Getting Help

If you get stuck:

1. Check the **Troubleshooting** sections in each document
2. Review the **Quick Start Guide** for simple examples
3. Consult the **Implementation Checklist** for systematic approach
4. QGIS/SNAP forums for software-specific issues
5. Python package documentation for library issues
6. Remote sensing textbooks for methodological questions

Good luck with your AMD mineral mapping project!

---

*Last updated: November 2025*
