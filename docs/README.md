# Documentation

## Quick Start

📖 **Start here:** [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Complete documentation overview and summaries

## Folder Structure

```
docs/
├── README.md                      (You are here)
├── DOCUMENTATION_INDEX.md         (Main index with summaries)
│
├── guides/                        (5 documents)
│   ├── implementation_checklist.md    - Complete project checklist
│   ├── qgis_snap_workflows.md        - GUI software workflows
│   ├── quick_start_guide.md          - Code snippets
│   ├── WORKFLOW_DIAGRAM.md           - Visual flowcharts
│   └── SETUP_AND_USAGE_GUIDE.md      - Setup instructions
│
└── changelogs/                    (5 documents)
    ├── SAM_Implementation_Report.md   - SAM debugging
    ├── INTEGRATION_SUMMARY.md         - USGS library integration
    ├── ENDMEMBER_FIX_SUMMARY.md       - SNAP library compatibility
    ├── SAM_OUTPUT_FIX_SUMMARY.md      - SNAP output compatibility
    └── NULL_MODEL_THRESHOLD.md        - Statistical null-model thresholding
```

## What's What

### 📚 Guides (How-to Documentation)
User-facing documentation for learning and using the system:
- **Implementation Checklist:** Full workflow from setup to final products
- **QGIS/SNAP Workflows:** Using GUI software for visualization
- **Quick Start Guide:** Copy-paste code examples
- **Workflow Diagram:** Visual representations and flowcharts
- **Setup & Usage:** Environment setup and execution instructions

### 📝 Changelogs (Technical Reports)
Technical documentation of fixes and improvements:
- **SAM Implementation Report:** Fixing 0% classification issue
- **Integration Summary:** Adding USGS spectral library
- **Endmember Fix:** Making library SNAP-compatible
- **SAM Output Fix:** Making classification results SNAP-compatible
- **Null-Model Threshold:** Replacing the arbitrary min-angle threshold with a background-anchored statistical threshold

## For Different Users

### 🆕 New to the Project?
1. Read [SETUP_AND_USAGE_GUIDE.md](guides/SETUP_AND_USAGE_GUIDE.md)
2. Review [WORKFLOW_DIAGRAM.md](guides/WORKFLOW_DIAGRAM.md)
3. Follow [implementation_checklist.md](guides/implementation_checklist.md)

### 💻 Need Code Examples?
→ [quick_start_guide.md](guides/quick_start_guide.md)

### 🗺️ Using QGIS or SNAP?
→ [qgis_snap_workflows.md](guides/qgis_snap_workflows.md)

### 🐛 Troubleshooting?
→ Check [changelogs/](changelogs/) for known issues and fixes

### 🔍 Want Complete Overview?
→ [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)

## Project Info

**Project:** Hyperion AMD Mineral Mapping
**Location:** Rio Tinto, Spain
**Data:** Hyperion Hyperspectral Imagery
**Methods:** SAM, MTMF, FCLS Classification
**Target Minerals:** Jarosite, Goethite, Hematite, Schwertmannite

---

*Last updated: January 13, 2026*
