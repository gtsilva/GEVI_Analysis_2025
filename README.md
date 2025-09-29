# ASAP Analysis - Voltage Indicator Performance Comparison

A suite of Python tools for analyzing genetically encoded voltage indicators (GEVIs) using fluorescence imaging data. Includes scripts for fluorescence extraction, event detection, statistical comparison, and publication-quality visualization.

## Overview

This repository provides tools for GEVI performance analysis, from raw imaging data to publication-ready figures. The pipeline emphasizes robust statistical methods, conservative event detection, and reproducible analysis.

## Scripts

### Fluorescence Extraction

#### `simple_conservative_extract_fluo.py` ⭐ *Current favorite*
Conservative fluorescence spike extraction with comprehensive data source support.

**Features:**
- **Data Sources**: RigDataV2 camera data, TIFF stacks, or single trace files (.npy, .csv)
- **Method**: Rolling percentile baseline detrending + 5σ threshold detection
- **Processing**: Auto-selection of silent windows, optional SOS Butterworth filtering
- **Output**: Per-ROI event CSVs and comprehensive diagnostic plots

**Usage:**
```bash
# RigDataV2 camera data
python simple_conservative_extract_fluo.py --source rig --data-dir /path/to/experiment

# TIFF stack
python simple_conservative_extract_fluo.py --source tiff --tif /path/to/stack.tif

# Single trace file
python simple_conservative_extract_fluo.py --input trace.npy --fs 1000 --out_csv events.csv --out_plot diag.png
```

#### `extract_fluorescence_cellpose.py`
Complete fluorescence extraction pipeline with Cellpose segmentation.

**Features:**
- Cellpose-based cell detection with customizable models
- Motion correction via phase correlation-based frame registration
- Neuropil correction with ring-shaped mask subtraction
- Supports RigDataV2 camera data or TIFF stacks
- Outputs fluorescence traces, cell masks, and QC plots

### Cell Detection Utilities

#### `diagnose_segmentation.py`
Diagnostic tool to troubleshoot segmentation failures.

**Features:**
- Tests multiple reference image projections (mean, max)
- Evaluates different thresholding approaches (Otsu, Li, percentile-based)
- Applies various image processing filters (DoG, equalization)
- Generates comprehensive diagnostic plots showing:
  - Raw projections and histograms
  - Intensity profiles
  - Different threshold levels and coverage
- Provides recommendations for parameter adjustment

**Usage:**
```bash
python diagnose_segmentation.py /path/to/experiment [camera_name]
```

#### `manual_cell_detection.py`
Manual cell detection using simple thresholding for problematic datasets.

**Features:**
- Tests multiple thresholding methods (Otsu, Triangle, Li, percentile-based)
- Filters regions by size and circularity
- Automatically selects best segmentation result
- Generates visualization and saves masks

**Usage:**
```bash
# Edit script to set data_dir, then run:
python manual_cell_detection.py
```

### Statistical Comparison

#### `compare_gevis.py`
Publication-grade GEVI comparison pipeline with robust statistics.

**Features:**
- Multi-GEVI performance comparison (2-4 indicators supported)
- Bootstrap confidence intervals (BCa method)
- Effect size calculations (Cliff's delta, median differences)
- Statistical testing (Mann-Whitney U, Kolmogorov-Smirnov) with FDR correction
- Quality control with outlier detection and artifact removal
- Spike shape analysis with temporal alignment and interpolation
- Export to multiple formats (PNG, PDF, SVG, CSV, JSON)

**Usage:**
```bash
# Two-way comparison
python compare_gevis.py \
  --gevi1 ASAP6.2 /path/to/asap62/data \
  --gevi2 ASAP6.3 /path/to/asap63/data \
  --output results/comparison \
  --config config_example.json

# Three-way comparison
python compare_gevis.py \
  --gevi1 ASAP4 /data/asap4 \
  --gevi2 ASAP6.2 /data/asap62 \
  --gevi3 ASAP6.3 /data/asap63 \
  --frame-rate 1000 \
  --bootstrap-n 20000 \
  --output results/
```

### Data Loading

#### `RigDataV2.py`
Data loading and instrumentation interface by **Shane Nichols** ([LinkedIn](https://www.linkedin.com/in/shane-nichols/)).

**Features:**
- Loading waveform data from instrumentation systems
- Camera data integration with parameters and binary frames
- Support for galvo scanning and camera acquisition modes
- Temporal alignment and synchronization tools

### Configuration

#### `config_example.json`
Configuration file with analysis parameters:
```json
{
  "frame_rate": 1000.0,
  "min_events_per_roi": 3,
  "bootstrap_n": 10000,
  "artifact_threshold": 5.0,
  "winsorize_percentile": 2.5,
  "gevi_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
}
```

## Analysis Results

The repository contains two complete analysis runs comparing ASAP voltage indicators:

### `gevi_comparison/` - Standard Analysis
Comparison of ASAP4, ASAP6.2, and ASAP6.3 with standard parameters:
- **Sample Size**: 4,034 events across 29 ROIs
- **Generated**: September 25, 2025
- **Bootstrap**: 10,000 iterations with BCa confidence intervals

### `gevi_conservative_comparison/` - Conservative Analysis
Same GEVI comparison using more stringent quality control parameters:
- Demonstrates robustness of findings across parameter settings
- Conservative thresholding and filtering criteria

### Contents of Each Results Directory:
**Figures:**
- `gevi_comparison_main.png/pdf/svg` - Main comparison figure with key metrics
- `gevi_comparison_supplement.png/pdf` - Supplementary quality control analyses  
- `gevi_spike_shape_comparison.png/pdf/svg` - Spike shape and kinetics overlay

**Data Tables:**
- `summary_table.csv` - Publication-ready summary with effect sizes and CIs
- `per_event_metrics.csv` - Individual event measurements
- `per_roi_metrics.csv` - Region-of-interest summary statistics

**Analysis Files:**
- `statistical_results.json` - Complete statistical test results
- `run_metadata.json` - Analysis parameters and metadata
- `analysis.log` - Processing workflow log
- `README.txt` - Methods and interpretation guide

## Key Metrics Analyzed

### Event-Level Metrics
- Amplitude: ΔF/F (%) relative to baseline
- Width: Event duration in milliseconds 
- Signal-to-Noise Ratio: Peak amplitude divided by baseline noise (MAD)
- Prominence: Height of peaks above surrounding baseline
- Area Under Curve: Integrated ΔF/F over time

### ROI-Level Metrics
- Event Rate: Events per second per region of interest
- Amplitude Variability: Coefficient of variation across events
- Baseline Brightness: Median fluorescence intensity
- Detection Consistency: Inter-quartile range of event metrics

## Usage Examples

### GEVI Performance Comparison
```bash
# Basic comparison of multiple GEVIs
python compare_gevis.py \
  --gevi1 ASAP6.2 /path/to/asap62/data \
  --gevi2 ASAP6.3 /path/to/asap63/data \
  --output results/comparison \
  --config config_example.json

# Three-way comparison with custom parameters
python compare_gevis.py \
  --gevi1 ASAP4 /data/asap4 \
  --gevi2 ASAP6.2 /data/asap62 \
  --gevi3 ASAP6.3 /data/asap63 \
  --frame-rate 1000 \
  --min-events 5 \
  --bootstrap-n 20000 \
  --output publication_results/
```

### Conservative Spike Detection
```bash
# Experimental data from RigDataV2
python simple_conservative_extract_fluo.py \
  --source rig \
  --data-dir /path/to/experiment

# TIFF stack analysis
python simple_conservative_extract_fluo.py \
  --source tiff \
  --tif /path/to/stack.tif

# Single trace file analysis
python simple_conservative_extract_fluo.py \
  --input trace.npy \
  --fs 1000 \
  --out_csv events.csv \
  --out_plot diagnostics.png

# CSV file with custom parameters
python simple_conservative_extract_fluo.py \
  --input trace.csv \
  --fs 500 \
  --column 2 \
  --silent-sec 3 \
  --th-mult 5 \
  --min-width-ms 1
```

### Complete Fluorescence Extraction Pipeline
```bash
# RigDataV2 experimental data
python extract_fluorescence_cellpose.py \
  --source rig \
  --data-dir /path/to/experiment \
  --cellpose-model cyto2 \
  --diameter 15 \
  --register \
  --neuropil-correction

# TIFF stack processing
python extract_fluorescence_cellpose.py \
  --source tiff \
  --tif /path/to/stack.tif \
  --cellpose-model nuclei \
  --diameter 20
```

## Methods

### Statistical Analysis
- Bootstrap Confidence Intervals: BCa method with configurable iterations
- Nonparametric Tests: Mann-Whitney U, Kolmogorov-Smirnov
- Effect Sizes: Cliff's delta and median differences with CIs
- Multiple Comparisons: Benjamini-Hochberg FDR correction

### Quality Control
- MAD-based outlier detection and removal
- Winsorization of extreme values
- Minimum event count per ROI filtering
- Motion artifact detection and correction

### Data Analysis
- Spike shape analysis using fluorescence traces
- Event alignment by onset or peak detection
- Temporal interpolation to common time grids
- Kinetics extraction (rise time, decay time, FWHM)

## Dependencies

### Core Requirements
```bash
pip install numpy pandas matplotlib scipy
```

### For Fluorescence Extraction Pipeline
```bash
pip install cellpose opencv-python tifffile scikit-image
```

### Optional Packages
```bash
pip install xarray           # For RigDataV2 data structures
pip install PyYAML           # For YAML configuration support  
pip install pyarrow          # For efficient parquet file export
pip install statsmodels      # For advanced statistical analysis
```

### System Requirements
- Python 3.8+
- For GPU acceleration with Cellpose: CUDA-compatible GPU with appropriate drivers

## Output Files

### Figures
- Main Comparison: Multi-panel figure with distributions, scatter plots, and effect sizes
- Supplementary Analyses: Quality control plots, sample size summaries, distribution comparisons
- Spike Shape Analysis: Trace overlays with confidence intervals and kinetics

### Data Tables
- Summary Table: Metrics with confidence intervals
- Per-Event Metrics: Individual event measurements
- Per-ROI Metrics: Region-of-interest summary statistics
- Statistical Results: Test results in JSON format

### Documentation
- Run Metadata: Parameters, timestamps, and analysis configuration
- README Files: Methods and interpretation guides
- Log Files: Analysis workflow documentation

## Implementation

### Output Quality
- Vector graphics (SVG, PDF)
- Statistical annotation and effect size reporting
- Reproducible analysis with fixed random seeds

### Analysis Features
- Bootstrap confidence intervals for metrics
- Multiple comparison corrections
- Outlier detection and handling
- Quality control filtering

### Data Integration
- Fluorescence trace analysis
- Event alignment and temporal interpolation
- Kinetics extraction
- Motion correction and artifact removal

## Citation

If you use this analysis pipeline in your research, please cite the repository and relevant methods papers for the tools used (Cellpose, statistical methods, etc.).

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with tests and documentation

