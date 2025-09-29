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

## Configuration

### `config_example.json`
Example configuration file for `compare_gevis.py`:
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

Two complete analysis runs are included comparing ASAP voltage indicators:

### `gevi_comparison/`
Standard analysis of ASAP4, ASAP6.2, and ASAP6.3:
- **Sample**: 4,034 events across 29 ROIs
- **Date**: September 25, 2025
- **Method**: 10,000 bootstrap iterations with BCa confidence intervals

### `gevi_conservative_comparison/`
Conservative analysis with more stringent quality control:
- Demonstrates robustness across parameter settings
- Stricter thresholding and filtering criteria

### Output Files (in each results directory):
- **Figures**: `gevi_comparison_main.[png/pdf/svg]`, `gevi_comparison_supplement.[png/pdf]`, `gevi_spike_shape_comparison.[png/pdf/svg]`
- **Data**: `summary_table.csv`, `per_event_metrics.csv`, `per_roi_metrics.csv`
- **Metadata**: `statistical_results.json`, `run_metadata.json`, `analysis.log`, `README.txt`

## Metrics

### Event-Level
- **Amplitude**: ΔF/F (%) relative to baseline
- **Width**: Event duration (ms)
- **SNR**: Peak amplitude / baseline noise (MAD)
- **Prominence**: Height above surrounding baseline
- **AUC**: Integrated ΔF/F over time

### ROI-Level
- **Event Rate**: Events per second per ROI
- **Amplitude CV**: Coefficient of variation across events
- **Baseline Brightness**: Median fluorescence intensity
- **Detection Consistency**: Inter-quartile range of metrics

## Methods

### Statistical Analysis
- Bootstrap confidence intervals (BCa method)
- Nonparametric tests (Mann-Whitney U, Kolmogorov-Smirnov)
- Effect sizes (Cliff's delta, median differences)
- Multiple comparison correction (Benjamini-Hochberg FDR)

### Quality Control
- MAD-based outlier detection
- Winsorization of extreme values
- Minimum event count filtering
- Motion artifact correction

### Spike Analysis
- Event alignment (onset or peak)
- Temporal interpolation to common time grid
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

