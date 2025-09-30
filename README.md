# ASAP Analysis - Voltage Indicator Performance Comparison

Python tools for analyzing genetically encoded voltage indicators (GEVIs) using fluorescence imaging data. Includes fluorescence extraction, event detection, statistical comparison, and visualization.

## Overview

This repository contains tools for GEVI performance analysis. The pipeline processes raw imaging data, detects events, and generates statistical comparisons with figures.

## Scripts

### `simple_conservative_extract_fluo.py`
Fluorescence spike extraction from imaging data.

**Data Sources:**
- RigDataV2 camera data
- TIFF stacks
- Single trace files (.npy, .csv)

**Method:**
- Rolling percentile baseline detrending
- 5σ threshold detection
- Auto-selection of silent windows
- Optional SOS Butterworth filtering

**Output:**
- Per-ROI event CSVs
- Diagnostic plots

**Usage:**
```bash
# RigDataV2 camera data
python simple_conservative_extract_fluo.py --source rig --data-dir /path/to/experiment

# TIFF stack
python simple_conservative_extract_fluo.py --source tiff --tif /path/to/stack.tif

# Single trace file
python simple_conservative_extract_fluo.py --input trace.npy --fs 1000 --out_csv events.csv --out_plot diag.png
```

### `extract_fluorescence_cellpose.py`
Fluorescence extraction pipeline with Cellpose segmentation.

**Components:**
- Cell detection via Cellpose models
- Motion correction using phase correlation-based frame registration
- Neuropil correction using ring-shaped mask subtraction

**Data Sources:**
- RigDataV2 camera data
- TIFF stacks

**Output:**
- Fluorescence traces
- Cell masks
- Quality control plots

### `compare_gevis.py`
GEVI comparison pipeline with statistical analysis.

**Analysis:**
- Multi-GEVI performance comparison (2-4 indicators)
- Bootstrap confidence intervals (BCa method)
- Effect size calculations (Cliff's delta, median differences)
- Statistical testing (Mann-Whitney U, Kolmogorov-Smirnov) with FDR correction
- Outlier detection and artifact removal
- Spike shape analysis with temporal alignment and interpolation

**Output Formats:**
- PNG, PDF, SVG, CSV, JSON

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

### `RigDataV2.py`
Data loading and instrumentation interface by **Shane Nichols** ([LinkedIn](https://www.linkedin.com/in/shane-nichols/)).

**Functions:**
- Waveform data loading from instrumentation systems
- Camera data integration with parameters and binary frames
- Galvo scanning and camera acquisition modes
- Temporal alignment and synchronization

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

Two analysis runs comparing ASAP voltage indicators:

### `gevi_comparison/`
Analysis of ASAP4, ASAP6.2, and ASAP6.3:
- **Sample**: 4,034 events across 29 ROIs
- **Date**: September 25, 2025
- **Method**: 10,000 bootstrap iterations with BCa confidence intervals

### `gevi_conservative_comparison/`
Analysis with modified quality control parameters:
- Alternative parameter settings
- Modified thresholding and filtering criteria

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
pip install numpy pandas matplotlib scipy scikit-image
```

### For Fluorescence Extraction
```bash
pip install cellpose torch tifffile
```

### Optional
```bash
pip install xarray statsmodels pyarrow PyYAML
```

### System Requirements
- Python 3.8+
- GPU optional (CUDA or MPS) for Cellpose acceleration

## Workflow

```
Raw Data (RigDataV2/TIFF/traces)
    ↓
[extract_fluorescence_cellpose.py] or [simple_conservative_extract_fluo.py]
    ↓
Event CSVs + Fluorescence traces
    ↓
[compare_gevis.py]
    ↓
Statistical comparison + Figures
```

## Citation

If you use this pipeline, please cite relevant tools:
- **Cellpose**: Stringer, C. et al. (2021). Nat Methods 18, 100-106
- **RigDataV2**: Shane Nichols

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

**Note**: `RigDataV2.py` was authored by Shane Nichols and may have separate licensing terms.
