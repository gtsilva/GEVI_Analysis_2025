# GEVI Performance Comparison Analysis

Generated on: 2025-09-25T12:11:45.938583

## Overview
This analysis compares the performance of 2 voltage indicators:
ASAP6.2, ASAP6.3

## Sample Sizes
- Total events analyzed: 874
- Total ROIs analyzed: 7

## Preprocessing Steps
- baseline_correction
- artifact_removal
- quality_control
- unit_conversion

## Configuration
- Frame rate: 1000.0 Hz
- Minimum events per ROI: 3
- Bootstrap iterations: 10000
- Random seed: 42

## Output Files

### Figures
- `gevi_comparison_main.png/pdf/svg`: Main comparison figure with all key metrics
- `gevi_comparison_supplement.png/pdf`: Supplementary analyses
- `gevi_spike_shape_comparison.png/pdf/svg`: Spike shape and kinetics overlay (if generated)

### Data Tables
- `summary_table.csv`: Publication-ready summary with effect sizes and CIs
- `per_roi_metrics.csv`: Detailed per-ROI metrics
- `per_event_metrics.csv`: Individual event data
- `per_event_metrics.parquet`: Same as CSV but more efficient format

### Analysis Results
- `statistical_results.json`: Complete statistical analysis results
- `run_metadata.json`: Analysis parameters and metadata

## Methods

### Metrics Calculated
- **Amplitude**: ΔF/F (%) relative to baseline
- **Width**: Event duration in milliseconds
- **SNR**: Signal-to-noise ratio (amplitude / baseline MAD)
- **Event Rate**: Events per second per ROI

### Statistical Analysis
- Bootstrap confidence intervals (BCa method, 10,000 iterations)
- Effect sizes as differences in medians with 95% CIs
- Nonparametric tests (Mann-Whitney U, Kolmogorov-Smirnov)
- Multiple testing correction (Benjamini-Hochberg FDR)
- Cliff's delta for effect size interpretation

### Quality Control
- Minimum 3 events per ROI
- Artifact removal using MAD-based outlier detection
- Winsorization at 2.5th percentiles

## Interpretation
- Effect sizes > 0.5 are considered large
- p-values are FDR-corrected for multiple comparisons
- Bootstrap CIs that don't include 0 indicate significant differences
- Raw data points are shown alongside summary statistics

For questions about this analysis, refer to the statistical_results.json file
for detailed test statistics and confidence intervals.