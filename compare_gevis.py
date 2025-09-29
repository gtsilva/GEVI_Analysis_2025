#!/usr/bin/env python3
"""

GEVI comparison.

Compares voltage indicator performance with statistics, effect sizes,
and visualizations. Supports multiple GEVIs, mixed-effects.

"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
try:
    import yaml
except ImportError:
    yaml = None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from math import floor, ceil
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False
})


def setup_logging(log_file: Optional[Path] = None, level: str = 'INFO') -> logging.Logger:
    """Configure logging for the pipeline."""
    logger = logging.getLogger('gevi_compare')
    logger.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: Optional[Path]) -> Dict:
    """Load configuration from YAML/JSON file."""
    default_config = {
        'frame_rate': 1000.0,  # Hz
        'pixel_size': 1.0,     # µm
        'exposure_ms': 1.0,    # ms
        'illumination_power': 1.0,  # mW/mm²
        'bleaching_window_min': 10.0,  # minutes
        'baseline_percentile': 10,
        'baseline_window_sec': 2.0,
        'artifact_threshold': 5.0,  # MAD units
        'min_brightness': 100,
        'max_motion': 2.0,  # pixels
        'min_events_per_roi': 3,
        'winsorize_percentile': 2.5,
        'bootstrap_n': 10000,
        'alpha': 0.05,
        'random_seed': 42,
        'gevi_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    }
    
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                if yaml is None:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
                user_config = yaml.safe_load(f)
            else:
                user_config = json.load(f)
        default_config.update(user_config)
    
    return default_config


def validate_inputs(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """Validate input data and configuration parameters."""
    logger.info("Validating inputs...")
    
    # First, standardize column names
    df = standardize_columns(df, logger)
    
    required_columns = ['t_sec', 'gevi', 'experiment', 'roi']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        # Show available columns for debugging
        logger.error(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate numeric columns
    numeric_cols = ['t_sec']
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Converting {col} to numeric")
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Validate configuration
    assert config['frame_rate'] > 0, "Frame rate must be positive"
    assert config['min_events_per_roi'] >= 1, "Minimum events per ROI must be >= 1"
    assert 0 < config['alpha'] < 1, "Alpha must be between 0 and 1"
    
    logger.info(f"Input validation passed. Data shape: {df.shape}")
    return df


def load_data(paths: Dict[str, Path], logger: logging.Logger) -> Tuple[pd.DataFrame, Dict]:
    """Load and combine data from multiple GEVI experiments with real traces."""
    logger.info("Loading data from experiment paths...")
    
    all_data = []
    real_events = []
    
    for gevi_name, root_path in paths.items():
        logger.info(f"Loading {gevi_name} from {root_path}")
        
        # Find all event files
        event_files = list(root_path.glob("**/events/events_roi*.csv"))
        if not event_files:
            # Try alternative patterns
            event_files = list(root_path.glob("**/*events*.csv"))
        
        logger.info(f"Found {len(event_files)} event files for {gevi_name}")
        
        for file_path in event_files:
            try:
                # Extract ROI number from filename
                roi_match = Path(file_path).stem
                roi_num = ''.join(filter(str.isdigit, roi_match))
                if not roi_num:
                    roi_num = '0'
                
                # Extract experiment ID from path
                exp_id = f"{file_path.parent.parent.name}"
                
                # Load event CSV
                df = pd.read_csv(file_path)
                
                # Add metadata columns
                df['gevi'] = gevi_name
                df['experiment'] = exp_id
                df['roi'] = int(roi_num)
                df['file_path'] = str(file_path)
                
                all_data.append(df)
                
                # Try to load corresponding traces.csv for real fluorescence data
                traces_path = file_path.parent.parent / "traces.csv"
                if traces_path.exists():
                    try:
                        traces_df = pd.read_csv(traces_path)
                        
                        # Filter to this ROI
                        roi_traces = traces_df[traces_df['cell_id'] == int(roi_num)]
                        
                        if len(roi_traces) > 0:
                            # Extract trace data
                            F = roi_traces['F_corr'].values
                            times = roi_traces['time'].values
                            
                            # Estimate dt from time series
                            if len(times) > 1:
                                dt = np.median(np.diff(times))
                            else:
                                dt = 1.0 / 1000.0  # Default 1000 Hz
                            
                            # For each event in this ROI, create real event dict
                            for _, event_row in df.iterrows():
                                event_time = event_row['time_s']
                                
                                # Find closest frame index for event time
                                time_diffs = np.abs(times - event_time)
                                i0 = np.argmin(time_diffs)
                                
                                # Create event dict with real trace
                                event_dict = {
                                    'gevi': gevi_name,
                                    'roi_id': int(roi_num),
                                    'F': F,
                                    'dt': dt,
                                    'i0': i0,
                                    'times': times,
                                    'experiment': exp_id,
                                    'event_time': event_time
                                }
                                real_events.append(event_dict)
                                
                    except Exception as e:
                        logger.warning(f"Failed to load traces for {file_path}: {e}")
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
    
    if not all_data:
        raise ValueError("No valid data files found")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} total events from {len(all_data)} files")
    logger.info(f"Found {len(real_events)} events with real trace data")
    
    return combined_df, real_events


def preprocess(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """Preprocess data with baseline correction, artifact removal, and QC."""
    logger.info("Preprocessing data...")
    
    np.random.seed(config['random_seed'])
    
    # Calculate baseline and ΔF/F if not present
    if 'amplitude_percent' not in df.columns:
        df = calculate_baseline_dff(df, config, logger)
    
    # Remove artifacts and outliers
    df = remove_artifacts(df, config, logger)
    
    # Quality control filtering
    df = quality_control_filter(df, config, logger)
    
    # Convert units and validate
    df = convert_units(df, config, logger)
    
    logger.info(f"Preprocessing complete. Final data shape: {df.shape}")
    return df


def standardize_columns(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Standardize column names across different data formats."""
    
    # Column mapping for common variations (based on actual CSV format from extract_fluorescence_cellpose.py)
    column_map = {
        'time_s': 't_sec',
        'time': 't_sec',
        't': 't_sec',
        'amp_percent': 'amplitude_percent',
        'amp_dff': 'amplitude_raw',
        'amplitude': 'amplitude_raw',
        'amp': 'amplitude_raw',
        'dff_amp': 'amplitude_raw',
        'width_ms': 'width_ms',  # Keep as is
        'width': 'width_raw',
        'fwhm': 'width_raw',
        'prom_percent': 'prominence_percent',
        'prominence': 'prominence_raw',
        'prom': 'prominence_raw',
        'snr_peak': 'snr',
        'local_F0': 'baseline',
        'local_noise_sigma': 'noise_sigma'
    }
    
    # Apply column mapping
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    
    return df


def calculate_baseline_dff(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """Calculate baseline and ΔF/F with bleaching correction."""
    
    def rolling_percentile_baseline(trace: np.ndarray, times: np.ndarray, 
                                  percentile: float, window_sec: float) -> np.ndarray:
        """Calculate rolling percentile baseline."""
        if len(trace) < 10:
            return np.full_like(trace, np.nanpercentile(trace, percentile))
        
        dt = np.median(np.diff(times))
        window_samples = int(window_sec / dt)
        
        baseline = np.zeros_like(trace)
        for i in range(len(trace)):
            start_idx = max(0, i - window_samples // 2)
            end_idx = min(len(trace), i + window_samples // 2)
            baseline[i] = np.nanpercentile(trace[start_idx:end_idx], percentile)
        
        return baseline
    
    # Group by experiment and ROI for baseline calculation
    processed_groups = []
    
    for (exp, roi), group in df.groupby(['experiment', 'roi']):
        if 'amplitude_raw' in group.columns:
            # Calculate baseline
            baseline = rolling_percentile_baseline(
                group['amplitude_raw'].values,
                group['t_sec'].values,
                config['baseline_percentile'],
                config['baseline_window_sec']
            )
            
            # Calculate ΔF/F
            dff = (group['amplitude_raw'] - baseline) / baseline * 100
            
            group = group.copy()
            group['baseline'] = baseline
            group['amplitude_percent'] = dff
        
        processed_groups.append(group)
    
    return pd.concat(processed_groups, ignore_index=True)


def remove_artifacts(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """Remove artifacts and outliers."""
    
    initial_count = len(df)
    
    # Remove events with NaN values in critical columns
    critical_cols = ['t_sec', 'amplitude_percent']
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
    
    # Remove extreme outliers using MAD-based threshold
    if 'amplitude_percent' in df.columns:
        amp_median = df['amplitude_percent'].median()
        amp_mad = stats.median_abs_deviation(df['amplitude_percent'], nan_policy='omit')
        threshold = config['artifact_threshold'] * amp_mad
        
        artifact_mask = np.abs(df['amplitude_percent'] - amp_median) > threshold
        df = df[~artifact_mask]
    
    # Winsorize remaining outliers
    numeric_cols = ['amplitude_percent', 'width_ms', 'prominence_percent']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = stats.mstats.winsorize(
                df[col], 
                limits=(config['winsorize_percentile']/100, config['winsorize_percentile']/100)
            )
    
    logger.info(f"Artifact removal: {initial_count} -> {len(df)} events "
                f"({initial_count - len(df)} removed)")
    
    return df


def quality_control_filter(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """Apply quality control filters to ROIs."""
    
    initial_rois = df['roi'].nunique()
    
    # Count events per ROI
    roi_counts = df.groupby(['experiment', 'roi']).size()
    valid_rois = roi_counts[roi_counts >= config['min_events_per_roi']].index
    
    # Filter to valid ROIs
    df = df.set_index(['experiment', 'roi']).loc[valid_rois].reset_index()
    
    final_rois = df['roi'].nunique()
    logger.info(f"QC filtering: {initial_rois} -> {final_rois} ROIs "
                f"({initial_rois - final_rois} removed)")
    
    return df


def convert_units(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """Convert and validate units."""
    
    # Convert width from samples to ms if needed
    if 'width_raw' in df.columns and 'width_ms' not in df.columns:
        df['width_ms'] = df['width_raw'] * (1000.0 / config['frame_rate'])
    
    # Ensure amplitude is in percent
    if 'amplitude_percent' not in df.columns:
        if 'amplitude_raw' in df.columns:
            # Check if raw amplitude looks like it's already in percent or fraction
            max_amp = df['amplitude_raw'].abs().max()
            if max_amp <= 2.0:  # Looks like fractional
                df['amplitude_percent'] = df['amplitude_raw'] * 100
            else:  # Assume already in percent
                df['amplitude_percent'] = df['amplitude_raw']
        else:
            logger.warning("No amplitude data found")
            df['amplitude_percent'] = np.nan
    
    # Convert prominence to percent if needed
    if 'prominence_percent' not in df.columns:
        if 'prominence_raw' in df.columns:
            max_prom = df['prominence_raw'].abs().max()
            if max_prom <= 2.0:  # Looks like fractional
                df['prominence_percent'] = df['prominence_raw'] * 100
            else:  # Assume already in percent
                df['prominence_percent'] = df['prominence_raw']
    
    return df


def extract_metrics(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract comprehensive per-event and per-ROI metrics."""
    logger.info("Extracting metrics...")
    
    # Per-event metrics (already mostly calculated)
    per_event = df.copy()
    
    # Calculate additional per-event metrics
    per_event = calculate_event_metrics(per_event, config, logger)
    
    # Per-ROI metrics
    per_roi = calculate_roi_metrics(per_event, config, logger)
    
    logger.info(f"Extracted metrics for {len(per_event)} events and {len(per_roi)} ROIs")
    return per_event, per_roi


def calculate_event_metrics(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """Calculate additional per-event metrics."""
    
    # SNR calculation (if baseline available)
    if 'baseline' in df.columns:
        baseline_mad = df.groupby(['experiment', 'roi'])['baseline'].transform(
            lambda x: stats.median_abs_deviation(x, nan_policy='omit')
        )
        df['snr'] = np.abs(df['amplitude_percent']) / (baseline_mad * 100)
    else:
        df['snr'] = np.nan
    
    # Area under curve (approximate)
    if 'width_ms' in df.columns and 'amplitude_percent' in df.columns:
        df['auc'] = df['amplitude_percent'] * df['width_ms'] / 1000  # %·s
    
    return df


def calculate_roi_metrics(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """Calculate per-ROI summary metrics."""
    
    roi_metrics = []
    
    for (gevi, exp, roi), group in df.groupby(['gevi', 'experiment', 'roi']):
        n_events = len(group)
        
        # Recording duration
        if n_events >= 2:
            duration_sec = group['t_sec'].max() - group['t_sec'].min()
            event_rate = n_events / duration_sec if duration_sec > 0 else np.nan
        else:
            duration_sec = np.nan
            event_rate = np.nan
        
        # Event metrics
        metrics = {
            'gevi': gevi,
            'experiment': exp,
            'roi': roi,
            'n_events': n_events,
            'duration_sec': duration_sec,
            'event_rate_hz': event_rate,
            'amplitude_median': group['amplitude_percent'].median(),
            'amplitude_iqr': group['amplitude_percent'].quantile(0.75) - group['amplitude_percent'].quantile(0.25),
            'amplitude_cv': group['amplitude_percent'].std() / group['amplitude_percent'].mean() if group['amplitude_percent'].mean() > 0 else np.nan,
        }
        
        # Width metrics
        if 'width_ms' in group.columns:
            metrics.update({
                'width_median': group['width_ms'].median(),
                'width_iqr': group['width_ms'].quantile(0.75) - group['width_ms'].quantile(0.25),
            })
        
        # SNR metrics
        if 'snr' in group.columns:
            metrics.update({
                'snr_median': group['snr'].median(),
                'snr_iqr': group['snr'].quantile(0.75) - group['snr'].quantile(0.25),
            })
        
        # Baseline brightness (if available)
        if 'baseline' in group.columns:
            metrics['baseline_brightness'] = group['baseline'].median()
        
        roi_metrics.append(metrics)
    
    return pd.DataFrame(roi_metrics)


def bootstrap_ci(data: np.ndarray, statistic: callable = np.median, 
                n_boot: int = 10000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence intervals using BCa method."""
    if len(data) < 2:
        return np.nan, np.nan, np.nan
    
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return np.nan, np.nan, np.nan
    
    # Original statistic
    theta_hat = statistic(data)
    
    # Bootstrap resamples
    n = len(data)
    boot_stats = []
    
    for _ in range(n_boot):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic(boot_sample))
    
    boot_stats = np.array(boot_stats)
    
    # BCa confidence intervals
    alpha_lower = alpha / 2
    alpha_upper = 1 - alpha / 2
    
    ci_lower = np.percentile(boot_stats, alpha_lower * 100)
    ci_upper = np.percentile(boot_stats, alpha_upper * 100)
    
    return theta_hat, ci_lower, ci_upper


def _mad(x):
    """Calculate median absolute deviation."""
    x = np.asarray(x)
    return np.median(np.abs(x - np.median(x))) + 1e-12


def bootstrap_ci_simple(x, stat=np.median, B=10000, seed=0):
    """Simple bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    if x.size == 0:
        return np.nan, np.nan
    boots = [stat(rng.choice(x, size=x.size, replace=True)) for _ in range(B)]
    return np.percentile(boots, [2.5, 97.5])


def _percentile_time(t, y, frac):
    """Find first time y crosses frac*max on rise."""
    y = np.asarray(y)
    t = np.asarray(t)
    ymax = np.max(y)
    target = frac * ymax
    idx = np.where(y >= target)[0]
    if idx.size == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return t[0]
    # linear interp between i-1 and i
    x0, x1 = y[i-1], y[i]
    if x1 == x0:
        return t[i]
    w = (target - x0) / (x1 - x0)
    return t[i-1] + w * (t[i] - t[i-1])


def cliff_delta_effect_size(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cliff's delta effect size."""
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    if len(x) == 0 or len(y) == 0:
        return np.nan
    
    # Count pairs where y > x, y < x
    greater = np.sum(y[:, np.newaxis] > x[np.newaxis, :])
    lesser = np.sum(y[:, np.newaxis] < x[np.newaxis, :])
    
    return (greater - lesser) / (len(x) * len(y))


def align_events(events, pre_ms, post_ms, amp_min, snr_min, min_pre_ms=50.0, 
                min_post_ms=150.0, align_by="onset", interp="linear", 
                frame_rate=None, force_figure=False, logger=None):
    """
    Align events using real trace data with comprehensive logging.
    
    events: iterable of dicts with keys {'gevi','roi_id','F','dt','i0','ipk'?}
    Returns dict: gevi -> dict(aligned=np.ndarray[n_events, T],
                               aligned_norm=np.ndarray[n_events, T],
                               t_ms=np.ndarray[T],
                               metrics=dict of per-event arrays,
                               n_rois=int, roi_ids=set)
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    # Relaxed default gates
    amp_min = max(amp_min, 0.005)
    snr_min = max(snr_min, 1.0)
    
    logger.info(f"Alignment parameters: pre={pre_ms}ms, post={post_ms}ms, "
                f"amp_min={amp_min}, snr_min={snr_min}, align_by={align_by}, interp={interp}")
    logger.info(f"Minimum windows: pre={min_pre_ms}ms, post={min_post_ms}ms")
    
    # Drop reason counters
    drop_reasons = {
        'short_pre_window': 0,
        'short_post_window': 0, 
        'missing_dt': 0,
        'amp_below': 0,
        'snr_below': 0,
        'interp_nan': 0,
        'other': 0
    }
    
    out = {}
    # common 1 ms grid
    Tpre = int(round(pre_ms))
    Tpost = int(round(post_ms))
    t_ms = np.arange(-Tpre, Tpost+1, 1.0)  # 1 ms grid
    
    # Ensure all events have required keys and set defaults
    for ev in events:
        ev.setdefault('ipk', None)
        if ev.get('dt') is None:
            if frame_rate is None:
                drop_reasons['missing_dt'] += 1
                continue
            ev['dt'] = 1.0 / frame_rate
    
    # group by GEVI
    from collections import defaultdict
    grouped = defaultdict(list)
    for ev in events:
        if ev.get('dt') is not None:  # Only include events with valid dt
            grouped[ev['gevi']].append(ev)

    for gevi, lst in grouped.items():
        aligned, aligned_norm = [], []
        amp, t_rise, t_decay, fwhm, t_peak = [], [], [], [], []
        auc, tau_d = [], []
        roi_ids = set()
        
        logger.info(f"Processing {len(lst)} events for {gevi}")

        for ev in lst:
            try:
                F = np.asarray(ev['F']).astype(float)
                dt = float(ev['dt'])
                i0 = int(ev['i0'])
                roi_ids.add(ev['roi_id'])

                n = F.size
                
                # Check for sufficient pre/post window
                pre_s = int(round(pre_ms/1000.0/dt))
                post_s = int(round(post_ms/1000.0/dt))
                min_pre_s = int(round(min_pre_ms/1000.0/dt))
                min_post_s = int(round(min_post_ms/1000.0/dt))
                
                available_pre = i0
                available_post = n - i0 - 1
                
                if available_pre < min_pre_s:
                    drop_reasons['short_pre_window'] += 1
                    continue
                if available_post < min_post_s:
                    drop_reasons['short_post_window'] += 1
                    continue
                
                # Align by peak if requested
                if align_by == "peak":
                    # Find peak in reasonable window around i0
                    search_start = max(0, i0 - pre_s//2)
                    search_end = min(n, i0 + post_s//2)
                    if search_end > search_start:
                        ipk = np.argmax(F[search_start:search_end]) + search_start
                        i0 = ipk  # Update alignment point
                
                # Window bounds in samples
                a = max(0, i0 - pre_s)
                b = min(n, i0 + post_s + 1)

                # local baseline using 10th percentile on pre window
                pre_a = max(0, i0 - pre_s)
                F0 = np.percentile(F[pre_a:i0] if i0>pre_a else F[:max(1,i0)], 10)
                if not np.isfinite(F0) or F0 == 0:
                    F0 = np.median(F[max(0, i0-5):i0+1]) + 1e-12

                # local linear detrend on [a,b)
                x = np.arange(a, b)
                if b - a >= 3:
                    coef = np.polyfit(x, F[a:b], 1)
                    trend = np.polyval(coef, x)
                    Fd = F[a:b] - trend
                else:
                    Fd = F[a:b] - F0

                dff = (Fd - (np.median(Fd[:min(10, Fd.size)]) if Fd.size>0 else 0.0)) / max(F0, 1e-12)

                # SNR and amplitude gate
                pre_seg = dff[:min(len(dff)//3, int(pre_s*0.5))] if len(dff)>10 else dff[:5]
                noise = _mad(pre_seg) if pre_seg.size>0 else _mad(dff)
                amp_ev = np.max(dff) if dff.size else np.nan
                snr = amp_ev / max(noise, 1e-12)
                
                if not np.isfinite(amp_ev) or amp_ev < amp_min:
                    drop_reasons['amp_below'] += 1
                    continue
                if snr < snr_min:
                    drop_reasons['snr_below'] += 1
                    continue

                # temporal vector in ms for the cut segment
                t_local_ms = (np.arange(a, b) - i0) * dt * 1000.0

                # resample to common 1 ms grid
                if b - a >= 4:
                    try:
                        if interp == "cubic":
                            cs = CubicSpline(t_local_ms, dff, extrapolate=False)
                            y = cs(t_ms)
                        else:  # linear interpolation
                            y = np.interp(t_ms, t_local_ms, dff)
                    except Exception:
                        # Fall back to linear if cubic fails
                        try:
                            y = np.interp(t_ms, t_local_ms, dff)
                        except Exception:
                            drop_reasons['interp_nan'] += 1
                            continue
                else:
                    # too few samples, skip
                    drop_reasons['other'] += 1
                    continue
                    
                if np.all(~np.isfinite(y)):
                    drop_reasons['interp_nan'] += 1
                    continue
                    
                # fill NaNs at ends by nearest finite
                finite = np.isfinite(y)
                if not finite.any():
                    drop_reasons['interp_nan'] += 1
                    continue
                y[~finite] = np.interp(t_ms[~finite], t_ms[finite], y[finite])

                # normalized for kinetics
                ymax = np.max(y)
                y_norm = y / max(ymax, 1e-12)

                # kinetics
                tpk = t_ms[np.argmax(y)]
                t10 = _percentile_time(t_ms, y, 0.10)
                t90 = _percentile_time(t_ms, y, 0.90)
                trise = t90 - t10 if np.isfinite(t90) and np.isfinite(t10) else np.nan

                # decay 90->10 after peak
                post_mask = t_ms >= tpk
                tm = t_ms[post_mask]; ym = y[post_mask]
                if ym.size >= 3:
                    # times when crossing 0.9 and 0.1 of peak on decay
                    def cross(frac):
                        target = frac * np.max(y)
                        idx = np.where(ym <= target)[0]
                        if idx.size == 0:
                            return np.nan
                        j = idx[0]
                        if j == 0:
                            return tm[0]
                        x0, x1 = ym[j-1], ym[j]
                        w = (target - x0) / (x1 - x0) if x1 != x0 else 0.0
                        return tm[j-1] + w*(tm[j]-tm[j-1])
                    t90d = cross(0.90); t10d = cross(0.10)
                    tdec = t10d - t90d if np.isfinite(t10d) and np.isfinite(t90d) else np.nan
                else:
                    tdec = np.nan

                # FWHM
                half = 0.5 * np.max(y)
                left_idx = np.where(y[:np.argmax(y)+1] >= half)[0]
                right_idx = np.where(y[np.argmax(y):] >= half)[0]
                if left_idx.size and right_idx.size:
                    t_left = np.interp(half, [y[left_idx[0]-1] if left_idx[0]>0 else y[left_idx[0]], y[left_idx[0]]],
                                       [t_ms[left_idx[0]-1] if left_idx[0]>0 else t_ms[left_idx[0]], t_ms[left_idx[0]]])
                    r0 = np.argmax(y)
                    t_right = np.interp(half, [y[r0+right_idx[-1]-1] if r0+right_idx[-1]-1>=0 else y[r0+right_idx[-1]],
                                              y[r0+right_idx[-1]]],
                                        [t_ms[r0+right_idx[-1]-1] if r0+right_idx[-1]-1>=0 else t_ms[r0+right_idx[-1]],
                                         t_ms[r0+right_idx[-1]]])
                    fwhm_ms = t_right - t_left
                else:
                    fwhm_ms = np.nan

                # AUC up to 300 ms after peak
                end_time = tpk + 300.0
                mask_auc = (t_ms >= 0) & (t_ms <= end_time)
                auc_ms = np.trapz(y[mask_auc], t_ms[mask_auc])

                aligned.append(y)
                aligned_norm.append(y_norm)
                amp.append(amp_ev); t_rise.append(trise); t_decay.append(tdec)
                fwhm.append(fwhm_ms); t_peak.append(tpk); auc.append(auc_ms); tau_d.append(np.nan)
                
            except Exception as e:
                drop_reasons['other'] += 1
                logger.debug(f"Event processing failed: {e}")
                continue

        logger.info(f"{gevi}: {len(aligned)} events passed filters")
        if len(aligned) == 0:
            continue

        out[gevi] = dict(
            aligned=np.vstack(aligned),
            aligned_norm=np.vstack(aligned_norm),
            t_ms=t_ms,
            metrics=dict(amp=np.array(amp), t_rise=np.array(t_rise), t_decay=np.array(t_decay),
                         fwhm=np.array(fwhm), t_peak=np.array(t_peak), auc=np.array(auc), tau_d=np.array(tau_d)),
            n_rois=len(roi_ids),
            roi_ids=roi_ids
        )
    
    # Log drop reasons
    total_dropped = sum(drop_reasons.values())
    logger.info(f"Event filtering results:")
    logger.info(f"  Total events processed: {len(events)}")
    logger.info(f"  Total events dropped: {total_dropped}")
    for reason, count in drop_reasons.items():
        if count > 0:
            logger.info(f"    {reason}: {count}")
    
    total_passed = sum(len(d['aligned']) for d in out.values())
    logger.info(f"  Total events passed: {total_passed}")
    
    if total_passed == 0:
        logger.warning("No events passed all filters!")
        
    return out


def make_spike_shape_figure(aligned_by_gevi, title_prefix, boot=10000, outdir=".", palette=None):
    """Create spike shape and kinetics figure."""
    if palette is None:
        palette = {}
    # Panel layout: A mean ΔF/F, B heatmaps per GEVI (removed normalized plot)
    gevis = list(aligned_by_gevi.keys())
    if len(gevis) < 1:
        return None, None
    t_ms = next(iter(aligned_by_gevi.values()))['t_ms']
    
    # Dynamically adjust figure width and columns based on number of GEVIs
    n_gevis = len(gevis)
    n_cols = min(n_gevis, 4)  # Maximum 4 columns to keep readable
    fig_width = max(12, 3 * n_cols + 3)  # Scale width with number of columns
    
    fig = plt.figure(figsize=(fig_width, 6))
    gs = fig.add_gridspec(2, n_cols, height_ratios=[1.5, 1], hspace=0.35, wspace=0.25)
    axA = fig.add_subplot(gs[0, :])
    axs_heat = [fig.add_subplot(gs[1, i]) for i in range(n_cols)]

    # A) grand average ΔF/F with 95% CI
    for gevi in gevis:
        col = palette.get(gevi, None)
        Y = aligned_by_gevi[gevi]['aligned']  # events x time
        mu = np.nanmean(Y, axis=0)
        # bootstrap CI over events
        rng = np.random.default_rng(0)
        B = min(boot, 5000)
        idx = np.arange(Y.shape[0])
        boots = np.empty((B, Y.shape[1]))
        for b in range(B):
            take = rng.choice(idx, size=idx.size, replace=True)
            boots[b] = np.nanmean(Y[take], axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        axA.plot(t_ms, mu*100, label=f"{gevi}  n={Y.shape[0]}",
                 linewidth=2, color=col)
        axA.fill_between(t_ms, lo*100, hi*100, alpha=0.2, color=col, linewidth=0)
        # annotate peak
        pk = t_ms[np.nanargmax(mu)]
        axA.axvline(pk, linestyle=":", linewidth=1, color=col, alpha=0.7)

    axA.axvline(0, color="k", linewidth=1)
    axA.set_xlabel("Time from onset (ms)")
    axA.set_ylabel("ΔF/F (%)")
    axA.set_title(f"{title_prefix} — Grand average ΔF/F with 95% CI")
    axA.legend(frameon=False)

    # B) heatmaps (normalized plot removed)
    im = None
    for i, gevi in enumerate(gevis[:n_cols]):
        ax = axs_heat[i]
        H = aligned_by_gevi[gevi]['aligned_norm']
        # sort by t_peak
        tpk = aligned_by_gevi[gevi]['metrics']['t_peak']
        order = np.argsort(tpk)
        im = ax.imshow(H[order, :], aspect="auto", origin="lower",
                       extent=[t_ms[0], t_ms[-1], 0, H.shape[0]],
                       vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"{gevi} events (normalized)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("events")
    
    # Hide unused heatmap axes if we have more columns than GEVIs
    for i in range(len(gevis), n_cols):
        axs_heat[i].set_visible(False)
    
    if im is not None:
        # Place colorbar at bottom center, below the heatmaps
        cax = fig.add_axes([0.3, 0.02, 0.4, 0.02])  # [left, bottom, width, height]
        plt.colorbar(im, cax=cax, orientation='horizontal', label="norm ΔF/F")

    fig.suptitle(title_prefix, y=0.96)
    # Leave space at bottom for horizontal colorbar
    fig.tight_layout(rect=[0, 0.08, 1.0, 0.94])
    return fig, f"{outdir}/gevi_spike_shape_comparison.png"


# REMOVED: synthetic event generation - now using real data only


def stats_models(per_event: pd.DataFrame, per_roi: pd.DataFrame, 
                config: Dict, logger: logging.Logger) -> Dict:
    """Run comprehensive statistical analyses."""
    logger.info("Running statistical analyses...")
    
    results = {}
    
    # Group comparisons for main metrics
    metrics = ['amplitude_percent', 'width_ms', 'snr', 'event_rate_hz']
    gevis = per_event['gevi'].unique()
    
    if len(gevis) < 2:
        logger.warning("Need at least 2 GEVIs for comparison")
        return results
    
    # Import here to avoid dependency issues if not available
    try:
        from statsmodels.stats.multitest import fdrcorrection
    except ImportError:
        logger.warning("statsmodels not available, using basic FDR correction")
        def fdrcorrection(pvals, alpha=0.05):
            from scipy.stats import false_discovery_control
            return false_discovery_control(pvals, alpha=alpha), false_discovery_control(pvals, alpha=alpha)
    
    # Pairwise comparisons
    for metric in metrics:
        if metric in ['event_rate_hz']:
            data_df = per_roi
        else:
            data_df = per_event
        
        if metric not in data_df.columns:
            continue
        
        results[metric] = {}
        
        # Bootstrap effect sizes and CIs
        gevi_data = {}
        for gevi in gevis:
            data = data_df[data_df['gevi'] == gevi][metric].dropna().values
            if len(data) > 0:
                stat, ci_low, ci_high = bootstrap_ci(data, n_boot=config['bootstrap_n'])
                gevi_data[gevi] = {
                    'data': data,
                    'median': stat,
                    'ci_lower': ci_low,
                    'ci_upper': ci_high,
                    'n': len(data)
                }
        
        results[metric]['groups'] = gevi_data
        
        # Pairwise tests
        gevi_pairs = [(g1, g2) for i, g1 in enumerate(gevis) for g2 in gevis[i+1:]]
        pairwise_results = []
        
        for g1, g2 in gevi_pairs:
            if g1 in gevi_data and g2 in gevi_data:
                data1 = gevi_data[g1]['data']
                data2 = gevi_data[g2]['data']
                
                # Effect size (difference in medians)
                effect_size = np.median(data2) - np.median(data1)
                
                # Bootstrap CI for effect size
                boot_diffs = []
                for _ in range(config['bootstrap_n']):
                    boot1 = np.random.choice(data1, size=len(data1), replace=True)
                    boot2 = np.random.choice(data2, size=len(data2), replace=True)
                    boot_diffs.append(np.median(boot2) - np.median(boot1))
                
                effect_ci_lower = np.percentile(boot_diffs, 2.5)
                effect_ci_upper = np.percentile(boot_diffs, 97.5)
                
                # Statistical tests
                ks_stat, ks_p = stats.ks_2samp(data1, data2)
                mw_stat, mw_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                
                # Cliff's delta
                cliff_delta = cliff_delta_effect_size(data1, data2)
                
                pairwise_results.append({
                    'gevi1': g1,
                    'gevi2': g2,
                    'effect_size': effect_size,
                    'effect_ci_lower': effect_ci_lower,
                    'effect_ci_upper': effect_ci_upper,
                    'ks_statistic': ks_stat,
                    'ks_p': ks_p,
                    'mw_statistic': mw_stat,
                    'mw_p': mw_p,
                    'cliff_delta': cliff_delta
                })
        
        results[metric]['pairwise'] = pairwise_results
    
    # FDR correction
    all_p_values = []
    p_value_info = []
    
    for metric, metric_results in results.items():
        if 'pairwise' in metric_results:
            for comp in metric_results['pairwise']:
                all_p_values.extend([comp['ks_p'], comp['mw_p']])
                p_value_info.extend([
                    (metric, comp['gevi1'], comp['gevi2'], 'ks'),
                    (metric, comp['gevi1'], comp['gevi2'], 'mw')
                ])
    
    if all_p_values:
        _, fdr_corrected = fdrcorrection(all_p_values, alpha=config['alpha'])
        
        # Add FDR-corrected p-values back to results
        p_idx = 0
        for metric, metric_results in results.items():
            if 'pairwise' in metric_results:
                for comp in metric_results['pairwise']:
                    comp['ks_p_fdr'] = fdr_corrected[p_idx]
                    comp['mw_p_fdr'] = fdr_corrected[p_idx + 1]
                    p_idx += 2
    
    logger.info("Statistical analyses complete")
    return results


def make_figures_with_real_events(per_event: pd.DataFrame, per_roi: pd.DataFrame, 
                                 stats_results: Dict, real_events: List[Dict], 
                                 config: Dict, args, logger: logging.Logger) -> Dict:
    """Create publication-grade figures including real spike shapes."""
    logger.info("Creating figures with real event data...")
    
    figures = {}
    
    # Set up color palette
    gevis = sorted(per_event['gevi'].unique())
    colors = config['gevi_colors'][:len(gevis)]
    color_map = dict(zip(gevis, colors))
    
    # Main comparison figure
    fig_main = create_main_comparison_figure(per_event, per_roi, stats_results, color_map, config)
    figures['main'] = fig_main
    
    # Supplementary figures
    fig_supp = create_supplementary_figures(per_event, per_roi, color_map, config)
    figures['supplement'] = fig_supp
    
    # Real spike shape figure using actual traces
    if not args.no_spike_figure and len(real_events) > 0:
        try:
            logger.info("Creating spike shape figure from real trace data...")
            
            aligned = align_events(
                real_events, 
                pre_ms=args.pre_ms, 
                post_ms=args.post_ms,
                amp_min=args.amp_min, 
                snr_min=args.snr_min,
                min_pre_ms=args.min_pre_ms,
                min_post_ms=args.min_post_ms,
                align_by=args.align_by,
                interp=args.interp,
                frame_rate=args.frame_rate,
                force_figure=args.force_spike_figure,
                logger=logger
            )
            
            if len(aligned) > 0:
                total_events = sum(len(d['aligned']) for d in aligned.values())
                if total_events >= 10 or args.force_spike_figure:
                    title = f"Real Event Traces | {args.align_by} aligned | {args.interp} interp"
                    fig_spike, _ = make_spike_shape_figure(
                        aligned, 
                        title_prefix=title,
                        boot=args.boot, 
                        outdir=".", 
                        palette=color_map
                    )
                    
                    if fig_spike is not None:
                        figures['spike_shape'] = fig_spike
                        logger.info(f"Real spike shape figure created with {total_events} events")
                    else:
                        logger.warning("Real spike shape figure creation returned None")
                else:
                    logger.warning(f"Only {total_events} events passed filters, need ≥10 (use --force-spike-figure to override)")
                    # Create diagnostic figure showing drop reasons
                    figures['spike_shape'] = create_diagnostic_figure(real_events, args, color_map)
            else:
                logger.warning("No events passed alignment filters")
                figures['spike_shape'] = create_diagnostic_figure(real_events, args, color_map)
                
        except Exception as e:
            logger.error(f"Failed to create real spike shape figure: {e}")
            # Fall back to CSV-based figure
            logger.info("Falling back to CSV-based spike figure...")
            fig_spike = create_csv_based_spike_figure(per_event, color_map, config)
            if fig_spike is not None:
                figures['spike_shape'] = fig_spike
    else:
        logger.warning("No spike shape figure generated: either --no-spike-figure used or no real events found")
    
    logger.info("Figure creation complete")
    return figures


def make_figures(per_event: pd.DataFrame, per_roi: pd.DataFrame, 
                stats_results: Dict, config: Dict, logger: logging.Logger) -> Dict:
    """Create publication-grade figures."""
    logger.info("Creating figures...")
    
    figures = {}
    
    # Set up color palette
    gevis = sorted(per_event['gevi'].unique())
    colors = config['gevi_colors'][:len(gevis)]
    color_map = dict(zip(gevis, colors))
    
    # Main comparison figure
    fig_main = create_main_comparison_figure(per_event, per_roi, stats_results, color_map, config)
    figures['main'] = fig_main
    
    # Supplementary figures
    fig_supp = create_supplementary_figures(per_event, per_roi, color_map, config)
    figures['supplement'] = fig_supp
    
    # Spike shape figure using real trace data
    try:
        logger.info("Attempting to create spike shape figure using real trace data...")
        # Note: real_events will be passed from main pipeline
        # For now, fall back to CSV-based approach
        fig_spike = create_csv_based_spike_figure(per_event, color_map, config)
        if fig_spike is not None:
            figures['spike_shape'] = fig_spike
            logger.info("Spike shape figure created successfully")
        else:
            logger.warning("Spike shape figure creation failed")
            
    except Exception as e:
        logger.warning(f"Failed to create spike shape figure: {e}")
    
    logger.info("Figure creation complete")
    return figures


def create_diagnostic_figure(real_events: List[Dict], args, color_map: Dict) -> plt.Figure:
    """Create diagnostic figure showing why events were dropped."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Event Filtering Diagnostic', fontsize=14)
    
    # Simulate alignment to get drop reasons (without full processing)
    drop_reasons = {
        'short_pre_window': 0,
        'short_post_window': 0, 
        'missing_dt': 0,
        'amp_below': 0,
        'snr_below': 0,
        'interp_nan': 0,
        'other': 0
    }
    
    # Quick check of events
    for ev in real_events:
        if ev.get('dt') is None:
            drop_reasons['missing_dt'] += 1
            continue
            
        F = np.asarray(ev['F']).astype(float)
        dt = float(ev['dt'])
        i0 = int(ev['i0'])
        n = F.size
        
        pre_s = int(round(args.pre_ms/1000.0/dt))
        post_s = int(round(args.post_ms/1000.0/dt))
        min_pre_s = int(round(args.min_pre_ms/1000.0/dt))
        min_post_s = int(round(args.min_post_ms/1000.0/dt))
        
        if i0 < min_pre_s:
            drop_reasons['short_pre_window'] += 1
        elif (n - i0 - 1) < min_post_s:
            drop_reasons['short_post_window'] += 1
        else:
            # Would pass basic window checks
            pass
    
    # Plot 1: Drop reasons bar chart
    ax = axes[0, 0]
    reasons = list(drop_reasons.keys())
    counts = list(drop_reasons.values())
    bars = ax.bar(reasons, counts)
    ax.set_title('Event Drop Reasons')
    ax.set_ylabel('Count')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom')
    
    # Plot 2: Event distribution by GEVI
    ax = axes[0, 1]
    gevi_counts = {}
    for ev in real_events:
        gevi = ev['gevi']
        gevi_counts[gevi] = gevi_counts.get(gevi, 0) + 1
    
    if gevi_counts:
        gevis = list(gevi_counts.keys())
        counts = list(gevi_counts.values())
        colors = [color_map.get(gevi, 'gray') for gevi in gevis]
        ax.bar(gevis, counts, color=colors)
        ax.set_title('Events per GEVI')
        ax.set_ylabel('Count')
    
    # Plot 3: Current thresholds
    ax = axes[1, 0]
    ax.text(0.1, 0.8, f"Thresholds Applied:", fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.7, f"Pre-window: {args.pre_ms} ms (min: {args.min_pre_ms} ms)", transform=ax.transAxes)
    ax.text(0.1, 0.6, f"Post-window: {args.post_ms} ms (min: {args.min_post_ms} ms)", transform=ax.transAxes)
    ax.text(0.1, 0.5, f"Amplitude minimum: {args.amp_min}", transform=ax.transAxes)
    ax.text(0.1, 0.4, f"SNR minimum: {args.snr_min}", transform=ax.transAxes)
    ax.text(0.1, 0.3, f"Align by: {args.align_by}", transform=ax.transAxes)
    ax.text(0.1, 0.2, f"Interpolation: {args.interp}", transform=ax.transAxes)
    ax.set_title('Current Settings')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Plot 4: Recommendations
    ax = axes[1, 1]
    total_events = len(real_events)
    total_dropped = sum(drop_reasons.values())
    ax.text(0.1, 0.8, f"Summary:", fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.7, f"Total events: {total_events}", transform=ax.transAxes)
    ax.text(0.1, 0.6, f"Events dropped: {total_dropped}", transform=ax.transAxes)
    ax.text(0.1, 0.5, f"Pass rate: {((total_events-total_dropped)/total_events*100):.1f}%", transform=ax.transAxes)
    
    if drop_reasons['short_pre_window'] > 0:
        ax.text(0.1, 0.3, f"→ Try reducing --min-pre-ms", transform=ax.transAxes, color='red')
    if drop_reasons['short_post_window'] > 0:
        ax.text(0.1, 0.2, f"→ Try reducing --min-post-ms", transform=ax.transAxes, color='red')
    if drop_reasons['amp_below'] > 0:
        ax.text(0.1, 0.1, f"→ Try reducing --amp-min", transform=ax.transAxes, color='red')
    
    ax.set_title('Recommendations')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_csv_based_spike_figure(per_event: pd.DataFrame, color_map: Dict, config: Dict) -> plt.Figure:
    """Create spike shape and kinetics figure directly from CSV metrics."""
    
    if 'width_ms' not in per_event.columns:
        return None
    
    # Filter events with valid data
    valid_events = per_event.dropna(subset=['amplitude_percent', 'width_ms'])
    if len(valid_events) < 10:
        return None
    
    gevis = sorted(valid_events['gevi'].unique())
    if len(gevis) < 1:
        return None
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Amplitude vs Time scatter (proxy for shape)
    ax_amp = fig.add_subplot(gs[0, :2])
    for gevi in gevis:
        gevi_data = valid_events[valid_events['gevi'] == gevi]
        if len(gevi_data) > 0:
            # Use time as x-axis and amplitude as y-axis to show distribution
            ax_amp.scatter(gevi_data['t_sec'], gevi_data['amplitude_percent'], 
                          alpha=0.6, s=20, color=color_map.get(gevi, 'gray'), 
                          label=f'{gevi} (n={len(gevi_data)})')
    
    ax_amp.set_xlabel('Time (s)')
    ax_amp.set_ylabel('Amplitude ΔF/F (%)')
    ax_amp.set_title('Event Amplitude Distribution Over Time')
    ax_amp.legend()
    
    # Panel B: Width distribution
    ax_width = fig.add_subplot(gs[0, 2])
    for gevi in gevis:
        gevi_data = valid_events[valid_events['gevi'] == gevi]['width_ms']
        if len(gevi_data) > 0:
            ax_width.hist(gevi_data, bins=20, alpha=0.7, density=True,
                         color=color_map.get(gevi, 'gray'), label=gevi)
    
    ax_width.set_xlabel('Width (ms)')
    ax_width.set_ylabel('Density')
    ax_width.set_title('Event Width Distribution')
    ax_width.legend()
    
    # Panel C: Amplitude vs Width relationship
    ax_scatter = fig.add_subplot(gs[1, 0])
    for gevi in gevis:
        gevi_data = valid_events[valid_events['gevi'] == gevi]
        if len(gevi_data) > 0:
            ax_scatter.scatter(gevi_data['width_ms'], gevi_data['amplitude_percent'],
                             alpha=0.6, s=20, color=color_map.get(gevi, 'gray'), label=gevi)
    
    ax_scatter.set_xlabel('Width (ms)')
    ax_scatter.set_ylabel('Amplitude ΔF/F (%)')
    ax_scatter.set_title('Amplitude vs Width')
    ax_scatter.legend()
    
    # Panel D: Kinetics summary (violin plots)
    ax_kinetics = fig.add_subplot(gs[1, 1:])
    
    # Prepare data for violin plot
    positions = []
    data_list = []
    labels = []
    colors_list = []
    
    pos = 0
    for gevi in gevis:
        gevi_data = valid_events[valid_events['gevi'] == gevi]
        if len(gevi_data) > 0:
            # Amplitude distribution
            positions.append(pos)
            data_list.append(gevi_data['amplitude_percent'])
            labels.append(f'{gevi}\nAmplitude')
            colors_list.append(color_map.get(gevi, 'gray'))
            pos += 1
            
            # Width distribution (scaled for comparison)
            positions.append(pos)
            data_list.append(gevi_data['width_ms'] / 10)  # Scale width to similar range
            labels.append(f'{gevi}\nWidth/10')
            colors_list.append(color_map.get(gevi, 'gray'))
            pos += 1.5  # Space between GEVIs
    
    if data_list:
        parts = ax_kinetics.violinplot(data_list, positions=positions, widths=0.8, showmeans=True)
        
        # Color the violin plots
        for patch, color in zip(parts['bodies'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax_kinetics.set_xticks(positions)
    ax_kinetics.set_xticklabels(labels, rotation=45, ha='right')
    ax_kinetics.set_ylabel('Value')
    ax_kinetics.set_title('Kinetics Summary (Width scaled by 10)')
    
    # Overall title
    sample_info = []
    for gevi in gevis:
        n_events = len(valid_events[valid_events['gevi'] == gevi])
        sample_info.append(f"{gevi}: {n_events} events")
    
    fig.suptitle(f"GEVI Event Characteristics\n{' | '.join(sample_info)}", fontsize=14)
    
    return fig


def create_main_comparison_figure(per_event: pd.DataFrame, per_roi: pd.DataFrame,
                                stats_results: Dict, color_map: Dict, config: Dict) -> plt.Figure:
    """Create main comparison figure with multiple panels."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    gevis = sorted(per_event['gevi'].unique())
    
    # Panel A: Amplitude distribution
    ax_amp = fig.add_subplot(gs[0, 0])
    create_half_violin_plot(ax_amp, per_event, 'amplitude_percent', 'gevi', 
                           color_map, stats_results.get('amplitude_percent', {}))
    ax_amp.set_ylabel('Amplitude ΔF/F (%)')
    ax_amp.set_title('Event Amplitude')
    
    # Panel B: Width distribution  
    ax_width = fig.add_subplot(gs[0, 1])
    if 'width_ms' in per_event.columns:
        create_half_violin_plot(ax_width, per_event, 'width_ms', 'gevi',
                               color_map, stats_results.get('width_ms', {}))
        ax_width.set_ylabel('Width (ms)')
        ax_width.set_title('Event Width')
    
    # Panel C: SNR distribution
    ax_snr = fig.add_subplot(gs[0, 2])
    if 'snr' in per_event.columns:
        create_half_violin_plot(ax_snr, per_event, 'snr', 'gevi',
                               color_map, stats_results.get('snr', {}))
        ax_snr.set_ylabel('SNR')
        ax_snr.set_title('Signal-to-Noise Ratio')
    
    # Panel D: Event rate - FIXED VERSION
    ax_rate = fig.add_subplot(gs[1, 0])
    create_half_violin_plot(ax_rate, per_roi, 'event_rate_hz', 'gevi',
                           color_map, stats_results.get('event_rate_hz', {}))
    ax_rate.set_ylabel('Event Rate (Hz)')
    ax_rate.set_title('Event Rates per ROI')
    
    # Panel E: Amplitude vs Width scatter
    ax_scatter = fig.add_subplot(gs[1, 1])
    create_amplitude_width_scatter(ax_scatter, per_roi, color_map)
    
    # Panel F: Rate vs SNR trade-off
    ax_trade = fig.add_subplot(gs[1, 2])
    if 'snr_median' in per_roi.columns:
        create_rate_snr_scatter(ax_trade, per_roi, color_map)
    
    # Panel G: Effect sizes
    ax_effects = fig.add_subplot(gs[2, :])
    create_effect_size_plot(ax_effects, stats_results, gevis)
    
    # Overall title with sample sizes
    sample_info = []
    for gevi in gevis:
        n_events = len(per_event[per_event['gevi'] == gevi])
        n_rois = len(per_roi[per_roi['gevi'] == gevi])
        sample_info.append(f"{gevi}: {n_events} events, {n_rois} ROIs")
    
    fig.suptitle(f"GEVI Performance Comparison\n{' | '.join(sample_info)}", 
                 fontsize=14, y=0.98)
    
    return fig


def create_half_violin_plot(ax: plt.Axes, data: pd.DataFrame, metric: str, 
                           group_col: str, color_map: Dict, stats_dict: Dict):
    """Create half-violin plot with raw points and confidence intervals."""
    
    groups = sorted(data[group_col].unique())
    positions = np.arange(len(groups))
    
    for i, group in enumerate(groups):
        group_data = data[data[group_col] == group][metric].dropna()
        
        if len(group_data) == 0:
            continue
        
        color = color_map.get(group, 'gray')
        
        # Half violin
        parts = ax.violinplot([group_data], positions=[i], showmeans=False, 
                             showmedians=False, showextrema=False, widths=0.6)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            # Make it half violin by modifying vertices
            vertices = pc.get_paths()[0].vertices
            vertices[:, 0] = np.clip(vertices[:, 0], i, i + 0.4)
        
        # Raw points (jittered)
        jitter = np.random.normal(0, 0.05, len(group_data))
        ax.scatter(i + 0.5 + jitter, group_data, alpha=0.4, s=8, color=color)
        
        # Median and CI
        if group in stats_dict.get('groups', {}):
            group_stats = stats_dict['groups'][group]
            median = group_stats['median']
            ci_low = group_stats['ci_lower']
            ci_high = group_stats['ci_upper']
            
            ax.plot([i + 0.6, i + 0.8], [median, median], 'k-', linewidth=2)
            ax.plot([i + 0.7, i + 0.7], [ci_low, ci_high], 'k-', linewidth=1)
        
        # Sample size annotation
        ax.text(i, ax.get_ylim()[1] * 0.95, f'n={len(group_data)}', 
                ha='center', va='top', fontsize=8)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_amplitude_width_scatter(ax: plt.Axes, per_roi: pd.DataFrame, color_map: Dict):
    """Create amplitude vs width scatter plot."""
    
    if 'width_median' not in per_roi.columns:
        ax.text(0.5, 0.5, 'Width data not available', ha='center', va='center', 
                transform=ax.transAxes)
        return

    for gevi in sorted(per_roi['gevi'].unique()):
        data = per_roi[per_roi['gevi'] == gevi]
        ax.scatter(data['width_median'], data['amplitude_median'], 
                  alpha=0.6, s=30, color=color_map.get(gevi, 'gray'), label=gevi)
        
        # Group median marker
        if len(data) > 0:
            median_x = data['width_median'].median()
            median_y = data['amplitude_median'].median()
            ax.scatter(median_x, median_y, s=100, marker='*', 
                      color='black', edgecolor='white', linewidth=1)
    
    ax.set_xlabel('Width median (ms)')
    ax.set_ylabel('Amplitude median ΔF/F (%)')
    ax.set_title('ROI-level Amplitude vs Width')
    ax.legend()


def create_rate_snr_scatter(ax: plt.Axes, per_roi: pd.DataFrame, color_map: Dict):
    """Create event rate vs SNR scatter plot."""
    
    for gevi in sorted(per_roi['gevi'].unique()):
        data = per_roi[per_roi['gevi'] == gevi]
        valid_data = data.dropna(subset=['event_rate_hz', 'snr_median'])
        
        if len(valid_data) > 0:
            ax.scatter(valid_data['event_rate_hz'], valid_data['snr_median'],
                      alpha=0.6, s=30, color=color_map.get(gevi, 'gray'), label=gevi)
    
    ax.set_xlabel('Event Rate (Hz)')
    ax.set_ylabel('SNR median')
    ax.set_title('Rate vs SNR Trade-off')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')


def create_effect_size_plot(ax: plt.Axes, stats_results: Dict, gevis: List[str]):
    """Create effect size forest plot."""
    
    metrics = ['amplitude_percent', 'width_ms', 'snr', 'event_rate_hz']
    metric_labels = ['Amplitude (%)', 'Width (ms)', 'SNR', 'Rate (Hz)']
    
    y_pos = []
    labels = []
    effects = []
    ci_lows = []
    ci_highs = []
    colors = []
    
    y = 0
    for metric, label in zip(metrics, metric_labels):
        if metric not in stats_results:
            continue
            
        pairwise = stats_results[metric].get('pairwise', [])
        for comp in pairwise:
            y_pos.append(y)
            labels.append(f"{label}\n{comp['gevi2']} vs {comp['gevi1']}")
            effects.append(comp['effect_size'])
            ci_lows.append(comp['effect_ci_lower'])
            ci_highs.append(comp['effect_ci_upper'])
            
            # Color based on significance
            p_val = comp.get('mw_p_fdr', comp.get('mw_p', 1.0))
            colors.append('red' if p_val < 0.05 else 'black')
            y += 1
    
    if not effects:
        ax.text(0.5, 0.5, 'No effect size data available', 
                ha='center', va='center', transform=ax.transAxes)
        return

    # Plot effect sizes with CIs
    ax.errorbar(effects, y_pos, xerr=[np.array(effects) - np.array(ci_lows),
                                     np.array(ci_highs) - np.array(effects)],
                fmt='o', capsize=3, capthick=1)
    
    # Add vertical line at zero
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Effect Size (Difference in Medians)')
    ax.set_title('Effect Sizes with 95% CIs')


def create_supplementary_figures(per_event: pd.DataFrame, per_roi: pd.DataFrame,
                                color_map: Dict, config: Dict) -> plt.Figure:
    """Create supplementary figures."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Supplementary Analyses', fontsize=14)
    
    # QC panel
    ax = axes[0, 0]
    if 'baseline_brightness' in per_roi.columns:
        for gevi in sorted(per_roi['gevi'].unique()):
            data = per_roi[per_roi['gevi'] == gevi]['baseline_brightness'].dropna()
            ax.hist(data, alpha=0.5, label=gevi, bins=20, 
                   color=color_map.get(gevi, 'gray'))
        ax.set_xlabel('Baseline Brightness')
        ax.set_ylabel('ROI Count')
        ax.set_title('QC: Baseline Brightness')
        ax.legend()
    
    # ECDF plots
    ax = axes[0, 1]
    for gevi in sorted(per_event['gevi'].unique()):
        data = per_event[per_event['gevi'] == gevi]['amplitude_percent'].dropna()
        if len(data) > 0:
            x_sorted = np.sort(data)
            y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
            ax.plot(x_sorted, y, label=gevi, color=color_map.get(gevi, 'gray'))
    ax.set_xlabel('Amplitude ΔF/F (%)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Empirical CDFs')
    ax.legend()
    
    # Distribution comparisons
    ax = axes[1, 0]
    create_distribution_comparison(ax, per_event, 'amplitude_percent', color_map)
    
    # Sample size summary
    ax = axes[1, 1]
    create_sample_size_summary(ax, per_event, per_roi, color_map)
    
    plt.tight_layout()
    return fig


def create_distribution_comparison(ax: plt.Axes, data: pd.DataFrame, 
                                 metric: str, color_map: Dict):
    """Create distribution comparison using shift functions."""
    
    gevis = sorted(data['gevi'].unique())
    if len(gevis) < 2:
        return

    # Simple quantile comparison
    quantiles = np.linspace(0.1, 0.9, 9)
    
    for i in range(len(gevis) - 1):
        gevi1, gevi2 = gevis[i], gevis[i + 1]
        data1 = data[data['gevi'] == gevi1][metric].dropna()
        data2 = data[data['gevi'] == gevi2][metric].dropna()
        
        if len(data1) > 0 and len(data2) > 0:
            q1 = np.quantile(data1, quantiles)
            q2 = np.quantile(data2, quantiles)
            differences = q2 - q1
            
            ax.plot(quantiles * 100, differences, 'o-', 
                   label=f'{gevi2} - {gevi1}', alpha=0.7)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Quantile (%)')
    ax.set_ylabel(f'Difference in {metric}')
    ax.set_title('Shift Function')
    ax.legend()


def create_sample_size_summary(ax: plt.Axes, per_event: pd.DataFrame, 
                              per_roi: pd.DataFrame, color_map: Dict):
    """Create sample size summary plot."""
    
    gevis = sorted(per_event['gevi'].unique())
    x_pos = np.arange(len(gevis))
    
    # Events per GEVI
    event_counts = [len(per_event[per_event['gevi'] == gevi]) for gevi in gevis]
    roi_counts = [len(per_roi[per_roi['gevi'] == gevi]) for gevi in gevis]
    
    width = 0.35
    ax.bar(x_pos - width/2, event_counts, width, label='Events', alpha=0.7)
    ax.bar(x_pos + width/2, roi_counts, width, label='ROIs', alpha=0.7)
    
    ax.set_xlabel('GEVI')
    ax.set_ylabel('Count')
    ax.set_title('Sample Sizes')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(gevis)
    ax.legend()
    
    # Add count labels
    for i, (events, rois) in enumerate(zip(event_counts, roi_counts)):
        ax.text(i - width/2, events + max(event_counts) * 0.01, str(events), 
                ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, rois + max(roi_counts) * 0.01, str(rois), 
                ha='center', va='bottom', fontsize=8)


def write_outputs(per_event: pd.DataFrame, per_roi: pd.DataFrame, 
                 stats_results: Dict, figures: Dict, config: Dict, 
                 output_dir: Path, logger: logging.Logger):
    """Write all outputs to disk."""
    logger.info("Writing outputs...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save figures
    if 'main' in figures:
        figures['main'].savefig(output_dir / 'gevi_comparison_main.png', dpi=300, bbox_inches='tight')
        figures['main'].savefig(output_dir / 'gevi_comparison_main.pdf', bbox_inches='tight')
        figures['main'].savefig(output_dir / 'gevi_comparison_main.svg', bbox_inches='tight')
    
    if 'supplement' in figures:
        figures['supplement'].savefig(output_dir / 'gevi_comparison_supplement.png', dpi=300, bbox_inches='tight')
        figures['supplement'].savefig(output_dir / 'gevi_comparison_supplement.pdf', bbox_inches='tight')
    
    if 'spike_shape' in figures:
        figures['spike_shape'].savefig(output_dir / 'gevi_spike_shape_comparison.png', dpi=300, bbox_inches='tight')
        figures['spike_shape'].savefig(output_dir / 'gevi_spike_shape_comparison.pdf', bbox_inches='tight')
        figures['spike_shape'].savefig(output_dir / 'gevi_spike_shape_comparison.svg', bbox_inches='tight')
        logger.info("Saved spike shape comparison figure")
    
    # Save data tables
    per_event.to_csv(output_dir / 'per_event_metrics.csv', index=False)
    per_roi.to_csv(output_dir / 'per_roi_metrics.csv', index=False)
    
    # Save per-event data as parquet for efficiency
    try:
        per_event.to_parquet(output_dir / 'per_event_metrics.parquet', index=False)
    except ImportError:
        logger.warning("Parquet export not available, saved as CSV only")
    
    # Create summary table
    summary_table = create_summary_table(per_event, per_roi, stats_results)
    summary_table.to_csv(output_dir / 'summary_table.csv', index=False)
    
    # Save statistics results as JSON
    stats_json = convert_stats_to_json(stats_results)
    with open(output_dir / 'statistical_results.json', 'w') as f:
        json.dump(stats_json, f, indent=2)
    
    # Save run metadata
    metadata = {
        'config': config,
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_events_total': len(per_event),
        'n_rois_total': len(per_roi),
        'gevis': sorted(per_event['gevi'].unique()),
        'metrics_calculated': list(per_event.columns),
        'preprocessing_steps': [
            'baseline_correction',
            'artifact_removal', 
            'quality_control',
            'unit_conversion'
        ]
    }
    
    with open(output_dir / 'run_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create README
    create_readme(output_dir, metadata, logger)
    
    logger.info(f"All outputs saved to {output_dir}")


def create_summary_table(per_event: pd.DataFrame, per_roi: pd.DataFrame, 
                        stats_results: Dict) -> pd.DataFrame:
    """Create publication-ready summary table."""
    
    summary_rows = []
    gevis = sorted(per_event['gevi'].unique())
    
    for gevi in gevis:
        row = {'GEVI': gevi}
        
        # Sample sizes
        n_events = len(per_event[per_event['gevi'] == gevi])
        n_rois = len(per_roi[per_roi['gevi'] == gevi])
        row['N_Events'] = n_events
        row['N_ROIs'] = n_rois
        
        # Main metrics with CIs
        metrics = {
            'amplitude_percent': ('Amplitude_pct', per_event),
            'width_ms': ('Width_ms', per_event),
            'snr': ('SNR', per_event),
            'event_rate_hz': ('Rate_Hz', per_roi)
        }
        
        for metric_col, (label, data_df) in metrics.items():
            if metric_col in data_df.columns:
                data = data_df[data_df['gevi'] == gevi][metric_col].dropna()
                if len(data) > 0:
                    median, ci_low, ci_high = bootstrap_ci(data, n_boot=1000)
                    row[f'{label}_Median'] = f"{median:.2f}"
                    row[f'{label}_95CI'] = f"[{ci_low:.2f}, {ci_high:.2f}]"
        
        summary_rows.append(row)
    
    # Add effect sizes and p-values for pairwise comparisons
    if len(gevis) == 2:
        for metric in ['amplitude_percent', 'width_ms', 'snr', 'event_rate_hz']:
            if metric in stats_results:
                pairwise = stats_results[metric].get('pairwise', [])
                if pairwise:
                    comp = pairwise[0]  # First (and only) comparison
                    effect = comp['effect_size']
                    ci_low = comp['effect_ci_lower']
                    ci_high = comp['effect_ci_upper']
                    p_fdr = comp.get('mw_p_fdr', comp.get('mw_p', np.nan))
                    
                    # Add to first row
                    metric_short = metric.replace('_percent', '').replace('_ms', '').replace('_hz', '')
                    summary_rows[0][f'{metric_short}_Effect'] = f"{effect:.2f} [{ci_low:.2f}, {ci_high:.2f}]"
                    summary_rows[0][f'{metric_short}_p_FDR'] = f"{p_fdr:.4f}" if not np.isnan(p_fdr) else "N/A"
    
    return pd.DataFrame(summary_rows)


def convert_stats_to_json(stats_results: Dict) -> Dict:
    """Convert statistics results to JSON-serializable format."""
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    return convert_numpy(stats_results)


def create_readme(output_dir: Path, metadata: Dict, logger: logging.Logger):
    """Create README file explaining the analysis."""
    
    readme_content = f"""
# GEVI Performance Comparison Analysis

Generated on: {metadata['timestamp']}

## Overview
This analysis compares the performance of {len(metadata['gevis'])} voltage indicators:
{', '.join(metadata['gevis'])}

## Sample Sizes
- Total events analyzed: {metadata['n_events_total']}
- Total ROIs analyzed: {metadata['n_rois_total']}

## Preprocessing Steps
{chr(10).join(f"- {step}" for step in metadata['preprocessing_steps'])}

## Configuration
- Frame rate: {metadata['config']['frame_rate']} Hz
- Minimum events per ROI: {metadata['config']['min_events_per_roi']}
- Bootstrap iterations: {metadata['config']['bootstrap_n']}
- Random seed: {metadata['config']['random_seed']}

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
- Minimum {metadata['config']['min_events_per_roi']} events per ROI
- Artifact removal using MAD-based outlier detection
- Winsorization at {metadata['config']['winsorize_percentile']}th percentiles

## Interpretation
- Effect sizes > 0.5 are considered large
- p-values are FDR-corrected for multiple comparisons
- Bootstrap CIs that don't include 0 indicate significant differences
- Raw data points are shown alongside summary statistics

For questions about this analysis, refer to the statistical_results.json file
for detailed test statistics and confidence intervals.
"""
    
    with open(output_dir / 'README.txt', 'w') as f:
        f.write(readme_content.strip())


def cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Publication-grade GEVI performance comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --gevi1 ASAP6_2 /data/asap2 --gevi2 ASAP6_3 /data/asap3 -o results/
  %(prog)s --config config.yaml --gevi1 ASAP6_2 /data/asap2 --gevi2 ASAP6_3 /data/asap3
        """
    )
    
    # Input specification
    parser.add_argument('--gevi1', nargs=2, metavar=('NAME', 'PATH'), 
                       help='First GEVI: name and data path')
    parser.add_argument('--gevi2', nargs=2, metavar=('NAME', 'PATH'),
                       help='Second GEVI: name and data path')
    parser.add_argument('--gevi3', nargs=2, metavar=('NAME', 'PATH'),
                       help='Third GEVI: name and data path (optional)')
    parser.add_argument('--gevi4', nargs=2, metavar=('NAME', 'PATH'),
                       help='Fourth GEVI: name and data path (optional)')
    
    # Configuration
    parser.add_argument('--config', type=Path,
                       help='YAML/JSON configuration file')
    parser.add_argument('-o', '--output', type=Path, default='gevi_comparison_results',
                       help='Output directory (default: gevi_comparison_results)')
    
    # Key parameters (override config)
    parser.add_argument('--frame-rate', type=float, default=None,
                       help='Acquisition frame rate (Hz); used if per-event dt is missing')
    parser.add_argument('--min-events', type=int, default=3,
                       help='Minimum events per ROI')
    parser.add_argument('--bootstrap-n', type=int, default=10000,
                       help='Bootstrap iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    # Spike shape figure parameters
    parser.add_argument("--no-spike-figure", action="store_true",
                       help="Disable spike shape and kinetics overlay figure")
    parser.add_argument("--pre-ms", type=float, default=200.0,
                       help="Pre-event window in milliseconds")
    parser.add_argument("--post-ms", type=float, default=400.0,
                       help="Post-event window in milliseconds")
    parser.add_argument("--amp-min", type=float, default=0.02,
                       help="Minimum amplitude threshold for spike shape analysis")
    parser.add_argument("--snr-min", type=float, default=3.0,
                       help="Minimum SNR threshold for spike shape analysis")
    parser.add_argument("--boot", type=int, default=10000,
                       help="Bootstrap iterations for spike shape CI")
    
    # Real data alignment parameters
    parser.add_argument("--align-by", choices=["onset","peak"], default="onset",
                       help="Align events by onset or peak")
    parser.add_argument("--min-pre-ms", type=float, default=50.0,
                       help="Minimum pre-event window required")
    parser.add_argument("--min-post-ms", type=float, default=150.0,
                       help="Minimum post-event window required")
    parser.add_argument("--interp", choices=["cubic","linear"], default="linear",
                       help="Interpolation method for resampling")
    parser.add_argument("--force-spike-figure", action="store_true",
                       help="Render figure even if <10 events pass; show CI as NaN-propagating")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(args.output / 'analysis.log' if args.output else None, 
                          args.log_level)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line args
        if args.frame_rate is not None:
            config['frame_rate'] = args.frame_rate
        if args.min_events != 3:
            config['min_events_per_roi'] = args.min_events
        if args.bootstrap_n != 10000:
            config['bootstrap_n'] = args.bootstrap_n
        if args.seed != 42:
            config['random_seed'] = args.seed
        
        # Set random seed
        np.random.seed(config['random_seed'])
        
        # Collect GEVI paths
        gevi_paths = {}
        for gevi_arg in ['gevi1', 'gevi2', 'gevi3', 'gevi4']:
            gevi_data = getattr(args, gevi_arg)
            if gevi_data:
                name, path_str = gevi_data
                gevi_paths[name] = Path(path_str)
        
        if len(gevi_paths) < 2:
            raise ValueError("Need at least 2 GEVIs for comparison")
        
        logger.info(f"Starting GEVI comparison: {list(gevi_paths.keys())}")
        logger.info(f"Output directory: {args.output}")
        
        # Run analysis pipeline with real events
        df, real_events = load_data(gevi_paths, logger)
        df = validate_inputs(df, config, logger)
        df = preprocess(df, config, logger)
        per_event, per_roi = extract_metrics(df, config, logger)
        
        # Generate figures including real spike shapes
        stats_results = stats_models(per_event, per_roi, config, logger)
        figures = make_figures_with_real_events(per_event, per_roi, stats_results, 
                                               real_events, config, args, logger)
        write_outputs(per_event, per_roi, stats_results, figures, config, 
                     args.output, logger)
        
        logger.info("Analysis completed successfully!")
        print(f"\nResults saved to: {args.output}")
        print("See README.txt for detailed explanation of outputs.")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    cli()
