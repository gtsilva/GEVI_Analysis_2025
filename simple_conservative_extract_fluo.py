#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conservative fluorescence spike extraction with comprehensive data loading.

This script supports the same data sources as extract_fluorescence_cellpose.py:
  1) RigDataV2.CameraData (frames binary + camera parameters)
  2) TIFF stack on disk
  3) Single trace files (.npy, .csv, etc.)

Method:
  1) Detrend using a rolling percentile baseline (same behavior as the main pipeline).
  2) Optional SOS Butterworth high/low/band-pass filtering (stable filtfilt).
  3) Auto-select a "silent" window (low outlier count then lowest std).
  4) Let σ = std of that silent window. Threshold = 5*σ.
  5) Detect peaks above threshold with width ≥ 1 ms at half-prominence.
  6) Save per-ROI event CSVs and comprehensive summary plots.

CLI:
  python simple_conservative_extract_fluo.py --source rig --data-dir /path/to/experiment
  python simple_conservative_extract_fluo.py --source tiff --tif /path/to/stack.tif
  python simple_conservative_extract_fluo.py --input trace.npy --fs 1000 --out_csv events.csv --out_plot diag.png
"""

from __future__ import annotations

# Early environment controls to prevent MPS/BLAS segfaults
import os
# tame thread explosions that trigger segfaults with MPS/BLAS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Allow CPU fallback for kernels unsupported by MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# block SAM codepaths and stale model hints
os.environ["CELLPOSE_DISABLE_SAM"] = "1"
for k in ("CELLPOSE_MODEL", "CELLPOSE_MODELS_PATH"):
    os.environ.pop(k, None)

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, Optional, Tuple, Dict
import inspect
import platform

import numpy as np
import pandas as pd
import sys
try:
    import torch
except Exception as e:
    print("Torch failed to import. Likely NumPy/ABI mismatch. See note at bottom.", file=sys.stderr)
    raise

try:
    from scipy import signal
except Exception as e:
    raise SystemExit("SciPy is required: pip install scipy") from e

# Optional imports handled at runtime
try:
    import tifffile as tiff
except Exception:
    tiff = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Cellpose
try:
    from cellpose import models, io as cp_io, utils as cp_utils
except Exception:
    raise SystemExit("Cellpose is required: pip install cellpose")

# Limit math library threads further in PyTorch
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Device picker with macOS torch 2.2.* guard
def pick_device(pref: str = "auto"):
    is_macos = (platform.system() == "Darwin")
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "mps":
        return torch.device("mps") if mps_ok else torch.device("cpu")
    # auto
    if is_macos and torch.__version__.startswith("2.2."):
        return torch.device("cpu")    # avoid known 2.2.x MPS issues
    return torch.device("mps") if mps_ok else torch.device("cpu")

# Centralized device/dtype picker
def _pick_device_and_dtype(pref=None, force_cpu=False, prefer_fp16=False):
    # choose device
    if force_cpu:
        dev = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    # choose dtype safely
    if prefer_fp16 and dev.type in ("cuda", "mps"):
        dt = torch.float16
    else:
        dt = torch.float32
    return dev, dt

# Skimage for morphology/registration
try:
    from skimage.measure import regionprops_table, label as sk_label, regionprops
    from skimage.morphology import remove_small_objects, remove_small_holes, dilation, disk, erosion
    from skimage.segmentation import watershed, find_boundaries, relabel_sequential, expand_labels, clear_border
    from skimage.feature import blob_log
    from scipy.ndimage import distance_transform_edt, maximum_filter, binary_dilation
    from scipy import ndimage as ndi
    # Additional imports for robust segmentation
    from skimage import filters, morphology, exposure, feature, segmentation, measure, util
    
    # Handle different versions of peak finding functions
    try:
        from skimage.feature import peak_local_maxima, peak_local_max
    except ImportError:
        try:
            from skimage.feature.peak import peak_local_maxima, peak_local_max
        except ImportError:
            # Fallback for very old versions
            from scipy.ndimage import maximum_filter
            def peak_local_maxima(image, min_distance=1, threshold_abs=None, labels=None):
                """Fallback implementation for peak_local_maxima."""
                if threshold_abs is None:
                    threshold_abs = 0.0
                
                # Apply threshold
                mask = image >= threshold_abs
                if labels is not None:
                    mask = mask & (labels > 0)
                
                # Find local maxima using maximum filter
                local_max = (image == maximum_filter(image, size=2*min_distance+1)) & mask
                
                # Return coordinates
                coords = np.argwhere(local_max)
                return coords
            
            def peak_local_max(image, min_distance=1, threshold_abs=None, labels=None):
                """Fallback implementation for peak_local_max."""
                return peak_local_maxima(image, min_distance, threshold_abs, labels)
            
except Exception:
    raise SystemExit("Scikit-image is required: pip install scikit-image")


@dataclass(frozen=True)
class Events:
    idx: np.ndarray           # peak sample indices
    t_s: np.ndarray           # peak times in seconds
    amp: np.ndarray           # peak amplitudes (processed units, e.g., dF/F)
    width_ms: np.ndarray      # widths at half-prominence (ms)
    polarity: np.ndarray      # 'pos' or 'neg'
    noise_sigma: float        # σ estimated from silent window
    threshold: float          # absolute threshold used


@dataclass
class DataStack:
    frames: np.ndarray  # shape (T, H, W), dtype float32 or uint16/uint8
    frame_rate: Optional[float] = None
    times: Optional[np.ndarray] = None  # shape (T,)

    @staticmethod
    def from_rig(data_dir: str, camera_name: str) -> "DataStack":
        """Load stack from RigDataV2.CameraData. Expects CameraData.frames with shape (H, W, T)."""
        try:
            import RigDataV2 as RigData  # local module provided by the user
        except Exception as e:
            raise RuntimeError("RigDataV2 import failed. Ensure RigDataV2.py is on PYTHONPATH.") from e

        cam = RigData.CameraData(name=camera_name,
                                 params_file=f"camera-parameters-{camera_name}.txt",
                                 bin_file=f"frames-{camera_name}.bin",
                                 data_dir=data_dir)
        frames = cam.frames  # likely (H, W, T)
        if frames.ndim != 3:
            raise ValueError(f"Expected 3D stack from RigDataV2, got shape {frames.shape}")

        # Move time to axis 0 for consistency
        if frames.shape[0] in (frames.shape[1], frames.shape[2]) or frames.shape[0] < 8:
            # Heuristic: RigData tends to be (H, W, T); put T first
            frames_t = np.moveaxis(frames, -1, 0)  # (T, H, W)
        else:
            frames_t = frames  # already T,H,W

        # Frame rate & times if available
        frame_rate = getattr(cam, "frameRate", None)
        frametimes = getattr(cam, "frametimes", None)
        times = None
        if frametimes is not None and np.ndim(frametimes) == 1 and len(frametimes) == frames_t.shape[0]:
            times = np.asarray(frametimes).astype(float)
        elif frame_rate is not None:
            times = np.arange(frames_t.shape[0]) / float(frame_rate)

        return DataStack(frames=frames_t, frame_rate=frame_rate, times=times)

    @staticmethod
    def from_tif(path: str) -> "DataStack":
        if tiff is None:
            raise RuntimeError("tifffile not available. Install tifffile to read TIFF stacks.")
        arr = tiff.imread(path)
        if arr.ndim != 3:
            raise ValueError(f"Only 3D stacks are supported, got shape {arr.shape}")
        # Assume either (T, H, W) or (H, W, T). Put time first.
        if arr.shape[0] in (arr.shape[1], arr.shape[2]) or arr.shape[0] < 8:
            arr = np.moveaxis(arr, -1, 0)
        times = None  # no timing in TIFF by default
        return DataStack(frames=arr, frame_rate=None, times=times)


# ---- Preprocessing kept minimal and consistent with existing pipeline ----

def detrend_percentile(x: np.ndarray, fs: float, win_s: float = 45.0, q: float = 10.0) -> np.ndarray:
    """Robust rolling-percentile detrend with reflect padding."""
    x = np.asarray(x, dtype=np.float64)
    win_samples = int(max(1, round(win_s * fs)))
    if win_samples >= x.size:
        baseline = np.percentile(x, q)
        return x - baseline
    pad = win_samples // 2
    xp = np.pad(x, pad, mode="reflect")
    out = np.empty_like(x)
    for i in range(x.size):
        seg = xp[i:i + win_samples]
        out[i] = x[i] - np.percentile(seg, q)
    return out


def make_bandpass_sos(fs: float,
                      low_hz: Optional[float] = 0.8,
                      high_hz: Optional[float] = None,
                      order: int = 3):
    """Stable Butterworth SOS. If only one cutoff is set, this becomes high- or low-pass."""
    fs = float(fs)
    nyq = 0.5 * fs
    if low_hz is None and high_hz is None:
        return None
    if low_hz is not None and high_hz is not None:
        low = max(0.1, float(low_hz)) / nyq
        high = min(0.45 * fs, float(high_hz)) / nyq
        if high <= low:
            low, high = 0.5 / nyq, min(120.0, 0.45 * fs) / nyq
        sos = signal.butter(order, [low, high], btype="band", output="sos")
    elif low_hz is not None:
        sos = signal.butter(order, max(0.1, float(low_hz)) / nyq, btype="highpass", output="sos")
    else:
        sos = signal.butter(order, min(0.45 * fs, float(high_hz)) / nyq, btype="lowpass", output="sos")
    return sos


def apply_sos(x: np.ndarray, sos) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if sos is None:
        return x
    n = x.size
    if n < 10:
        return x - np.median(x)
    padlen = min(30, n - 1)
    try:
        return signal.sosfiltfilt(sos, x, padlen=padlen)
    except Exception:
        return signal.sosfilt(sos, x)


def preprocess_trace(x: np.ndarray, fs: float,
                     detrend_win_s: float = 45.0, detrend_q: float = 10.0,
                     low_hz: Optional[float] = 0.8, high_hz: Optional[float] = None,
                     order: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, copy=False)
    d = detrend_percentile(x, fs, win_s=detrend_win_s, q=detrend_q)
    sos = make_bandpass_sos(fs, low_hz=low_hz, high_hz=high_hz, order=order)
    y = apply_sos(d, sos)
    return y


# ---- Noise estimation from an auto-selected silent window ----

def robust_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return 1.4826 * mad


def find_silent_window(y: np.ndarray, fs: float, win_s: float = 3.0) -> Tuple[int, int, float]:
    """
    Slide a window of win_s seconds.
    Score = (#|y|>3*global_robust_sigma, then local std). Pick min.
    Return (start, end, std_in_window).
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    w = int(max(1, round(win_s * fs)))
    if w >= n:
        return 0, n, float(np.std(y, ddof=1))
    sig_r = robust_sigma(y)
    best_start = 0
    best_score = (10**9, float("inf"))
    best_std = float("inf")
    step = max(1, w // 10)
    for s in range(0, n - w + 1, step):
        seg = y[s:s + w]
        n_above = int(np.sum(np.abs(seg) > 3.0 * sig_r))
        sd = float(np.std(seg, ddof=1))
        score = (n_above, sd)
        if score < best_score:
            best_score = score
            best_start = s
            best_std = sd
    return best_start, best_start + w, best_std


# ---- Conservative spike detection ----

def detect_spikes_simple(y: np.ndarray, fs: float,
                         sigma: float,
                         thresh_mult: float = 5.0,
                         min_width_ms: float = 1.2,
                         polarity: Literal["pos", "neg", "both"] = "pos",
                         refractory_ms: float = 3.0,
                         prominence_mult: float = 3.0,
                         merge_ms: Optional[float] = None) -> Events:
    """
    Peak detector with prominence and consolidation to avoid double counting.
    Rules:
      height >= thresh_mult * sigma
      prominence >= prominence_mult * sigma
      width >= min_width_ms (half‑prominence)
      inter‑peak distance >= refractory_ms
      and merge peaks closer than merge_ms (default: max(refractory_ms, 1.5*min_width_ms)).
    """
    y = np.asarray(y, dtype=np.float64)
    thr = float(thresh_mult * sigma)
    prom = float(prominence_mult * sigma)
    wmin = max(1, int(round(min_width_ms * 1e-3 * fs)))
    dist = max(1, int(round(refractory_ms * 1e-3 * fs)))

    peaks, amps, widths_ms, pols = [], [], [], []

    def _pick(v, pol: str):
        pk, props = signal.find_peaks(v, height=thr, prominence=prom,
                                      width=(wmin, None), distance=dist)
        if pk.size == 0:
            return [], [], []
        heights = props.get("peak_heights", v[pk])
        widths_samples = props.get("widths", np.full(pk.size, np.nan))
        return pk.tolist(), heights.tolist(), (1000.0 * widths_samples / fs).tolist()

    if polarity in ("pos", "both"):
        pk, h, wms = _pick(y, "pos")
        peaks += pk; amps += h; widths_ms += wms; pols += ["pos"] * len(pk)
    if polarity in ("neg", "both"):
        pk, h, wms = _pick(-y, "neg")
        peaks += pk; amps += (-np.asarray(h)).tolist(); widths_ms += wms; pols += ["neg"] * len(pk)

    if not peaks:
        return Events(np.array([], int), np.array([], float), np.array([], float),
                      np.array([], float), np.array([], "<U3"), float(sigma), thr)

    # sort by index
    order = np.argsort(peaks)
    peaks = np.asarray(peaks, int)[order]
    amps = np.asarray(amps, float)[order]
    widths_ms = np.asarray(widths_ms, float)[order]
    pols = np.asarray(pols, dtype="<U3")[order]

    # consolidate peaks that are too close; keep the larger |amp|
    if merge_ms is None:
        merge_ms = max(refractory_ms, 1.5 * min_width_ms)
    merge = max(1, int(round(merge_ms * 1e-3 * fs)))
    keep = [0]
    for i in range(1, peaks.size):
        if peaks[i] - peaks[keep[-1]] <= merge:
            if abs(amps[i]) > abs(amps[keep[-1]]):
                keep[-1] = i
        else:
            keep.append(i)
    keep = np.asarray(keep, int)
    peaks, amps, widths_ms, pols = peaks[keep], amps[keep], widths_ms[keep], pols[keep]

    t = peaks / float(fs)
    return Events(peaks, t, amps, widths_ms, pols, float(sigma), thr)


# ---- Baseline computation for voltage summary ----

def compute_dff0(raw: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute dF/F0 exactly matching preprint methodology."""
    raw = np.asarray(raw, dtype=np.float64)
    
    # Rolling percentile baseline
    window = max(1, int(0.5 * fs))  # 0.5s windows
    n = len(raw)
    baseline = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        segment = raw[start:end]
        p30, p80 = np.percentile(segment, [30, 80])
        mask = (segment >= p30) & (segment <= p80)
        baseline[i] = np.mean(segment[mask]) if np.any(mask) else np.mean(segment)
    
    F0 = np.maximum(baseline, np.finfo(np.float64).eps)  # Prevent division by zero
    dff = (raw - F0) / F0
    percent_dff = 100.0 * dff
    return dff, percent_dff, F0


# ---- Trace extraction from imaging data ----

def extract_roi_traces(frames: np.ndarray, masks: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract mean fluorescence traces from ROIs defined by masks.
    Returns dict with F traces; shapes (N_cells, T).
    """
    T, H, W = frames.shape
    lab = masks.astype(np.int32)
    n_cells = lab.max()
    if n_cells == 0:
        return {"F": np.array([]).reshape(0, T)}

    F = np.zeros((n_cells, T), dtype=np.float32)

    # Precompute boolean masks per ROI
    roi_masks = [(lab == i) for i in range(1, n_cells + 1)]

    for t in range(T):
        frame = frames[t].astype(np.float32)
        for i, roi in enumerate(roi_masks):
            vals = frame[roi]
            F[i, t] = vals.mean() if vals.size else np.nan

    return {"F": F}


def build_reference(frames: np.ndarray, method: str = "mean") -> np.ndarray:
    """Compute a reference image from the stack (T,H,W)."""
    if method == "mean":
        ref = frames.mean(axis=0)
    elif method == "max":
        ref = frames.max(axis=0)
    else:
        raise ValueError(f"Unknown reference method: {method}")
    # Normalize to [0,1] float
    ref = ref.astype(np.float32)
    if ref.max() > 0:
        ref = ref / ref.max()
    return ref


# ========================== SMART SEGMENTATION IMPROVEMENTS ==========================

# ----------------------------- helpers -----------------------------
def _to_gray(A):
    A = np.asarray(A)
    return A.mean(axis=2).astype(np.float32) if A.ndim == 3 else A.astype(np.float32)

def _norm01(I):
    lo, hi = np.percentile(I, (0.5, 99.5))
    return np.clip((I - lo) / max(hi - lo, 1e-6), 0, 1)

def _adaptive_norm(I, clip_limit=0.02):
    """Adaptive normalization that handles varying contrast conditions."""
    I = np.asarray(I, dtype=np.float32)
    
    # Try multiple normalization strategies and pick best
    methods = []
    
    # Method 1: Robust percentile normalization
    lo, hi = np.percentile(I, (1, 99))
    if hi > lo:
        norm1 = np.clip((I - lo) / (hi - lo), 0, 1)
        methods.append(("percentile_1_99", norm1))
    
    # Method 2: Adaptive histogram equalization
    try:
        from skimage import exposure
        norm2 = exposure.equalize_adapthist(I, clip_limit=clip_limit)
        methods.append(("clahe", norm2))
    except:
        pass
    
    # Method 3: Standard 0.5-99.5 percentile (original)
    lo, hi = np.percentile(I, (0.5, 99.5))
    norm3 = np.clip((I - lo) / max(hi - lo, 1e-6), 0, 1)
    methods.append(("percentile_0.5_99.5", norm3))
    
    # Select best based on contrast and structure preservation
    best_method = "percentile_0.5_99.5"
    best_norm = norm3
    best_score = -1
    
    for name, norm_img in methods:
        # Score based on contrast and edge content
        gradient = filters.sobel(norm_img)
        contrast = np.std(norm_img)
        edge_content = np.mean(gradient > 0.1)
        score = contrast * 0.6 + edge_content * 0.4
        
        if score > best_score:
            best_score = score
            best_norm = norm_img
            best_method = name
    
    return best_norm

def _assess_image_quality(I):
    """Assess image quality metrics to guide segmentation strategy."""
    I_norm = _norm01(I)
    
    # Basic statistics
    mean_intensity = np.mean(I_norm)
    contrast = np.std(I_norm)
    
    # Edge content
    gradient = filters.sobel(I_norm)
    edge_content = np.mean(gradient > 0.05)
    
    # Signal-to-noise ratio estimate
    # High-frequency content as noise proxy
    I_smooth = filters.gaussian(I_norm, sigma=1.0)
    noise_proxy = np.std(I_norm - I_smooth)
    snr_estimate = contrast / (noise_proxy + 1e-6)
    
    # Texture analysis
    try:
        from skimage.feature import graycomatrix, graycoprops
        # Convert to uint8 for GLCM
        I_uint8 = (I_norm * 255).astype(np.uint8)
        glcm = graycomatrix(I_uint8, [1], [0], levels=256, symmetric=True, normed=True)
        contrast_glcm = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    except:
        contrast_glcm = contrast
        homogeneity = 0.5
    
    # Detect membrane-like structures
    try:
        ridges = filters.sato(I_norm, sigmas=[0.5, 1.0, 1.5])
        membrane_score = np.percentile(ridges, 95)
    except:
        membrane_score = 0.0
    
    quality_metrics = {
        'mean_intensity': float(mean_intensity),
        'contrast': float(contrast),
        'edge_content': float(edge_content),
        'snr_estimate': float(snr_estimate),
        'contrast_glcm': float(contrast_glcm),
        'homogeneity': float(homogeneity),
        'membrane_score': float(membrane_score),
        'image_shape': I.shape
    }
    
    return quality_metrics

def _estimate_diam_px(I, quality_metrics=None):
    """Enhanced diameter estimation with multiple robust methods."""
    I2 = _adaptive_norm(I)  # Use adaptive normalization
    
    estimates = []
    weights = []
    
    # Get image quality metrics if not provided
    if quality_metrics is None:
        quality_metrics = _assess_image_quality(I)
    
    # Method 1: Multi-scale LoG blob detection with adaptive thresholds
    snr = quality_metrics.get('snr_estimate', 3.0)
    base_thresh = max(0.001, 0.01 / snr)  # Adapt threshold to SNR
    
    for i, thresh in enumerate([base_thresh * f for f in [0.5, 1.0, 2.0, 4.0]]):
        try:
            blobs = feature.blob_log(I2, min_sigma=0.8, max_sigma=12, num_sigma=20, threshold=thresh)
            if blobs.size >= 4:  # Need sufficient blobs for reliable estimate
                diameters = blobs[:, 2] * np.sqrt(2.0) * 2.0
                # Filter out outliers
                q25, q75 = np.percentile(diameters, [25, 75])
                iqr = q75 - q25
                valid_diams = diameters[(diameters >= q25 - 1.5*iqr) & (diameters <= q75 + 1.5*iqr)]
                if len(valid_diams) >= 2:
                    est = float(np.median(valid_diams))
                    weight = len(valid_diams) / len(diameters)  # Fraction of valid detections
                    estimates.append(est)
                    weights.append(weight)
                    print(f"LoG (thresh={thresh:.4f}): {len(blobs)} blobs, {len(valid_diams)} valid, diameter={est:.1f}px")
        except Exception as e:
            print(f"LoG method failed at threshold {thresh}: {e}")
    
    # Method 2: Multi-threshold distance transform
    for percentile in [75, 85, 95]:  # Try different foreground thresholds
        try:
            threshold = np.percentile(I2, percentile)
            m = I2 > threshold
            if 100 <= m.sum() <= 0.8 * I2.size:  # Reasonable foreground size
                d = ndi.distance_transform_edt(m)
                v = d[d > 2]  # Exclude border artifacts
                if v.size > 20:
                    # Use multiple statistics for robustness
                    median_dist = np.median(v)
                    p90_dist = np.percentile(v, 90)
                    est = float(2.0 * median_dist)
                    weight = min(1.0, v.size / 100)  # More points = higher weight
                    estimates.append(est)
                    weights.append(weight)
                    print(f"Distance transform (p{percentile}): {m.sum()} fg pixels, diameter={est:.1f}px")
        except Exception as e:
            print(f"Distance transform failed at percentile {percentile}: {e}")
    
    # Method 3: Granulometry-based estimation (opening-closing with varying sizes)
    try:
        sizes = np.arange(1, min(20, min(I2.shape)//10))
        granulometry = []
        for s in sizes:
            opened = morphology.opening(I2 > np.percentile(I2, 85), morphology.disk(s))
            granulometry.append(opened.sum())
        
        # Find the size that maximally preserves foreground
        granulometry = np.array(granulometry)
        if len(granulometry) > 3:
            # Look for "elbow" in granulometry curve
            diffs = np.diff(granulometry)
            if len(diffs) > 1:
                second_diffs = np.diff(diffs)
                if len(second_diffs) > 0:
                    optimal_idx = np.argmax(second_diffs) + 1
                    est = float(sizes[optimal_idx] * 2.0)  # Convert radius to diameter
                    weight = 0.7  # Moderate confidence
                    estimates.append(est)
                    weights.append(weight)
                    print(f"Granulometry: optimal size={sizes[optimal_idx]}, diameter={est:.1f}px")
    except Exception as e:
        print(f"Granulometry method failed: {e}")
    
    # Method 4: Peak spacing in auto-correlation
    try:
        # Compute 2D auto-correlation
        I_centered = I2 - np.mean(I2)
        autocorr = signal.fftconvolve(I_centered, I_centered[::-1, ::-1], mode='same')
        autocorr = autocorr / autocorr.max()
        
        # Find peaks in the central region
        center = np.array(autocorr.shape) // 2
        search_radius = min(center) // 2
        y_slice = slice(center[0] - search_radius, center[0] + search_radius)
        x_slice = slice(center[1] - search_radius, center[1] + search_radius)
        autocorr_crop = autocorr[y_slice, x_slice]
        
        # Find local maxima
        peaks = peak_local_maxima(autocorr_crop, min_distance=3, threshold_abs=0.1)
        if len(peaks) > 1:
            # Calculate distances from center
            center_crop = np.array(autocorr_crop.shape) // 2
            distances = np.sqrt(np.sum((peaks - center_crop)**2, axis=1))
            distances = distances[distances > 2]  # Exclude central peak
            if len(distances) > 0:
                typical_spacing = np.median(distances)
                est = float(typical_spacing * 2.0)  # Spacing to diameter
                weight = 0.6
                estimates.append(est)
                weights.append(weight)
                print(f"Auto-correlation: spacing={typical_spacing:.1f}, diameter={est:.1f}px")
    except Exception as e:
        print(f"Auto-correlation method failed: {e}")
    
    # Method 5: Adaptive fallback based on image characteristics
    img_size = min(I2.shape)
    contrast = quality_metrics.get('contrast', 0.1)
    edge_content = quality_metrics.get('edge_content', 0.1)
    
    # Adjust fallback based on image properties - MUCH MORE CONSERVATIVE for small images
    if img_size < 150:  # Small field of view like yours (100x134)
        if contrast > 0.1 and edge_content > 0.08:
            # Good contrast in small image - cells likely 15-25% of image size
            fallback_factor = 0.20  # Larger fallback for small images
        else:
            fallback_factor = 0.25  # Even larger for low contrast
    else:
        # Original logic for larger images
        if contrast > 0.15 and edge_content > 0.1:
            fallback_factor = 0.08
        elif contrast < 0.05:
            fallback_factor = 0.15
        else:
            fallback_factor = 0.1
    
    conservative_est = float(img_size * fallback_factor)
    estimates.append(conservative_est)
    weights.append(0.4)  # Higher weight for fallback in small images
    print(f"Adaptive fallback: diameter={conservative_est:.1f}px ({fallback_factor*100:.1f}% of {img_size}px)")
    
    # Weighted combination of estimates
    if estimates:
        estimates = np.array(estimates)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        # Use weighted median for robustness
        sorted_indices = np.argsort(estimates)
        cumulative_weights = np.cumsum(weights[sorted_indices])
        median_index = sorted_indices[np.searchsorted(cumulative_weights, 0.5)]
        weighted_median = estimates[median_index]
        
        # Also compute weighted mean for comparison
        weighted_mean = np.sum(estimates * weights)
        
        # Choose between median and mean based on spread
        spread = np.std(estimates)
        if spread > 5.0:  # High variance - prefer robust median
            final_estimate = weighted_median
            method = "weighted_median"
        else:  # Low variance - use mean
            final_estimate = weighted_mean
            method = "weighted_mean"
        
        # Apply reasonable bounds with adaptive limits
        min_size = max(6.0, img_size * 0.02)
        max_size = min(40.0, img_size * 0.25)
        final_estimate = float(np.clip(final_estimate, min_size, max_size))
        
        print(f"Final diameter: {final_estimate:.1f}px ({method}, {len(estimates)} methods)")
        print(f"  Estimates: {[f'{e:.1f}' for e in estimates]}")
        print(f"  Weights: {[f'{w:.2f}' for w in weights]}")
        
        return final_estimate
    else:
        print("All estimation methods failed, using fallback 12.0px")
        return 12.0

def _ring_ratio(I, mask, r_px):
    # mean intensity on a thin ring vs interior; >1 means membrane brighter than cytosol
    w = max(1, int(round(0.25 * (r_px/2))))
    inner = morphology.erosion(mask, morphology.disk(w))
    outer = morphology.dilation(mask, morphology.disk(w))
    ring = np.logical_and(outer, np.logical_not(inner))
    m_in = I[inner].mean() if inner.any() else 0.0
    m_rg = I[ring].mean() if ring.any() else 0.0
    return float(m_rg / (m_in + 1e-6))

def _enhanced_post_filter(I, lbl, d_px, quality_metrics=None, ring_min=1.20):
    """Enhanced post-processing with adaptive filtering and overlap resolution."""
    if lbl.max() == 0: 
        return lbl
    
    # First apply overlap resolution
    lbl = _resolve_overlapping_segments(lbl, I, d_px)
    
    if lbl.max() == 0:
        return lbl
    
    I2 = _adaptive_norm(I)  # Use adaptive normalization
    r = d_px / 2.0
    expected_area = np.pi * r**2
    
    # Adaptive area bounds based on image quality and size
    if quality_metrics is not None:
        contrast = quality_metrics.get('contrast', 0.1)
        snr = quality_metrics.get('snr_estimate', 3.0)
        img_shape = quality_metrics.get('image_shape', (100, 100))
        img_area = img_shape[0] * img_shape[1]
        
        # More lenient for small images where precision is harder
        if img_area < 20000:  # Small images like yours (100x134 = 13400)
            area_tolerance = 2.5  # 250% tolerance for small images (even more lenient)
        elif contrast > 0.15 and snr > 4.0:
            area_tolerance = 0.8  # 80% tolerance (more lenient)
        elif contrast < 0.05 or snr < 2.0:
            area_tolerance = 1.8  # 180% tolerance for low quality (more lenient)
        else:
            area_tolerance = 1.2  # 120% default (more lenient)
    else:
        area_tolerance = 1.0
    
    min_area = int(expected_area * max(0.1, 1 - area_tolerance))  # Never go below 10% of expected
    max_area = int(expected_area * (1 + 3 * area_tolerance))  # Allow up to 3x tolerance above
    
    # Adaptive shape constraints - much more lenient for small images
    if quality_metrics and img_area < 20000:
        # Very lenient for small images
        min_circularity = 0.05
        max_eccentricity = 0.99
        min_solidity = 0.40
    else:
        # Original constraints for larger images
        min_circularity = 0.08 if quality_metrics and quality_metrics.get('snr_estimate', 3) < 2.0 else 0.12
        max_eccentricity = 0.98 if quality_metrics and quality_metrics.get('contrast', 0.1) < 0.05 else 0.95
        min_solidity = 0.50 if quality_metrics and quality_metrics.get('edge_content', 0.1) < 0.05 else 0.60
    
    # Enhanced filtering with multiple criteria
    out = np.zeros_like(lbl, np.int32)
    cur = 1
    rejected_reasons = []
    
    for p in measure.regionprops(lbl, intensity_image=I2):
        m = (lbl == p.label)
        reason = None
        
        # 1. Area filter
        if p.area < min_area:
            reason = f"too_small_area_{p.area}<{min_area}"
        elif p.area > max_area:
            reason = f"too_large_area_{p.area}>{max_area}"
        
        # 2. Shape filters - more lenient for membrane-based segmentation
        elif p.solidity < min_solidity:
            reason = f"low_solidity_{p.solidity:.2f}<{min_solidity}"
        elif p.eccentricity > max_eccentricity:
            reason = f"high_eccentricity_{p.eccentricity:.2f}>{max_eccentricity}"
        else:
            circ = 4.0 * np.pi * p.area / (p.perimeter**2 + 1e-6)
            if circ < min_circularity:
                reason = f"low_circularity_{circ:.2f}<{min_circularity}"
            
            # 3. Intensity-based filters
            elif hasattr(p, 'mean_intensity'):
                # Ring ratio test for membrane-like structures - skip for small images
                if quality_metrics and img_area >= 20000:  # Only apply ring ratio test for larger images
                    ring_ratio = _ring_ratio(I2, m, r_px=d_px)
                    if ring_ratio < ring_min:
                        reason = f"low_ring_ratio_{ring_ratio:.2f}<{ring_min}"
                
                # Intensity consistency test
                elif quality_metrics and quality_metrics.get('snr_estimate', 3) > 3.0:
                    # For high SNR, check intensity uniformity within ROI
                    roi_intensities = I2[m]
                    intensity_cv = np.std(roi_intensities) / (np.mean(roi_intensities) + 1e-6)
                    if intensity_cv > 0.8:  # Too variable intensity
                        reason = f"high_intensity_variation_{intensity_cv:.2f}>0.8"
        
        if reason is None:
            # Passed all filters
            out[m] = cur
            cur += 1
        else:
            rejected_reasons.append(reason)
    
    # Print filtering summary
    initial_count = lbl.max()
    final_count = cur - 1
    print(f"Post-filtering: {initial_count} -> {final_count} ROIs")
    if rejected_reasons:
        from collections import Counter
        reason_counts = Counter([r.split('_')[0] + '_' + r.split('_')[1] for r in rejected_reasons])
        print(f"  Rejection reasons: {dict(reason_counts)}")
    
    return measure.label(out > 0, connectivity=1).astype(np.int32)

def _post_filter(I, lbl, d_px, ring_min=1.20):
    """Legacy post-filter for compatibility."""
    return _enhanced_post_filter(I, lbl, d_px, quality_metrics=None, ring_min=ring_min)

# ----------------------------- classic blobs -----------------------------
def _segment_classic(I, d_px):
    # DoG + watershed, strict, with border crop
    I0 = _norm01(I)
    H, W = I0.shape
    margin = int(round(0.03 * min(H, W)))
    valid = np.zeros_like(I0, bool); valid[margin:H-margin, margin:W-margin] = True

    Ieq = exposure.equalize_adapthist(I0, clip_limit=0.01)
    Ism = filters.gaussian(Ieq, 1.0, preserve_range=True)
    s_small = max(0.6, 0.6 * (d_px/2)); s_large = max(s_small + 0.5, 1.8 * (d_px/2))
    dog = filters.gaussian(Ism, s_small) - filters.gaussian(Ism, s_large)
    dog = (dog - dog.mean()) / (dog.std() + 1e-6)

    mask = (dog > np.percentile(dog[valid], 98.0)) & valid
    r1 = max(1, int(round(0.18 * (d_px/2)))); r2 = max(1, int(round(0.25 * (d_px/2))))
    mask = morphology.opening(mask, morphology.disk(r1))
    mask = morphology.closing(mask, morphology.disk(r2))
    mask = morphology.remove_small_objects(mask, min_size=int(0.25 * np.pi * (d_px/2)**2))
    mask = ndi.binary_fill_holes(mask)
    mask = clear_border(mask)
    if not mask.any(): return np.zeros_like(I0, np.int32)

    # More aggressive minimum distance to prevent over-merging
    min_dist = max(4, int(round(1.2 * (d_px/2))))  # Increased from 0.8 to 1.2
    seeds = peak_local_max(dog, labels=mask, min_distance=min_dist, threshold_abs=1.0)  # Higher threshold
    if seeds.size == 0: 
        # Try with more relaxed parameters if no seeds found
        min_dist = max(3, int(round(0.8 * (d_px/2))))
        seeds = peak_local_max(dog, labels=mask, min_distance=min_dist, threshold_abs=0.5)
        if seeds.size == 0:
            return np.zeros_like(I0, np.int32)
    
    mk = np.zeros_like(I0, int); mk[tuple(seeds.T)] = np.arange(1, seeds.shape[0] + 1)
    mk, _ = ndi.label(mk > 0)
    
    # Use original image gradient instead of smoothed, with high compactness
    grad = filters.sobel(I0)  # Use original normalized image
    lbl = watershed(grad, mk, mask=mask, compactness=1.0)  # Much higher compactness
    lbl = measure.label(lbl, connectivity=1)
    quality_metrics = _assess_image_quality(I0)
    return _enhanced_post_filter(I0, lbl, d_px, quality_metrics)

# ----------------------------- membrane mode -----------------------------
def _segment_membrane(I, d_px, strict=0.0):
    # enhance bright ridges > close loops > fill > split interiors
    I = _norm01(I)
    Ieq = exposure.equalize_adapthist(I, clip_limit=0.01)
    rid = filters.sato(Ieq, sigmas=np.linspace(0.6, 1.2*(d_px/6), 6), black_ridges=False)
    thr = np.percentile(rid, 85 + 8*strict)
    E = rid > thr
    E = morphology.binary_closing(E, morphology.disk(max(1,int(0.15*(d_px/2)))))
    E = morphology.remove_small_objects(E, min_size=int(0.08*np.pi*(d_px/2)**2))
    F = ndi.binary_fill_holes(E)
    F = clear_border(F)
    if not F.any(): return np.zeros_like(I, np.int32)
    # interiors - be more conservative about erosion for small cells
    erosion_size = max(1, int(0.1*(d_px/2)))  # Smaller erosion
    inner = morphology.erosion(F, morphology.disk(erosion_size))
    dist = ndi.distance_transform_edt(inner)
    min_distance = max(3, int(0.6*(d_px/2)))  # Ensure seeds are well separated
    seeds = peak_local_max(dist, labels=inner, min_distance=min_distance, threshold_abs=0.3*dist.max())
    if seeds.size==0: return np.zeros_like(I, np.int32)
    # Create seeds with better separation
    mk = np.zeros_like(I, int)
    mk[tuple(seeds.T)] = np.arange(1, seeds.shape[0] + 1)
    mk, _ = ndi.label(mk > 0)
    
    # Use more conservative watershed with compactness to prevent over-merging
    lbl = watershed(-dist, mk, mask=inner, compactness=0.01)
    quality_metrics = _assess_image_quality(I)
    return _enhanced_post_filter(I, measure.label(lbl, connectivity=1), d_px, quality_metrics)

# ----------------------------- Overlap Resolution Functions -----------------------------
def _resolve_overlapping_segments(masks, ref_img, diameter_px):
    """Resolve overlapping segments that likely belong to the same cell."""
    if masks.max() == 0:
        return masks
        
    from skimage.measure import regionprops
    from scipy.spatial.distance import cdist
    
    # Get properties of all segments
    props = regionprops(masks, intensity_image=ref_img)
    if len(props) <= 1:
        return masks
    
    # Calculate centroids and other properties
    centroids = np.array([prop.centroid for prop in props])
    areas = np.array([prop.area for prop in props])
    
    # Calculate distance matrix between centroids
    distances = cdist(centroids, centroids)
    
    # Define merge criteria
    min_merge_distance = diameter_px * 0.7  # Segments closer than 70% of diameter
    area_ratio_threshold = 3.0  # Don't merge if one segment is 3x larger than the other
    
    # Find segments to merge
    segments_to_merge = []
    merged_labels = set()
    
    for i in range(len(props)):
        if i + 1 in merged_labels:  # Already processed
            continue
            
        current_group = [i]
        
        for j in range(i + 1, len(props)):
            if j + 1 in merged_labels:  # Already processed
                continue
                
            # Check if segments should be merged
            if distances[i, j] < min_merge_distance:
                # Check area ratio to avoid merging very different sized segments
                area_ratio = max(areas[i], areas[j]) / (min(areas[i], areas[j]) + 1e-6)
                
                if area_ratio < area_ratio_threshold:
                    current_group.append(j)
                    merged_labels.add(j + 1)
        
        if len(current_group) > 1:
            segments_to_merge.append(current_group)
    
    # Apply merging
    output_masks = masks.copy()
    
    for group in segments_to_merge:
        # Merge all segments in group to the first label
        primary_label = group[0] + 1  # +1 because props are 0-indexed but labels are 1-indexed
        
        for idx in group[1:]:
            secondary_label = idx + 1
            output_masks[output_masks == secondary_label] = primary_label
            print(f"    Merged overlapping segments: {secondary_label} -> {primary_label}")
    
    # Relabel to ensure contiguous numbering
    return measure.label(output_masks > 0, connectivity=1).astype(np.int32)

def _score_cellpose_result(masks, ref_img, diameter_px, model_type, img_variant):
    """Enhanced scoring for Cellpose results."""
    if masks.max() == 0:
        return 0.0
    
    from skimage.measure import regionprops
    props = regionprops(masks, intensity_image=ref_img)
    n_cells = len(props)
    
    # Base score from cell count
    if n_cells < 2:
        return 10.0  # Very low score for too few cells
    elif n_cells > 20:
        return 5.0   # Penalize too many (likely over-segmented)
    
    score = n_cells * 15  # Base score
    
    # Bonus for ideal cell count range
    if 3 <= n_cells <= 10:
        score += 50
    elif 5 <= n_cells <= 8:
        score += 100  # Sweet spot
    
    # Model preference (newer models get slight bonus)
    model_bonus = {"cyto3": 20, "cyto2": 10, "cyto": 0}
    score += model_bonus.get(model_type, 0)
    
    # Image variant preference
    variant_bonus = {"enhanced": 15, "denoised": 10, "original": 5, "inverted": 0}
    score += variant_bonus.get(img_variant, 0)
    
    # Quality-based scoring
    expected_area = np.pi * (diameter_px / 2) ** 2
    area_scores = []
    shape_scores = []
    
    for prop in props:
        # Area score (closer to expected = better)
        area_ratio = prop.area / expected_area
        if 0.3 <= area_ratio <= 3.0:  # Reasonable size range
            area_score = 1.0 - abs(np.log(area_ratio))
            area_scores.append(max(0, area_score))
        else:
            area_scores.append(0)
        
        # Shape score (more circular = better for cells)
        circularity = 4 * np.pi * prop.area / (prop.perimeter ** 2 + 1e-6)
        shape_scores.append(circularity)
    
    # Add quality bonuses
    if area_scores:
        avg_area_score = np.mean(area_scores)
        score += avg_area_score * 30
    
    if shape_scores:
        avg_shape_score = np.mean(shape_scores)
        score += avg_shape_score * 25
    
    return score

# ----------------------------- Enhanced Cellpose with Advanced Features -----------------------------
def _segment_cellpose(I, d_px):
    """MPS-safe Cellpose v4 segmentation on a 2D reference image."""
    try:
        # Choose device (auto)
        device = pick_device("auto")
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        print(f"[cellpose] torch {torch.__version__}, device={device.type}")

        # Build v4 model with pretrained weights; no model_type
        try:
            cp = models.CellposeModel(
                gpu=False,
                device=device,
                pretrained_model="cyto3"
            )
        except Exception as e:
            raise RuntimeError(f"Cellpose init failed on {device}: {e}")

        # Prepare 2D float32 C-contiguous image; gentle standardization for dim images
        ref_img = np.asarray(I, dtype=np.float32)
        if ref_img.ndim != 2:
            ref_img = np.squeeze(ref_img)
        assert ref_img.ndim == 2, "Cellpose expects a 2D image for 2D segmentation."
        if not ref_img.flags['C_CONTIGUOUS']:
            ref_img = np.ascontiguousarray(ref_img)
        mean_v = float(ref_img.mean()); std_v = float(ref_img.std()) + 1e-6
        ref_img = (ref_img - mean_v) / std_v
        min_v = float(ref_img.min()); ptp_v = float(ref_img.ptp()) + 1e-6
        ref_img = np.clip((ref_img - min_v) / ptp_v, 0.0, 1.0).astype(np.float32, copy=False)

        # diameter
        try:
            cp_diam = float(d_px)
        except Exception:
            cp_diam = 0.0

        with torch.inference_mode():
            masks, flows, styles = cp.eval(
                ref_img,
                diameter=cp_diam,
                do_3D=False,
                augment=False,
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )

        if masks is None or (np.asarray(masks).max(initial=0) == 0):
            print("Cellpose produced no masks.")
            return np.zeros_like(ref_img, np.int32)

        return np.asarray(masks, dtype=np.int32)
    except Exception as e:
        print(f"Cellpose failed completely: {e}")
        return np.zeros_like(np.asarray(I, dtype=np.float32), np.int32)

def _smart_backend_selection(quality_metrics):
    """Select optimal segmentation backend based on image characteristics."""
    
    # Extract key metrics
    contrast = quality_metrics.get('contrast', 0.1)
    edge_content = quality_metrics.get('edge_content', 0.1)
    snr_estimate = quality_metrics.get('snr_estimate', 3.0)
    membrane_score = quality_metrics.get('membrane_score', 0.0)
    homogeneity = quality_metrics.get('homogeneity', 0.5)
    
    # Decision logic based on image characteristics
    scores = {}
    
    # Cellpose scoring - much higher preference for the advanced version
    cellpose_score = 5.0  # Base boost for Cellpose 4.0
    if snr_estimate > 1.5:  # Good SNR favors Cellpose
        cellpose_score += 3.0
    if contrast > 0.05:  # Even low contrast works with new Cellpose
        cellpose_score += 2.0
    if edge_content > 0.03:  # Visible structures
        cellpose_score += 2.0
    if homogeneity < 0.9:  # Not too uniform (has structure)
        cellpose_score += 2.0
    scores['cellpose'] = cellpose_score
    
    # Classic (DoG+watershed) scoring
    classic_score = 0.0
    if edge_content > 0.08:  # Good edge content favors DoG
        classic_score += 2.0
    if contrast > 0.15:  # High contrast
        classic_score += 1.5
    if snr_estimate < 4.0:  # Not too noisy
        classic_score += 1.0
    if membrane_score < 0.3:  # Not strongly membrane-like
        classic_score += 0.5
    scores['classic'] = classic_score
    
    # Membrane-aware scoring
    membrane_score_method = 0.0
    if membrane_score > 0.2:  # Strong membrane signal
        membrane_score_method += 3.0
    if edge_content > 0.1:  # Good boundaries
        membrane_score_method += 1.0
    if contrast > 0.08:  # Sufficient contrast
        membrane_score_method += 1.0
    scores['membrane'] = membrane_score_method
    
    # Select best backend
    best_backend = max(scores, key=scores.get)
    best_score = scores[best_backend]
    
    # Minimum score threshold - if all are low, use fallback order
    if best_score < 1.5:
        print("Low confidence in all methods, using fallback order: cellpose -> classic -> membrane")
        return ['cellpose', 'classic', 'membrane']
    
    # Create ordered list starting with best
    backends = list(scores.keys())
    backends.sort(key=lambda x: scores[x], reverse=True)
    
    print(f"Backend selection scores: {scores}")
    print(f"Recommended order: {backends}")
    
    return backends

def _evaluate_segmentation_quality(masks, ref_img, diameter_estimate):
    """Evaluate the quality of a segmentation result."""
    if masks.max() == 0:
        return 0.0
    
    try:
        from skimage.measure import regionprops
        props = regionprops(masks)
        n_cells = len(props)
        
        if n_cells == 0:
            return 0.0
        
        # Basic metrics
        areas = [p.area for p in props]
        circularities = [4 * np.pi * p.area / (p.perimeter**2 + 1e-6) for p in props]
        solidities = [p.solidity for p in props]
        eccentricities = [p.eccentricity for p in props]
        
        # Expected area based on diameter
        expected_area = np.pi * (diameter_estimate / 2)**2
        
        # Scoring components
        scores = []
        
        # 1. Number of cells (prefer 5-50 cells)
        if 5 <= n_cells <= 50:
            n_score = 1.0
        elif 3 <= n_cells <= 80:
            n_score = 0.7
        elif n_cells > 0:
            n_score = 0.3
        else:
            n_score = 0.0
        scores.append(n_score)
        
        # 2. Area consistency (how close to expected size)
        area_ratios = np.array(areas) / expected_area
        area_score = 1.0 - np.mean(np.abs(np.log(np.clip(area_ratios, 0.1, 10))))
        area_score = max(0, min(1, area_score))
        scores.append(area_score)
        
        # 3. Shape quality (circularity, solidity)
        circ_score = min(1.0, np.mean(circularities))  # Cap at 1.0
        solid_score = np.mean(solidities)
        shape_score = (circ_score + solid_score) / 2
        scores.append(shape_score)
        
        # 4. Eccentricity (prefer round objects)
        eccent_score = 1.0 - np.mean(eccentricities)
        scores.append(eccent_score)
        
        # 5. Size variability (prefer consistent sizes)
        size_cv = np.std(areas) / (np.mean(areas) + 1e-6)
        size_score = max(0, 1.0 - size_cv)
        scores.append(size_score)
        
        # 6. Coverage (reasonable fraction of image)
        total_area = sum(areas)
        img_area = ref_img.size
        coverage = total_area / img_area
        if 0.05 <= coverage <= 0.3:
            coverage_score = 1.0
        elif 0.02 <= coverage <= 0.5:
            coverage_score = 0.7
        else:
            coverage_score = 0.3
        scores.append(coverage_score)
        
        # Weighted average
        weights = [0.2, 0.25, 0.2, 0.1, 0.15, 0.1]  # Sum = 1.0
        final_score = sum(s * w for s, w in zip(scores, weights))
        
        print(f"  Quality metrics: n_cells={n_cells}, areas={np.mean(areas):.1f}±{np.std(areas):.1f}, "
              f"circ={np.mean(circularities):.2f}, coverage={coverage:.3f}")
        print(f"  Component scores: {[f'{s:.2f}' for s in scores]} -> {final_score:.3f}")
        
        return final_score
        
    except Exception as e:
        print(f"Quality evaluation failed: {e}")
        return 0.0

# ----------------------------- orchestrator -----------------------------
def segment_cells(ref_img, backend="auto", diameter_px="auto", save_debug=None):
    """Smart segmentation with adaptive backend selection and quality assessment."""
    I = _to_gray(ref_img)
    
    # Assess image quality first
    quality_metrics = _assess_image_quality(I)
    print(f"Image quality assessment: contrast={quality_metrics['contrast']:.3f}, "
          f"SNR={quality_metrics['snr_estimate']:.1f}, edge_content={quality_metrics['edge_content']:.3f}")
    
    # Estimate diameter with quality-aware methods
    if isinstance(diameter_px, str) and diameter_px.lower() == "auto":
        d_est = _estimate_diam_px(I, quality_metrics)
    else:
        d_est = float(diameter_px)
    
    best_masks = np.zeros_like(I, np.int32)
    best_score = -1.0
    best_method = "none"
    
    if backend == "auto":
        # Intelligent backend selection
        backend_order = _smart_backend_selection(quality_metrics)
        test_backends = backend_order
    elif backend in ["cellpose", "classic", "membrane"]:
        test_backends = [backend]
    else:
        # Fallback for unknown backend
        test_backends = ["cellpose", "classic", "membrane"]
    
    # Try each backend and evaluate quality
    results = []
    for method in test_backends:
        print(f"\nTrying segmentation method: {method}")
        try:
            if method == "cellpose":
                masks = _segment_cellpose(I, d_est)
            elif method == "classic":
                masks = _segment_classic(I, d_est)
            elif method == "membrane":
                masks = _segment_membrane(I, d_est)
            else:
                continue
            
            if masks.max() > 0:
                quality_score = _evaluate_segmentation_quality(masks, I, d_est)
                results.append((method, masks, quality_score))
                print(f"  {method}: {masks.max()} cells, quality={quality_score:.3f}")
                
                if quality_score > best_score:
                    best_score = quality_score
                    best_masks = masks
                    best_method = method
            else:
                print(f"  {method}: No cells detected")
                # If specifically cellpose requested and failed, auto-fallback to classic then membrane
                if backend in ("cellpose", "auto") and method == "cellpose":
                    print("  cellpose yielded 0 cells, trying classic fallback...")
                    try:
                        masks = _segment_classic(I, d_est)
                        if masks.max() == 0:
                            print("  classic produced 0 cells, trying membrane fallback...")
                            masks = _segment_membrane(I, d_est)
                    except Exception as fb_e:
                        print(f"  Fallback segmentation failed: {fb_e}")
                    if masks.max() > 0:
                        quality_score = _evaluate_segmentation_quality(masks, I, d_est)
                        results.append(("fallback", masks, quality_score))
                        print(f"  fallback: {masks.max()} cells, quality={quality_score:.3f}")
                        if quality_score > best_score:
                            best_score = quality_score
                            best_masks = masks
                            best_method = "fallback"
                
        except Exception as e:
            print(f"  {method}: Failed with error: {e}")
    
    # If auto mode and we have multiple decent results, potentially combine them
    if backend == "auto" and len(results) > 1:
        # Check if results are very different - if so, might want to ensemble
        scores = [r[2] for r in results]
        if len(scores) >= 2 and max(scores) - min(scores) < 0.2:
            # Results are similar quality - stick with best
            print(f"Multiple similar quality results, using best: {best_method} (score={best_score:.3f})")
        else:
            print(f"Using best result: {best_method} (score={best_score:.3f})")
    
    print(f"\nFinal segmentation: {best_method} with {best_masks.max()} cells (quality={best_score:.3f})")
    
    # Enhanced debug snapshot
    if save_debug is not None:
        import matplotlib.pyplot as plt
        
        # Create comprehensive debug figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(_adaptive_norm(I), cmap='gray')
        axes[0, 0].set_title("Adaptive Normalized Input")
        axes[0, 0].axis('off')
        
        # DoG filter response
        s_small = max(0.6, 0.6*(d_est/2))
        s_large = max(s_small+0.5, 1.8*(d_est/2))
        dog = filters.gaussian(_adaptive_norm(I), s_small) - filters.gaussian(_adaptive_norm(I), s_large)
        dog = (dog - dog.mean()) / (dog.std() + 1e-6)
        
        im = axes[0, 1].imshow(dog, cmap='magma')
        axes[0, 1].set_title(f"DoG Filter (σ={s_small:.1f}-{s_large:.1f})")
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
        
        # Seeds/peaks
        seeds = peak_local_maxima(dog, min_distance=max(2,int(0.8*(d_est/2))), threshold_abs=0.6)
        axes[0, 2].imshow(_adaptive_norm(I), cmap='gray')
        if seeds.size > 0:
            axes[0, 2].scatter(seeds[:,1], seeds[:,0], s=20, c='cyan', marker='x')
        axes[0, 2].set_title(f"Detected Seeds (n={len(seeds)})")
        axes[0, 2].axis('off')
        
        # Final segmentation
        axes[1, 0].imshow(_adaptive_norm(I), cmap='gray')
        if best_masks.max() > 0:
            contours = measure.find_contours(best_masks > 0, 0.5)
            for c in contours:
                axes[1, 0].plot(c[:, 1], c[:, 0], 'r-', linewidth=1.5)
        axes[1, 0].set_title(f"Final: {best_method} ({best_masks.max()} cells)")
        axes[1, 0].axis('off')
        
        # Quality metrics text
        axes[1, 1].axis('off')
        quality_text = f"""Quality Assessment:
Method: {best_method}
Score: {best_score:.3f}
Cells: {best_masks.max()}
Diameter: {d_est:.1f}px

Image Metrics:
Contrast: {quality_metrics['contrast']:.3f}
SNR: {quality_metrics['snr_estimate']:.1f}
Edge content: {quality_metrics['edge_content']:.3f}
Membrane score: {quality_metrics['membrane_score']:.3f}

All Methods Tried:
{chr(10).join([f'{r[0]}: {r[1].max()} cells (q={r[2]:.3f})' for r in results])}"""
        
        axes[1, 1].text(0.05, 0.95, quality_text, transform=axes[1, 1].transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Labeled masks with colors
        if best_masks.max() > 0:
            from matplotlib.colors import ListedColormap
            import matplotlib.cm as cm
            
            # Create a colormap for labels
            n_labels = best_masks.max()
            colors = cm.tab20(np.linspace(0, 1, min(n_labels, 20)))
            if n_labels > 20:
                colors = np.tile(colors, (n_labels // 20 + 1, 1))[:n_labels]
            
            # Add transparent background
            colors_with_bg = np.vstack([[0, 0, 0, 0], colors[:n_labels]])
            cmap = ListedColormap(colors_with_bg)
            
            axes[1, 2].imshow(_adaptive_norm(I), cmap='gray')
            axes[1, 2].imshow(best_masks, alpha=0.6, cmap=cmap, vmin=0, vmax=n_labels)
            axes[1, 2].set_title(f"Labeled Masks (colored)")
            axes[1, 2].axis('off')
        else:
            axes[1, 2].imshow(_adaptive_norm(I), cmap='gray')
            axes[1, 2].set_title("No cells detected")
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_debug, dpi=300, bbox_inches='tight')
        plt.close()
    
    return best_masks.astype(np.int32)
# ======================== END RELIABLE SEGMENTATION ========================

# Helper to build Cellpose v4 model (cyto3->cyto2 fallback)
def _build_cellpose_model(device, prefer="cyto3"):
    # v4 API: no model_type, no net_avg, no channels here, and no SAM
    try:
        return models.CellposeModel(
            gpu=False,
            device=device,
            pretrained_model=prefer,
            sam_model=None  # ensure no SAM even if present
        )
    except Exception as e1:
        print(f"[cellpose] {prefer} load failed: {e1}")
        return models.CellposeModel(
            gpu=False,
            device=device,
            pretrained_model="cyto2",
            sam_model=None
        )

def run_cellpose(args, ref_img: np.ndarray, est_diam_px: Optional[float]):
    """Run Cellpose on a single reference image -> labeled mask (H,W)."""
    device = torch.device("cpu")
    print(f"[cellpose] torch {torch.__version__}, device={device.type}")

    # enforce 2D single-channel float32 C-contiguous input
    ref_img = np.asarray(ref_img)
    if ref_img.ndim == 3:
        ref_img = ref_img.mean(axis=-1)   # collapse to 2D
    elif ref_img.ndim != 2:
        raise ValueError(f"Cellpose expects a 2D image; got {ref_img.shape}")
    ref_img = np.ascontiguousarray(ref_img.astype(np.float32, copy=False))

    # mild normalization
    m, s = float(ref_img.mean()), float(ref_img.std()) + 1e-6
    ref_img = (ref_img - m) / s
    mn, ptp = float(ref_img.min()), float(ref_img.ptp()) + 1e-6
    ref_img = np.clip((ref_img - mn) / ptp, 0, 1).astype(np.float32, copy=False)

    cp = _build_cellpose_model(device, prefer="cyto3")
    cp_diam = 0.0 if (est_diam_px is None or str(est_diam_px).lower() == "auto") else float(est_diam_px)

    with torch.inference_mode():
        masks, flows, styles = cp.eval(
            ref_img,
            diameter=cp_diam,
            do_3D=False,
            normalize=False,
            augment=False,
            flow_threshold=0.4,
            cellprob_threshold=0.0
            # NOTE: no `channels` in v4 for 2D single-channel; no `tile`
        )
    return masks


def clean_masks(masks: np.ndarray,
                min_area: int = 30,
                fill_holes: bool = True,
                remove_border: bool = False) -> np.ndarray:
    """Clean labeled masks; re-label sequentially (1..N)."""
    lab = masks.astype(np.int32)
    lab_bin = lab > 0
    lab_bin = remove_small_objects(lab_bin, min_size=int(min_area))
    if fill_holes:
        lab_bin = remove_small_holes(lab_bin, area_threshold=int(min_area))
    lab = sk_label(lab_bin)
    
    if remove_border:
        # Zero-out objects touching the border
        H, W = lab.shape
        border = np.zeros_like(lab, dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        # Any label that appears on border becomes background
        border_labels = np.unique(lab[border])
        for lb in border_labels:
            lab[lab == lb] = 0
        lab = sk_label(lab > 0)
    return lab.astype(np.int32)


# ---- I/O, plotting, and CLI ----

def load_trace(path: Path, column: Optional[int]) -> np.ndarray:
    """Load single trace file for basic spike detection."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        key = list(data.keys())[0]
        arr = data[key]
    elif ext in {".csv", ".tsv", ".txt"}:
        delim = "," if ext == ".csv" else None
        arr = np.loadtxt(path, delimiter=delim)
    else:
        raise ValueError(f"Unsupported input format: {ext}")
    if arr.ndim == 1:
        return arr.astype(np.float64)
    if arr.ndim == 2:
        col = int(0 if column is None else column)
        return np.asarray(arr[:, col], dtype=np.float64)
    raise ValueError("Unsupported array shape")

def save_events_csv(out_path: Path, ev: Events, fs: float, roi_id: Optional[int] = None):
    """Save events to CSV in format compatible with extract_fluorescence_cellpose.py"""
    import csv
    out = Path(out_path)
    
    if roi_id is not None:
        # Per-ROI format (for imaging data)
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "amp_percent", "amp_dff", "prom_percent", "width_ms", 
                       "rise_ms", "decay_ms", "snr_peak", "local_F0", "local_noise_sigma"])
            for i in range(ev.idx.size):
                amp_percent = float(ev.amp[i]) * 100  # Convert to percent
                amp_dff = float(ev.amp[i])
                prom_percent = amp_percent * 0.8  # Approximate prominence
                snr_peak = amp_dff / (ev.noise_sigma + 1e-12)
                w.writerow([float(ev.t_s[i]), amp_percent, amp_dff, prom_percent, 
                           float(ev.width_ms[i]), 2.0, 5.0, snr_peak, 0.0, float(ev.noise_sigma)])
    else:
        # Simple format (for single trace analysis)
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["peak_index", "t_sec", "amplitude", "width_ms", "polarity",
                        "noise_sigma", "threshold", "fs_hz"])
            for i in range(ev.idx.size):
                w.writerow([int(ev.idx[i]), float(ev.t_s[i]), float(ev.amp[i]),
                            float(ev.width_ms[i]), ev.polarity[i],
                            float(ev.noise_sigma), float(ev.threshold), float(fs)])


def plot_diagnostics(raw: np.ndarray, proc: np.ndarray, fs: float, ev: Events,
                     save_path: Optional[Path] = None, show: bool = False):
    """Basic diagnostic plot for single trace analysis."""
    if plt is None:
        return
    t = np.arange(raw.size) / float(fs)
    plt.figure(figsize=(10, 5))
    plt.plot(t, raw, linewidth=0.7, label="raw")
    plt.plot(t, proc, linewidth=0.8, label="processed")
    plt.axhline(ev.threshold, linestyle="--", linewidth=0.8, label=f"threshold={ev.threshold:.3g}")
    if ev.idx.size:
        plt.scatter(ev.t_s, proc[ev.idx], s=20, marker="o", label="peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("ΔF/F (after detrend)")
    plt.legend(loc="best")
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_roi_summary(raw_trace: np.ndarray, proc_trace: np.ndarray, events: Events, 
                     fs: float, roi_id: int, save_path: Path, title_suffix: str = ""):
    """Create ROI summary plot matching extract_fluorescence_cellpose.py style."""
    if plt is None:
        return
        
    t = np.arange(raw_trace.size) / float(fs)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Raw fluorescence
    axes[0, 0].plot(t, raw_trace, 'k-', linewidth=0.8)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('F (raw)')
    axes[0, 0].set_title(f'ROI {roi_id} - Raw Fluorescence{title_suffix}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Processed ΔF/F
    axes[0, 1].plot(t, proc_trace * 100, 'b-', linewidth=0.8)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('ΔF/F (%)')
    axes[0, 1].set_title(f'ROI {roi_id} - Processed ΔF/F')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Filtered with events (black ticks)
    axes[1, 0].plot(t, proc_trace * 100, 'r-', linewidth=0.8)
    if events.idx.size:
        # Add black tick marks at event times
        y0, y1 = axes[1, 0].get_ylim()
        ytick = y1 - 0.03 * (y1 - y0)
        axes[1, 0].plot(events.t_s, np.full_like(events.t_s, ytick),
                       linestyle='None', marker='|', color='k',
                       markersize=9, markeredgewidth=1.6, clip_on=False, zorder=10)
    axes[1, 0].set_xlabel('Time (s)'); axes[1, 0].set_ylabel('ΔF/F (%)')
    axes[1, 0].set_title(f'ROI {roi_id} - Events (n={events.idx.size})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Power spectrum (same method as extract_fluorescence_cellpose.py)
    if len(proc_trace) > fs * 2:  # Need at least 2 seconds for reliable PSD
        try:
            # Use the same parameters as extract_fluorescence_cellpose.py
            f, Pxx = signal.welch(proc_trace, fs=fs, nperseg=min(4096, len(proc_trace)))
            axes[1, 1].semilogx(f, Pxx, color='g', lw=1.0)
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_ylabel('Power')
            axes[1, 1].set_title(f'ROI {roi_id} - Power Spectrum')
            axes[1, 1].grid(True, alpha=0.3)
        except Exception as e:
            print(f"Warning: PSD computation failed for ROI {roi_id}: {e}")
            # Fall back to metrics table
            metrics_text = f"""ROI {roi_id} Metrics:
Noise σ: {events.noise_sigma:.4f}
Threshold: {events.threshold:.4f}
Event Count: {events.idx.size}
Event Rate: {events.idx.size / (len(raw_trace) / fs):.2f} Hz"""
            
            axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title(f'ROI {roi_id} - Metrics')
            axes[1, 1].axis('off')
    else:
        # Too short for PSD - show metrics table
        metrics_text = f"""ROI {roi_id} Metrics:
Noise σ: {events.noise_sigma:.4f}
Threshold: {events.threshold:.4f}
Event Count: {events.idx.size}
Event Rate: {events.idx.size / (len(raw_trace) / fs):.2f} Hz
Duration: {len(raw_trace) / fs:.1f} s"""
        
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title(f'ROI {roi_id} - Metrics (too short for PSD)')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_overlay_plot(ref_img: np.ndarray, masks: np.ndarray, save_path: Path):
    """Create overlay plot showing ROIs on reference image."""
    if plt is None:
        return
        
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(ref_img, cmap="gray")
    
    # Find ROI boundaries and plot them
    from skimage.segmentation import find_boundaries
    boundaries = find_boundaries(masks, mode='outer')
    
    # Color each ROI
    n_cells = masks.max()
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_cells, 20)))
    if n_cells > 20:
        colors = np.tile(colors, (n_cells // 20 + 1, 1))[:n_cells]
    
    # Outline each ROI with different colors (same as extract_fluorescence_cellpose.py)
    outlines = cp_utils.outlines_list(masks)
    for i, ol in enumerate(outlines):
        color = colors[i % len(colors)] if i < len(colors) else 'white'
        ax.plot(ol[:, 0], ol[:, 1], linewidth=1.5, color=color, label=f'Cell {i+1}')
    
    # Add cell number labels at centroids
    props = regionprops(masks)
    for prop in props:
        centroid = prop.centroid
        cell_id = prop.label
        color = colors[(cell_id-1) % len(colors)] if (cell_id-1) < len(colors) else 'white'
        ax.text(centroid[1], centroid[0], str(cell_id), 
               fontsize=10, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color))
    
    ax.set_title(f'Conservative ROI Segmentation ({n_cells} ROIs)', fontsize=14, fontweight='bold')
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_voltage_summary(traces_dict, masks, img_overlay, times, out_dir, fps=None, events_by_roi=None):
    import os
    import numpy as np
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from skimage import measure, segmentation, color

    # ---- style ----
    mpl.rcParams.update({
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.pad_inches': 0.02,
        'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.facecolor': 'white', 'figure.facecolor': 'white',
        'legend.frameon': False,
    })

    if traces_dict is None or 'F_corr' not in traces_dict:
        return {}
    F_corr = np.asarray(traces_dict['F_corr'])
    N, T = F_corr.shape

    if fps is None:
        dt = float(np.median(np.diff(times)))
        fps = 1.0 / dt
    t = np.asarray(times) if (times is not None and len(times) == T) else np.arange(T) / float(fps)

    # ΔF/F0 in %
    def dff_percent(x):
        x = np.asarray(x, float)
        # fast rolling baseline: 0.5 s trimmed mean
        w = max(1, int(round(0.5 * fps)))
        pad = np.pad(x, (w, w), mode='edge')
        # 30–80% trimmed mean
        baseline = np.empty_like(x)
        for i in range(x.size):
            seg = pad[i:i + 2*w + 1]
            p, q = np.percentile(seg, (30, 80))
            core = seg[(seg >= p) & (seg <= q)]
            baseline[i] = core.mean() if core.size else seg.mean()
        dff = (x - baseline) / (baseline + 1e-12)
        return 100.0 * dff

    Yp = np.column_stack([dff_percent(F_corr[i]) for i in range(N)])  # T x N

    # events from caller (no re-detection)
    event_times = []
    counts = np.zeros(N, int)
    if isinstance(events_by_roi, dict) and len(events_by_roi):
        for i in range(N):
            ev = events_by_roi.get(i, None)
            if ev is None:
                event_times.append(np.array([]))
            else:
                ti = getattr(ev, 't_s', None)
                if ti is None and isinstance(ev, dict):
                    ti = ev.get('t_s', [])
                ti = np.asarray(ti, float)
                event_times.append(ti)
                c = getattr(ev, 'idx', None)
                if c is None and isinstance(ev, dict):
                    c = ev.get('idx', [])
                counts[i] = int(np.size(c)) if c is not None else int(ti.size)
    else:
        event_times = [np.array([]) for _ in range(N)]
    duration = T / float(fps)
    rates = counts / duration
    best = int(np.argmax(rates + 1e-6*np.std(Yp, 0)))
    eventful = [i for i in range(N) if counts[i] >= 3]

    # ---- figure scaffolding ----
    import math
    fig = plt.figure(figsize=(8.2, 4.8), constrained_layout=False)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.06, right=0.995, bottom=0.09, top=0.97,
                           wspace=0.45, hspace=0.45)

    # A) Field + ROIs with neat outlines + scale bar
    ax0 = fig.add_subplot(gs[0, 0])
    if img_overlay is None:
        img_overlay = F_corr.mean(0)
    if img_overlay.ndim == 2:
        ax0.imshow(img_overlay, cmap='gray', interpolation='nearest')
    else:
        ax0.imshow(img_overlay, interpolation='nearest')
    if masks is not None:
        # draw crisp outlines
        outlines = measure.find_contours(masks > 0, 0.5)
        for c in outlines:
            ax0.plot(c[:, 1], c[:, 0], lw=1.0)
        # highlight eventful and best ROI
        props = measure.regionprops(masks)
        if props:
            cx = [p.centroid[1] for p in props]
            cy = [p.centroid[0] for p in props]
            ax0.scatter(cx, cy, s=10, c='white', lw=0)
            for i in eventful:
                if i < len(props):
                    ax0.scatter([cx[i]], [cy[i]], s=28, facecolors='none', edgecolors='tab:red', linewidths=1.0)
            if best < len(props):
                ax0.scatter([cx[best]], [cy[best]], s=32, marker='x', c='tab:blue', linewidths=1.0)
    # simple scale bar (assumes ~1 px/µm if no metadata)
    L = img_overlay.shape[1]
    bar = max(20, int(0.1 * L))
    ax0.plot([10, 10 + bar], [img_overlay.shape[0] - 10]*2, lw=2.0, c='white')
    ax0.text(10, img_overlay.shape[0] - 14, f'{bar} px', color='white', va='bottom', ha='left')
    ax0.set_title('Field + ROIs')
    ax0.set_xticks([]); ax0.set_yticks([])

    # B) Example ROI trace (clean look)
    ax1 = fig.add_subplot(gs[0, 1])
    y = Yp[:, best]
    ax1.plot(t, y, lw=1.0, label='ΔF/F₀ (%)')
    # bandpassed overlay, faint
    from scipy import signal
    ny = 0.5 * fps
    hi = min(120.0, 0.45 * fps) / ny
    lo = max(0.8, 0.5) / ny
    sos = signal.butter(3, [lo, hi], btype='band', output='sos')
    ybp = signal.sosfiltfilt(sos, y)
    ax1.plot(t, ybp, lw=0.9, alpha=0.7, label='bandpassed')
    # event ticks at the top 8% of the axis
    if len(event_times[best]) > 0:
        y0, y1 = ax1.get_ylim()
        ax1.vlines(event_times[best], y1 - 0.08*(y1 - y0), y1, lw=0.6, alpha=0.8)
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('ΔF/F₀ (%)')
    ax1.set_title(f'ROI {best+1}  spikes={counts[best]}  rate={rates[best]:.2f} Hz')
    ax1.legend(loc='upper left', frameon=False)
    ax1.grid(alpha=0.25, linewidth=0.5)

    # C) Event raster (sorted by rate)
    ax2 = fig.add_subplot(gs[0, 2])
    order = np.argsort(-rates)
    for row, j in enumerate(order, start=1):
        if len(event_times[j]) > 0:
            ax2.vlines(event_times[j], row-0.4, row+0.4, lw=0.6)
    ax2.set_ylim(0.5, N + 0.5)
    ax2.set_yticks(range(1, min(N, 15) + 1))
    ax2.set_yticklabels([f'ROI {order[i-1]+1}' for i in ax2.get_yticks()])
    ax2.set_xlabel('Time (s)'); ax2.set_title('Event raster')

    # D) Per‑ROI rate vs SNR (simple SNR proxy)
    ax3 = fig.add_subplot(gs[1, 0])
    # noise via high‑pass MAD
    def hp20_sigma(vec):
        sos = signal.butter(3, min(120.0, 0.45*fps)/ny, btype='highpass', output='sos')
        r = signal.sosfiltfilt(sos, vec - np.median(vec))
        return np.percentile(np.abs(r), 80) / 0.6745 + 1e-12
    noise = np.array([hp20_sigma(Yp[:, i]) for i in range(N)])
    snr = (np.percentile(Yp, 95, axis=0) - np.median(Yp, axis=0)) / noise
    ax3.scatter(snr, rates, s=14)
    ax3.set_xlabel('SNR (95th−median)/σ(HP20)')
    ax3.set_ylabel('Rate (Hz)')
    ax3.set_title('Per-ROI rate vs SNR')

    # E) Event synchrony placeholder
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.text(0.5, 0.5, 'Need ≥2 eventful ROIs', ha='center', va='center')
    ax4.set_title('Event synchrony')

    # F) Population activity with 1/Fs binning
    ax5 = fig.add_subplot(gs[1, 2])
    # Safely concatenate event times
    valid_event_times = [e for e in event_times if len(e) > 0]
    if valid_event_times:
        all_t = np.concatenate(valid_event_times)
        if all_t.size:
            bins = np.arange(t[0], t[-1] + 1.0/fps, 1.0/fps)
            counts_pop, edges = np.histogram(all_t, bins=bins)
            ax5.plot(edges[:-1], counts_pop, lw=0.9)
            ax5.set_ylabel('Pop. rate (Hz/ROI)')
        else:
            ax5.text(0.5, 0.5, 'No events detected', ha='center', va='center', transform=ax5.transAxes)
    else:
        ax5.text(0.5, 0.5, 'No events detected', ha='center', va='center', transform=ax5.transAxes)
    ax5.set_xlabel('Time (s)'); ax5.set_title('Population activity')

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'voltage_summary.png'), bbox_inches='tight')
    plt.close(fig)

    return {
        'rates': rates.tolist(),
        'snr': snr.tolist(),
        'counts': counts.tolist(),
        'best_roi': int(best),
        'fps': float(fps),
    }


def run_single_trace_pipeline(input_path: Path,
                              fs: float,
                              column: Optional[int],
                              output_csv: Path,
                              output_plot: Optional[Path],
                              silent_win_s: float = 3.0,
                              thresh_mult: float = 5.0,
                              min_width_ms: float = 1.0,
                              polarity: Literal["pos", "neg", "both"] = "pos",
                              refractory_ms: float = 1.0,
                              detrend_win_s: float = 45.0,
                              detrend_q: float = 10.0,
                              low_hz: Optional[float] = 0.8,
                              high_hz: Optional[float] = None,
                              filter_order: int = 3) -> Events:
    """Pipeline for single trace analysis."""
    raw = load_trace(Path(input_path), column=column)
    proc = preprocess_trace(raw, fs, detrend_win_s=detrend_win_s, detrend_q=detrend_q,
                            low_hz=low_hz, high_hz=high_hz, order=filter_order)
    _, _, sigma = find_silent_window(proc, fs, win_s=silent_win_s)
    ev = detect_spikes_simple(proc, fs, sigma=sigma, thresh_mult=thresh_mult,
                              min_width_ms=min_width_ms, polarity=polarity, refractory_ms=refractory_ms)
    save_events_csv(output_csv, ev, fs)
    if output_plot is not None:
        plot_diagnostics(raw, proc, fs, ev, save_path=output_plot, show=False)
    return ev


def run_imaging_pipeline(args,
                        data_stack: DataStack,
                        out_dir: str,
                        seg_backend: str = "auto",
                        diameter_px: str = "auto",
                        reference_method: str = "mean",
                        silent_win_s: float = 3.0,
                        thresh_mult: float = 5.0,
                        min_width_ms: float = 1.0,
                        polarity: Literal["pos", "neg", "both"] = "pos",
                        refractory_ms: float = 1.0,
                        detrend_win_s: float = 45.0,
                        detrend_q: float = 10.0,
                        low_hz: Optional[float] = 0.8,
                        high_hz: Optional[float] = None,
                        filter_order: int = 3,
                        baseline_percentile: float = 20.0,
                        save_roi_summaries: bool = True,
                        plot_summary: bool = True) -> Dict:
    """Pipeline for imaging data analysis with ROI extraction."""
    
    # Setup output directories
    os.makedirs(out_dir, exist_ok=True)
    events_dir = Path(out_dir) / "events"
    roi_dir = Path(out_dir) / "roi_summaries"
    events_dir.mkdir(exist_ok=True)
    roi_dir.mkdir(exist_ok=True)
    
    frames = data_stack.frames.astype(np.float32)
    if frames.max() > 0:
        frames /= frames.max()  # Normalize to [0,1]
    
    # Create reference image for segmentation
    ref_img = build_reference(frames, method=reference_method)
    print(f"Using {reference_method} projection for segmentation reference")
    
    # Determine frame rate first (needed for segmentation)
    T = frames.shape[0]
    if data_stack.frame_rate is not None:
        fs = data_stack.frame_rate
    elif data_stack.times is not None and len(data_stack.times) > 1:
        fs = 1.0 / np.median(np.diff(data_stack.times))
    else:
        fs = 30.0  # Default fallback
    
    # Use improved segmentation with intelligent backend selection
    print(f"Running smart segmentation with backend={seg_backend}, diameter={diameter_px}")
    debug_path = os.path.join(out_dir, "seg_debug.png") if save_roi_summaries else None
    masks = segment_cells(ref_img, 
                          backend=seg_backend, 
                          diameter_px=diameter_px,
                          save_debug=debug_path)
    
    has_cells = (masks is not None) and (np.asarray(masks).max(initial=0) > 0)
    if not has_cells:
        msg = "Error: No cells detected with any method."
        if getattr(args, "strict_cellpose", False) and seg_backend == "cellpose":
            raise RuntimeError("Cellpose produced no masks and --strict-cellpose is set.")
        print(msg + " Check seg_debug.png for diagnostics.")
        return {
            "events": {},
            "event_counts": np.array([]),
            "event_rates": np.array([]),
            "eventful_rois": [],
            "metadata": {"error": "No cells detected"}
        }
    
    # Extract traces
    traces_dict = extract_roi_traces(frames, masks)
    F_traces = traces_dict["F"]  # (N_cells, T)
    n_cells, T = F_traces.shape
    
    times = data_stack.times if data_stack.times is not None else np.arange(T) / fs
    
    # Process each ROI
    all_events = {}
    event_counts = np.zeros(n_cells)
    event_rates = np.zeros(n_cells)
    eventful_rois = []
    
    print(f"Processing {n_cells} ROIs at {fs:.1f} Hz...")
    
    for roi_idx in range(n_cells):
        raw_trace = F_traces[roi_idx]
        
        # Skip ROIs with insufficient data
        if np.isnan(raw_trace).sum() > 0.5 * len(raw_trace):
            continue
            
        # Compute F0 and dF/F
        F0 = np.percentile(raw_trace, baseline_percentile)
        dff_trace = (raw_trace - F0) / (F0 + 1e-6)
        
        # Preprocess
        proc_trace = preprocess_trace(dff_trace, fs, detrend_win_s=detrend_win_s, detrend_q=detrend_q,
                                     low_hz=low_hz, high_hz=high_hz, order=filter_order)
        
        # Detect events
        _, _, sigma = find_silent_window(proc_trace, fs, win_s=silent_win_s)
        events = detect_spikes_simple(proc_trace, fs, sigma=sigma, thresh_mult=thresh_mult,
                                     min_width_ms=min_width_ms, polarity=polarity, 
                                     refractory_ms=refractory_ms)
        
        all_events[roi_idx] = events
        event_counts[roi_idx] = events.idx.size
        event_rates[roi_idx] = events.idx.size / (T / fs)
        
        # Save events CSV for eventful ROIs
        if events.idx.size >= 3:  # Minimum 3 events
            eventful_rois.append(roi_idx)
            csv_path = events_dir / f"events_roi{roi_idx+1:03d}.csv"
            save_events_csv(csv_path, events, fs, roi_id=roi_idx+1)
        
        # Save ROI summary plot
        if save_roi_summaries:
            title_suffix = "" if events.idx.size >= 3 else " [LOW ACTIVITY]"
            plot_path = roi_dir / f"roi_{roi_idx+1:03d}.png"
            plot_roi_summary(raw_trace, proc_trace, events, fs, roi_idx+1, plot_path, title_suffix)
        
        print(f"ROI {roi_idx+1}: {events.idx.size} events, rate={event_rates[roi_idx]:.2f} Hz, "
              f"σ={events.noise_sigma:.4f}")
    
    # Create overlay plot
    if plot_summary:
        overlay_path = Path(out_dir) / "overlay.png"
        save_overlay_plot(ref_img, masks, overlay_path)
    
    # Generate voltage summary (same as extract_fluorescence_cellpose.py)
    if plot_summary:
        try:
            traces_for_summary = {"F_corr": F_traces}  # Use F_corr key as expected
            voltage_metrics = plot_voltage_summary(
                traces_for_summary, masks, ref_img, times, out_dir, fs, events_by_roi=all_events
            )
        except Exception as e:
            print(f"Warning: Voltage summary generation failed: {e}")
            voltage_metrics = {}
    
    # Save traces CSV
    rows = []
    for i in range(n_cells):
        for t in range(T):
            rows.append({
                "cell_id": int(i + 1),
                "frame": int(t),
                "time": float(times[t]),
                "F": float(F_traces[i, t]),
                "F_neu": np.nan,  # Not computed in conservative analysis
                "F_corr": float(F_traces[i, t]),
                "F0": float(np.percentile(F_traces[i], baseline_percentile)),
                "dFF": float((F_traces[i, t] - np.percentile(F_traces[i], baseline_percentile)) / 
                           (np.percentile(F_traces[i], baseline_percentile) + 1e-6)),
            })
    
    traces_df = pd.DataFrame(rows)
    traces_csv_path = Path(out_dir) / "traces.csv"
    traces_df.to_csv(traces_csv_path, index=False)
    
    # Save metadata
    metadata = {
        "n_cells": int(n_cells),
        "n_frames": int(T),
        "frame_rate": float(fs),
        "segmentation_backend": seg_backend,
        "diameter_px": diameter_px,
        "baseline_percentile": float(baseline_percentile),
        "detection_method": "conservative_silent_window",
        "threshold_multiplier": float(thresh_mult),
        "min_width_ms": float(min_width_ms),
        "eventful_rois": [int(i+1) for i in eventful_rois],
        "n_eventful": len(eventful_rois)
    }
    
    with open(Path(out_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nAnalysis complete:")
    print(f"- {len(eventful_rois)} eventful ROIs (≥3 events) out of {n_cells}")
    print(f"- Event CSVs saved to: {events_dir}")
    print(f"- ROI summaries saved to: {roi_dir}")
    print(f"- Traces saved to: {traces_csv_path}")
    if plot_summary:
        print(f"- Overlay saved to: {Path(out_dir) / 'overlay.png'}")
    
    # Save masks (same as extract_fluorescence_cellpose.py)
    np.save(Path(out_dir) / "masks.npy", masks)
    if tiff is not None:
        tiff.imwrite(Path(out_dir) / "masks.tif", masks.astype(np.uint16), compression="zlib")
        print(f"- Masks saved to: {Path(out_dir) / 'masks.npy'} and masks.tif")
    else:
        print(f"- Masks saved to: {Path(out_dir) / 'masks.npy'}")
    
    return {
        "events": all_events,
        "event_counts": event_counts,
        "event_rates": event_rates,
        "eventful_rois": eventful_rois,
        "metadata": metadata
    }

def build_args():
    p = argparse.ArgumentParser(
        description="Conservative spike extraction from fluorescence using noise from a silent segment.",
        epilog="""
Examples:
  # Single trace analysis
  %(prog)s --input trace.npy --fs 1000 --out_csv events.csv --out_plot diag.png
  %(prog)s --input trace.csv --fs 500 --column 2 --silent-sec 3 --th-mult 5 --min-width-ms 1
  
  # Imaging data analysis  
  %(prog)s --source rig --data-dir /path/to/experiment --camera-name flash
  %(prog)s --source tiff --tif /path/to/stack.tif --roi-size 60
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data source selection
    p.add_argument("--source", choices=["trace", "rig", "tiff"], default="trace", 
                   help="Input source type: trace (single file), rig (RigDataV2), or tiff (TIFF stack)")
    
    # Single trace mode arguments
    p.add_argument("--input", type=Path, help="Trace file (.npy, .npz, .csv, .tsv) - required for --source trace")
    p.add_argument("--fs", type=float, help="Sampling rate (Hz) - required for --source trace")
    p.add_argument("--column", type=int, default=None, help="If input is 2D, which column to use (0-based)")
    p.add_argument("--out_csv", type=Path, help="Output CSV with detected events - required for --source trace")
    p.add_argument("--out_plot", type=Path, default=None, help="Optional diagnostic PNG path")
    
    # Imaging data mode arguments
    p.add_argument("--data-dir", type=str, help="RigDataV2 data directory (for --source rig)")
    p.add_argument("--camera-name", type=str, default="flash", help="Camera name (RigDataV2)")
    p.add_argument("--tif", type=str, help="Path to a TIFF stack (for --source tiff)")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: data-dir/analysis_output)")
    
    # Segmentation parameters
    p.add_argument("--seg-backend", default="auto", choices=["auto","cellpose","classic","membrane"],
                   help="auto tries intelligent backend selection; specific backends: cellpose, classic, membrane")
    p.add_argument("--diameter-px", default="auto", help="cell diameter in pixels or 'auto' for smart estimation")
    p.add_argument("--reference-method", choices=["mean", "max"], default="mean", 
                   help="Reference image method: mean projection (default) or max")
    p.add_argument("--baseline-percentile", type=float, default=20.0, help="Percentile for F0 baseline (per ROI)")
    p.add_argument("--seg-quality-threshold", type=float, default=0.0, 
                   help="Minimum segmentation quality score (0.0-1.0, 0=accept all)")
    p.add_argument("--adaptive-preprocessing", action="store_true", default=True,
                   help="Use adaptive image preprocessing based on quality metrics")
    p.add_argument("--enhanced-debug", action="store_true", default=False,
                   help="Generate detailed segmentation debug plots with quality metrics")
    
    # Detection parameters
    p.add_argument("--silent-sec", type=float, default=3.0, help="Silent window length for noise estimation (s)")
    p.add_argument("--th-mult", type=float, default=5.0, help="Threshold multiple of sigma")
    p.add_argument("--min-width-ms", type=float, default=1.2, help="Minimum peak width (ms) at half prominence")
    p.add_argument("--polarity", type=str, choices=["pos", "neg", "both"], default="pos", help="Peak polarity")
    p.add_argument("--refractory-ms", type=float, default=3.0, help="Minimum distance between peaks (ms)")
    
    # Preprocessing parameters
    p.add_argument("--detrend-win-s", type=float, default=45.0, help="Rolling percentile window (s)")
    p.add_argument("--detrend-q", type=float, default=10.0, help="Percentile q for baseline")
    p.add_argument("--low-hz", type=float, default=0.8, help="High-pass cutoff (Hz), use 0 to disable")
    p.add_argument("--high-hz", type=float, default=120.0, help="Low-pass cutoff (Hz); clamped to 0.45*fs")
    p.add_argument("--filter-order", type=int, default=3, help="Butterworth order")
    
    # Output options
    p.add_argument("--save-roi-summaries", action="store_true", default=True, help="Save individual ROI summary PNGs")
    p.add_argument("--plot-summary", action="store_true", default=True, help="Generate summary plots (overlay)")
    
    # New argument for strict Cellpose
    p.add_argument("--strict-cellpose", action="store_true")
    
    # New CLI arguments
    p.add_argument("--device", choices=["auto","cpu","mps"], default="auto",
                   help="Compute device for Cellpose. Default auto; on macOS with torch 2.2.* auto=>CPU.")
    p.add_argument("--force-mps", action="store_true",
                   help="Force MPS even on torch 2.2.* (not recommended; may segfault).")
    
    return p.parse_args()

def main():
    args = build_args()
    
    # Handle None values for filtering
    low_hz = None if args.low_hz == 0 else args.low_hz
    high_hz = args.high_hz
    
    if args.source == "trace":
        # Single trace analysis mode
        if not args.input:
            raise SystemExit("--input is required for --source trace")
        if not args.fs:
            raise SystemExit("--fs is required for --source trace")
        if not args.out_csv:
            raise SystemExit("--out_csv is required for --source trace")
            
        print(f"Analyzing single trace: {args.input}")
        run_single_trace_pipeline(
            input_path=args.input,
            fs=args.fs,
            column=args.column,
            output_csv=args.out_csv,
            output_plot=args.out_plot,
            silent_win_s=args.silent_sec,
            thresh_mult=args.th_mult,
            min_width_ms=args.min_width_ms,
            polarity=args.polarity,
            refractory_ms=args.refractory_ms,
            detrend_win_s=args.detrend_win_s,
            detrend_q=args.detrend_q,
            low_hz=low_hz,
            high_hz=high_hz,
            filter_order=args.filter_order
        )
        print(f"Single trace analysis complete. Results saved to: {args.out_csv}")
        
    else:
        # Imaging data analysis mode
        if args.source == "rig":
            if not args.data_dir:
                raise SystemExit("--data-dir is required for --source rig")
            print(f"Loading RigDataV2 data from: {args.data_dir}")
            stack = DataStack.from_rig(args.data_dir, args.camera_name)
        else:  # tiff
            if not args.tif:
                raise SystemExit("--tif is required for --source tiff")
            print(f"Loading TIFF stack from: {args.tif}")
            stack = DataStack.from_tif(args.tif)
        
        # Set up output directory
        if args.out_dir is None:
            if args.data_dir:
                args.out_dir = os.path.join(args.data_dir, "analysis_output")
            else:
                args.out_dir = "./analysis_output"
        
        print(f"Output directory: {args.out_dir}")
        print(f"Data shape: {stack.frames.shape}, Frame rate: {stack.frame_rate}")
        
        # Run imaging analysis
        results = run_imaging_pipeline(
            args,
            data_stack=stack,
            out_dir=args.out_dir,
            seg_backend=args.seg_backend,
            diameter_px=args.diameter_px,
            reference_method=args.reference_method,
            silent_win_s=args.silent_sec,
            thresh_mult=args.th_mult,
            min_width_ms=args.min_width_ms,
            polarity=args.polarity,
            refractory_ms=args.refractory_ms,
            detrend_win_s=args.detrend_win_s,
            detrend_q=args.detrend_q,
            low_hz=low_hz,
            high_hz=high_hz,
            filter_order=args.filter_order,
            baseline_percentile=args.baseline_percentile,
            save_roi_summaries=args.save_roi_summaries,
            plot_summary=args.plot_summary
        )
        
        print(f"Imaging analysis complete!")
        print(f"Results saved to: {args.out_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
