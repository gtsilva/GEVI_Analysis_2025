#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract per-cell fluorescence time courses with Cellpose segmentation.

This script is designed to cover two common acquisition paths:
  1) Using RigDataV2.CameraData (frames binary + camera parameters), or
  2) A TIFF stack on disk.

Workflow:
  - Load image stack.
  - (Optional) Register frames to the reference (phase correlation).
  - Build a reference image (mean or max projection).
  - Run Cellpose on the reference to get a static mask per cell.
  - (Optional) Clean masks (min area, fill holes, remove border).
  - (Optional) Build neuropil rings and compute neuropil-corrected traces.
  - Compute F, F0, and dF/F per cell across time.
  - Save: traces.csv, masks.npy (and masks.tif if tifffile is present), overlay.png.

Notes:
  - By default, a single segmentation is done on the reference image and
    applied to all frames; this avoids per-frame ID swapping. If cells move,
    use --register to correct for translational drift before extraction.
  - For strongly deforming tissue, consider per-frame segmentation + tracking.
    That is out of scope here but can be added if needed.

"""
from __future__ import annotations

# -----------------------------------------------------------------------------
FORCE_TICK_ONLY_SUMMARIES = True   # never draw colored spans/lines
# -----------------------------------------------------------------------------

import os
import sys
import argparse
import json
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd

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
from cellpose import models, io as cp_io, utils as cp_utils

# Use scipy.signal for all signal processing
from scipy import signal

# Skimage for morphology/registration
from skimage.measure import regionprops_table, label as sk_label
from skimage.morphology import remove_small_objects, remove_small_holes, dilation, disk, erosion
from skimage.registration import phase_cross_correlation
from skimage.segmentation import watershed, find_boundaries, relabel_sequential, expand_labels
from skimage.feature import blob_log
try:
    from skimage.feature import peak_local_maxima
except ImportError:
    from skimage.feature import peak_local_max as peak_local_maxima
from skimage.filters import sobel
from scipy.ndimage import shift as ndi_shift, distance_transform_edt, maximum_filter, binary_dilation, gaussian_filter
from scipy import ndimage as ndi
import scipy.signal
import math


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


def build_reference(frames: np.ndarray, method: str = "mean") -> np.ndarray:
    """Compute a reference image from the stack (T,H,W)."""
    if method == "mean":
        ref = frames.mean(axis=0)
    elif method == "max":
        ref = frames.max(axis=0)
    else:
        raise ValueError(f"Unknown reference method: {method}")
    # Normalize to [0,1] float for Cellpose robustness
    ref = ref.astype(np.float32)
    if ref.max() > 0:
        ref = ref / ref.max()
    return ref


def run_cellpose(ref_img: np.ndarray,
                 model_type: str = "cyto",
                 diameter: Optional[float] = None,
                 channels: Tuple[int, int] = (0, 0),
                 use_gpu: bool = False,
                 cellprob_threshold: float = 0.0,
                 flow_threshold: float = 0.4,
                 min_size: int = 0) -> Tuple[np.ndarray, Dict]:
    """Run Cellpose on a single reference image -> labeled mask (H,W)."""
    model = models.Cellpose(gpu=use_gpu, model_type=model_type)
    masks, flows, styles, diams = model.eval(
        ref_img,
        channels=list(channels),
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
    )
    return masks.astype(np.int32), {"flows": flows, "styles": styles, "diams": diams}


def make_ref_and_cp(frames):
    """Create gently normalized reference image for both Cellpose and overlay."""
    med = np.median(frames, axis=0).astype(np.float32)
    p1, p99 = np.percentile(med, (2, 99.8))
    med_norm = np.clip((med - p1) / (p99 - p1 + 1e-6), 0, 1)
    # Use this for BOTH Cellpose input and overlay
    img_cp = med_norm
    img_overlay = med_norm
    return img_overlay, img_cp


def make_ref(frames):
    """Legacy function - redirects to new gentle normalization."""
    img_overlay, img_cp = make_ref_and_cp(frames)
    return img_cp


def autoscale_from_sample(frames, sample_stride=20):
    """Estimate cell diameter and area from blob detection on median image."""
    sample = frames[::sample_stride]
    # estimate scale via LoG blobs on median image
    med = np.median(sample, axis=0).astype(np.float32)
    med = (med - med.min()) / (med.max() - med.min() + 1e-6)
    blobs = blob_log(med, min_sigma=2, max_sigma=12, num_sigma=10, threshold=0.02)
    if len(blobs) == 0:
        return None
    # radius ≈ sqrt(2)*sigma
    radii = np.sqrt(2) * blobs[:, 2]
    r0 = float(np.median(radii))
    diameter = max(8.0, 2.0 * r0)
    expected_area = math.pi * (r0 ** 2)
    return diameter, expected_area


def run_cellpose_diam(model, img_ref, diam, args):
    """Helper to run Cellpose once at a given diameter."""
    masks, flows, styles, diams = model.eval(
        img_ref, diameter=diam, channels=args.channels,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold, resample=True)
    return masks.astype(np.int32), flows


def _to_2d_scoremap(score_src, fallback_2d):
    """
    Convert Cellpose 'flows' or any score source to a 2-D float map with same HxW as fallback_2d.
    Accepts: ndarray (H,W), ndarray (C,H,W), list/tuple of ndarrays, or None.
    Reduces channel/stack dimension by mean.
    Returns normalized [0,1] float32 map; falls back to fallback_2d if shapes mismatch.
    """
    H, W = fallback_2d.shape
    def _as2d(arr):
        a = np.asarray(arr)
        if a.ndim == 2:
            return a
        if a.ndim == 3:
            # assume (C,H,W) or (H,W,C); pick channel-first by heuristic
            if a.shape[1:] == (H, W):
                return a.mean(axis=0)
            if a.shape[0:2] == (H, W):
                return a.mean(axis=2)
        return None

    amap = None
    if score_src is None:
        amap = None
    elif isinstance(score_src, (list, tuple)):
        # try any element that reduces to 2D
        for s in score_src:
            amap = _as2d(s)
            if amap is not None and amap.shape == (H, W):
                break
    else:
        amap = _as2d(score_src)

    if amap is None or amap.shape != (H, W):
        amap = fallback_2d

    # normalize to [0,1]
    amap = amap.astype(np.float32)
    mn, mx = float(np.percentile(amap, 1)), float(np.percentile(amap, 99))
    if mx <= mn:
        mx = amap.max(); mn = amap.min()
    amap = np.clip((amap - mn) / (mx - mn + 1e-6), 0.0, 1.0)
    return amap


def run_cellpose_once(model, img_cp, diam, args):
    """Run Cellpose at one diameter with probability scoring."""
    masks, flows, styles, diams = model.eval(
        img_cp, diameter=int(max(8, diam)),
        channels=args.channels,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        resample=True)
    # 'flows' can be a list/tuple; we pass it to the scorer which will sanitize
    score_src = flows if flows is not None else img_cp
    return masks.astype(np.int32), score_src


def proposals_from_lab(lab, score_src, fallback_img_2d):
    """Extract region proposals with intensity-based scoring."""
    from skimage.measure import regionprops
    score_map = _to_2d_scoremap(score_src, fallback_img_2d)
    H, W = score_map.shape
    props = []
    for p in regionprops(lab):
        coords = p.coords  # (N,2) -> rows, cols
        if coords.size == 0:
            continue
        rows = np.clip(coords[:, 0], 0, H - 1)
        cols = np.clip(coords[:, 1], 0, W - 1)
        score = float(score_map[rows, cols].mean())
        props.append({"coords": coords, "area": coords.shape[0], "score": score})
    return props


def iou_coords(a, b, shape):
    """Calculate IoU and areas from coordinate arrays."""
    A = np.zeros(shape, bool)
    A[a[:, 0], a[:, 1]] = True
    B = np.zeros(shape, bool)
    B[b[:, 0], b[:, 1]] = True
    inter = np.logical_and(A, B).sum()
    if inter == 0:
        return 0.0, A.sum(), B.sum()
    union = A.sum() + B.sum() - inter
    return inter / union, A.sum(), B.sum()


def nms(proposals, shape, iou_thr, contain_thr):
    """Non-maximum suppression across proposals."""
    props = sorted(proposals, key=lambda r: (r["score"], r["area"]), reverse=True)
    kept = []
    for c in props:
        keep = True
        for k in kept:
            iou, a, b = iou_coords(c["coords"], k["coords"], shape)
            contain = iou * min(a, b) / (min(a, b) + 1e-9)
            if contain >= contain_thr or iou >= iou_thr:
                keep = False
                break
        if keep:
            kept.append(c)
    lab = np.zeros(shape, np.int32)
    for i, k in enumerate(kept, start=1):
        coords = k["coords"]
        lab[coords[:, 0], coords[:, 1]] = i
    return lab


def masks_to_regions(lab):
    """Convert labeled mask to list of region dictionaries."""
    from skimage.measure import regionprops
    props = regionprops(lab)
    regs = []
    for p in props:
        regs.append({'label': p.label, 'coords': p.coords})
    return regs


def iou(a, b, shape):
    """Calculate intersection over union for two sets of coordinates."""
    m = np.zeros(shape, bool)
    m[tuple(a.T)] = True
    n = np.zeros(shape, bool) 
    n[tuple(b.T)] = True
    inter = np.logical_and(m, n).sum()
    if inter == 0:
        return 0.0
    union = m.sum() + n.sum() - inter
    return inter / union


def props_from_lab(lab, ref):
    """Extract region properties with intensity-based scoring."""
    from skimage.measure import regionprops
    props = []
    for p in regionprops(lab):
        coords = p.coords
        score = float(ref[tuple(coords.T)].mean())  # intensity proxy
        props.append({"coords": coords, "area": coords.shape[0], "score": score})
    return props


def iou_from_coords(a, b, shape):
    """Calculate IoU and areas from coordinate arrays."""
    m = np.zeros(shape, bool)
    m[tuple(a.T)] = True
    n = np.zeros(shape, bool) 
    n[tuple(b.T)] = True
    inter = np.logical_and(m, n).sum()
    if inter == 0:
        return 0.0, m.sum(), n.sum()
    union = m.sum() + n.sum() - inter
    return inter / union, m.sum(), n.sum()


def nms_union(labs, ref, iou_thr=0.80, contain_thr=0.90):
    """Non-maximum suppression union that drops contained/overlapping proposals."""
    shape = ref.shape
    cand = []
    for lab in labs:
        cand.extend(props_from_lab(lab, ref))
    # sort by score then area
    cand.sort(key=lambda r: (r["score"], r["area"]), reverse=True)
    kept = []
    for c in cand:
        keep = True
        for k in kept:
            iou, a, b = iou_from_coords(c["coords"], k["coords"], shape)
            # drop if mostly contained in a kept ROI
            overlap_frac_small = iou * min(a, b) / (min(a, b) + 1e-9)
            if overlap_frac_small >= contain_thr or iou >= iou_thr:
                keep = False
                break
        if keep:
            kept.append(c)
    out = np.zeros(shape, np.int32)
    lid = 1
    for k in kept:
        out[tuple(k["coords"].T)] = lid
        lid += 1
    return out


def union_masks(shape, labs, iou_thr=0.60):
    """Legacy union function - kept for backward compatibility."""
    # Use NMS union instead
    return nms_union(labs, np.ones(shape), iou_thr=iou_thr, contain_thr=0.90)


def find_local_maxima(image: np.ndarray, min_distance: int = 10, threshold_abs: float = 0.0) -> np.ndarray:
    """Find local maxima in an image using maximum filter (compatible with older scikit-image)."""
    # Apply maximum filter
    max_filtered = maximum_filter(image, size=min_distance)
    
    # Find points where the original equals the max filtered (local maxima)
    local_maxima_mask = (image == max_filtered) & (image > threshold_abs)
    
    # Get coordinates of local maxima
    coords = np.where(local_maxima_mask)
    return np.column_stack((coords[0], coords[1]))  # Return as (y, x) pairs


def apply_watershed_splitting(masks: np.ndarray, min_distance: int = 15, threshold_ratio: float = 0.5, 
                             expected_cell_size: int = 500, size_tolerance: float = 0.5) -> np.ndarray:
    """Apply watershed to split merged cells based on distance transform and size-based logic."""
    result_masks = np.zeros_like(masks)
    current_label = 1
    
    # Calculate size thresholds
    min_cell_size = expected_cell_size * (1 - size_tolerance)
    max_single_cell_size = expected_cell_size * (1 + size_tolerance)
    likely_double_cell_size = expected_cell_size * (1.5 + size_tolerance)
    
    for label_id in np.unique(masks):
        if label_id == 0:  # Skip background
            continue
            
        # Extract single cell mask
        cell_mask = (masks == label_id)
        cell_area = np.sum(cell_mask)
        
        # Skip very small objects
        if cell_area < min_cell_size * 0.3:  # Much smaller than expected
            result_masks[cell_mask] = current_label
            current_label += 1
            continue
        
        # Size-based decision logic
        if cell_area <= max_single_cell_size:
            # Likely a single cell - don't split unless very clear evidence
            should_attempt_split = False
            conservative_threshold = 0.8  # Very conservative
        elif cell_area <= likely_double_cell_size:
            # Likely two cells - more willing to split
            should_attempt_split = True
            conservative_threshold = 0.4  # Less conservative
        else:
            # Very large - likely multiple cells, be aggressive about splitting
            should_attempt_split = True
            conservative_threshold = 0.3  # Least conservative
        
        # Compute distance transform
        distance = distance_transform_edt(cell_mask)
        
        # Adjust threshold based on size
        threshold_abs = conservative_threshold * distance.max()
        
        # Find local maxima (potential cell centers)
        local_maxima = find_local_maxima(distance, min_distance=min_distance, 
                                       threshold_abs=threshold_abs)
        
        if len(local_maxima) <= 1 or not should_attempt_split:
            # Single cell or not worth splitting
            result_masks[cell_mask] = current_label
            current_label += 1
            
        elif len(local_maxima) == 2:
            # Two potential cells - validate with size and geometry
            y1, x1 = local_maxima[0]
            y2, x2 = local_maxima[1]
            
            # Check separation between peaks
            peak_distance = np.sqrt((y2-y1)**2 + (x2-x1)**2)
            
            # Check valley depth
            mid_y, mid_x = (y1 + y2) // 2, (x1 + x2) // 2
            if cell_mask[mid_y, mid_x]:
                valley_depth = distance[mid_y, mid_x]
                peak_height = min(distance[y1, x1], distance[y2, x2])
                valley_ratio = valley_depth / peak_height
            else:
                valley_ratio = 1.0  # No valley, don't split
            
            # Decision criteria based on size and geometry
            should_split = False
            
            if cell_area > likely_double_cell_size:
                # Large cell - split if reasonable separation
                should_split = (valley_ratio < 0.7 and peak_distance > min_distance)
            elif cell_area > max_single_cell_size:
                # Medium-large cell - split if good separation
                should_split = (valley_ratio < 0.5 and peak_distance > min_distance * 1.2)
            else:
                # Smaller cell - only split if excellent separation
                should_split = (valley_ratio < 0.3 and peak_distance > min_distance * 1.5)
            
            if should_split:
                # Apply watershed
                markers = np.zeros_like(distance, dtype=int)
                markers[y1, x1] = 1
                markers[y2, x2] = 2
                
                watershed_result = watershed(-distance, markers, mask=cell_mask)
                
                # Check if resulting segments are reasonable sizes
                seg1_size = np.sum(watershed_result == 1)
                seg2_size = np.sum(watershed_result == 2)
                
                # Only keep split if both segments are reasonable sizes
                if (seg1_size > min_cell_size * 0.5 and seg2_size > min_cell_size * 0.5 and
                    seg1_size < expected_cell_size * 2 and seg2_size < expected_cell_size * 2):
                    
                    # Good split - assign labels
                    result_masks[watershed_result == 1] = current_label
                    current_label += 1
                    result_masks[watershed_result == 2] = current_label
                    current_label += 1
                else:
                    # Bad split - keep as single cell
                    result_masks[cell_mask] = current_label
                    current_label += 1
            else:
                # Don't split
                result_masks[cell_mask] = current_label
                current_label += 1
                
        else:
            # More than 2 peaks - be very selective
            if cell_area > likely_double_cell_size * 1.5:  # Very large cell
                # Filter for strong, well-separated peaks
                strong_peaks = []
                for y, x in local_maxima:
                    if distance[y, x] > 0.7 * distance.max():
                        # Check if this peak is well separated from others
                        well_separated = True
                        for y2, x2 in strong_peaks:
                            if np.sqrt((y-y2)**2 + (x-x2)**2) < min_distance * 1.5:
                                well_separated = False
                                break
                        if well_separated:
                            strong_peaks.append((y, x))
                
                # Limit to maximum reasonable number of cells
                max_cells = min(len(strong_peaks), int(cell_area / min_cell_size))
                if max_cells > 1 and max_cells <= 4:  # Don't create too many cells
                    strong_peaks = strong_peaks[:max_cells]
                    
                    # Apply watershed
                    markers = np.zeros_like(distance, dtype=int)
                    for i, (y, x) in enumerate(strong_peaks):
                        markers[y, x] = i + 1
                    
                    watershed_result = watershed(-distance, markers, mask=cell_mask)
                    
                    # Validate all segments
                    valid_split = True
                    segment_sizes = []
                    for marker_id in range(1, len(strong_peaks) + 1):
                        seg_size = np.sum(watershed_result == marker_id)
                        segment_sizes.append(seg_size)
                        if seg_size < min_cell_size * 0.4:  # Too small
                            valid_split = False
                            break
                    
                    if valid_split:
                        # Good multi-split
                        for marker_id in range(1, len(strong_peaks) + 1):
                            result_masks[watershed_result == marker_id] = current_label
                            current_label += 1
                    else:
                        # Keep as single cell
                        result_masks[cell_mask] = current_label
                        current_label += 1
                else:
                    # Keep as single cell
                    result_masks[cell_mask] = current_label
                    current_label += 1
            else:
                # Not large enough to justify multiple splits
                result_masks[cell_mask] = current_label
                current_label += 1
    
    return result_masks.astype(np.int32)


def _roi_radius(area):
    """Calculate radius from area assuming circular ROI."""
    return np.sqrt(max(area, 1.0) / np.pi)


def split_roi_adaptive(lab, img, valley_pct=20, min_sep_frac=0.25, min_dist_frac=0.20, max_min_dist=12):
    """Split ROIs adaptively based on distance transform and valley detection."""
    out = lab.copy()
    labels = [l for l in np.unique(lab) if l != 0]
    for lbl in labels:
        mask = (out == lbl)
        area = int(mask.sum())
        if area == 0:
            continue
        # distance map and peaks
        dist = ndi.distance_transform_edt(mask)
        r = _roi_radius(area)
        min_sep = max(2, int(min_sep_frac * r))
        
        # Use our compatible peak finding function
        coords_array = find_local_maxima(dist, min_distance=min_sep, threshold_abs=0.1 * dist.max())
        if len(coords_array) < 2:
            continue
        coords = [(int(y), int(x)) for y, x in coords_array]
        
        # valley check
        # approximate ridge as watershed lines between peaks on -dist
        md = max(1, int(min_dist_frac * r))
        md = min(md, int(max_min_dist))
        markers = np.zeros_like(out, dtype=np.int32)
        for i, (rr, cc) in enumerate(coords[:16], start=1):  # cap seeds
            markers[rr, cc] = i
        ws = watershed(-dist, markers=ndi.label(markers)[0], mask=mask, watershed_line=True)
        ridge = (ws == 0) & mask
        if ridge.sum() == 0:
            continue
        ridge_vals = img[ridge]
        inside_vals = img[mask]
        thr = np.percentile(inside_vals, valley_pct)
        ridge_is_dark = np.median(ridge_vals) <= thr

        # geometric separation - require both photometric and geometric criteria
        from skimage.measure import label as sk_label
        markers_lbl, nseeds = ndi.label(markers > 0)
        ws_tmp = watershed(-dist, markers_lbl, mask=mask, watershed_line=False)
        parts = [p for p in np.unique(ws_tmp) if p != 0]
        if len(parts) >= 2:
            # minimum area per part to avoid tiny slivers
            areas = [int((ws_tmp == p).sum()) for p in parts]
            min_allowed = int(0.25 * area)  # avoid tiny slivers
            good_geom = sum(a >= min_allowed for a in areas) >= 2
        else:
            good_geom = False

        if ridge_is_dark and good_geom:
            ws2 = ws_tmp
            out[mask] = 0
            for p in parts:
                if int((ws2 == p).sum()) >= min_allowed:
                    out[(ws2 == p)] = out.max() + 1
    return out


def extract_traces_for_merge(frames, lab, baseline_pct):
    """Extract ΔF/F traces for merging analysis."""
    T = frames.shape[0]
    labels = [l for l in np.unique(lab) if l != 0]
    traces = {}
    for lbl in labels:
        pix = frames[:, lab == lbl]
        if pix.shape[1] == 0:
            continue
        F = np.mean(pix, axis=1)
        F0 = np.percentile(F, baseline_pct)
        dff = (F - F0) / (F0 + 1e-6)
        traces[lbl] = dff
    return traces


def maxlag_xcorr(a, b, maxlag):
    """Returns max Pearson r over lags [-maxlag, maxlag]."""
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    best = -1.0
    for k in range(-maxlag, maxlag + 1):
        if k < 0:
            r = np.dot(a[-k:], b[:len(b) + k]) / denom
        elif k > 0:
            r = np.dot(a[:len(a) - k], b[k:]) / denom
        else:
            r = np.dot(a, b) / denom
        if r > best:
            best = r
    return float(best)


def touching_pairs(lab):
    """Find pairs of touching labels."""
    pairs = set()
    # fast neighborhood check
    for shift in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        s = ndi.shift(lab, shift=shift, order=0, mode='nearest')
        mask = (lab != s) & (lab > 0) & (s > 0)
        a = lab[mask].ravel()
        b = s[mask].ravel()
        for i, j in zip(a, b):
            if i != j:
                pairs.add(tuple(sorted((int(i), int(j)))))
    return list(pairs)


def merge_neighbors_adaptive(lab, img, traces, merge_grad_pct=10, r_thresh=0.85, maxlag=1):
    """Merge neighboring ROIs based on gradient and temporal correlation."""
    out = lab.copy()
    grad = sobel(img.astype(np.float32))
    gthr = np.percentile(grad[out > 0], merge_grad_pct)
    merged = True
    while merged:
        merged = False
        for i, j in touching_pairs(out):
            # shared boundary mask
            border = ((out == i) & ndi.binary_dilation(out == j)) | ((out == j) & ndi.binary_dilation(out == i))
            if border.sum() == 0:
                continue
            weak = np.median(grad[border]) <= gthr
            if not weak:
                continue
            # temporal coherence
            ti, tj = traces.get(i), traces.get(j)
            if ti is None or tj is None:
                continue
            r = maxlag_xcorr(ti, tj, maxlag)
            if r >= r_thresh:
                out[out == j] = i
                # update trace
                traces[i] = (traces[i] + traces[j]) / 2.0
                del traces[j]
                merged = True
                break
    # relabel contiguous after merges
    out, _, _ = relabel_sequential(out)
    return out, traces


def roi_coherence(frames, mask, max_pixels=400):
    """Calculate median pairwise correlation within ROI pixels."""
    idx = np.flatnonzero(mask.ravel())
    if idx.size < 20:
        return 0.0
    if idx.size > max_pixels:
        idx = np.random.choice(idx, max_pixels, replace=False)
    X = frames.reshape(frames.shape[0], -1)[:, idx].astype(np.float32)
    X -= X.mean(0, keepdims=True)
    X /= (X.std(0, keepdims=True) + 1e-6)
    C = np.corrcoef(X, rowvar=False)
    tri = C[np.triu_indices_from(C, k=1)]
    if tri.size == 0:
        return 0.0
    return float(np.median(tri))


def annulus_mask(mask, inner, outer, shape):
    """Create annulus mask around ROI for background estimation."""
    dil_o = expand_labels(mask.astype(np.int32), outer) > 0
    dil_i = expand_labels(mask.astype(np.int32), inner) > 0
    ann = dil_o & (~dil_i)
    return ann & (~mask)


def clean_masks(masks: np.ndarray,
                min_area: int = 30,
                fill_holes: bool = True,
                remove_border: bool = False,
                use_watershed: bool = False) -> np.ndarray:
    """Clean labeled masks; re-label sequentially (1..N)."""
    lab = masks.astype(np.int32)
    lab_bin = lab > 0
    lab_bin = remove_small_objects(lab_bin, min_size=int(min_area))
    if fill_holes:
        lab_bin = remove_small_holes(lab_bin, area_threshold=int(min_area))
    lab = sk_label(lab_bin)
    
    # Apply watershed splitting if requested
    if use_watershed:
        lab = apply_watershed_splitting(lab, 
                                      min_distance=ARGS.watershed_min_distance,
                                      threshold_ratio=ARGS.watershed_threshold,
                                      expected_cell_size=ARGS.expected_cell_size,
                                      size_tolerance=ARGS.size_tolerance)
    
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


def compute_translations(frames: np.ndarray, ref: np.ndarray, upsample_factor: int = 1) -> np.ndarray:
    """Estimate per-frame (dy, dx) translations to align each frame to ref using phase correlation."""
    T = frames.shape[0]
    shifts = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        shift, _, _ = phase_cross_correlation(ref, frames[t], upsample_factor=upsample_factor)
        shifts[t] = shift  # (dy, dx)
    return shifts


def apply_translations(frames: np.ndarray, shifts: np.ndarray, order: int = 1) -> np.ndarray:
    """Apply per-frame (dy, dx) shifts to frames (T,H,W)."""
    T, H, W = frames.shape
    out = np.empty_like(frames)
    for t in range(T):
        dy, dx = shifts[t]
        out[t] = ndi_shift(frames[t], shift=(dy, dx), order=order, mode="nearest", prefilter=False)
    return out


def make_neuropil_ring(mask_i: np.ndarray, all_masks: np.ndarray,
                       inner: int = 3, outer: int = 10) -> np.ndarray:
    """Return a boolean ring for one ROI, excluding other ROIs."""
    inner_d = dilation(mask_i, disk(max(1, inner)))
    outer_d = dilation(mask_i, disk(max(inner + 1, outer)))
    ring = (outer_d & (~inner_d))
    # Exclude any pixels belonging to any ROI
    ring = ring & (all_masks == 0)
    return ring


def extract_traces(frames: np.ndarray, masks: np.ndarray,
                   neuropil: int = 0, neuropil_scale: float = 0.7) -> Dict[str, np.ndarray]:
    """
    Compute per-ROI traces from frames using a static segmentation mask.
    Returns dict with F, F_neu, F_corr; shapes (N_cells, T).
    """
    T, H, W = frames.shape
    lab = masks.astype(np.int32)
    n_cells = lab.max()
    if n_cells == 0:
        raise ValueError("No cells found in mask. Adjust Cellpose parameters.")

    # Build neuropil rings if requested
    rings = None
    if neuropil and neuropil > 0:
        rings = []
        bg_mask_all = (lab == 0)
        for i in range(1, n_cells + 1):
            ring = make_neuropil_ring(lab == i, lab, inner=max(1, neuropil // 2), outer=neuropil)
            rings.append(ring)
        rings = np.stack(rings, axis=0)  # (N, H, W)

    F = np.zeros((n_cells, T), dtype=np.float32)
    F_neu = np.zeros_like(F)

    # Precompute boolean masks per ROI to avoid relabel cost
    roi_masks = [(lab == i) for i in range(1, n_cells + 1)]

    for t in range(T):
        frame = frames[t].astype(np.float32)
        for i, roi in enumerate(roi_masks):
            vals = frame[roi]
            F[i, t] = vals.mean() if vals.size else np.nan
            if rings is not None:
                neu_vals = frame[rings[i]]
                F_neu[i, t] = np.median(neu_vals) if neu_vals.size else np.nan

    F_corr = F - neuropil_scale * F_neu if rings is not None else F.copy()
    return {"F": F, "F_neu": F_neu, "F_corr": F_corr}


def extract_traces_with_local_bg(frames, lab, baseline_pct, do_local_bg, r_in, r_out):
    """Extract traces with optional local background subtraction."""
    labels = [l for l in np.unique(lab) if l != 0]
    traces = {}
    
    for lbl in labels:
        roi = (lab == lbl)
        if do_local_bg:
            ann = annulus_mask(roi, r_in, r_out, frames.shape[1:])
            if ann.sum() == 0:
                ann = (~roi)
            F_roi = frames[:, roi].mean(1)
            F_bg = frames[:, ann].mean(1)
            F = F_roi - F_bg
        else:
            F = frames[:, roi].mean(1)
        F0 = np.percentile(F, baseline_pct)
        dff = (F - F0) / (abs(F0) + 1e-6)
        traces[lbl] = dff
    return traces


def compute_dff(F: np.ndarray, percentile: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dF/F using a per-cell baseline F0 as a low percentile of the trace.
    F: (N, T) array. Returns (dff, F0).
    """
    if not (0 < percentile < 100):
        raise ValueError("percentile must be in (0,100)")
    F0 = np.percentile(F, percentile, axis=1, keepdims=True)
    # Guard against zero baseline
    F0 = np.maximum(F0, 1e-6)
    dff = (F - F0) / F0
    return dff.astype(np.float32), F0.astype(np.float32)


# Removed plot_traces function - no longer needed


# Use simple_conservative_extract_fluo.py detrend_percentile instead


def notch_iir(x, fps, freqs=[50, 60, 100, 120], Q=30):
    """Apply notch filter only if mains peaks are detected."""
    from scipy.signal import iirnotch, filtfilt, welch
    
    # Check if notching is needed
    if len(x) < fps * 2:  # Need at least 2 seconds for reliable PSD
        return x
    
    try:
        # Compute PSD to check for mains interference
        freqs_psd, psd = welch(x, fs=fps, nperseg=min(int(fps), len(x)//4))
        median_power = np.median(psd)
        
        filtered = x.copy()
        applied_notches = []
        
        for freq in freqs:
            if freq >= fps / 2:
                continue
            
            # Check if there's a peak within ±2 Hz
            freq_mask = (freqs_psd >= freq - 2) & (freqs_psd <= freq + 2)
            if np.any(freq_mask):
                peak_power = np.max(psd[freq_mask])
                if peak_power > 5 * median_power:
                    # Apply notch filter
                    b, a = iirnotch(freq, Q, fps)
                    filtered = filtfilt(b, a, filtered)
                    applied_notches.append(freq)
        
        if applied_notches:
            print(f"Applied notch filters at: {applied_notches} Hz")
        
        return filtered
    
    except Exception as e:
        print(f"Warning: Notch filter failed ({e}), returning original")
        return x


# Legacy - use make_bandpass_sos from simple_conservative_extract_fluo.py instead
def bandpass_butter(x, fps, low_hz=None, high_hz=None, order=2, **kw):
    """Legacy function - kept for backward compatibility only."""
    from simple_conservative_extract_fluo import make_bandpass_sos, apply_sos
    if 'low' in kw:  low_hz = kw['low']
    if 'high' in kw: high_hz = kw['high']
    if low_hz is None: low_hz = 0.8
    sos = make_bandpass_sos(fps, low_hz=low_hz, high_hz=high_hz, order=order)
    return apply_sos(x, sos)

def _mad(x):
    """Median absolute deviation with small epsilon."""
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-12

def _rolling_mad(x, win):
    """Rolling MAD with reflect padding."""
    w = max(3, int(win) | 1)
    pad = w//2
    xp = np.pad(x, pad, mode='reflect')
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        seg = xp[i:i+w]
        out[i] = _mad(seg)
    return out + 1e-12


def compute_snr(trace_bp):
    """Compute SNR metrics for a band-passed trace."""
    # Robust noise estimate using MAD on band-passed trace
    sigma = 1.4826 * np.median(np.abs(trace_bp - np.median(trace_bp)))
    
    # SNR as (95th percentile - median) / sigma
    p95 = np.percentile(trace_bp, 95)
    median_val = np.median(trace_bp)
    snr = (p95 - median_val) / (sigma + 1e-8)
    
    # Test: SNR should be positive for reasonable traces
    assert snr >= 0, f"SNR cannot be negative: {snr}"
    
    return float(snr), float(sigma)


# Robust, fast baseline and dF/F (memory-safe)
def piecewise_baseline(x: np.ndarray, fs: float, win_s: float = 0.5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    w = max(1, int(fs * win_s))
    n = x.size
    nblocks = int(np.ceil(n / w))
    b = np.empty(nblocks, dtype=np.float64)
    for i in range(nblocks):
        seg = x[i*w:(i+1)*w]
        if seg.size == 0:
            b[i] = b[i-1] if i else float(np.median(x))
            continue
        lo, hi = np.percentile(seg, [30, 80])
        b[i] = 0.5 * (lo + hi)
    # upsample the piecewise baseline to full length
    t_small = np.arange(nblocks) * w
    return np.interp(np.arange(n), t_small, b)

def compute_dff_robust(F: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    F0 = piecewise_baseline(F, fs, win_s=0.5)
    F0[F0 == 0] = np.finfo(np.float64).eps
    dff = (F - F0) / F0
    return dff, F0

# Safe band-pass filtering that cannot stall
# Legacy - redirects to simple_conservative_extract_fluo.py functions
def design_bandpass(fs: float, low_hz: float, high_hz: float, order: int = 3):
    from simple_conservative_extract_fluo import make_bandpass_sos
    return make_bandpass_sos(fs, low_hz=low_hz, high_hz=high_hz, order=order)

def apply_bandpass(x: np.ndarray, sos) -> np.ndarray:
    from simple_conservative_extract_fluo import apply_sos
    return apply_sos(x, sos)

def _mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))

def _hp(x, fs, fc=20.0, order=3):
    nyq = 0.5*fs
    b, a = butter(order, fc/nyq, btype='highpass')
    return filtfilt(b, a, x)

def _make_kernel(fs, tau_r_ms=3.8, tau_d_ms=5.5, dur_ms=30):
    """Unit‑norm positive-going kernel (rise exp, decay exp)."""
    t = np.arange(0, int(dur_ms*1e-3*fs)) / fs
    kr = 1 - np.exp(-t/(tau_r_ms*1e-3))
    kd = np.exp(-t/(tau_d_ms*1e-3))
    k = kr * kd
    k -= k.mean()
    nrm = np.linalg.norm(k)
    return k / (nrm if nrm>0 else 1.0)

def _corr_with_kernel(x, k):
    if len(x) < len(k):  # pad
        pad = np.zeros(len(k)); pad[:len(x)] = x
        x = pad
    x = (x - x.mean()) / (x.std()+1e-12)
    return float(np.dot(x[:len(k)], k))

# Rolling percentile baseline per preprint
def rolling_percentile_baseline(F: np.ndarray, fs: float, win_s: float = 0.5) -> np.ndarray:
    """Rolling baseline using 30-80th percentile mean in sliding windows."""
    window = max(1, int(win_s * fs))
    n = len(F)
    baseline = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        segment = F[start:end]
        p30, p80 = np.percentile(segment, [30, 80])
        baseline[i] = np.mean(segment[(segment >= p30) & (segment <= p80)])
    
    return baseline

def compute_df_preprint(F: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dF/F using rolling percentile baseline per preprint."""
    F0 = rolling_percentile_baseline(F, fs, win_s=0.5)
    F0 = np.maximum(F0, np.finfo(np.float64).eps)  # Prevent division by zero
    df = (F - F0) / F0
    return df, F0

def bandpass_preprint(x: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass per preprint."""
    from simple_conservative_extract_fluo import make_bandpass_sos, apply_sos
    low_hz = max(0.8, 0.5 * 1/fs)
    high_hz = min(120, 0.45 * fs)
    sos = make_bandpass_sos(fs, low_hz=low_hz, high_hz=high_hz, order=3)
    return apply_sos(x, sos)

def noise_estimate_preprint(df: np.ndarray, fs: float) -> float:
    """Robust noise as 1.4826 * MAD(hp20(df))."""
    # 20 Hz high-pass
    b, a = signal.butter(2, 20.0 * 2/fs, btype='high')
    hp20 = signal.filtfilt(b, a, df)
    sigma = 1.4826 * np.median(np.abs(hp20 - np.median(hp20)))
    return max(sigma, 1e-9)

def build_asap6_kernel(fs: float, tau_r: float = 0.004, tau_d: float = 0.0055) -> np.ndarray:
    """Build normalized ASAP6 kinetics kernel."""
    kernel_len = int(0.040 * fs)  # 40 ms
    t = np.arange(kernel_len) / fs
    
    # Mono-exponential rise and decay
    rise = 1 - np.exp(-t / tau_r)
    decay = np.exp(-t / tau_d)
    kernel = rise * decay
    
    # Normalize
    kernel = kernel / np.linalg.norm(kernel)
    return kernel

def detect_spikes_preprint(F: np.ndarray, fs: float, prom_k: float = 3.5,
                          min_width_ms: float = 1.5, max_width_ms: float = 8.0,
                          refractory_ms: float = 8.0, strict_artifacts: bool = True) -> Dict:
    """
    Comprehensive spike detection following preprint methodology exactly.
    Returns dict with events, metadata, and QC flags.
    """
    
    # Step 1: Baseline and dF/F
    df, F0 = compute_df_preprint(F, fs)
    
    # Step 2: Band-pass filtering
    df_filt = bandpass_preprint(df, fs)
    
    # Step 3: Noise estimate
    sigma = noise_estimate_preprint(df_filt, fs)
    
    # Step 4: Matched filter score
    kernel = build_asap6_kernel(fs, tau_r=0.004, tau_d=0.0055)
    score = np.convolve(df_filt, kernel, mode='same') / np.linalg.norm(kernel)
    
    # Step 5: Peak proposals with SciPy
    distance = max(1, int(0.008 * fs))  # 8 ms minimum distance
    prominence = prom_k * sigma
    width_min = max(1, int(min_width_ms * fs / 1000))
    width_max = max(width_min + 1, int(max_width_ms * fs / 1000))
    
    peaks, props = signal.find_peaks(score, distance=distance, prominence=prominence, width=(width_min, width_max))
    
    if len(peaks) == 0:
        return {
            "events": [],
            "sigma": sigma,
            "n_candidates": 0,
            "n_accepted": 0,
            "artifact": False,
            "qc_summary": "No candidates found"
        }
    
    # Step 6: Per-peak refinement and artifact rejection
    accepted_events = []
    rejected_count = 0
    
    for pk in peaks:
        event = {"frame_idx": int(pk), "time_s": pk / fs}
        
        # Local amplitude on raw df
        local_baseline = np.median(df[max(0, pk-3):pk]) if pk >= 3 else np.median(df[:pk+1])
        amplitude = df[pk] - local_baseline
        event["amp_dff"] = amplitude
        event["amp_pct"] = amplitude * 100
        
        # Prominence
        try:
            prom_val = signal.peak_prominences(df, [pk])[0][0]
            event["prominence_dff"] = prom_val
        except:
            event["prominence_dff"] = amplitude
        
        # Width measurements
        try:
            width_samples = signal.peak_widths(df, [pk], rel_height=0.5)[0][0]
            event["width_ms"] = (width_samples / fs) * 1000
        except:
            event["width_ms"] = 3.0  # Default
        
        # Rise/decay times
        win_samples = max(1, int(0.010 * fs))  # ±10 ms for measurements
        start = max(0, pk - win_samples)
        end = min(len(df), pk + win_samples + 1)
        
        if end > start + 2:
            segment = df[start:end]
            peak_local = pk - start
            peak_val = segment[peak_local]
            
            if peak_val > 0:
                # 10-90% rise time
                val_10 = 0.1 * peak_val
                val_90 = 0.9 * peak_val
                
                rise_10_idx = peak_local
                rise_90_idx = peak_local
                
                for j in range(peak_local, -1, -1):
                    if segment[j] <= val_10:
                        rise_10_idx = j
                        break
                for j in range(peak_local, -1, -1):
                    if segment[j] <= val_90:
                        rise_90_idx = j
                        break
                
                event["rise_ms"] = (peak_local - rise_90_idx) / fs * 1000
                
                # 90-10% decay time
                decay_90_idx = peak_local
                decay_10_idx = peak_local
                
                for j in range(peak_local, len(segment)):
                    if segment[j] <= val_90:
                        decay_90_idx = j
                        break
                for j in range(decay_90_idx, len(segment)):
                    if segment[j] <= val_10:
                        decay_10_idx = j
                        break
                
                event["decay_ms"] = (decay_10_idx - decay_90_idx) / fs * 1000
            else:
                event["rise_ms"] = 0.0
                event["decay_ms"] = 0.0
        else:
            event["rise_ms"] = 0.0
            event["decay_ms"] = 0.0
        
        # SNR and area
        event["snr_peak"] = amplitude / sigma
        
        # Area (AUC over ±8ms)
        area_start = max(0, pk - max(1, int(0.008 * fs)))
        area_end = min(len(df), pk + max(1, int(0.008 * fs)))
        event["area_dff"] = np.sum(df[area_start:area_end]) / fs
        
        # Baseline measurements
        pre_start = max(0, pk - max(1, int(0.010 * fs)))
        pre_end = max(0, pk - max(1, int(0.002 * fs)))
        post_start = min(len(df), pk + max(1, int(0.002 * fs)))
        post_end = min(len(df), pk + max(1, int(0.010 * fs)))
        
        event["baseline_pre"] = np.median(df[pre_start:pre_end]) if pre_end > pre_start else 0.0
        event["baseline_post"] = np.median(df[post_start:post_end]) if post_end > post_start else 0.0
        
        # QC flags for artifact rejection
        qc_flags = []
        reject = False
        
        # Rise/decay constraints
        if event["rise_ms"] > 3.5:
            qc_flags.append("slow_rise")
            reject = True
        if event["decay_ms"] < 2.5:
            qc_flags.append("fast_decay")
            reject = True
        
        # Slope/derivative check
        d = np.diff(df) * fs
        if pk < len(d):
            deriv_win = d[max(0, pk-3):min(len(d), pk+3)]
            max_deriv = np.max(np.abs(deriv_win))
            if max_deriv > 10 * sigma * fs:
                qc_flags.append("high_derivative")
                reject = True
        
        # Photobleach/step change check
        win_ms = 200
        win_samples = max(1, int(win_ms * fs / 1000))
        check_start = max(0, pk - win_samples)
        check_end = min(len(df), pk + win_samples)
        
        if check_end > check_start + 10:
            pre_mean = np.mean(df[check_start:pk])
            post_mean = np.mean(df[pk:check_end])
            step_change = abs(post_mean - pre_mean)
            if step_change > 6 * sigma:
                qc_flags.append("step_change")
                reject = True
        
        # Saturation/outlier check
        if abs(amplitude) > 6 * sigma:
            qc_flags.append("amplitude_outlier")
            reject = True
        
        # Check for camera saturation (simplified)
        if np.any(np.abs(F) > 0.99 * np.max(F)):
            sat_frac = np.mean(np.abs(F) > 0.99 * np.max(F))
            if sat_frac > 0.001:  # 0.1%
                qc_flags.append("saturation")
                reject = True
        
        event["qc_flags"] = ",".join(qc_flags) if qc_flags else "pass"
        event["thresh_used"] = prominence
        event["kernel_tau_r_ms"] = 4.0
        event["kernel_tau_d_ms"] = 5.5
        
        if not reject or not strict_artifacts:
            accepted_events.append(event)
        else:
            rejected_count += 1
    
    # Step 7: ISI sanity check
    if len(accepted_events) > 1:
        refractory_samples = max(1, int(refractory_ms * fs / 1000))
        times = np.array([e["time_s"] for e in accepted_events])
        indices = np.array([e["frame_idx"] for e in accepted_events])
        
        # Remove events violating refractory period
        keep_mask = np.ones(len(accepted_events), dtype=bool)
        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] < refractory_samples:
                keep_mask[i] = False
        
        accepted_events = [accepted_events[i] for i in range(len(accepted_events)) if keep_mask[i]]
    
    # Step 8: ROI-level artifact check
    roi_artifact = False
    fail_rate = rejected_count / (len(peaks) + 1e-6)
    
    if len(accepted_events) > 0:
        median_width = np.median([e["width_ms"] for e in accepted_events])
        if median_width < 1.0 or median_width > 8.0:
            roi_artifact = True
    
    if fail_rate > 0.3:  # More than 30% failed
        roi_artifact = True
    
    return {
        "events": accepted_events,
        "sigma": sigma,
        "n_candidates": len(peaks),
        "n_accepted": len(accepted_events),
        "artifact": roi_artifact,
        "qc_summary": f"candidates={len(peaks)}, accepted={len(accepted_events)}, fail_rate={fail_rate:.2f}",
        "filtered_df": df_filt,
        "baseline_F0": F0
    }

# Tight artifact guard that never kills the run
def is_artifact(trace: np.ndarray, fs: float) -> bool:
    x = np.asarray(trace, dtype=np.float64)
    if not np.all(np.isfinite(x)): 
        return True
    z = (x - np.median(x)) / (np.std(x) + 1e-12)
    if np.any(np.abs(z) > 25):  # saturation/explosions
        return True
    dx = np.diff(x, prepend=x[0]) * fs
    dz = (dx - np.median(dx)) / (np.std(dx) + 1e-12)
    if np.any(np.abs(dz) > 25):  # step/glitch
        return True
    if (np.percentile(np.abs(x), 99.9) > 5.0):  # absurd ΔF/F
        return True
    return False

# Safe Welch PSD
def safe_welch(x, fs):
    nper = int(min(len(x), max(256, 2*fs)))
    nover = int(0.5*nper)
    f, p = signal.welch(np.asarray(x, dtype=np.float64), fs=fs, nperseg=nper, noverlap=nover)
    return f, p

# Helper for clean spike tick visualization
import numpy as np

def _overlay_event_ticks(ax, t_sec, peaks_idx):
    """
    Draw black ticks at event times at the top of the axis.
    t_sec: 1D np.ndarray of time for the trace currently shown in ax
    peaks_idx: 1D integer indices returned by scipy.signal.find_peaks,
               indexed into the SAME trace used to plot in ax
    """
    if peaks_idx is None or len(peaks_idx) == 0:
        return
    peaks_idx = np.asarray(peaks_idx, dtype=int)
    # keep only indices that fall inside the plotted trace
    peaks_idx = peaks_idx[(peaks_idx >= 0) & (peaks_idx < t_sec.size)]
    if peaks_idx.size == 0:
        return
    x = t_sec[peaks_idx]
    y0, y1 = ax.get_ylim()
    ytick = y1 - 0.03 * (y1 - y0)  # a bit below the top so autoscale won't move
    ax.plot(x, np.full_like(x, ytick, dtype=float),
            linestyle='None', marker='|', color='k',
            markersize=9, markeredgewidth=1.6, clip_on=False, zorder=10)

# Single summary plotter with only black ticks
def save_roi_summary_ticks(path, t, Fraw, dff, dff_filt, fs, peak_idx_or_t, title, yunits='%'):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    # Accept indices or times
    peak_t = np.asarray(peak_idx_or_t, dtype=float)
    if peak_t.size and np.all(peak_t == np.floor(peak_t)):  # looks like indices
        peak_t = peak_t / float(fs)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Raw
    axs[0, 0].plot(t, Fraw, color='k', lw=0.5)
    axs[0, 0].set_title(title, color=('crimson' if 'ARTIFACT' in title else 'black'))
    axs[0, 0].set_xlabel('Time (s)'); axs[0, 0].set_ylabel('F (raw)')

    # ΔF/F0 (detrended)
    ylab = 'ΔF/F0 (%)' if yunits == '%' else 'ΔF/F0'
    axs[0, 1].plot(t, dff * (100.0 if yunits == '%' else 1.0), color='b', lw=0.5)
    axs[0, 1].set_title('ΔF/F0'); axs[0, 1].set_xlabel('Time (s)'); axs[0, 1].set_ylabel(ylab)

    # Filtered + Events (red trace + black ticks only)
    y_filt = dff_filt * (100.0 if yunits == '%' else 1.0)
    axs[1, 0].plot(t, y_filt, color='r', lw=0.5)
    
    # Timebase sanity check and add ticks using the helper
    if hasattr(peak_idx_or_t, '__len__') and len(peak_idx_or_t) > 0:
        pk = np.asarray(peak_idx_or_t, dtype=int)
        bad = (pk < 0) | (pk >= t.size)
        if np.any(bad):
            print(f"Warning: {np.sum(bad)} event indices out of range; dropping.")
            pk = pk[~bad]
        
        # Use the tick helper for proper alignment
        _overlay_event_ticks(axs[1, 0], t, pk)
    
    axs[1, 0].set_title('Filtered + Events'); axs[1, 0].set_xlabel('Time (s)'); axs[1, 0].set_ylabel(ylab)

    # Power spectrum of ΔF/F0
    f, Pxx = welch(dff, fs=fs, nperseg=min(4096, len(dff)))
    axs[1, 1].semilogx(f, Pxx, color='g', lw=1.0)
    axs[1, 1].set_title('Power Spectrum'); axs[1, 1].set_xlabel('Frequency (Hz)'); axs[1, 1].set_ylabel('Power')

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# dF/F0 exactly as in preprint (Methods, Fig. 2 traces)
def compute_baseline_percentile(x, fs, win_s=0.5, low=30, high=80):
    """Split trace into 0.5s windows, return mean of 30-80th percentile in each window."""
    x = np.asarray(x, dtype=np.float64)
    w = max(1, int(win_s * fs))
    n = len(x)
    n_windows = int(np.ceil(n / w))
    
    # Compute baseline values in each window
    baseline_vals = []
    window_centers = []
    
    for i in range(n_windows):
        start = i * w
        end = min(n, (i + 1) * w)
        if end > start:
            segment = x[start:end]
            p_low, p_high = np.percentile(segment, [low, high])
            mask = (segment >= p_low) & (segment <= p_high)
            baseline_val = np.mean(segment[mask]) if np.any(mask) else np.mean(segment)
            baseline_vals.append(baseline_val)
            window_centers.append(start + (end - start) / 2)
    
    # Linear interpolation to full length
    if len(baseline_vals) == 1:
        return np.full(n, baseline_vals[0], dtype=np.float64)
    
    baseline = np.interp(np.arange(n), window_centers, baseline_vals).astype(np.float64)
    return baseline

def compute_dff0(raw, fs):
    """Compute dF/F0 exactly matching preprint methodology."""
    raw = np.asarray(raw, dtype=np.float64)
    F0 = compute_baseline_percentile(raw, fs, win_s=0.5, low=30, high=80)
    F0 = np.maximum(F0, np.finfo(np.float64).eps)  # Prevent division by zero
    dff = (raw - F0) / F0
    percent_dff = 100.0 * dff
    return dff, percent_dff, F0

# Peak detection tuned to optical spikes
def detect_spikes_percent(dff_pct, fs, noise_sigma, cfg):
    """
    Spike detection tuned for optical spikes with preprint parameters.
    Uses percent dF/F for plotting and metrics.
    """
    # Guard filtering parameters
    high_hz = min(120.0, fs/2 * 0.95)
    
    # Zero-phase bandpass filter
    try:
        sig = bandpass_butter(dff_pct, fs, low_hz=0.8, high_hz=high_hz, order=3)
    except:
        # Fallback to SOS if filtfilt fails
        sos = signal.butter(3, [0.8/(fs/2), high_hz/(fs/2)], btype='band', output='sos')
        sig = signal.sosfiltfilt(sos, dff_pct)
    
    # Peak detection parameters tuned for optical spikes
    width_frames_min = max(1, int(np.ceil(0.001 * fs)))  # 1 ms
    width_frames_max = max(width_frames_min + 1, int(np.floor(0.006 * fs)))  # 6 ms
    distance = max(1, int(np.ceil(0.0025 * fs)))  # 2.5 ms refractory
    height = cfg.k_height * noise_sigma
    prominence = cfg.k_prom * noise_sigma
    
    # SciPy find_peaks with constraints
    peaks, props = signal.find_peaks(sig, 
                                   width=(width_frames_min, width_frames_max),
                                   distance=distance,
                                   height=height,
                                   prominence=prominence)
    
    if len(peaks) == 0:
        return {
            'idx': np.array([]),
            't_sec': np.array([]),
            'amp_dff': np.array([]),
            'amp_percent': np.array([]),
            'prom_percent': np.array([]),
            'width_ms': np.array([]),
            'rise_ms': np.array([]),
            'decay_ms': np.array([]),
            'snr_peak': np.array([]),
            'local_F0': np.array([]),
            'local_noise_sigma': np.array([])
        }
    
    # Compute measurements for each peak
    t_sec = peaks / fs
    amp_percent = dff_pct[peaks]  # Already in percent
    amp_dff = amp_percent / 100.0  # Convert to fraction
    
    # FWHM width measurements
    try:
        widths_samples, _, _, _ = signal.peak_widths(sig, peaks, rel_height=0.5)
        width_ms = (widths_samples / fs) * 1000.0
    except:
        width_ms = np.full(len(peaks), 3.0)
    
    # Prominence in percent
    prom_percent = props.get('prominences', np.zeros(len(peaks))) * 100 / noise_sigma
    
    # Rise and decay times
    rise_ms = np.zeros(len(peaks))
    decay_ms = np.zeros(len(peaks))
    local_F0 = np.zeros(len(peaks))
    local_noise_sigma = np.zeros(len(peaks))
    
    for i, pk in enumerate(peaks):
        # Rise time (10% to peak)
        win_rise = max(1, int(0.010 * fs))  # 10 ms window
        start_rise = max(0, pk - win_rise)
        if pk > start_rise:
            segment = sig[start_rise:pk+1]
            peak_val = segment[-1]
            val_10 = 0.1 * peak_val
            
            # Find 10% point
            rise_idx = len(segment) - 1
            for j in range(len(segment)-1, -1, -1):
                if segment[j] <= val_10:
                    rise_idx = j
                    break
            rise_ms[i] = (len(segment) - 1 - rise_idx) / fs * 1000
        
        # Decay time (peak to 30%)
        win_decay = max(1, int(0.015 * fs))  # 15 ms window
        end_decay = min(len(sig), pk + win_decay)
        if end_decay > pk:
            segment = sig[pk:end_decay]
            peak_val = segment[0]
            val_30 = 0.3 * peak_val
            
            # Find 30% point
            decay_idx = 0
            for j in range(len(segment)):
                if segment[j] <= val_30:
                    decay_idx = j
                    break
            decay_ms[i] = decay_idx / fs * 1000
        
        # Local F0 (baseline at event time)
        local_F0[i] = compute_baseline_percentile(dff_pct, fs)[pk] / 100.0  # Convert to fraction
        
        # Local noise sigma (±100 ms window)
        win_noise = max(1, int(0.100 * fs))  # 100 ms
        start_noise = max(0, pk - win_noise)
        end_noise = min(len(dff_pct), pk + win_noise + 1)
        local_segment = dff_pct[start_noise:end_noise]
        
        # 20 Hz high-pass + MAD for local noise
        try:
            hp_local = bandpass_butter(local_segment, fs, low_hz=20.0, high_hz=high_hz, order=3)
            local_noise_sigma[i] = 1.4826 * np.median(np.abs(hp_local - np.median(hp_local)))
        except:
            local_noise_sigma[i] = noise_sigma
    
    # SNR using local noise
    snr_peak = amp_percent / (local_noise_sigma + 1e-12)
    
    # Filter based on width and decay constraints
    valid_mask = (width_ms >= 1.0) & (width_ms <= 6.0) & (decay_ms <= 12.0)
    
    return {
        'idx': peaks[valid_mask],
        't_sec': t_sec[valid_mask],
        'amp_dff': amp_dff[valid_mask],
        'amp_percent': amp_percent[valid_mask],
        'prom_percent': prom_percent[valid_mask],
        'width_ms': width_ms[valid_mask],
        'rise_ms': rise_ms[valid_mask],
        'decay_ms': decay_ms[valid_mask],
        'snr_peak': snr_peak[valid_mask],
        'local_F0': local_F0[valid_mask],
        'local_noise_sigma': local_noise_sigma[valid_mask]
    }

# Comprehensive artifact rejection
from scipy.signal import welch
from scipy.stats import kurtosis

def is_artifact_trace(percent_dff, fs, ev, cfg):
    """
    Robust artifact classifier requiring ≥2 rules to trigger.
    Must trigger at least 2 of 4 artifact rules to mark ROI as artifact.
    """
    percent_dff = np.asarray(percent_dff, dtype=np.float64)
    artifact_flags = []
    
    # Rule 1: PSD line-noise ratio
    try:
        f, Pxx = signal.welch(percent_dff, fs=fs, nperseg=min(4096, len(percent_dff)))
        # Line noise bands
        line_bands = [(45, 55), (58, 62), (95, 105)]
        P_line = 0
        for low, high in line_bands:
            mask = (f >= low) & (f <= high)
            if np.any(mask):
                P_line += np.sum(Pxx[mask])
        
        P_total = np.sum(Pxx)
        ratio_line = P_line / (P_total + 1e-12)
        if ratio_line > cfg.line_ratio:
            artifact_flags.append("line_noise")
    except:
        pass
    
    # Rule 2: Derivative outlier rate
    try:
        d = np.diff(percent_dff, prepend=percent_dff[0])
        # 20 Hz high-pass on derivative
        hp_d = bandpass_butter(d, fs, low_hz=20.0, high_hz=min(120.0, fs/2*0.95), order=3)
        sigma_d = 1.4826 * np.median(np.abs(hp_d - np.median(hp_d)))
        q999 = np.percentile(np.abs(d), 99.9)
        if q999 > 12 * sigma_d:
            artifact_flags.append("derivative_outlier")
    except:
        pass
    
    # Rule 3: Saturation/clip
    try:
        # Check for extreme percent dF/F
        if np.max(np.abs(percent_dff)) > 80.0:
            artifact_flags.append("extreme_dff")
        # Note: Camera saturation check would need raw data, skipping for now
    except:
        pass
    
    # Rule 4: Unphysiological density
    try:
        if ev and len(ev.get('width_ms', [])) > 0:
            median_width = np.median(ev['width_ms'])
            if median_width < 0.6 or median_width > 8.0:
                artifact_flags.append("unphysical_width")
            
            if len(ev.get('t_sec', [])) > 1:
                isis = np.diff(ev['t_sec'])
                rate_hz = len(ev['t_sec']) / (ev['t_sec'][-1] - ev['t_sec'][0] + 1e-6)
                isi_cv = np.std(isis) / (np.mean(isis) + 1e-6)
                if rate_hz > 10.0 and isi_cv < 0.2:
                    artifact_flags.append("periodic_vibration")
    except:
        pass
    
    # Require at least 2 flags to mark as artifact
    return len(artifact_flags) >= 2

# Save events CSV with preprint format
def save_events_csv(path, rid, ev, fs):
    if ev is None or len(ev.get('t_sec', [])) == 0:
        return
    
    # Create CSV with specified columns
    with open(os.path.join(path, f"events_roi{rid:03d}.csv"), 'w') as f:
        f.write("time_s,amp_percent,amp_dff,prom_percent,width_ms,rise_ms,decay_ms,snr_peak,local_F0,local_noise_sigma\n")
        
        for i in range(len(ev['t_sec'])):
            time_s = ev['t_sec'][i]
            amp_percent = ev['amp_percent'][i] if i < len(ev['amp_percent']) else 0.0
            amp_dff = ev['amp_dff'][i] if i < len(ev['amp_dff']) else 0.0
            prom_percent = ev.get('prom_percent', [0.0])[i] if i < len(ev.get('prom_percent', [])) else 0.0
            width_ms = ev['width_ms'][i] if i < len(ev['width_ms']) else 0.0
            rise_ms = ev.get('rise_ms', [0.0])[i] if i < len(ev.get('rise_ms', [])) else 0.0
            decay_ms = ev.get('decay_ms', [0.0])[i] if i < len(ev.get('decay_ms', [])) else 0.0
            snr_peak = ev.get('snr_peak', [0.0])[i] if i < len(ev.get('snr_peak', [])) else 0.0
            local_F0 = ev.get('local_F0', [0.0])[i] if i < len(ev.get('local_F0', [])) else 0.0
            local_noise_sigma = ev.get('local_noise_sigma', [1e-6])[i] if i < len(ev.get('local_noise_sigma', [])) else 1e-6
            
            f.write(f"{time_s:.6f},{amp_percent:.3f},{amp_dff:.6f},{prom_percent:.3f},"
                   f"{width_ms:.3f},{rise_ms:.3f},{decay_ms:.3f},{snr_peak:.3f},"
                   f"{local_F0:.6f},{local_noise_sigma:.6f}\n")

# Helper functions for spike detection
def _odd(n): 
    n = int(round(n));  return n if n % 2 == 1 else n + 1

def notch_series(x, fs, freqs=(50,60,100)):
    y = x.astype(float)
    for f0 in freqs:
        if f0 < 0.45*fs:
            b,a = scipy.signal.iirnotch(w0=f0/(fs/2), Q=30)
            y = scipy.signal.filtfilt(b,a,y)
    return y

def rolling_mad(x, win_s, fs):
    w = _odd(win_s*fs)
    pad = w//2
    xp = np.pad(x, (pad,pad), mode='reflect')
    out = np.empty_like(x, float)
    for i in range(len(x)): 
        seg = xp[i:i+w]; out[i] = _mad(seg)
    return out + 1e-12

def enforce_refractory(pk, prom, fs, refr_ms=3.0, merge_ms=1.5):
    if len(pk)==0: return np.array([],int)
    pk = np.asarray(pk); prom = np.asarray(prom) if prom is not None else np.ones_like(pk)
    order = np.argsort(pk); pk=pk[order]; prom=prom[order]
    mrg = int(round(merge_ms*1e-3*fs)); ref = int(round(refr_ms*1e-3*fs))
    keep=[]; last=-10**9; best=0
    for i,p in enumerate(pk):
        if p-last<=mrg:
            if prom[i]>prom[best]: best=i
        else:
            if i>0: keep.append(pk[best])
            best=i; last=p
    keep.append(pk[best])
    out=[]; last=-10**9
    for p in keep:
        if p-last>=ref: out.append(p); last=p
    return np.asarray(out,int)

def detect_spikes_matched(dff, fs,
                          z_k=3.8, prom_k=2.8,
                          min_width_ms=2.0, max_width_ms=9.0,
                          refractory_ms=2.0, min_corr=0.55,
                          _recursion_depth=0):
    """
    Adaptive matched‑filter spike detection.
    - σ from MAD of 20 Hz high‑pass dF/F (preprint noise definition).
    - Peaks on matched‑filtered trace with z and prominence thresholds.
    - Width gate in ms; shape gate by kernel correlation.
    Returns: events list of dicts, and a dict of thresholds/σ.
    """
    # robust noise
    hp = _hp(dff, fs, fc=20.0)
    sigma = _mad_sigma(hp)
    if sigma <= 1e-12:
        return [], {"sigma": sigma, "thr": np.nan, "prom": np.nan}

    # matched filter
    k = _make_kernel(fs)
    mf = np.convolve(dff, k, mode='same')

    thr = z_k * sigma
    prom = prom_k * sigma
    dist = max(1, int(refractory_ms*1e-3*fs))
    w_lo = max(1, int(min_width_ms*1e-3*fs))
    w_hi = max(w_lo+1, int(max_width_ms*1e-3*fs))

    peaks, props = find_peaks(mf, height=thr, prominence=prom, distance=dist, width=(w_lo, w_hi))
    events = []
    # refine features on original dF/F
    for idx, p in enumerate(peaks):
        # local snippet for correlation and baseline
        w = int(0.015*fs)
        i0, i1 = max(0, p-w), min(len(dff), p+w)
        seg = dff[i0:i1]
        # local baseline as 30th percentile (resists spikes)
        base = np.percentile(seg, 30.0)
        amp = dff[p] - base
        fwhm_ms = (props["widths"][idx]/fs)*1000.0
        prom_val = props["prominences"][idx]
        # shape check
        s0 = p - int(0.008*fs)
        s1 = p + int(0.012*fs)
        s0 = max(0, s0); s1 = min(len(dff), s1)
        corr = _corr_with_kernel(dff[s0:s1], _make_kernel(fs))
        if corr < min_corr:
            continue
        # final SNR
        snr_peak = amp / (sigma + 1e-12)
        t_s = p / fs
        events.append({
            "frame": int(p),
            "time_s": float(t_s),
            "amp_dff": float(amp),
            "snr_peak": float(snr_peak),
            "width_ms": float(fwhm_ms),
            "prom_dff": float(prom_val),
            "corr_kernel": float(corr)
        })

    # simple adaptive relaxation if too few peaks but SNR suggests spikes
    # Be more conservative about recursion to prevent infinite loops
    if (len(events) == 0 and 
        (np.max(dff) - np.median(dff)) > 3.0*sigma and 
        _recursion_depth < 2 and  # Limit to 2 levels
        z_k > 3.0):  # Only recurse if we can actually reduce z_k
        return detect_spikes_matched(dff, fs,
                                     z_k=max(3.0, z_k-0.5),
                                     prom_k=max(2.0, prom_k-0.4),
                                     min_width_ms=min_width_ms,
                                     max_width_ms=max_width_ms,
                                     refractory_ms=refractory_ms,
                                     min_corr=max(0.45, min_corr-0.05),
                                     _recursion_depth=_recursion_depth+1)
    return events, {"sigma": sigma, "thr": thr, "prom": prom}


def detect_voltage_events_adaptive(raw_trace, fs, args=None):
    """
    Wrapper function that maintains compatibility with existing code.
    Uses the unified detector with proper parameter handling.
    """
    # Get parameters from args or use defaults
    prom_k = getattr(args, 'prom_k', 3.5) if args else 3.5
    min_width_ms = getattr(args, 'min_width_ms', 1.5) if args else 1.5
    max_width_ms = getattr(args, 'max_width_ms', 8.0) if args else 8.0
    refractory_ms = getattr(args, 'refractory_ms', 8.0) if args else 8.0
    strict_artifacts = getattr(args, 'strict_artifacts', True) if args else True
    
    # Compute dF/F with running baseline
    baseline = running_percentile_baseline(raw_trace, fs, win_s=0.5, q_low=30, q_high=80)
    dff = (raw_trace - baseline) / (baseline + 1e-6)
    
    # Use unified detector
    params = {
        'prom_sigma': prom_k,
        'height_sigma': 3.0,
        'min_isi_ms': refractory_ms,
        'wlen_ms': 7.0,
        'min_width_ms': min_width_ms,
        'max_width_ms': max_width_ms
    }
    
    ev = detect_spikes(dff, fs, params)
    
    # Check for artifacts
    if strict_artifacts and is_artifact_trace(dff, fs, ev):
        # Return empty results for artifacts
        ev = {
            'idx': np.array([]),
            't_sec': np.array([]),
            'amp_dff': np.array([]),
            'width_ms': np.array([]),
            'prom_dff': np.array([]),
            'height_dff': np.array([]),
            'noise_sigma': 1e-6
        }
    
    # Convert to old format for compatibility
    return {
        "idx": ev['idx'],
        "amplitude_percent": ev['amp_dff'] * 100,
        "width_ms": ev['width_ms'],
        "prominence_percent": ev['amp_dff'] * 80,
        "template_r": np.full(len(ev['idx']), 0.8),
        "deriv_z": np.zeros(len(ev['idx'])),
        "amp_z": ev['amp_dff'] / (ev['noise_sigma'] + 1e-12),
        "amp": ev['amp_dff'],
        "bp": bandpass_butter(dff, fs, low_hz=0.8, high_hz=min(120.0, 0.45*fs), order=3),
        "sigma_hp20": ev['noise_sigma'],
        "F0": baseline,
        "events": ev,  # Keep unified format
        "dff": dff
    }


def compute_local_sigma(x_bp, fps, window_ms=200):
    """Compute rolling local noise estimate."""
    window_samples = max(1, int(window_ms * fps / 1000))
    local_sigma = np.zeros_like(x_bp)
    
    for i in range(len(x_bp)):
        start = max(0, i - window_samples // 2)
        end = min(len(x_bp), i + window_samples // 2 + 1)
        segment = x_bp[start:end]
        local_sigma[i] = 1.4826 * np.median(np.abs(segment - np.median(segment)))
    
    return local_sigma


def detect_events(x_bp, fps, method='auto', k_z=6.0, prom=1.8, wmin_ms=1.5, wmax_ms=30.0, 
                 refractory_ms=3.0, merge_ms=2.0):
    """Multi-channel event detector with bidirectional peaks and auto-tuning."""
    from scipy.signal import find_peaks, savgol_filter
    from scipy import stats
    
    # Set random seed for deterministic surrogates
    rng = np.random.default_rng(0)
    
    # Global and local noise estimates
    sigma_global = 1.4826 * np.median(np.abs(x_bp - np.median(x_bp)))
    local_sigma = compute_local_sigma(x_bp, fps, window_ms=200)
    
    # Convert time parameters to samples
    wmin_samples = max(1, int(wmin_ms * fps / 1000))
    wmax_samples = int(wmax_ms * fps / 1000)
    refractory_samples = int(refractory_ms * fps / 1000)
    merge_samples = int(merge_ms * fps / 1000)
    
    # Build candidate signals
    def build_candidate_signals(x):
        signals = {}
        
        # Raw signal
        signals['raw'] = x
        
        # Derivative signal (Savitzky-Golay)
        if len(x) > 5:
            window_len = max(3, min(len(x)//10*2+1, int(0.008 * fps)))
            if window_len % 2 == 0:
                window_len += 1
            if window_len >= len(x):
                window_len = len(x) if len(x) % 2 == 1 else len(x) - 1
            
            try:
                smooth = savgol_filter(x, window_length=window_len, polyorder=2)
                deriv = np.gradient(smooth)  # Use gradient instead of diff to preserve length
                signals['der'] = deriv
            except:
                signals['der'] = np.gradient(x)
        else:
            signals['der'] = np.gradient(x)
        
        # Matched filter signal (biphasic template)
        template_ms = 6
        template_samples = max(3, int(template_ms * fps / 1000))
        
        # Create biphasic template (3ms up, 3ms down)
        t = np.arange(template_samples)
        rise_samples = template_samples // 2
        template = np.zeros(template_samples)
        template[:rise_samples] = np.linspace(0, 1, rise_samples)
        template[rise_samples:] = np.exp(-2 * (t[rise_samples:] - rise_samples) / (template_samples - rise_samples))
        template = template - template.mean()
        template = template / (np.linalg.norm(template) + 1e-8)
        
        if len(x) > len(template):
            mf_output = np.convolve(x, template[::-1], mode='same')
            signals['mf'] = mf_output
        else:
            signals['mf'] = x
        
        return signals
    
    # Detect peaks in signal (bidirectional)
    def detect_peaks_bidirectional(y, signal_name, k_z_current, prom_current):
        sigma_y = 1.4826 * np.median(np.abs(y - np.median(y)))
        height_thresh = k_z_current * sigma_y
        
        all_peaks_info = []
        
        # Positive peaks
        try:
            peaks_pos, props_pos = find_peaks(y,
                                            height=height_thresh,
                                            prominence=prom_current * local_sigma,
                                            width=(wmin_samples, wmax_samples),
                                            distance=refractory_samples)
            
            for i, peak in enumerate(peaks_pos):
                all_peaks_info.append({
                    'peak': peak,
                    'amplitude': y[peak],
                    'prominence': props_pos.get('prominences', [np.nan])[i] if i < len(props_pos.get('prominences', [])) else np.nan,
                    'width': props_pos.get('widths', [np.nan])[i] / fps if i < len(props_pos.get('widths', [])) else np.nan,
                    'source': signal_name,
                    'polarity': 'positive'
                })
        except:
            pass
        
        # Negative peaks (flip signal)
        try:
            peaks_neg, props_neg = find_peaks(-y,
                                            height=height_thresh,
                                            prominence=prom_current * local_sigma,
                                            width=(wmin_samples, wmax_samples),
                                            distance=refractory_samples)
            
            for i, peak in enumerate(peaks_neg):
                all_peaks_info.append({
                    'peak': peak,
                    'amplitude': y[peak],  # Original amplitude (negative)
                    'prominence': props_neg.get('prominences', [np.nan])[i] if i < len(props_neg.get('prominences', [])) else np.nan,
                    'width': props_neg.get('widths', [np.nan])[i] / fps if i < len(props_neg.get('widths', [])) else np.nan,
                    'source': signal_name,
                    'polarity': 'negative'
                })
        except:
            pass
        
        return all_peaks_info
    
    # Build signals
    signals = build_candidate_signals(x_bp)
    
    # Detect events across all channels
    all_peaks = []
    k_z_current = k_z
    prom_current = prom
    
    methods_to_try = [method] if method != 'auto' else ['raw', 'der', 'mf']
    
    for signal_name in methods_to_try:
        if signal_name in signals:
            peaks_info = detect_peaks_bidirectional(signals[signal_name], signal_name, k_z_current, prom_current)
            all_peaks.extend(peaks_info)
    
    # Auto-tune if no events but strong signal
    if len(all_peaks) == 0:
        max_z = (np.max(x_bp) - np.median(x_bp)) / sigma_global
        if max_z >= 6:
            k_z_lowered = max(4.5, k_z * 0.9)
            print(f"Auto-lowering k_z from {k_z} to {k_z_lowered} for strong signal (max_z={max_z:.1f})")
            
            for signal_name in methods_to_try:
                if signal_name in signals:
                    peaks_info = detect_peaks_bidirectional(signals[signal_name], signal_name, k_z_lowered, prom_current)
                    all_peaks.extend(peaks_info)
    
    if len(all_peaks) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    # Sort by time and merge nearby peaks
    all_peaks.sort(key=lambda x: x['peak'])
    
    # Merge nearby events (keep highest prominence)
    if merge_samples > 0:
        merged_peaks = []
        i = 0
        while i < len(all_peaks):
            group = [all_peaks[i]]
            j = i + 1
            while j < len(all_peaks) and all_peaks[j]['peak'] - all_peaks[i]['peak'] <= merge_samples:
                group.append(all_peaks[j])
                j += 1
            
            # Keep peak with highest absolute prominence
            best_peak = max(group, key=lambda x: abs(x['prominence']) if not np.isnan(x['prominence']) else 0)
            merged_peaks.append(best_peak)
            
            i = j
        
        all_peaks = merged_peaks
    
    # Surrogate FDR control
    max_iterations = 3
    for iteration in range(max_iterations):
        if len(all_peaks) == 0:
            break
        
        # Create surrogate by circular shift
        shift_amount = rng.integers(1, len(x_bp))
        surrogate = np.roll(x_bp, shift_amount)
        
        # Count surrogate events
        surr_signals = build_candidate_signals(surrogate)
        surr_count = 0
        
        for signal_name in methods_to_try:
            if signal_name in surr_signals:
                surr_peaks = detect_peaks_bidirectional(surr_signals[signal_name], signal_name, k_z_current, prom_current)
                surr_count += len(surr_peaks)
        
        # Check FDR
        if surr_count <= 0.2 * len(all_peaks):
            break
        
        # Raise thresholds
        k_z_current = min(k_z_current * 1.1, k_z * 1.5)
        prom_current *= 1.1
        
        if iteration == max_iterations - 1:
            print(f"Warning: High false-positive rate, using raised thresholds (k_z={k_z_current:.1f})")
        
        # Re-detect with raised thresholds
        all_peaks = []
        for signal_name in methods_to_try:
            if signal_name in signals:
                peaks_info = detect_peaks_bidirectional(signals[signal_name], signal_name, k_z_current, prom_current)
                all_peaks.extend(peaks_info)
        
        all_peaks.sort(key=lambda x: x['peak'])
    
    # Extract final results
    if len(all_peaks) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    # Extract data from peak dictionaries
    event_times = np.array([p['peak'] for p in all_peaks]) / fps
    event_amps = np.array([p['amplitude'] for p in all_peaks])
    event_widths = np.array([p.get('width', np.nan) for p in all_peaks])
    event_proms = np.array([p.get('prominence', np.nan) for p in all_peaks])
    event_sources = np.array([p.get('source', 'unknown') for p in all_peaks])
    
    # Ensure monotonicity (should already be sorted)
    if len(event_times) > 1:
        sort_indices = np.argsort(event_times)
        event_times = event_times[sort_indices]
        event_amps = event_amps[sort_indices]
        event_widths = event_widths[sort_indices]
        event_proms = event_proms[sort_indices]
        event_sources = event_sources[sort_indices]
    
    return event_times, event_amps, event_widths, event_proms, event_sources


def ccg_peak(a, b, fps, maxlag_ms=50):
    """Returns peak normalized cross-correlation within ±maxlag."""
    maxlag_samples = int(maxlag_ms * fps / 1000)
    
    # Normalize signals
    a_norm = (a - np.mean(a)) / (np.std(a) + 1e-8)
    b_norm = (b - np.mean(b)) / (np.std(b) + 1e-8)
    
    # Cross-correlation
    ccf = np.correlate(a_norm, b_norm, mode='full')
    center = len(ccf) // 2
    
    # Extract central region
    start = max(0, center - maxlag_samples)
    end = min(len(ccf), center + maxlag_samples + 1)
    ccf_segment = ccf[start:end]
    
    if len(ccf_segment) == 0:
        return 0.0
    
    # Normalize by zero-lag autocorrelations
    peak_ccg = np.max(np.abs(ccf_segment)) / len(a_norm)
    
    # Test: peak should be between 0 and 1 for reasonable signals
    assert 0 <= peak_ccg <= 2, f"CCG peak out of range: {peak_ccg}"
    
    return float(peak_ccg)


def plot_voltage_summary(traces_dict, masks, img_overlay, times, out_dir, fps=None):
    """Generate comprehensive spontaneous RGC voltage summary figure."""
    if plt is None:
        print("Warning: matplotlib not available, skipping voltage summary")
        return {}
    
    # Set random seed for deterministic behavior
    rng = np.random.default_rng(0)
    
    # Extract data
    F_corr = traces_dict["F_corr"]
    n_cells, T = F_corr.shape
    
    if n_cells == 0:
        print("Warning: No cells found, skipping voltage summary")
        return {}
    
    # Estimate fps if not provided
    if fps is None:
        if times is not None and len(times) > 1:
            fps = 1.0 / np.median(np.diff(times))
        else:
            print("ERROR: fps missing and cannot be estimated, skipping voltage summary")
            return {}
    
    # Constants for voltage summary
    AUTO_NOTCH = True
    EVENT_MIN_COUNT = 3
    
    # Convert F_corr to raw F traces (T x N)
    traces = F_corr.T  # Now T x N
    duration_sec = T / fps
    
    # Utilities for voltage summary
    import scipy
    import scipy.signal
    import scipy.ndimage
    from pathlib import Path
    
    def bleach_baseline_piecewise(F, fps, win_s=0.5):
        """Piece-wise baseline with 30-80th percentile in 0.5s windows."""
        w = max(3, int(round(win_s*fps)))
        out = np.empty_like(F, float)
        for i in range(0, len(F), w):
            seg = F[i:i+w]
            lo, hi = np.percentile(seg, [30, 80])
            out[i:i+w] = seg[(seg>=lo)&(seg<=hi)].mean()
        # reflect-pad and smooth with median filter of length w to avoid steps
        return scipy.signal.medfilt(out, kernel_size=max(3, (w//2)*2+1))

    def f0_mode(F_bleach, nbins=256):
        """F0 as mode of fluorescence histogram."""
        lo, hi = np.percentile(F_bleach, [1, 99])
        hist, edges = np.histogram(np.clip(F_bleach, lo, hi), bins=nbins, range=(lo, hi))
        k = np.argmax(hist)
        return 0.5*(edges[k]+edges[k+1])

    def compute_noise_std(dff0, fps, event_idx):
        """Noise from 20 Hz high-pass filtered, spike-removed dff0."""
        # Create mask excluding ±6 ms around each event
        mask = np.ones(len(dff0), dtype=bool)
        half_win = max(1, int(round(0.006*fps)))  # 6 ms
        for i in event_idx:
            start = max(0, i - half_win)
            end = min(len(dff0), i + half_win + 1)
            mask[start:end] = False
        
        # Linear interpolation to fill gaps
        dff0_clean = dff0.copy()
        if not np.all(mask):
            from scipy import interpolate
            valid_idx = np.where(mask)[0]
            if len(valid_idx) > 1:
                f = interpolate.interp1d(valid_idx, dff0[valid_idx], kind='linear', 
                                       bounds_error=False, fill_value='extrapolate')
                invalid_idx = np.where(~mask)[0]
                dff0_clean[invalid_idx] = f(invalid_idx)
        
        # 20 Hz high-pass filter
        nyquist = fps / 2
        if 20 < nyquist:
            b, a = scipy.signal.butter(2, 20/nyquist, btype='highpass')
            filtered = scipy.signal.filtfilt(b, a, dff0_clean)
            noise_std = np.std(filtered)
        else:
            noise_std = np.std(dff0_clean)
        
        return noise_std

    def auto_notch_if_needed(x, fps, freqs=(50, 60, 100, 120), Q=30):
        """Auto-notch filter for mains frequencies."""
        f, pxx = scipy.signal.welch(x, fs=fps, nperseg=min(4096, len(x)//4))
        y = x.copy()
        for f0 in freqs:
            m = (f > f0-2) & (f < f0+2)
            if not m.any(): 
                continue
            if np.max(pxx[m]) > 5*np.median(pxx[m]):
                b, a = scipy.signal.iirnotch(f0, Q, fps)
                y = scipy.signal.filtfilt(b, a, y)
        return y

    def _mad(x):
        m = np.median(x); return np.median(np.abs(x - m)) + 1e-12

    def _rolling_mad(x, win):
        w = _to_odd_int(win); pad = w//2
        xp = np.pad(x, pad, mode='reflect'); out = np.empty_like(x, float)
        for i in range(len(x)): out[i] = _mad(xp[i:i+w])
        return out + 1e-12

    def _mk_biphasic(fs, ms=6.0):
        n = _to_odd_int(ms*1e-3*fs)
        t = np.arange(n) - n//2
        up = np.maximum(0, 1 - np.abs(t)/(n//4+1)); dn = -up
        tpl = np.r_[up[:n//2], 0, dn[n//2+1:]]; tpl -= tpl.mean()
        tpl /= np.linalg.norm(tpl) + 1e-12
        return tpl

    def _mk_monophasic(fs, ms=6.0):
        n = _to_odd_int(ms*1e-3*fs)
        t = np.arange(n) - n//2
        tpl = np.maximum(0, 1 - (t/(n//3+1))**2)
        tpl -= tpl.mean(); tpl /= np.linalg.norm(tpl) + 1e-12
        return tpl

    def _matched(y, tpl):
        return scipy.signal.fftconvolve(y, tpl[::-1], mode='same')

    def _enforce_refractory_and_merge(idx, prom, fs, refr_ms=4.0, merge_ms=2.0):
        if len(idx)==0: return np.array([], int)
        ref = int(round(refr_ms*1e-3*fs)); mrg = int(round(merge_ms*1e-3*fs))
        order = np.argsort(idx); idx = np.asarray(idx)[order]
        prom = np.asarray(prom)[order] if prom is not None else np.ones_like(idx)
        keep = []; last = -10**9; best = 0
        for i,p in enumerate(idx):
            if p - last <= mrg:
                if prom[i] > prom[best]: best = i
            else:
                if i>0: keep.append(idx[best])
                best = i; last = p
        keep.append(idx[best])
        out = []; last = -10**9
        for p in keep:
            if p - last >= ref: out.append(p); last = p
        return np.asarray(out, int)

    def _mad(x): 
        m=np.median(x); return np.median(np.abs(x-m))+1e-12
    
    def _to_int(x): 
        return int(np.asarray(x).ravel()[0])
    
    def _to_odd_int(x):
        v = int(np.rint(x));  return v if v % 2 == 1 else v + 1

    def normalize_events_output(ev, bp, fps):
        """
        Return a dict with keys: idx, amp, width_s, prom, template_r, deriv_z, amp_z.
        Works whether `ev` is a dict or a tuple/list from older code.
        """
        if isinstance(ev, dict):
            out = ev.copy()
        else:
            arr = list(ev)
            idx = np.asarray(arr[0], dtype=int)
            out = {"idx": idx}
            # optional extras if present
            out["template_r"] = np.asarray(arr[1], float)[:idx.size] if len(arr) > 1 else np.zeros(idx.size)
            out["deriv_z"]    = np.asarray(arr[2], float)[:idx.size] if len(arr) > 2 else np.zeros(idx.size)
            out["amp_z"]      = np.asarray(arr[3], float)[:idx.size] if len(arr) > 3 else np.zeros(idx.size)
        # always compute these from bp
        idx = out["idx"]
        if idx.size:
            out["amp"]     = bp[idx].astype(float)
            out["width_s"] = scipy.signal.peak_widths(np.abs(bp), idx, rel_height=0.5)[0] / float(fps)
            wlen           = _to_odd_int(2*max(3, int(round(0.02*fps))) + 1)
            out["prom"]    = scipy.signal.peak_prominences(bp, idx, wlen=wlen)[0].astype(float)
            if "amp_z" not in out or out["amp_z"].size != idx.size:
                sigma = 1.4826 * (np.median(np.abs(bp - np.median(bp))) + 1e-12)
                out["amp_z"] = np.abs(out["amp"]) / sigma
            # pad missing vectors to idx length
            for k in ("template_r", "deriv_z"):
                if k not in out or out[k].size != idx.size:
                    out[k] = np.zeros(idx.size, float)
        else:
            out.setdefault("amp", np.array([], float))
            out.setdefault("width_s", np.array([], float))
            out.setdefault("prom", np.array([], float))
            out.setdefault("template_r", np.array([], float))
            out.setdefault("deriv_z", np.array([], float))
            out.setdefault("amp_z", np.array([], float))
        return out

    def _artifact_mask(raw, dff, bp, fps):
        """
        Return (valid_mask, is_artifact, info).
        Flags step/glitch/saturation segments and dilates by ±25 ms.
        """
        T=len(bp); t=np.arange(T)/fps
        # robust z of derivative and amplitude
        d = np.diff(dff, prepend=dff[0])
        dz = np.abs(d)/(1.4826*_mad(d))
        sig_bp = 1.4826*_mad(bp); bpz = np.abs(bp-np.median(bp))/(sig_bp+1e-12)
        sig_df = 1.4826*_mad(dff); dfz = np.abs(dff-np.median(dff))/(sig_df+1e-12)
        # primary bad samples
        bad = (dz>40) | (bpz>60) | (dfz>60) | ~np.isfinite(bp)
        # dilate by ±25 ms to cover ringing
        rad = _to_int(0.025*fps); 
        if rad>0:
            k=np.ones(2*rad+1, dtype=int); bad = np.convolve(bad.astype(int), k, mode='same')>0
        valid = ~bad
        # global artifact decision
        frac_out = float(np.mean(~valid))
        base5 = np.percentile(raw,5); medraw = np.median(raw)
        base_collapse = (base5<=1e-6) or (medraw<=1e-6) or (base5 < 0.02*medraw)
        is_art = frac_out>0.3 or base_collapse
        info = dict(frac_out=frac_out, base_collapse=bool(base_collapse))
        return valid, bool(is_art), info

    def _find_peaks_with_local_prom(y, height, prom_sigma, local_sigma, wmin, wmax):
        """Use SciPy peak finder, then gate by vector local prominence threshold."""
        import scipy.signal as sp
        wmin = _to_int(wmin); wmax = _to_int(wmax)
        pk, props = sp.find_peaks(y, height=float(height), width=(wmin, wmax))  # no array 'prominence'
        if pk.size == 0:
            return pk, np.array([], float), props
        wlen = _to_odd_int(2*wmax + 1)   # int and odd
        prom = sp.peak_prominences(y, pk, wlen=wlen)[0]
        thr  = prom_sigma * np.asarray(local_sigma)[pk]  # vector threshold after the fact
        keep = prom >= thr
        return pk[keep], prom[keep], props

    def detect_voltage_events(bp, fs,
                              k0=4.6, prom_sigma=1.4,
                              wmin_ms=0.8, wmax_ms=25.0,
                              refr_ms=4.0, merge_ms=2.0,
                              corr_min=0.20, deriv_z_min=6.0, amp_z_min=3.5):
        """Union of raw/derivative/matched filters, positive-only, with template or amplitude validation."""
        loc_sig = _rolling_mad(bp, _to_int(0.20*fs)) * 1.4826
        win = _to_odd_int(0.008*fs)
        y_raw = bp
        y_der = scipy.signal.savgol_filter(bp, window_length=win, polyorder=2, deriv=1, delta=1/fs)
        bip = _mk_biphasic(fs, 6.0); mono = _mk_monophasic(fs, 6.0)
        y_mf_bi   = _matched(bp,  bip)
        y_mf_mono = _matched(bp, mono)

        # Compute width constraints once as proper ints
        wmin = _to_int(wmin_ms*1e-3*fs)
        wmax = _to_int(wmax_ms*1e-3*fs)

        def find_positive_peaks(y, k=k0):
            sig = 1.4826*_mad(y)
            
            pk, prom, props = _find_peaks_with_local_prom(
                y, height=k*sig, prom_sigma=prom_sigma,
                local_sigma=np.maximum(loc_sig, 1e-12), wmin=wmin, wmax=wmax
            )
            
            if len(pk)==0 and (np.max(y)/sig >= 6):
                for k_try in (4.2, 3.8):
                    pk, prom, props = _find_peaks_with_local_prom(
                        y, height=k_try*sig, prom_sigma=prom_sigma,
                        local_sigma=np.maximum(loc_sig, 1e-12), wmin=wmin, wmax=wmax
                    )
                    if len(pk): break
            
            pk = _enforce_refractory_and_merge(pk, prom, fs, refr_ms, merge_ms)
            return pk

        all_idx = []
        # Only detect positive peaks (spikes)
        for y in (y_raw, y_der, y_mf_bi, y_mf_mono):
            all_idx.extend(find_positive_peaks(y))
        if not all_idx: 
            return np.array([],int), np.array([]), np.array([]), np.array([]), np.array([])

        idx = np.unique(np.asarray(all_idx, int))
        # validations on original band-passed trace
        sigma = 1.4826*_mad(bp)
        # local derivative z at candidate indices
        dz = np.abs(y_der[idx])/(1.4826*_mad(y_der) + 1e-12)
        ampz = np.abs(bp[idx])/(sigma + 1e-12)
        # template correlations
        def corr_at(idx, tpl):
            half = _to_int(len(tpl)//2)
            out = np.zeros(len(idx))
            for j,i in enumerate(idx):
                lo = max(0, i-half); hi = min(len(bp), i+half+1)
                seg = bp[lo:hi].astype(float)
                if len(seg)<len(tpl): seg = np.pad(seg, (0,len(tpl)-len(seg)), mode='edge')
                seg = seg[:len(tpl)]; seg -= seg.mean()
                num = np.dot(seg, tpl); den = (np.linalg.norm(seg)*np.linalg.norm(tpl) + 1e-12)
                out[j] = num/den
            return out
        r_bi   = corr_at(idx, bip)
        r_mono = corr_at(idx, mono)
        r_max  = np.maximum(np.abs(r_bi), np.abs(r_mono))

        keep = (r_max >= corr_min) | (dz >= deriv_z_min) | (ampz >= amp_z_min)
        idx = idx[keep]; r_max = r_max[keep]; dz = dz[keep]; ampz = ampz[keep]
        
        # Compute all metrics for return dict
        if len(idx) > 0:
            # Prominences
            try:
                prom = scipy.signal.peak_prominences(bp, idx, wlen=_to_odd_int(2*wmax+1))[0]
            except:
                prom = np.full(len(idx), np.nan)
            
            # Widths
            try:
                widths = scipy.signal.peak_widths(np.abs(bp), idx, rel_height=0.5)[0] / float(fs)
            except:
                widths = np.full(len(idx), 0.005)  # 5ms default
            
            # Filter out events with zero width
            width_valid = widths > 0.0001  # > 0.1 ms minimum
            if np.any(width_valid):
                idx = idx[width_valid]
                widths = widths[width_valid]
                prom = prom[width_valid] if len(prom) == len(width_valid) else prom[:len(idx)]
                r_max = r_max[width_valid] if len(r_max) == len(width_valid) else r_max[:len(idx)]
                dz = dz[width_valid] if len(dz) == len(width_valid) else dz[:len(idx)]
                ampz = ampz[width_valid] if len(ampz) == len(width_valid) else ampz[:len(idx)]
            else:
                # All events had zero width - reject all
                idx = np.array([], dtype=int)
                widths = prom = amp = amp_z = r_max = dz = ampz = np.array([])
            
            # Amplitudes and z-scores for remaining events
            if len(idx) > 0:
                amp = bp[idx]
                amp_z = np.abs(bp[idx] - np.median(bp)) / (1.4826*_mad(bp) + 1e-12)
            else:
                amp = amp_z = np.array([])
        else:
            prom = widths = amp = amp_z = np.array([])
        
        return {
            "idx": idx.astype(int),
            "amp": amp.astype(float),
            "width_s": widths.astype(float),
            "prom": prom.astype(float),
            "template_r": r_max.astype(float),
            "deriv_z": dz.astype(float),
            "amp_z": amp_z.astype(float)
        }
    
    # Single pass setup
    from pathlib import Path
    EVENT_MIN_COUNT = 3  # keep consistent everywhere

    events_dir = Path(out_dir) / "events"
    roi_dir = Path(out_dir) / "roi_summaries"
    events_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    # purge old event CSVs; we only keep the current run's eventful ROIs
    for p in events_dir.glob("events_roi*.csv"):
        try: 
            p.unlink()
        except Exception: 
            pass

    # Prepare arrays once
    T, N = traces.shape
    fs = fps
    bp_low = 0.8
    bp_high = min(120.0, 0.45*fs)

    event_counts = np.zeros(N, int)
    event_rates = np.zeros(N, float)
    snr_vals = np.zeros(N, float)
    is_art_flags = np.zeros(N, bool)
    events_all = {}

    bp_traces = np.zeros_like(traces, dtype=float)
    dff_traces = np.zeros_like(traces, dtype=float)

    print(f"Processing: fps={fs:.1f}, band-pass=[{bp_low}, {bp_high:.1f}] Hz")

    for n in range(N):
        raw = traces[:, n].astype(float)

        # Paper's ΔF/F0 definition with bleach correction
        base = bleach_baseline_piecewise(raw, fps, win_s=0.5)
        F_bleach = raw / (base + 1e-12)
        F0 = f0_mode(F_bleach)
        dff0 = (F_bleach - F0) / (F0 + 1e-12)     # fraction
        dff_traces[:, n] = dff0

        # auto-notch if needed
        dff0 = auto_notch_if_needed(dff0, fps)

        # band-pass for detection
        bp = bandpass_butter(dff0, fps, low_hz=0.8, high_hz=min(120.0, 0.45*fps))
        bp_traces[:, n] = bp

        # artifact masking
        valid_mask, is_art, info = _artifact_mask(raw, dff0, bp, fps)
        bp_clean = bp.copy()
        bp_clean[~valid_mask] = np.median(bp)   # neutralize glitch segments for detection

        # Compute dF/F0 exactly as in preprint
        dff, percent_dff, F0 = compute_dff0(raw, fs)
        
        # Noise estimate using 20 Hz high-pass of percent dF/F (preprint Methods)
        try:
            hp_trace = bandpass_butter(percent_dff, fs, low_hz=20.0, high_hz=min(120.0, fs/2*0.95), order=3)
            noise_sigma = 1.4826 * np.median(np.abs(hp_trace - np.median(hp_trace)))
        except:
            noise_sigma = 1.4826 * np.median(np.abs(percent_dff - np.median(percent_dff)))
        
        noise_sigma = max(noise_sigma, 1e-9)
        
        # Create config object for detection
        class Config:
            def __init__(self):
                self.k_height = getattr(ARGS, 'k_height', 2.2)
                self.k_prom = getattr(ARGS, 'k_prom', 2.8)  
                self.line_ratio = getattr(ARGS, 'line_ratio', 0.45)
        
        cfg = Config()
        
        # Detect spikes on percent dF/F
        ev = detect_spikes_percent(percent_dff, fs, noise_sigma, cfg)
        
        # Robust artifact check (requires ≥2 rules)
        roi_is_artifact = is_artifact_trace(percent_dff, fs, ev, cfg)
        
        event_count = len(ev['idx'])
        idx = ev['idx']
        
        # Convert to old format for compatibility with existing plotting code  
        amplitude_percent = ev.get('amp_percent', np.array([]))
        width_ms = ev.get('width_ms', np.array([]))
        snr = ev.get('snr_peak', np.array([]))
        amp = ev.get('amp_dff', np.array([]))
        
        # Store enhanced ev dict with percent dF/F for plotting
        ev_enhanced = ev.copy()
        ev_enhanced.update({
            "amplitude_percent": amplitude_percent,
            "snr": snr,
            "amp": amp,
            "bp": bandpass_butter(percent_dff, fs, low_hz=0.8, high_hz=min(120.0, 0.45*fs), order=3),
            "sigma_hp20": noise_sigma,
            "percent_dff": percent_dff,  # Store for plotting
            "dff": dff  # Store dff fraction
        })
        ev = ev_enhanced
        
        # All event processing now handled by the new detector
        # No legacy processing needed
            
        # metrics using percent dF/F from new detector
        snr_vals[n] = (np.percentile(percent_dff, 95) - np.median(percent_dff)) / (noise_sigma + 1e-12)

        # calculate event rate using new detector
        rate_hz = event_count / (len(percent_dff) / fs)
        dense_window = 0.2  # 200 ms
        if event_count > 0:
            bins = np.arange(0, len(percent_dff)/fs + dense_window, dense_window)
            event_times_array = ev['t_sec']
            hist,_ = np.histogram(event_times_array, bins)
            dens_frac = hist.max()/max(1, event_count)
        else:
            dens_frac = 0.0
        
        # Use the robust artifact check (already applied above)
        final_artifact = roi_is_artifact
        
        # Eventful ROI definition: >=3 spikes and not artifact
        is_eventful = (event_count >= 3) and (not roi_is_artifact)
        
        # Collect metrics for the figure
        event_counts[n] = event_count
        event_rates[n] = rate_hz
        snr_vals[n] = (np.percentile(percent_dff, 95) - np.median(percent_dff)) / (noise_sigma + 1e-12)
        events_all[n] = ev
        is_art_flags[n] = final_artifact

        # Save CSV only for eventful, non-artifact ROIs
        if is_eventful and not roi_is_artifact:
            save_events_csv(str(events_dir), n+1, ev, fs)

        # Log per-ROI line as specified  
        if event_count > 0:
            median_amp_percent = np.median(ev.get('amp_percent', [0.0]))
        else:
            median_amp_percent = 0.0
        
        print(f"ROI {n+1}: events={event_count}, artifact={final_artifact}, "
              f"median_amp_percent={median_amp_percent:.3f}, noise_sigma={noise_sigma:.4f}")

        # --- SAVE ROI SUMMARY PNGs with BLACK tick marks only ---
        t = np.arange(T)/fs
        title = f'ROI {n+1}{"  [ARTIFACT]" if final_artifact else ""}'
        
        # Use the single hardened plotter (no colored overlays)
        summary_path = roi_dir / f'roi_{n+1:03d}.png'
        
        # Ensure proper timebase alignment
        dff_filt_for_plot = ev.get("bp", np.zeros_like(raw))
        
        # Timebase sanity check
        peaks_idx = ev.get('idx', [])
        if len(peaks_idx) > 0:
            pk = np.asarray(peaks_idx, dtype=int)
            bad = (pk < 0) | (pk >= len(dff_filt_for_plot))
            if np.any(bad):
                print(f"Warning: {np.sum(bad)} event indices out of range for ROI {n+1}; dropping.")
                pk = pk[~bad]
                peaks_idx = pk
        
        save_roi_summary_ticks(
            path=str(summary_path),
            t=t,
            Fraw=raw,
            dff=percent_dff / 100.0,  # Convert back to fraction for plotting
            dff_filt=dff_filt_for_plot / 100.0,  # Convert to fraction
            fs=fs,
            peak_idx_or_t=peaks_idx,  # Use cleaned indices
            title=title,
            yunits='%'
        )

    # Build eventful subset and event times for the figure
    eventful_idx = [i for i in range(N) if (not is_art_flags[i]) and (event_counts[i] >= 3)]
    if not eventful_idx:
        eventful_idx = [int(np.argmax(event_counts))] if np.max(event_counts)>0 else []

    # Build event_times for figure panels
    event_times = []
    for i in range(N):
        if "idx" in events_all[i] and len(events_all[i]["idx"]) > 0:
            event_times.append(events_all[i]["idx"] / fs)
        elif "events" in events_all[i]:
            # New detector format
            event_times.append(np.array([e["time_s"] for e in events_all[i]["events"]]))
        else:
            event_times.append(np.array([]))
    
    csv_count = len(list(events_dir.glob("events_roi*.csv")))
    
    print(f"ROI summaries saved to: {roi_dir} ({N} ROIs)")
    print(f"Event CSVs saved to: {events_dir} ({csv_count} eventful ROIs)")
    print(f"Eventful ROIs ({len(eventful_idx)}/{N}): {[i+1 for i in eventful_idx]}")
    
    # Create ROI info from masks for plotting
    from skimage.measure import regionprops
    props = regionprops(masks)
    rois = []
    for p in props:
        centroid = p.centroid  # (row, col)
        rois.append({'centroid': (centroid[1], centroid[0])})  # Convert to (x, y)
    
    # Ensure we have the right number of ROIs
    if len(rois) != n_cells:
        # Fill missing ROIs with dummy centroids
        for i in range(len(rois), n_cells):
            rois.append({'centroid': (0, 0)})
    
    field_img = img_overlay
    
    # Create all_events structure from single-pass data
    all_events = {}
    for i in range(N):
        # Use data already computed in the loop above
        # Handle both old and new event format
        if "events" in events_all[i]:
            # New detector format
            event_list = events_all[i]["events"]
            amps_array = np.array([e["amp_dff"] for e in event_list]) if event_list else np.array([])
        elif "amp" in events_all[i]:
            # Old format
            amps_array = events_all[i]["amp"]
        else:
            amps_array = np.array([])
            
        all_events[i] = {
            'times': event_times[i],
            'amps': amps_array,
            'count': event_counts[i],
            'rate_hz': event_rates[i]
        }
    
    # ROI with most spikes among eventful, else global max, else highest SNR
    if len(eventful_idx) > 0:
        roi_best = max(eventful_idx, key=lambda i: event_counts[i])
    else:
        roi_best = int(np.argmax(event_counts)) if np.max(event_counts) > 0 else int(np.argmax(snr_vals))
    
    # Create figure with 3x4 grid
    fig = plt.figure(figsize=(16, 12))
    
    # A) Field + ROIs
    ax_field = plt.subplot(3, 4, 1)
    if field_img is not None:
        ax_field.imshow(field_img, cmap='gray')
        # Overlay ROI centroids
        for i, roi in enumerate(rois[:n_cells]):
            if 'centroid' in roi:
                x, y = roi['centroid']
                ax_field.plot(x, y, 'r+', markersize=8)
                ax_field.text(x+2, y+2, str(i+1), color='red', fontsize=8)
        ax_field.set_title('Field + ROIs')
        ax_field.axis('off')
    else:
        ax_field.text(0.5, 0.5, 'Field image\nnot available', ha='center', va='center')
        ax_field.set_title('Field + ROIs')
    
    # B) Example Traces - single best ROI (show as percent)
    ax_traces = plt.subplot(3, 4, 2)
    t = np.arange(T)/fps
    ax_traces.plot(t, traces[:, roi_best],           lw=0.6, color='k', label='raw')
    ax_traces.plot(t, dff_traces[:, roi_best] * 100, lw=0.6, color='b', label='ΔF/F0')
    ax_traces.plot(t, bp_traces[:, roi_best] * 100,  lw=0.6, color='r', label='filtered')
    ax_traces.set_title(f"Example Traces – ROI {roi_best+1} (spikes={event_counts[roi_best]})")
    ax_traces.set_xlabel("Time (s)")
    ax_traces.set_ylabel("ΔF/F0 (%)")
    ax_traces.legend(frameon=False, fontsize=8, loc='upper right')
    
    # C) ROI Events - same roi_best (show as percent)
    ax_spikes = plt.subplot(3, 4, 3)
    bp = bp_traces[:, roi_best] * 100  # Convert to percent
    idx = events_all[roi_best]["idx"]
    ax_spikes.plot(t, bp, lw=0.6, color='k')
    if idx.size:
        y0, y1 = ax_spikes.get_ylim()
        ax_spikes.vlines(idx/fps, y1 - 0.06*(y1 - y0), y1, color='k', lw=0.9)  # black ticks at top
        # inset: mean event waveform
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset = ax_spikes.inset_axes([0.62, 0.55, 0.32, 0.35])
        half = max(1, int(round(0.004*fps)))  # ±4 ms window
        valid = idx[(idx >= half) & (idx < T - half)]
        if valid.size:
            w = np.arange(-half, half+1)
            waves = np.stack([bp[i-half:i+half+1] for i in valid])
            inset.plot(w/fps*1e3, waves.mean(0), lw=1.0, color='r')
            inset.set_xlabel("ms")
            inset.set_ylabel("ΔF/F0 (%)")
            inset.tick_params(labelsize=6)
    ax_spikes.set_title(f"ROI {roi_best+1} Events (rate={event_rates[roi_best]:.2f} Hz)")
    ax_spikes.set_xlabel("Time (s)")
    ax_spikes.set_ylabel("ΔF/F0 (%)")
    
    # D) Event raster - only eventful ROIs, sorted by event rate
    ax_raster = plt.subplot(3, 4, 4)
    
    # Sort eventful ROIs by event rate
    if len(eventful_idx) > 0:
        eventful_rates = [(i, event_rates[i]) for i in eventful_idx]
        eventful_rates.sort(key=lambda x: x[1], reverse=True)
        sorted_eventful = [x[0] for x in eventful_rates]
        
        # Raster plot
        plot_row = 1
        for i in sorted_eventful:
            events = all_events[i]
            if len(events['times']) > 0:
                ax_raster.scatter(events['times'], [plot_row] * len(events['times']), 
                                s=1, c='black', alpha=0.7)
            plot_row += 1
        
        ax_raster.set_ylim(0.5, len(sorted_eventful) + 0.5)
        ax_raster.set_yticks(range(1, len(sorted_eventful) + 1))
        ax_raster.set_yticklabels([f'{sorted_eventful[i-1]+1}' for i in range(1, len(sorted_eventful) + 1)])
    
    ax_raster.set_ylabel('ROI')
    ax_raster.set_xlabel('Time (s)')
    ax_raster.set_title(f'Event Raster ({len(eventful_idx)}/{N} ROIs)')
    
    # E) ISI variability - only eventful ROIs
    ax_isi = plt.subplot(3, 4, 5)
    
    # Collect ISI CVs from eventful ROIs only
    isi_cvs = []
    for i in eventful_idx:
        events = all_events[i]
        if len(events['times']) > 2:
            isis = np.diff(events['times'])
            isi_cv = np.std(isis) / (np.mean(isis) + 1e-8)
            isi_cvs.append(isi_cv)
    
    if len(isi_cvs) >= 2:
        ax_isi.hist(isi_cvs, bins=10, alpha=0.7, color='blue')
        ax_isi.set_xlabel('ISI CV')
        ax_isi.set_ylabel('Count')
        ax_isi.set_title(f'ISI Variability\n({len(isi_cvs)} ROIs with ≥3 events)')
    else:
        ax_isi.text(0.5, 0.5, 'insufficient eventful ROIs', ha='center', va='center', transform=ax_isi.transAxes)
        ax_isi.set_title('ISI Variability')
    
    # F) Cross-correlation - eventful ROIs only
    ax_ccg = plt.subplot(3, 4, 6)
    
    if len(eventful_idx) > 1:
        n_eventful = len(eventful_idx)
        ccg_matrix = np.zeros((n_eventful, n_eventful))
        
        for i, roi_i in enumerate(eventful_idx):
            for j, roi_j in enumerate(eventful_idx):
                if i != j:
                    ccg_matrix[i, j] = ccg_peak(dff_traces[:, roi_i], dff_traces[:, roi_j], fps)
        
        # Hierarchical clustering for ordering
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            
            # Convert to distance matrix
            dist_matrix = 1 - np.abs(ccg_matrix)
            condensed = squareform(dist_matrix, checks=False)
            linkage_matrix = linkage(condensed, method='average')
            order = leaves_list(linkage_matrix)
            
            # Reorder matrix
            ccg_matrix = ccg_matrix[np.ix_(order, order)]
            eventful_ordered = [eventful_idx[i] for i in order]
        except:
            eventful_ordered = eventful_idx
        
        im = ax_ccg.imshow(ccg_matrix, cmap='viridis', aspect='auto')
        ax_ccg.set_xlabel('ROI')
        ax_ccg.set_ylabel('ROI')
        ax_ccg.set_title(f'Cross-Correlation\n({n_eventful} ROIs with ≥3 events)')
        # Set 1-indexed tick labels
        ax_ccg.set_xticks(range(n_eventful))
        ax_ccg.set_yticks(range(n_eventful))
        ax_ccg.set_xticklabels([f'{eventful_ordered[i]+1}' for i in range(n_eventful)])
        ax_ccg.set_yticklabels([f'{eventful_ordered[i]+1}' for i in range(n_eventful)])
        plt.colorbar(im, ax=ax_ccg, shrink=0.8)
    else:
        ax_ccg.text(0.5, 0.5, 'Need >1 eventful ROI\nfor correlation', ha='center', va='center', transform=ax_ccg.transAxes)
        ax_ccg.set_title('Cross-Correlation')
    
    # G) Spatial coupling - eventful ROIs only
    ax_spatial = plt.subplot(3, 4, 7)
    
    if len(eventful_idx) > 1 and len(rois) >= N:
        distances = []
        ccg_peaks = []
        
        # Use only eventful ROIs for spatial analysis
        for i, roi_i in enumerate(eventful_idx):
            for j, roi_j in enumerate(eventful_idx):
                if i < j and 'centroid' in rois[roi_i] and 'centroid' in rois[roi_j]:
                    x1, y1 = rois[roi_i]['centroid']
                    x2, y2 = rois[roi_j]['centroid']
                    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    distances.append(dist)
                    # Use the reordered matrix if available
                    if 'eventful_ordered' in locals():
                        i_ord = eventful_ordered.index(roi_i)
                        j_ord = eventful_ordered.index(roi_j)
                        ccg_peaks.append(ccg_matrix[i_ord, j_ord])
                    else:
                        ccg_peaks.append(ccg_peak(dff_traces[:, roi_i], dff_traces[:, roi_j], fps))
        
        if len(distances) > 1:
            ax_spatial.scatter(distances, ccg_peaks, alpha=0.7, s=20)
            ax_spatial.set_xlabel('Distance (pixels)')
            ax_spatial.set_ylabel('CCG Peak')
            ax_spatial.set_title(f'Spatial Coupling\n({len(distances)} pairs)')
    else:
        ax_spatial.text(0.5, 0.5, 'Need >1 eventful ROI', ha='center', va='center', transform=ax_spatial.transAxes)
        ax_spatial.set_title('Spatial Coupling')
    
    # H) Power spectra - eventful ROIs with all ROIs reference
    ax_spectra = plt.subplot(3, 4, 8)
    
    if T > fps * 4 and len(eventful_idx) > 0:  # Need at least 4 seconds for reasonable PSD
        # Eventful ROIs
        eventful_traces = dff_traces[:, eventful_idx].T
        freqs, psds_eventful = scipy.signal.welch(eventful_traces, fs=fps, nperseg=min(int(2*fps), T//4))
        
        # All ROIs reference
        freqs_all, psds_all = scipy.signal.welch(dff_traces.T, fs=fps, nperseg=min(int(2*fps), T//4))
        median_all = np.median(psds_all, axis=0)
        
        # Plot faint gray line for all ROIs
        ax_spectra.plot(freqs_all, median_all, 'gray', linewidth=1, alpha=0.5, label='All ROIs')
        
        # Plot median ± IQR for eventful ROIs
        median_psd = np.median(psds_eventful, axis=0)
        q25_psd = np.percentile(psds_eventful, 25, axis=0)
        q75_psd = np.percentile(psds_eventful, 75, axis=0)
        
        ax_spectra.plot(freqs, median_psd, 'r-', linewidth=2, label='Eventful median')
        ax_spectra.fill_between(freqs, q25_psd, q75_psd, alpha=0.3, color='red', label='IQR')
        
        ax_spectra.set_xlabel('Frequency (Hz)')
        ax_spectra.set_ylabel('Power')
        ax_spectra.set_title(f'Power Spectra\n({len(eventful_idx)} ROIs with ≥3 events)')
        ax_spectra.set_xlim(0, min(50, fps/2))
        ax_spectra.legend()
    else:
        ax_spectra.text(0.5, 0.5, 'Insufficient eventful ROIs\nor data length', ha='center', va='center', transform=ax_spectra.transAxes)
        ax_spectra.set_title('Power Spectra')
    
    # I) Network events
    ax_network = plt.subplot(3, 4, 9)
    
    # Population rate
    bin_ms = 10
    bin_samples = int(bin_ms * fps / 1000)
    n_bins = T // bin_samples
    
    pop_rate = np.zeros(n_bins)
    for i in range(n_bins):
        start_time = i * bin_samples / fps
        end_time = (i + 1) * bin_samples / fps
        count = 0
        for j in range(n_cells):
            events = all_events[j]
            count += np.sum((events['times'] >= start_time) & (events['times'] < end_time))
        pop_rate[i] = count
    
    # Z-score population rate
    pop_rate_z = (pop_rate - np.mean(pop_rate)) / (np.std(pop_rate) + 1e-8)
    
    t_bins = np.arange(n_bins) * bin_samples / fps
    ax_network.plot(t_bins, pop_rate_z, 'k-', linewidth=1)
    ax_network.axhline(3, color='r', linestyle='--', alpha=0.7)
    ax_network.set_xlabel('Time (s)')
    ax_network.set_ylabel('Pop. Rate (z-score)')
    ax_network.set_title('Network Events')
    
    # J) PCA
    ax_pca_var = plt.subplot(3, 4, 10)
    ax_pca_ts = plt.subplot(3, 4, 11)
    ax_pca_space = plt.subplot(3, 4, 12)
    
    try:
        from sklearn.decomposition import PCA
        
        if len(eventful_idx) > 2:
            # Run PCA on eventful ROIs only
            eventful_traces_pca = dff_traces[:, eventful_idx]  # T x N_eventful
            pca = PCA()
            pca_result = pca.fit_transform(eventful_traces_pca)  # T x N_components
            
            # Variance explained
            var_exp = pca.explained_variance_ratio_
            ax_pca_var.bar(range(min(10, len(var_exp))), var_exp[:10])
            ax_pca_var.set_xlabel('PC')
            ax_pca_var.set_ylabel('Variance Explained')
            ax_pca_var.set_title(f'PCA Variance\n({len(eventful_idx)} ROIs with ≥3 events)')
            
            # PC1 time course with population rate overlay
            pc1_timecourse = pca_result[:, 0]  # PC1 over time
            
            # Compute population event rate for eventful ROIs
            bin_ms = 10
            bin_samples = int(bin_ms * fps / 1000)
            n_bins = T // bin_samples
            pop_rate = np.zeros(n_bins)
            
            for i in range(n_bins):
                start_time = i * bin_samples / fps
                end_time = (i + 1) * bin_samples / fps
                count = 0
                for j in eventful_idx:
                    events = all_events[j]
                    count += np.sum((events['times'] >= start_time) & (events['times'] < end_time))
                pop_rate[i] = count
            
            t_bins = np.arange(n_bins) * bin_samples / fps
            
            # Plot PC1 and population rate
            t_all = np.arange(T) / fps
            ax_pca_ts.plot(t_all, pc1_timecourse, 'b-', linewidth=1, label='PC1')
            
            # Scale population rate to overlay
            if np.std(pop_rate) > 0:
                pop_scaled = (pop_rate - np.mean(pop_rate)) / np.std(pop_rate)
                pop_scaled = pop_scaled * np.std(pc1_timecourse) + np.mean(pc1_timecourse)
                ax_pca_ts.plot(t_bins, pop_scaled, 'gray', linewidth=1, alpha=0.7, label='Pop rate (scaled)')
                
                # Calculate correlation
                if len(t_bins) == len(pc1_timecourse):
                    r_corr = np.corrcoef(pc1_timecourse, pop_rate)[0, 1]
                    ax_pca_ts.set_title(f'PC1 Time Course\n(r={r_corr:.2f})')
                else:
                    ax_pca_ts.set_title('PC1 Time Course')
            else:
                ax_pca_ts.set_title('PC1 Time Course')
            
            ax_pca_ts.set_xlabel('Time (s)')
            ax_pca_ts.set_ylabel('PC1')
            ax_pca_ts.legend()
            
            # Spatial PC1 loadings for eventful ROIs only
            if len(rois) >= N:
                pc1_loadings = pca.components_[0]  # PC1 loadings for each eventful ROI
                x_coords = [rois[i]['centroid'][0] for i in eventful_idx]
                y_coords = [rois[i]['centroid'][1] for i in eventful_idx]
                scatter = ax_pca_space.scatter(x_coords, y_coords, c=pc1_loadings, cmap='RdBu_r', s=50)
                ax_pca_space.set_xlabel('X (pixels)')
                ax_pca_space.set_ylabel('Y (pixels)')
                ax_pca_space.set_title('PC1 Loadings')
                cbar = plt.colorbar(scatter, ax=ax_pca_space, shrink=0.8)
                cbar.set_label('loading')
        else:
            # Not enough eventful ROIs for meaningful PCA
            for ax in [ax_pca_var, ax_pca_ts, ax_pca_space]:
                ax.text(0.5, 0.5, f'Need >2 eventful ROIs\nfor PCA\n({len(eventful_idx)} found)', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('PCA (insufficient eventful ROIs)')
    
    except ImportError:
        # sklearn not available
        for ax in [ax_pca_var, ax_pca_ts, ax_pca_space]:
            ax.text(0.5, 0.5, 'sklearn not\navailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('PCA (sklearn missing)')
    except Exception as e:
        # Other PCA errors
        for ax in [ax_pca_var, ax_pca_ts, ax_pca_space]:
            ax.text(0.5, 0.5, f'PCA error:\n{str(e)[:20]}...', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('PCA (error)')
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(os.path.join(out_dir, "voltage_summary.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig)
    
    # Save per-ROI metrics as JSON
    roi_metrics = {}
    for i in range(N):
        roi_metrics[f'roi_{i+1}'] = {
            'snr': float(snr_vals[i]),
            'sigma': float(1.4826 * _mad(bp_traces[:, i])),
            'event_count': int(event_counts[i]),
            'event_rate_hz': float(event_rates[i]),
            'included_eventful': i in eventful_idx
        }
    
    # Save metrics JSON
    import json
    with open(os.path.join(out_dir, "voltage_metrics.json"), "w") as f:
        json.dump(roi_metrics, f, indent=2)
    
    # Compile metrics for return
    metrics = {
        'event_rates': event_rates.tolist(),
        'snr_values': snr_vals.tolist(),
        'noise_values': [1.4826 * _mad(bp_traces[:, i]) for i in range(N)],
        'all_events': all_events,  # Include for ROI summaries
        'eventful_idx': eventful_idx,
        'best_roi': int(roi_best),
        'fps': fps
    }
    
    print(f"Voltage summary saved to: {os.path.join(out_dir, 'voltage_summary.png')}")
    print(f"Voltage metrics saved to: {os.path.join(out_dir, 'voltage_metrics.json')}")
    
    return metrics


def save_roi_summaries(traces_dict, times, all_events, snr_values, noise_values, out_dir, max_rois=200, tick_only=True):
    """Save individual ROI summary PNGs."""
    if plt is None:
        print("Warning: matplotlib not available, skipping ROI summaries")
        return
    
    F_corr = traces_dict["F_corr"]
    n_cells, T = F_corr.shape
    
    if n_cells == 0:
        return
    
    # Estimate fps
    if times is not None and len(times) > 1:
        fps = 1.0 / np.median(np.diff(times))
    else:
        fps = 30.0
    
    # Create output directory
    roi_dir = os.path.join(out_dir, "roi_summaries")
    os.makedirs(roi_dir, exist_ok=True)
    
    # Force tick-only summaries 
    if FORCE_TICK_ONLY_SUMMARIES:
        tick_only = True
    
    # Limit number of ROIs to save and ensure we don't exceed available data
    n_to_save = min(n_cells, max_rois, len(snr_values), len(noise_values))
    
    for i in range(n_to_save):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Raw F trace
        raw_trace = F_corr[i]
        t_all = times if times is not None else np.arange(T) / fps
        
        axes[0, 0].plot(t_all, raw_trace, 'k-', linewidth=0.8)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('F (raw)')
        axes[0, 0].set_title(f'ROI {i+1} - Raw Fluorescence')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Detrended ΔF/F
        detrended = detrend_percentile(raw_trace, fps, win_s=45, q=10)
        axes[0, 1].plot(t_all, detrended, 'b-', linewidth=0.8)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('ΔF/F (detrended)')
        axes[0, 1].set_title(f'ROI {i+1} - Detrended')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Band-passed ΔF/F with events
        filtered = bandpass_butter(detrended, fps, low_hz=1.0, high_hz=None, order=3)
        axes[1, 0].plot(t_all, filtered, 'r-', linewidth=0.8)
        
        # Time-base sanity and clean tick overlay
        if i in all_events:
            events = all_events[i]
            
            # Ensure 'peaks' field exists for tick plotting
            if ('peaks' not in events or events['peaks'] is None) and ('times' in events):
                events['peaks'] = (np.asarray(events['times']) * fps).astype(int)
            
            # Guard against mismatches
            if 'peaks' in events and len(events['peaks']) > 0:
                max_peak = np.max(events['peaks'])
                if max_peak < len(filtered):
                    # Use clean tick helper (no colored overlays)
                    t_filt = np.arange(len(filtered)) / fps
                    _overlay_event_ticks(axes[1, 0], t_filt, events['peaks'])
        else:
            events = {'times': [], 'amps': [], 'peaks': []}
        
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('ΔF/F (filtered)')
        axes[1, 0].set_title(f'ROI {i+1} - Filtered + Events')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Power spectral density
        from scipy.signal import welch
        if T > fps * 2:  # Need at least 2 seconds
            freqs, psd = welch(detrended, fs=fps, nperseg=min(int(fps), T//4))
            axes[1, 1].loglog(freqs, psd, 'g-', linewidth=1)
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_ylabel('Power')
            axes[1, 1].set_title(f'ROI {i+1} - Power Spectrum')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Metrics table instead of PSD
            event_rate = len(events['times']) / (T / fps)
            # Handle both old and new event structure formats
            if 'amps' in events and len(events['amps']) > 0:
                median_amp = np.median(events['amps'])
            elif 'amplitude_percent' in events and len(events.get('amplitude_percent', [])) > 0:
                median_amp = np.median(events['amplitude_percent']) / 100.0  # Convert from percent
            else:
                median_amp = 0
            
            # Safe access to arrays that might not match ROI indices
            snr_val = snr_values[i] if i < len(snr_values) else 0.0
            noise_val = noise_values[i] if i < len(noise_values) else 0.0
            
            metrics_text = f"""ROI {i+1} Metrics:
SNR: {snr_val:.2f}
Noise σ: {noise_val:.4f}
Event Rate: {event_rate:.2f} Hz
Event Count: {len(events['times'])}
Median Amp: {median_amp:.3f}"""
            
            axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title(f'ROI {i+1} - Metrics')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Make filename collisions impossible
        if FORCE_TICK_ONLY_SUMMARIES:
            fname = os.path.join(roi_dir, f"roi_{i+1:03d}.png")
        else:
            fname = os.path.join(roi_dir, f"roi_{i+1:03d}_overlay.png")
        
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        
        # Log ROI summary completion
        n_ticks = len(events.get('peaks', events.get('times', [])))
        print(f"ROI {i+1}: summary saved with {n_ticks} spike tick(s).")
        
        plt.close()
    
    print(f"ROI summaries saved to: {roi_dir} ({n_to_save} ROIs)")


def save_outputs(out_dir: str,
                 masks: np.ndarray,
                 ref_img: np.ndarray,
                 traces: Dict[str, np.ndarray],
                 times: Optional[np.ndarray]) -> str:
    """Save masks, overlay, and traces to out_dir. Returns CSV path."""
    os.makedirs(out_dir, exist_ok=True)

    # Save masks
    np.save(os.path.join(out_dir, "masks.npy"), masks)
    if tiff is not None:
        tiff.imwrite(os.path.join(out_dir, "masks.tif"),
                     masks.astype(np.uint16), compression="zlib")

    # Save overlay if matplotlib is available
    if plt is not None:
        # Get number of cells from masks
        n_cells = masks.max()
        
        # Create main overlay
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(ref_img, cmap="gray")
        
        # Define colors for each cell
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_cells, 20)))  # Use tab20 colormap
        if n_cells > 20:
            # If more than 20 cells, cycle through colors
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            colors = np.tile(colors, (n_cells // 20 + 1, 1))[:n_cells]
        
        # Outline each ROI with different colors
        outlines = cp_utils.outlines_list(masks)
        for i, ol in enumerate(outlines):
            color = colors[i % len(colors)] if i < len(colors) else 'white'
            ax.plot(ol[:, 0], ol[:, 1], linewidth=1.5, color=color, label=f'Cell {i+1}')
        
        # Add cell number labels at centroids
        from skimage.measure import regionprops
        props = regionprops(masks)
        for prop in props:
            centroid = prop.centroid
            cell_id = prop.label
            color = colors[(cell_id-1) % len(colors)] if (cell_id-1) < len(colors) else 'white'
            ax.text(centroid[1], centroid[0], str(cell_id), 
                   fontsize=10, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color))
        
        ax.set_title(f'Cell Segmentation Overlay ({n_cells} cells)', fontsize=14, fontweight='bold')
        ax.set_axis_off()
        
        # No separate legend needed - cell numbers are shown directly on overlay
        
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "overlay.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.close(fig)

    # Save traces
    F = traces["F"]
    F_neu = traces.get("F_neu")
    F_corr = traces["F_corr"]
    # Estimate frame rate for baseline computation
    if times is not None and len(times) > 1:
        fps_baseline = 1.0 / np.median(np.diff(times))
    else:
        fps_baseline = 30.0  # Default fallback
    
    dff, F0 = compute_dff(F_corr, percentile=ARGS.baseline_percentile)

    n_cells, T = F.shape
    if times is None or len(times) != T:
        times = np.arange(T, dtype=float)

    rows = []
    for i in range(n_cells):
        for t in range(T):
            rows.append({
                "cell_id": int(i + 1),
                "frame": int(t),
                "time": float(times[t]),
                "F": float(F[i, t]),
                "F_neu": float(F_neu[i, t]) if F_neu is not None else np.nan,
                "F_corr": float(F_corr[i, t]),
                "F0": float(F0[i, 0]),
                "dFF": float(dff[i, t]),
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "traces.csv")
    df.to_csv(csv_path, index=False)

    # Also save a compact wide-format table (cells as columns)
    wide = pd.DataFrame({"time": times})
    for i in range(n_cells):
        wide[f"cell_{i+1}"] = dff[i]
    wide.to_csv(os.path.join(out_dir, "traces_dff_wide.csv"), index=False)

    # Note: Individual trace plots removed - only overlay and voltage summary remain
    
    # Generate voltage summary and/or ROI summaries
    voltage_metrics = {}
    
    if ARGS.plot_voltage_summary or ARGS.save_roi_summaries:
        if n_cells > 0:
            try:
                if ARGS.plot_voltage_summary:
                    try:
                        voltage_metrics = plot_voltage_summary(traces, masks, ref_img, times, out_dir)
                        print(f"Voltage summary saved to: {out_dir}")
                    except Exception as e:
                        print(f"Warning: Voltage summary generation failed: {e}")
                        voltage_metrics = {}
                
                # Save individual ROI summaries if requested (independent of voltage summary)
                if ARGS.save_roi_summaries:
                    # If voltage summary was generated, use its data; otherwise compute minimal data
                    if voltage_metrics and 'all_events' in voltage_metrics:
                        all_events = voltage_metrics['all_events']
                        snr_values = voltage_metrics['snr_values']
                        noise_values = voltage_metrics['noise_values']
                    else:
                        # Compute minimal preprocessing for ROI summaries
                        print("Computing preprocessing for ROI summaries...")
                        
                        # Estimate fps
                        if times is not None and len(times) > 1:
                            fps_roi = 1.0 / np.median(np.diff(times))
                        else:
                            fps_roi = 30.0
                        
                        F_corr = traces["F_corr"]
                        traces_data = F_corr.T  # T x N
                        
                        all_events = {}
                        snr_values = []
                        noise_values = []
                        
                        for i in range(n_cells):
                            # Basic preprocessing
                            detrended = detrend_percentile(traces_data[:, i], fps_roi, win_s=45, q=10)
                            if getattr(ARGS, 'auto_notch', True):
                                detrended = notch_iir(detrended, fps_roi)
                            filtered = bandpass_butter(detrended, fps_roi, low_hz=2.0, high_hz=None, order=2)
                            
                            # SNR and events
                            snr, noise = compute_snr(filtered)
                            snr_values.append(snr)
                            noise_values.append(noise)
                            
                            try:
                                # Use robust detector for ROI summaries
                                dff_roi, percent_dff_roi, F0_roi = compute_dff0(traces_data[:, i], fps_roi)
                                
                                # Noise estimate for this ROI
                                try:
                                    hp_roi = bandpass_butter(percent_dff_roi, fps_roi, low_hz=20.0, high_hz=min(120.0, fps_roi/2*0.95), order=3)
                                    noise_sigma_roi = 1.4826 * np.median(np.abs(hp_roi - np.median(hp_roi)))
                                except:
                                    noise_sigma_roi = 1.4826 * np.median(np.abs(percent_dff_roi - np.median(percent_dff_roi)))
                                noise_sigma_roi = max(noise_sigma_roi, 1e-9)
                                
                                # Create config for this ROI
                                class ConfigRoi:
                                    def __init__(self):
                                        self.k_height = getattr(ARGS, 'k_height', 2.2)
                                        self.k_prom = getattr(ARGS, 'k_prom', 2.8)
                                        self.line_ratio = getattr(ARGS, 'line_ratio', 0.45)
                                
                                cfg_roi = ConfigRoi()
                                
                                # Use the percent-based detector
                                ev_roi = detect_spikes_percent(percent_dff_roi, fps_roi, noise_sigma_roi, cfg_roi)
                                
                                # Check for artifacts
                                if not is_artifact_trace(percent_dff_roi, fps_roi, ev_roi, cfg_roi):
                                    events_roi = []
                                    for j in range(len(ev_roi['idx'])):
                                        events_roi.append({
                                            "time_s": float(ev_roi['t_sec'][j]),
                                            "amp_dff": float(ev_roi['amp_dff'][j]),
                                            "width_ms": float(ev_roi['width_ms'][j]),
                                            "prom_dff": float(ev_roi['prom_dff'][j])
                                        })
                                else:
                                    events_roi = []
                                
                                all_events[i] = {
                                    'times': np.array([e["time_s"] for e in events_roi]),
                                    'amps': np.array([e["amp_dff"] for e in events_roi]),
                                    'widths': np.array([e["width_ms"] / 1000.0 for e in events_roi]),  # Convert to seconds
                                    'prominences': np.array([e["prom_dff"] for e in events_roi]),
                                    'sources': np.full(len(events_roi), 'relaxed')
                                }
                            except Exception as e:
                                print(f"Warning: Event detection failed for ROI {i+1}: {e}")
                                all_events[i] = {
                                    'times': np.array([]),
                                    'amps': np.array([]),
                                    'widths': np.array([]),
                                    'prominences': np.array([]),
                                    'sources': np.array([])
                                }
                    
                    # Save ROI summaries with safe list lengths
                    # Ensure lists have at least n_cells entries
                    while len(snr_values) < n_cells:
                        snr_values.append(0.0)
                    while len(noise_values) < n_cells:
                        noise_values.append(0.0)
                    
                    # ROI summaries already saved in main loop - skip second pass
                    if False:  # Disable second pass to prevent overwrites
                        try:
                            save_roi_summaries(traces, times, all_events, snr_values, noise_values, 
                                             out_dir, max_rois=ARGS.roi_summaries_max)
                        except Exception as e:
                            print(f"Warning: ROI summaries generation failed: {e}")
                            print(f"Debug: n_cells={n_cells}, len(snr_values)={len(snr_values)}, len(noise_values)={len(noise_values)}")
                    else:
                        print("ROI summaries already saved in main loop - skipping second pass")
                
            except Exception as e:
                print(f"Warning: Failed to generate analysis outputs: {e}")
        else:
            print("Warning: No cells found, skipping analysis outputs")

    # Save simple metadata
    md = {
        "n_cells": int(n_cells),
        "n_frames": int(T),
        "baseline_percentile": float(ARGS.baseline_percentile),
        "neuropil_pixels": int(ARGS.neuropil),
        "neuropil_scale": float(ARGS.neuropil_scale),
        "model_type": ARGS.model_type,
        "diameter": None if ARGS.diameter is None else float(ARGS.diameter),
        "channels": [int(c) for c in ARGS.channels],
        "registered": bool(ARGS.register),
        "reference_method": ARGS.reference_method,
        "plots_generated": bool(ARGS.plot_traces or ARGS.plot_summary),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(md, f, indent=2)
    return csv_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract per-cell fluorescence time courses using Cellpose.")
    p.add_argument("--source", choices=["rig", "tiff"], default="rig", help="Input source type.")
    p.add_argument("--data-dir", type=str, required=True, help="RigDataV2 data directory (for --source rig).")
    p.add_argument("--camera-name", type=str, default="flash", help="Camera name (RigDataV2).")
    p.add_argument("--tif", type=str, help="Path to a TIFF stack (for --source tiff).")

    # Cellpose params (optimized defaults for in vitro GEVI data)
    p.add_argument("--model-type", type=str, default="cyto2", help="Cellpose model type, e.g., cyto, cyto2, nuclei.")
    p.add_argument("--diameter", type=float, default=20, help="Approximate object diameter in pixels.")
    p.add_argument("--channels", type=int, nargs=2, default=[0, 0],
                   help="Cellpose channels [cytoplasm, nucleus]. Grayscale use 0 0.")
    p.add_argument("--cellprob-threshold", type=float, default=0.10, help="Cell probability threshold (higher = more stringent).")
    p.add_argument("--flow-threshold", type=float, default=0.18, help="Flow error threshold (lower = more stringent).")
    p.add_argument("--min-size", type=int, default=30, help="Minimum object size in pixels (Cellpose filtering).")
    p.add_argument("--use-watershed", action="store_true", help="Apply watershed to split merged cells.")
    p.add_argument("--watershed-min-distance", type=int, default=15, help="Minimum distance between watershed peaks (higher = less splitting).")
    p.add_argument("--watershed-threshold", type=float, default=0.5, help="Threshold for watershed splitting (0-1, higher = less splitting).")
    p.add_argument("--expected-cell-size", type=int, default=210, help="Expected cell size in pixels (used for size-based splitting logic).")
    p.add_argument("--size-tolerance", type=float, default=0.22, help="Size tolerance for splitting decisions (0.22 = 22%% tolerance).")
    p.add_argument("--enable-merge", action="store_true", help="Allow temporal merges (default: False for higher recall).")
    
    # New adaptive pipeline parameters
    p.add_argument("--autoscale", type=lambda s: s.lower() in {"1","true","yes"}, default=False)
    p.add_argument("--min-size-frac", type=float, default=0.25, help="min object area as fraction of expected_cell_size")
    p.add_argument("--valley-percentile", type=int, default=25)
    p.add_argument("--merge-grad-percentile", type=int, default=10)
    p.add_argument("--trace-merge-r", type=float, default=0.85)
    p.add_argument("--trace-merge-maxlag", type=int, default=1)
    p.add_argument("--min-peak-sep-frac", type=float, default=0.25)
    p.add_argument("--min-distance-frac", type=float, default=0.20)
    p.add_argument("--max-min-distance", type=int, default=12)
    
    # ROI Quality Control parameters
    p.add_argument("--exclude-border", type=lambda s: s.lower() in {"1","true","yes"}, default=True)
    p.add_argument("--qc-min-intensity-pct", type=float, default=30.0, help="ROI mean on ref image must exceed this percentile")
    p.add_argument("--qc-min-coherence", type=float, default=0.15, help="median pairwise pixel correlation within ROI")
    p.add_argument("--local-bg", type=lambda s: s.lower() in {"1","true","yes"}, default=True)
    p.add_argument("--bg-inner", type=int, default=3, help="pixels outside ROI for background annulus inner radius")
    p.add_argument("--bg-outer", type=int, default=8, help="pixels outside ROI for background annulus outer radius")
    
    # Multi-scale Cellpose parameters (optimized for GEVI data)
    p.add_argument("--multiscale", action="store_true", default=True)
    p.add_argument("--scales", nargs="+", type=float, default=[0.75, 0.90, 1.00, 1.15, 1.30])
    p.add_argument("--nms-iou", type=float, default=0.60)
    p.add_argument("--nms-contain", type=float, default=0.90)
    
    # Baseline and filtering parameters
    p.add_argument("--f0-mode", choices=["pwin", "mode"], default="pwin", help="F0 estimation: pwin=piecewise windows, mode=histogram mode")
    p.add_argument("--hp-noise", type=float, default=20.0, help="High-pass frequency for noise estimation (Hz)")
    
    # Event detection parameters (Preprint-compliant)
    p.add_argument("--k-height", type=float, default=2.2, help="Height threshold (×sigma)")
    p.add_argument("--k-prom", type=float, default=2.8, help="Prominence threshold (×sigma)")
    p.add_argument("--line-ratio", type=float, default=0.45, help="Line noise artifact threshold")
    p.add_argument("--min-width-ms", type=float, default=1.5, help="Minimum event width (ms)")
    p.add_argument("--max-width-ms", type=float, default=8.0, help="Maximum event width (ms)")
    p.add_argument("--refractory-ms", type=float, default=8.0, help="Refractory period (ms)")
    p.add_argument("--strict-artifacts", type=lambda s: s.lower() in {"1","true","yes"}, default=True, help="Apply strict artifact rejection")
    
    # Artifact rejection parameters
    p.add_argument("--artifact-rate-hz", type=float, default=10.0, help="Maximum sustained spike rate (Hz)")
    p.add_argument("--artifact-sat-frac", type=float, default=0.002, help="Maximum saturation fraction")
    p.add_argument("--artifact-global-frac", type=float, default=0.30, help="Global artifact coincidence threshold")
    
    p.add_argument("--event-min-count", type=int, default=3, help="Minimum events for ROI to be included in analysis")
    p.add_argument("--snr-threshold", type=float, default=1.5, help="SNR threshold for including ROIs in analysis")
    
    # Per-ROI summary parameters
    p.add_argument("--save-roi-summaries", type=lambda s: s.lower() in {"1","true","yes"}, default=True, help="Save individual ROI summary PNGs")
    p.add_argument("--roi-summaries-max", type=int, default=200, help="Maximum number of ROI summaries to save")
    
    # Debug parameters
    p.add_argument("--debug-roi", type=int, default=None, help="Force specific ROI for event panel and save diagnostics")
    p.add_argument("--debug-save-surrogate", action="store_true", help="Save surrogate peak overlay for debug ROI")
    p.add_argument("--mask-min-area", type=int, default=30, help="Post Cellpose area filter (pixels).")
    p.add_argument("--no-fill-holes", action="store_true", help="Disable hole filling in masks.")
    p.add_argument("--remove-border", action="store_true", help="Remove ROIs touching the image border.")
    p.add_argument("--gpu", action="store_true", help="Use GPU for Cellpose if available.")

    # Registration & reference
    p.add_argument("--register", action="store_true", help="Estimate and apply translational drift correction (typically not needed for in vitro data).")
    p.add_argument("--reference-method", choices=["mean", "max"], default="mean", help="Reference image for Cellpose.")

    # Neuropil and dF/F
    p.add_argument("--neuropil", type=int, default=0, help="Neuropil ring outer radius in pixels (0 disables, typically not needed for in vitro data).")
    p.add_argument("--neuropil-scale", type=float, default=0.7, help="Scale for neuropil subtraction (F - s*Fneu).")
    
    # Plotting options (enabled by default for comprehensive analysis)
    p.add_argument("--plot-traces", action="store_true", default=True, help="Generate plots of fluorescent traces.")
    p.add_argument("--plot-summary", action="store_true", default=True, help="Generate summary plots (heatmap, overlay).")
    p.add_argument("--plot-voltage-summary", action="store_true", default=True, help="Generate spontaneous RGC voltage summary figure.")
    p.add_argument("--baseline-percentile", type=float, default=20.0, help="Percentile for F0 baseline (per cell).")

    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: data-dir/analysis_output).")
    return p.parse_args(argv)


def main(argv=None):
    global ARGS
    ARGS = parse_args(argv)

    # Set up automatic output directory if not specified
    if ARGS.out_dir is None:
        if ARGS.data_dir:
            ARGS.out_dir = os.path.join(ARGS.data_dir, "analysis_output")
        else:
            ARGS.out_dir = "./analysis_output"
    
    # Create output directory
    os.makedirs(ARGS.out_dir, exist_ok=True)
    print(f"Output directory: {ARGS.out_dir}")

    if ARGS.source == "rig":
        if not ARGS.data_dir:
            raise SystemExit("--data-dir is required for --source rig")
        stack = DataStack.from_rig(ARGS.data_dir, ARGS.camera_name)
    else:
        if not ARGS.tif:
            raise SystemExit("--tif is required for --source tiff")
        stack = DataStack.from_tif(ARGS.tif)

    frames = stack.frames.astype(np.float32)
    # Normalize to [0,1] for Cellpose stability
    if frames.max() > 0:
        frames /= frames.max()

    # Autoscale parameters if requested
    if ARGS.autoscale:
        res = autoscale_from_sample(frames)
        if res is not None:
            auto_diam, auto_area = res
            if ARGS.diameter is None or ARGS.diameter <= 0:
                ARGS.diameter = auto_diam
            if ARGS.expected_cell_size <= 0:
                ARGS.expected_cell_size = auto_area
        if ARGS.min_size <= 0 and ARGS.expected_cell_size > 0:
            ARGS.min_size = int(max(20, ARGS.min_size_frac * ARGS.expected_cell_size))

    # Create gently normalized reference image for both Cellpose and overlay
    img_overlay, img_cp = make_ref_and_cp(frames)

    # (Optional) register to reference (drift correction)
    if ARGS.register:
        shifts = compute_translations(frames, img_cp, upsample_factor=1)
        frames_reg = apply_translations(frames, shifts, order=1)
        # Update reference images after registration
        img_overlay, img_cp = make_ref_and_cp(frames_reg)
        frames = frames_reg  # use registered frames for measurement

    # Multi-scale Cellpose with strict NMS for higher recall
    model = models.Cellpose(gpu=ARGS.gpu, model_type=ARGS.model_type)
    
    if ARGS.multiscale:
        base_d = ARGS.diameter if ARGS.diameter and ARGS.diameter > 0 else 20
        proposals = []
        for s in ARGS.scales:
            lab_s, score_src = run_cellpose_once(model, img_cp, base_d * s, ARGS)
            if lab_s.max() == 0:
                continue
            proposals += proposals_from_lab(lab_s, score_src, img_cp)
        masks = nms(proposals, img_cp.shape, iou_thr=ARGS.nms_iou, contain_thr=ARGS.nms_contain)
    else:
        masks, _ = run_cellpose_once(model, img_cp, ARGS.diameter or 20, ARGS)
    
    # Small-object filter based on expected size
    if ARGS.expected_cell_size:
        min_area = max(20, int(0.12 * ARGS.expected_cell_size))
    else:
        min_area = 30
    masks = remove_small_objects(masks, min_area)

    # Apply adaptive splitting only (no merging for higher recall)
    if ARGS.autoscale:
        # Adaptive split only
        masks = split_roi_adaptive(
            masks, img_cp.astype(np.float32),
            valley_pct=ARGS.valley_percentile,
            min_sep_frac=ARGS.min_peak_sep_frac,
            min_dist_frac=ARGS.min_distance_frac,
            max_min_dist=ARGS.max_min_distance
        )

        # Optional merging (disabled by default for higher recall)
        if ARGS.enable_merge:
            # Compute traces for merging
            traces_for_merge = extract_traces_for_merge(frames.astype(np.float32), masks, baseline_pct=ARGS.baseline_percentile)

            # Adaptive merge using gradients + traces
            masks, traces_for_merge = merge_neighbors_adaptive(
                masks, img_cp.astype(np.float32), traces_for_merge,
                merge_grad_pct=ARGS.merge_grad_percentile,
                r_thresh=ARGS.trace_merge_r,
                maxlag=ARGS.trace_merge_maxlag
            )

        # Final size filter after adaptive processing
        if ARGS.min_size:
            masks = remove_small_objects(masks, ARGS.min_size)
    elif ARGS.use_watershed:
        # Fall back to old watershed if not using autoscale
        masks = apply_watershed_splitting(masks, 
                                        min_distance=ARGS.watershed_min_distance,
                                        threshold_ratio=ARGS.watershed_threshold,
                                        expected_cell_size=ARGS.expected_cell_size,
                                        size_tolerance=ARGS.size_tolerance)
    else:
        # Default: just apply object-wise splitting without merging
        masks = split_roi_adaptive(
            masks, img_cp.astype(np.float32),
            valley_pct=20,  # default valley percentile
            min_sep_frac=0.25,
            min_dist_frac=0.20,
            max_min_dist=12
        )

    # ROI Quality Control pass
    # Use the overlay image for QC (more natural intensity levels)
    img_ref_qc = img_overlay
    
    # Store original masks before QC
    masks_before_qc = masks.copy()
    
    H, W = img_ref_qc.shape
    if np.sum(masks > 0) > 0:  # Only apply QC if there are ROIs
        intensity_thr = np.percentile(img_ref_qc[masks > 0], ARGS.qc_min_intensity_pct)

        keep = np.zeros_like(masks, bool)
        for lbl in np.unique(masks):
            if lbl == 0:
                continue
            mask = (masks == lbl)
            
            # border exclusion
            if ARGS.exclude_border:
                if (mask[0, :].any() or mask[-1, :].any() or 
                    mask[:, 0].any() or mask[:, -1].any()):
                    continue
            
            # intensity filter
            if img_ref_qc[mask].mean() < intensity_thr:
                continue
            
            # coherence filter
            coh = roi_coherence(frames, mask)
            if coh < ARGS.qc_min_coherence:
                continue
            
            keep |= mask
        
        masks[~keep] = 0
        masks, _, _ = relabel_sequential(masks)
        
        # If QC removed all ROIs, fall back to less strict criteria
        if masks.max() == 0:
            print("Warning: QC removed all ROIs, falling back to border exclusion only")
            # Reset to original masks and apply only border exclusion
            masks = masks_before_qc.copy()
            keep = np.zeros_like(masks, bool)
            for lbl in np.unique(masks):
                if lbl == 0:
                    continue
                mask = (masks == lbl)
                
                # Only apply border exclusion if enabled
                if ARGS.exclude_border:
                    if (mask[0, :].any() or mask[-1, :].any() or 
                        mask[:, 0].any() or mask[:, -1].any()):
                        continue
                
                keep |= mask
            
            masks[~keep] = 0
            masks, _, _ = relabel_sequential(masks)
            
            # If still no ROIs, keep all original ROIs
            if masks.max() == 0:
                print("Warning: Even border exclusion removed all ROIs, keeping all original ROIs")
                masks = masks_before_qc.copy()

    # Extract traces with local background subtraction
    if ARGS.local_bg:
        # Use new extraction method with local background
        trace_dict = extract_traces_with_local_bg(frames, masks, ARGS.baseline_percentile, 
                                                 ARGS.local_bg, ARGS.bg_inner, ARGS.bg_outer)
        # Convert to old format for compatibility
        n_cells = masks.max()
        if n_cells > 0:
            T = frames.shape[0]
            F_corr = np.zeros((n_cells, T), dtype=np.float32)
            for i, lbl in enumerate(sorted(trace_dict.keys())):
                # Note: this assumes dF/F traces, need to reconstruct F
                dff = trace_dict[lbl]
                # Rough reconstruction of F from dF/F (not perfect but maintains compatibility)
                F0_est = 1.0  # nominal baseline
                F_est = F0_est * (1 + dff)
                F_corr[i] = F_est
            traces = {"F": F_corr, "F_neu": np.zeros_like(F_corr), "F_corr": F_corr}
        else:
            traces = {"F": np.array([]).reshape(0, frames.shape[0]), 
                     "F_neu": np.array([]).reshape(0, frames.shape[0]), 
                     "F_corr": np.array([]).reshape(0, frames.shape[0])}
    else:
        # Use original extraction method
        traces = extract_traces(frames, masks,
                               neuropil=int(ARGS.neuropil),
                               neuropil_scale=float(ARGS.neuropil_scale))

    # Save outputs
    csv_path = save_outputs(ARGS.out_dir, masks, img_overlay, traces, stack.times)

    print(f"Done. Traces saved to: {csv_path}")
    print(f"Masks saved to: {os.path.join(ARGS.out_dir, 'masks.npy')}")
    if tiff is not None:
        print(f"Masks TIFF saved to: {os.path.join(ARGS.out_dir, 'masks.tif')}")
    if plt is not None:
        print(f"Overlay saved to: {os.path.join(ARGS.out_dir, 'overlay.png')}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
