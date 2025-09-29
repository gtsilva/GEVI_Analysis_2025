#!/usr/bin/env python3
"""
Quick diagnostic script to understand why segmentation is failing.
Examines the reference image and tests different segmentation approaches.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from our scripts
from simple_conservative_extract_fluo import DataStack, build_reference
from skimage import exposure, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi

def diagnose_data(data_dir, camera_name="flash"):
    """Diagnose segmentation issues with the data."""
    
    print(f"Loading data from: {data_dir}")
    
    # Load the data
    try:
        stack = DataStack.from_rig(data_dir, camera_name)
        print(f"Data loaded: shape={stack.frames.shape}, frame_rate={stack.frame_rate}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    frames = stack.frames.astype(np.float32)
    if frames.max() > 0:
        frames /= frames.max()
    
    # Create both mean and max projections
    ref_mean = build_reference(frames, method="mean")
    ref_max = build_reference(frames, method="max")
    
    print(f"Reference images: mean range=[{ref_mean.min():.3f}, {ref_mean.max():.3f}], max range=[{ref_max.min():.3f}, {ref_max.max():.3f}]")
    
    # Basic image statistics
    print(f"Raw frames: min={frames.min():.3f}, max={frames.max():.3f}, mean={frames.mean():.3f}")
    print(f"Frame shape: {frames.shape[1:]} (H x W)")
    
    # Test different approaches on both projections
    for proj_name, ref_img in [("mean", ref_mean), ("max", ref_max)]:
        print(f"\n=== Testing {proj_name} projection ===")
        
        # Normalize
        I = ref_img.astype(np.float32)
        lo, hi = np.percentile(I, (0.5, 99.5))
        I_norm = np.clip((I - lo) / max(hi - lo, 1e-6), 0, 1)
        
        print(f"Normalized range: [{I_norm.min():.3f}, {I_norm.max():.3f}]")
        
        # Try blob detection
        blobs = feature.blob_log(I_norm, min_sigma=1.5, max_sigma=8, num_sigma=12, threshold=0.02)
        print(f"LoG blobs found: {len(blobs)}")
        if len(blobs) > 0:
            radii = blobs[:, 2] * np.sqrt(2.0)
            print(f"Blob radii: median={np.median(radii):.1f}, range=[{radii.min():.1f}, {radii.max():.1f}]")
        
        # Try thresholding approaches
        otsu_thresh = filters.threshold_otsu(I_norm)
        li_thresh = filters.threshold_li(I_norm)
        
        otsu_mask = I_norm > otsu_thresh
        li_mask = I_norm > li_thresh
        
        print(f"Otsu threshold: {otsu_thresh:.3f}, coverage: {100*otsu_mask.sum()/otsu_mask.size:.1f}%")
        print(f"Li threshold: {li_thresh:.3f}, coverage: {100*li_mask.sum()/li_mask.size:.1f}%")
        
        # Test adaptive histogram equalization
        Ieq = exposure.equalize_adapthist(I_norm, clip_limit=0.01)
        print(f"After equalization: range=[{Ieq.min():.3f}, {Ieq.max():.3f}]")
        
        # Test different percentile thresholds
        for pct in [90, 95, 97, 98, 99]:
            thr = np.percentile(I_norm, pct)
            mask = I_norm > thr
            print(f"{pct}th percentile: {thr:.3f}, coverage: {100*mask.sum()/mask.size:.1f}%")
    
    # Create comprehensive diagnostic plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Raw projections and histogram
    axes[0,0].imshow(ref_mean, cmap='gray')
    axes[0,0].set_title('Mean Projection')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(ref_max, cmap='gray')
    axes[0,1].set_title('Max Projection')
    axes[0,1].axis('off')
    
    axes[0,2].hist(ref_max.ravel(), bins=100, alpha=0.7, label='max proj')
    axes[0,2].hist(ref_mean.ravel(), bins=100, alpha=0.7, label='mean proj')
    axes[0,2].set_title('Intensity Histograms')
    axes[0,2].legend()
    
    # Show intensity profiles
    mid_row = ref_max.shape[0] // 2
    axes[0,3].plot(ref_max[mid_row, :], label='max proj')
    axes[0,3].plot(ref_mean[mid_row, :], label='mean proj')
    axes[0,3].set_title('Middle Row Profile')
    axes[0,3].legend()
    
    # Row 2: Processing steps on max projection
    I_norm = np.clip((ref_max - np.percentile(ref_max, 0.5)) / 
                     (np.percentile(ref_max, 99.5) - np.percentile(ref_max, 0.5) + 1e-6), 0, 1)
    
    axes[1,0].imshow(I_norm, cmap='gray')
    axes[1,0].set_title('Normalized')
    axes[1,0].axis('off')
    
    Ieq = exposure.equalize_adapthist(I_norm, clip_limit=0.01)
    axes[1,1].imshow(Ieq, cmap='gray')
    axes[1,1].set_title('Equalized')
    axes[1,1].axis('off')
    
    # DoG filtering
    s_small = 3.0  # Try reasonable values
    s_large = 6.0
    dog = filters.gaussian(Ieq, s_small) - filters.gaussian(Ieq, s_large)
    dog_norm = (dog - dog.mean()) / (dog.std() + 1e-6)
    
    axes[1,2].imshow(dog_norm, cmap='magma')
    axes[1,2].set_title('DoG Filter')
    axes[1,2].axis('off')
    
    # Threshold attempts
    otsu_mask = I_norm > filters.threshold_otsu(I_norm)
    axes[1,3].imshow(otsu_mask, cmap='gray')
    axes[1,3].set_title(f'Otsu Mask ({100*otsu_mask.sum()/otsu_mask.size:.1f}%)')
    axes[1,3].axis('off')
    
    # Row 3: Different threshold levels
    for i, pct in enumerate([95, 97, 98, 99]):
        if i < 4:
            thr = np.percentile(I_norm, pct)
            mask = I_norm > thr
            axes[2,i].imshow(mask, cmap='gray')
            axes[2,i].set_title(f'{pct}th %ile ({100*mask.sum()/mask.size:.1f}%)')
            axes[2,i].axis('off')
    
    plt.tight_layout()
    
    # Save diagnostic plot
    output_dir = Path(data_dir) / "analysis_output"
    output_dir.mkdir(exist_ok=True)
    diag_path = output_dir / "segmentation_diagnosis.png"
    plt.savefig(diag_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nDiagnostic plot saved to: {diag_path}")
    print("\nSuggestions:")
    print("1. Check if your cells are visible in the max projection")
    print("2. Look at the DoG filter - cells should appear as bright spots")
    print("3. Check threshold coverage - might need to adjust percentiles")
    print("4. Try manual diameter if auto-detection failed")

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diagnose_segmentation.py <data_dir> [camera_name]")
        print("Example: python diagnose_segmentation.py /path/to/experiment flash")
        return
    
    data_dir = sys.argv[1]
    camera_name = sys.argv[2] if len(sys.argv) > 2 else "flash"
    
    diagnose_data(data_dir, camera_name)

if __name__ == "__main__":
    main()
