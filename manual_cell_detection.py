#!/usr/bin/env python3
"""
Manual cell detection script for problematic data.
Uses simple thresholding and connected components.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure, segmentation
from simple_conservative_extract_fluo import DataStack, build_reference

def simple_cell_detection(img, min_size=50, max_size=1000):
    """
    Simple cell detection using adaptive thresholding.
    """
    print(f"Image shape: {img.shape}, range: {img.min():.3f} - {img.max():.3f}")
    
    # Normalize
    img_norm = (img - img.min()) / (img.max() - img.min())
    
    # Try multiple thresholding approaches
    methods = []
    
    # Method 1: Otsu
    try:
        thresh_otsu = filters.threshold_otsu(img_norm)
        mask1 = img_norm > thresh_otsu
        methods.append(("Otsu", mask1, thresh_otsu))
    except:
        pass
    
    # Method 2: Triangle
    try:
        thresh_tri = filters.threshold_triangle(img_norm)
        mask2 = img_norm > thresh_tri
        methods.append(("Triangle", mask2, thresh_tri))
    except:
        pass
    
    # Method 3: Li
    try:
        thresh_li = filters.threshold_li(img_norm)
        mask3 = img_norm > thresh_li
        methods.append(("Li", mask3, thresh_li))
    except:
        pass
    
    # Method 4: Manual percentile
    for p in [75, 80, 85, 90]:
        thresh_p = np.percentile(img_norm, p)
        mask_p = img_norm > thresh_p
        methods.append((f"P{p}", mask_p, thresh_p))
    
    # Test each method
    results = []
    for name, mask, thresh in methods:
        # Clean up mask
        mask_clean = morphology.remove_small_objects(mask, min_size=min_size//4)
        mask_clean = morphology.remove_small_holes(mask_clean, area_threshold=min_size//4)
        
        # Label connected components
        labeled = measure.label(mask_clean)
        props = measure.regionprops(labeled)
        
        # Filter by size
        good_regions = []
        for prop in props:
            if min_size <= prop.area <= max_size:
                # Check if roughly circular
                circularity = 4 * np.pi * prop.area / (prop.perimeter**2 + 1e-6)
                if circularity > 0.3:  # Very lenient
                    good_regions.append(prop)
        
        results.append({
            'name': name,
            'threshold': thresh,
            'mask': mask_clean,
            'labeled': labeled,
            'n_regions': len(good_regions),
            'regions': good_regions
        })
        
        print(f"{name} (thresh={thresh:.3f}): {len(props)} total, {len(good_regions)} good regions")
    
    # Pick best result (most regions in reasonable range)
    best = max(results, key=lambda x: x['n_regions'] if 2 <= x['n_regions'] <= 8 else 0)
    
    if best['n_regions'] == 0:
        print("No good segmentation found, trying most permissive...")
        best = max(results, key=lambda x: x['n_regions'])
    
    print(f"\nBest method: {best['name']} with {best['n_regions']} cells")
    
    # Create final mask
    final_mask = np.zeros_like(img, dtype=np.int32)
    for i, region in enumerate(best['regions'], 1):
        final_mask[best['labeled'] == region.label] = i
    
    return final_mask, best

def visualize_result(img, mask, result):
    """Create visualization of segmentation result."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Threshold mask
    axes[0, 1].imshow(result['mask'], cmap='gray')
    axes[0, 1].set_title(f'{result["name"]} Threshold (t={result["threshold"]:.3f})')
    axes[0, 1].axis('off')
    
    # Labeled regions
    axes[1, 0].imshow(img, cmap='gray')
    if mask.max() > 0:
        # Draw contours
        for i in range(1, mask.max() + 1):
            contours = measure.find_contours(mask == i, 0.5)
            for contour in contours:
                axes[1, 0].plot(contour[:, 1], contour[:, 0], linewidth=2)
    axes[1, 0].set_title(f'Final Result ({mask.max()} cells)')
    axes[1, 0].axis('off')
    
    # Colored mask
    axes[1, 1].imshow(mask, cmap='tab20')
    axes[1, 1].set_title('Colored Labels')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Load your data
    data_dir = "/Users/gts/Downloads/ASAP_experiments/ASAP6_3/240824220944_cluster_01-50mWpermm2"
    
    print("Loading data...")
    stack = DataStack.from_rig(data_dir, "flash")
    frames = stack.frames.astype(np.float32)
    if frames.max() > 0:
        frames /= frames.max()
    
    # Create reference
    ref_img = build_reference(frames, method="mean")
    print(f"Reference image shape: {ref_img.shape}")
    
    # Detect cells
    mask, result = simple_cell_detection(ref_img, min_size=30, max_size=500)
    
    # Visualize
    fig = visualize_result(ref_img, mask, result)
    
    # Save
    output_dir = f"{data_dir}/analysis_output"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(f"{output_dir}/manual_segmentation.png", dpi=300, bbox_inches='tight')
    np.save(f"{output_dir}/manual_masks.npy", mask)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"- manual_segmentation.png: Visualization")
    print(f"- manual_masks.npy: Segmentation masks")
    print(f"Detected {mask.max()} cells")
    
    plt.show()

if __name__ == "__main__":
    main()
