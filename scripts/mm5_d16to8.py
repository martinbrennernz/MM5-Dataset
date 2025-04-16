#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#  mm5_d16to8.py
#
#  LICENSE:
#    This file is distributed under the Creative Commons
#    Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
#    https://creativecommons.org/licenses/by-nc/4.0/
#
#    You are free to:
#      • Share — copy and redistribute the material in any medium or format
#      • Adapt — remix, transform, and build upon the material
#    under the following terms:
#      • Attribution — You must give appropriate credit, provide a link to
#        the license, and indicate if changes were made.
#      • NonCommercial — You may not use the material for commercial purposes.
#
#
#  DISCLAIMER:
#    This code is provided “AS IS,” without warranties or conditions of any kind,
#    express or implied. Use it at your own risk.
# -----------------------------------------------------------------------------

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from scipy.signal import peak_widths
from scipy.ndimage import gaussian_filter1d

import time

"""
Created on Sat Dec  2 23:17:32 2023

@author: MB
"""
  
def create_color_bar_rgb():
    # Red channel: 0 to 255, stay at 255, then 255 to 0
    red = np.concatenate([np.arange(256), np.full(256, 255), np.arange(255, -1, -1), np.full(256, 0)])

    # Green channel: 0, then 0 to 255, then 255 to 0
    green = np.concatenate([np.full(256, 0), np.arange(256), np.full(256, 255), np.arange(255, -1, -1)])

    # Blue channel: entirely zeros (could be adjusted if color blending is desired)
    #blue = np.zeros_like(red)

    # Combine red, green channels, clip first and last 20 values to avoid color distortions
    #color_bar = np.column_stack([red, green, blue])[20:-24]
    color_bar = np.column_stack([red, green])[20:-24]

    return color_bar   

def compress_depth_with_resolution(depth_values, resolution, mode="near", start_point=None, outlier_threshold=0):
    """
    Compress depth values based on the given resolution.

    Parameters:
    - depth_values: The depth values to be compressed.
    - resolution: The resolution for compression.
    - mode: "near" for near values, "far" for far values, and "center" for centering around a start point.
    - start_point: The starting point for numbering when mode is "center".
    - outlier_threshold: The occurrence count threshold for identifying outliers. If None, no outlier removal is done.

    Returns:
    - Compressed depth values.
    """

    # Remove outliers if outlier_threshold is provided
    if outlier_threshold>0:
        # Find unique depth values and their counts
        unique_depths, counts = np.unique(depth_values, return_counts=True)

        # Identify outlier depth values
        outliers = unique_depths[counts <= outlier_threshold]

        # For each outlier, replace it with the next value in the sorted unique depth values
        for outlier in outliers:
            # Find the index of the outlier in the sorted unique depths
            index = np.where(unique_depths == outlier)[0][0]

            # If it's the last value, replace with the previous value, else replace with the next value
            replacement_value = unique_depths[index - 1] if index == len(unique_depths) - 1 else unique_depths[index + 1]

            # Replace outlier with replacement_value in depth_values
            depth_values[depth_values == outlier] = replacement_value

    # Calculate the compressed values based on the resolution
    compressed_values = (depth_values // resolution) * resolution

    # If mode is "far", reassign the values starting from 255 and decreasing
    if mode == "far":
        unique_vals = np.unique(compressed_values)
        mapping = {val: 255 - i for i, val in enumerate(sorted(unique_vals))}
        for original, new in mapping.items():
            compressed_values[compressed_values == original] = new

    # If mode is "near" or "center", map the rounded values to start from the desired starting point
    elif mode in ["near", "center"]:
        unique_vals = np.unique(compressed_values)
        if mode == "near":
            start_val = 1
        else:
            if start_point is None:
                raise ValueError("For 'center' mode, a start_point must be provided.")
            start_val = start_point

        mapping = {val: start_val + i for i, val in enumerate(sorted(unique_vals))}
        for original, new in mapping.items():
            compressed_values[compressed_values == original] = new

    return compressed_values


def enhanced_16bit_to_8bit(depth, focus_range, peaks, widths, res_oof_near=15, res_oof_far=20, res_gap=10, res_focus=2, num_channels=1, debug=False):
    
    #print("num channels: " + str(num_channels))
    # Create a mask to identify non-zero values
    non_zero_mask = depth != 0

    # Create Out-Of-Focus (OOF) Masks, ignoring zero values
    oof_mask_near = np.logical_and(depth < focus_range[0], non_zero_mask)
    oof_mask_far = np.logical_and(depth > focus_range[1]-1, non_zero_mask)

    # Get min/max values for oof_mask_near
    if np.any(oof_mask_near):  # Check if the mask is not empty
        oof_near_min = depth[oof_mask_near].min()
        oof_near_max = depth[oof_mask_near].max()
    else:
        oof_near_min = None
        oof_near_max = None

    # Get min/max values for oof_mask_far
    if np.any(oof_mask_far):  # Check if the mask is not empty
        oof_far_min = depth[oof_mask_far].min()
        oof_far_max = depth[oof_mask_far].max()
    else:
        oof_far_min = None
        oof_far_max = None

    # Initialize a dictionary to hold min/max values for each peak
    peak_min_max_dict = {}

    # Create Peak Focus Mask
    peak_focus_mask = np.zeros_like(depth, dtype=bool)
    for (peak_center, _), peak_width in zip(peaks, widths):

        #replace static width with dynamic
        #individual_peak_mask = np.logical_and(equalized_depth >= (peak_center - focus_width), equalized_depth <= (peak_center + focus_width))
        individual_peak_mask = np.logical_and(depth >= (peak_center - peak_width), depth <= (peak_center + peak_width))

        # Update the overall peak focus mask
        peak_focus_mask = np.logical_or(peak_focus_mask, individual_peak_mask)

        # Mask zero values
        individual_peak_values = np.ma.masked_equal(depth[individual_peak_mask], 0)

        # Get min/max values for individual peaks, ignoring zeros
        if individual_peak_values.count() > 0:  # Check if there are any non-zero values
            peak_min = individual_peak_values.min()
            peak_max = individual_peak_values.max()
        else:
            peak_min = None
            peak_max = None

        # Store in dictionary
        peak_min_max_dict[peak_center] = {'min': peak_min, 'max': peak_max, 'width': peak_width}


    # Create a list of all regions with their min, max values, and masks
    regions = []

    # Handle OOF near
    if oof_near_min is None:
        oof_near_min = 1
    if oof_near_max is None:
        oof_near_max = focus_range[0] - 1
    oof_near_values = depth[oof_mask_near]
    regions.append({'name': 'oof_near', 'min': oof_near_min, 'max': oof_near_max, 'mask': oof_mask_near, 'values': oof_near_values, 'compress': compress_depth_with_resolution(oof_near_values,res_oof_near,"near")})
    
    if debug:
        print(f"DEBUG: MIN  min:{oof_near_min} max:{oof_near_max}")
    # Handle peaks
    previous_max = oof_near_max
    previous_Tmax = oof_near_max # if wider earlier peaks, make sure to capture the max
    if debug:
        print(f"DEBUG: previous_max:{previous_max} max:{oof_near_max}")    
    for peak_center, peak_info in peak_min_max_dict.items():
        if peak_info['min'] is None:
            peak_info['min'] = previous_max + 1
        if peak_info['max'] is None:
            peak_info['max'] = peak_center + peak_info['width']
        if peak_info['max'] > focus_range[1]:
            peak_info['max'] = focus_range[1]-1 # peak max can only be maximum of focus range
        # deal with overlap
        if peak_info['min'] < previous_max:
            peak_info['min'] = previous_max+1

        if (peak_info['min'] >= focus_range[1]) or (peak_info['min'] >= peak_info['max']):
            continue # skip peaks outside of focus or fully overlap
        reg_mask = np.logical_and(depth >= ( peak_info['min']), depth <= (peak_info['max']))
        regions.append({
            'name': f'peak_{peak_center}',
            'min': peak_info['min'],
            'max': peak_info['max'],
            'mask': reg_mask,
            'values': depth[reg_mask],
            'compress': [0,]
        })
        if debug:
            print(f"DEBUG: INSERT PEAK! name:{peak_center}  min:{peak_info['min']} max:{peak_info['max']}")   
        
        previous_max = peak_info['max']
        previous_Tmax = previous_max if previous_Tmax <= previous_max else previous_Tmax

    # Handle OOF far
    oof_far_min = focus_range[1] # max focus range
    if oof_far_min is None:
        oof_far_min = previous_Tmax + 1
    if oof_far_max is None:
        oof_far_max = previous_Tmax + 1
    if oof_far_max < oof_far_min: # can happen if max depth is smaller than max focus range
        oof_far_max = oof_far_min
    oof_far_values = depth[oof_mask_far]
    regions.append({'name': 'oof_far', 'min': oof_far_min, 'max': oof_far_max, 'mask': oof_mask_far, 'values': oof_far_values, 'compress': [0,]})


    # Sort regions by their min value
    regions = sorted(regions, key=lambda x: x['min'])

    # Determine gaps between regions and store them
    gaps = []
    inserted_gap = False
    last_gap_max_comp = 0
    for i in range(1, len(regions)):
        prev_region = regions[i-1]
        curr_region = regions[i]

        if curr_region['min'] > prev_region['max']+1:
            if debug:
                print(f"DEBUG: INSERT GAP! cur name:{curr_region['name']} cur min:{curr_region['min']} max:{curr_region['max']}")
            gap_mask = np.logical_and(depth > prev_region['max'], depth < curr_region['min'])
            gap_values = depth[gap_mask]
            if prev_region['compress'].size > 0:
                max_val = prev_region['compress'].max()
            else:
                max_val = 0
            if debug:
                print(f"DEBUG: {curr_region['name']} start compress val: {max_val+1} prevCompSize:{prev_region['compress'].size}")
                print("DEBUG: " + str(res_gap))
            gap_compress= compress_depth_with_resolution(gap_values,res_gap,"center", max_val+1)
            gaps.append({
                'name': f'gap_{i}',
                'min': prev_region['max']+1,
                'max': curr_region['min']-1,
                'mask': gap_mask,
                'values': gap_values,
                'compress': gap_compress
            })
            # Check if the gap_compress array is empty
            if gap_compress.size == 0:
                #last_gap_max_comp = oof_far_min
                last_gap_max_comp = oof_near_max
                if debug:
                    print(f"DEBUG: gap_compress was empty: {curr_region['name']} start compress val: {max_val+1} now set to: {oof_far_min}")
            else:
                last_gap_max_comp = gap_compress.max()
            inserted_gap = True

        # process peaks for compreession
        if inserted_gap:
            #print(f"inserted: {last_gap_max_comp}")
            starting_point = last_gap_max_comp + 1
            inserted_gap = False
        elif prev_region['compress'].size > 0:
            starting_point = prev_region['compress'].max() + 1
        else:
            starting_point = 1  # if off_near is empty set start to 1

        if curr_region['name'] == 'oof_far':
            curr_region['compress'] = compress_depth_with_resolution(curr_region['values'], res_oof_far, "center", starting_point)
        else:
            #print(f"DEBUG process compress: {curr_region['name']} starts at {curr_region['min']} and ends at {curr_region['max']}; starting point: {starting_point}")
            curr_region['compress'] = compress_depth_with_resolution(curr_region['values'], res_focus, "center", starting_point, 0)


    # Combine regions and gaps
    all_sections = regions + gaps
    # Sort all_sections based on min values
    all_sections = sorted(all_sections, key=lambda x: x['min'])

    # Now, all_sections contains all the regions (OOF, peaks) and gaps with their min, max values, masks, and values.
    new_depth_16bit = np.zeros_like(depth)
    for section in all_sections:
        new_depth_16bit[section['mask']] = section['compress']
        if debug:
            print(f"{section['name']} starts at {section['min']} and ends at {section['max']}; unique vals {len(np.unique(section['values']))}; unique Cvals {len(np.unique(section['compress']))}")

    # Extract values from the oof_mask_near of the 16-bit depth map
    oof_near_values = depth[oof_mask_near]
    oof_far_values = depth[oof_mask_far]

    if num_channels==1:
        # Convert to 8-bit
        depth_8bit = ((new_depth_16bit - new_depth_16bit.min()) / (new_depth_16bit.max() - new_depth_16bit.min()) * 255).astype(np.uint8)
        #print("doing one channel only")
        return depth_8bit, None
    else:
        color_bar_rgb = create_color_bar_rgb()  # Expected shape: (980, 3)
        
        # Always perform min–max normalization to span the full 0 to 980 range.
        depth_min = new_depth_16bit.min()
        depth_max = new_depth_16bit.max()

        normalized_depth = ((new_depth_16bit - depth_min) / (depth_max - depth_min) * 980).astype(np.uint16)
        
        # Ensure indices are within the valid range [0, 979]
        normalized_depth_clipped = np.clip(normalized_depth, 0, 979)
        
        if debug:
            print(f"DEBUG: Normalized depth min: {normalized_depth.min()}, max: {normalized_depth.max()}")
            print(f"DEBUG: Clipped normalized depth min: {normalized_depth_clipped.min()}, max: {normalized_depth_clipped.max()}")
        
        # Map the normalized indices to the color bar
        return color_bar_rgb[normalized_depth_clipped], normalized_depth_clipped

def showDepth(raw_depth, index, debug=False, hist=False, inpaint=False, inpaint_radius=3, ol_threshold=0
    , min_width = 100
    , max_width = 400
    , min_focus = 300
    , max_focus = 3800
    , res_oof_near=15, res_oof_far=20, res_gap=10, res_focus=2, num_channels=1, num_peaks=3, num_buckets=20
    , sigma_value=0.4
    , min_height = 0.0002       # Minimum height of a peak
    , min_prominence = 0   # Minimum prominence of a peak
    , min_distance = 1
    , useKDE = True):
    # Check if raw_depth is a PIL Image and convert to numpy array if true
    if isinstance(raw_depth, Image.Image):
        raw_depth = np.array(raw_depth)


    # # Display rawDepth image
    # plt.figure(figsize=(8, 8))
    # plt.imshow(raw_depth, extent=[0, raw_depth.shape[1], 0, raw_depth.shape[0]])
    # plt.title(f"RawDepth Image {index}")
    # plt.axis('off')
    # plt.show()
    if(hist):
        ol_threshold = 0
        start_time = time.time()

        # Flatten the raw_depth array, remove NaN values and exclude 0 values
        raw_depth_flat = raw_depth[~np.isnan(raw_depth) & (raw_depth != 0)].flatten()

        end_time_01 = time.time()
        elapsed_time_01 = end_time_01 - start_time

        # Plot histogram of rawDepth image
        plt.figure(figsize=(8, 4))
        plt.hist(raw_depth_flat, bins=100, color='c', density=True)  # Reduced bins to 40 for coarser and faster calc

        #------------------------------------------------------------------------------------------------------------
        start_time = time.time() # reset start-time
        # Identify depth values with occurrence count not higher than threshold_count
        if ol_threshold>0:
            # Find unique depth values and their counts
            unique_depths, counts = np.unique(raw_depth_flat, return_counts=True)

            # Identify outlier depth values
            outliers = unique_depths[counts <= ol_threshold]

            # For each outlier, replace it with the next value in the sorted unique depth values
            for outlier in outliers:
                # Find the index of the outlier in the sorted unique depths
                index = np.where(unique_depths == outlier)[0][0]

                # If it's the last value, replace with the previous value, else replace with the next value
                replacement_value = unique_depths[index - 1] if index == len(unique_depths) - 1 else unique_depths[index + 1]

                # Replace outlier with replacement_value in raw_depth
                raw_depth_flat[raw_depth_flat == outlier] = replacement_value
        end_time_02 = time.time()
        elapsed_time_02 = end_time_02 - start_time
        
        #------------------------------------------------------------------------------------------------------------
        if useKDE:
            #########################################################################################################################
            ### slower in python
            start_time = time.time() # reset start-time
            # Adjust the bandwidth and compute KDE
            kde = gaussian_kde(raw_depth_flat, bw_method=0.05)  # Adjust the bw_method 0.2

            # Reduce the number of points in linspace
            x_range = np.linspace(raw_depth_flat.min(), raw_depth_flat.max(), num_buckets)  # Reduced to 20 for coarser and faster calc

            end_time_0 = time.time()
            elapsed_time_0 = end_time_0 - start_time

            plt.plot(x_range, kde(x_range), color='r')
            plt.show()
    #         print(f"done plotting")
            # Find peaks
            start_time = time.time() # reset start-time
            peaks, properties = find_peaks(kde(x_range), height=0, prominence=0)
            peak_centers = x_range[peaks].astype(int)
            peak_values = kde(x_range)[peaks]
            # Filter peaks to only include those within the [min_focus, max_focus] range
            valid_idx = (peak_centers >= min_focus) & (peak_centers <= max_focus)
            peak_centers = peak_centers[valid_idx].astype(int)
            peak_values = peak_values[valid_idx]

            # Get the widths of the peaks at half their prominence
            widths, _, _, _ = peak_widths(kde(x_range), peaks, rel_height=0.5)
            end_time_1 = time.time()
            elapsed_time_1 = end_time_1 - start_time
        else:
            #########################################################################################################################
            ### faster
            start_time = time.time() # reset start-time
            # Compute a histogram of the data
            counts, bin_edges = np.histogram(raw_depth_flat, bins=num_buckets, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Smooth the histogram (this acts as an approximation to the KDE)
            smoothed_counts = gaussian_filter1d(counts, sigma=sigma_value)
            
            end_time_0 = time.time()
            elapsed_time_0 = end_time_0 - start_time
            
            plt.plot(bin_centers, smoothed_counts, color='r')
            plt.title("Fast Density Estimate")
            plt.show()
            
            # Find peaks on the smoothed density curve
            start_time = time.time() # reset start-time
            
            # Find peaks on the smoothed density curve with specified sensitivity
            peaks, properties = find_peaks(smoothed_counts, height=min_height, prominence=min_prominence, distance=min_distance)
            peak_centers = bin_centers[peaks].astype(int)
            peak_values = smoothed_counts[peaks]
            widths, _, _, _ = peak_widths(smoothed_counts, peaks, rel_height=0.5)
            # Filter peaks to only include those within the [min_focus, max_focus] range
            valid_idx = (peak_centers >= min_focus) & (peak_centers <= max_focus)
            peak_centers = peak_centers[valid_idx].astype(int)
            peak_values = peak_values[valid_idx]
            
            widths = widths[valid_idx]
            end_time_1 = time.time()
            elapsed_time_1 = end_time_1 - start_time
            #########################################################################################################################       

        start_time = time.time() # reset start-time
        # Sort peaks based on heights (values)
        sorted_indices = np.argsort(-peak_values)
        sorted_peaks = [(peak_centers[i], peak_values[i]) for i in sorted_indices]

        # Sort peak widths based on the sorted peak indices and apply constraints
        sorted_peak_widths = [min(max(widths[i]*5, min_width), max_width) for i in sorted_indices]

        # Limit the number of peaks (e.g., top 5 peaks)
        #num_peaks = 5
        limited_sorted_peaks = sorted_peaks[:num_peaks]
        limited_sorted_peak_widths = sorted_peak_widths[:num_peaks]
        end_time_2 = time.time()
        elapsed_time_2 = end_time_2 - start_time

        print(f"Peak centers: {limited_sorted_peaks}")
        
        start_time = time.time() # reset start-time
        
        focus_range = (min_focus, max_focus) 
        compressed_depth, c980N = enhanced_16bit_to_8bit(raw_depth, focus_range, limited_sorted_peaks, limited_sorted_peak_widths
                                                      , res_oof_near, res_oof_far, res_gap, res_focus, num_channels, debug)

        end_time_3 = time.time()
        elapsed_time_3 = end_time_3 - start_time

        start_time = time.time() # reset start-time
        # Normalize the 16-bit depth map to the range [0, 255]
        normalized_depth = ((raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255).astype(np.uint8)

        # Equalize the 8-bit depth map
        equalized_depth_8bit = cv2.equalizeHist(normalized_depth)

        end_time_4 = time.time()
        elapsed_time_4 = end_time_4 - start_time

        print(f"Time to flatten: {elapsed_time_01 * 1000:.3f} ms")
        print(f"Time to rem outliers: {elapsed_time_02 * 1000:.3f} ms")
        print(f"Time to create counts (or kde if enabled): {elapsed_time_0 * 1000:.3f} ms")
        print(f"Time to find peaks X: {elapsed_time_1 * 1000:.3f} ms")
        print(f"Time to do sorting: {elapsed_time_2 * 1000:.3f} ms")
        print(f"Time to focus compress: {elapsed_time_3 * 1000:.3f} ms")
        print(f"Time to Equalize: {elapsed_time_4 * 1000:.3f} ms")


        # Normalize the 16-bit raw depth to 8-bit for visualization
        normalized_depth_8bit = ((raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255).astype(np.uint8)

        # If inpaint flag is set, perform inpainting
        if inpaint:
            start_time = time.time()
            # Create a mask of invalid depth values (assuming invalid values are set to 0)
            mask = (raw_depth == 0).astype(np.uint8)
            # Inpainting cv2.INPAINT_TELEA
            #raw_depth = cv2.inpaint(raw_depth, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)
            compressed_depth = cv2.inpaint(compressed_depth, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)
            end_time = time.time()
            elapsed_time_5 = end_time - start_time
            if debug:
                print(f"Time inpaint: {elapsed_time_5} seconds.")

            equalized_depth_8bit = cv2.inpaint(equalized_depth_8bit, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)
            # filtered_depth = joint_bilateral_filter(depth_img, rgb_img)

            normalized_depth_8bit = cv2.inpaint(normalized_depth_8bit, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)

        # Display the images side by side
        fig, axes = plt.subplots(1, 4, figsize=(25, 15))

        # Set the min and max values for 16-bit and 8-bit images
        min_val_16bit = 0
        max_val_16bit = 65535  # 2^16 - 1

        min_val_8bit = 0
        max_val_8bit = 255

        # Display raw depth image
        im_raw = axes[0].imshow(raw_depth, cmap='gray')
        im_raw.set_clim(min_val_16bit, max_val_16bit)  # Set the color limits for 16-bit
        axes[0].set_title('Raw Depth 16bit')
        axes[0].axis('off')

        # Display normalized depth image
        im_normalized = axes[1].imshow(normalized_depth_8bit, cmap='gray')
        im_normalized.set_clim(min_val_8bit, max_val_8bit)  # Set the color limits for 8-bit
        axes[1].set_title('Normalized Depth 8bit')
        axes[1].axis('off')

        # Display compressed depth image
        im_equalized = axes[2].imshow(equalized_depth_8bit, cmap='gray')
        im_equalized.set_clim(min_val_8bit, max_val_8bit)  # Set the color limits for 8-bit
        axes[2].set_title('Equalized Depth 8bit')
        axes[2].axis('off')

        # Display the compressed depth image
        if compressed_depth.ndim == 3 and compressed_depth.shape[2] == 2:
            # Convert the 2-channel image (RG) into a 3-channel RGB-like format
            height, width, _ = compressed_depth.shape
            rgb_image = np.zeros((height, width, 3), dtype=compressed_depth.dtype)
            # Copy the two channels into R and G
            rgb_image[..., 0:2] = compressed_depth  # R, G assigned
            # B remains zero

            im_compressed = axes[3].imshow(rgb_image)  # Display as color
            axes[3].set_title('Compressed Focus Depth (RG -> RGB)')
        else:
            # Single-channel (grayscale) image
            im_compressed = axes[3].imshow(compressed_depth, cmap='gray')
            im_compressed.set_clim(min_val_8bit, max_val_8bit)  # Set 8-bit color limits
            axes[3].set_title('Compressed Focus Depth (Grayscale)')

        plt.tight_layout()
        plt.show()

        # Plot  histograms and fix scale to the maximum frequency out of the three
        fig, axes = plt.subplots(1, 4, figsize=(25, 5))

        # Compute histogram values and bins for each image
        hist_raw, bins_raw = np.histogram(raw_depth[raw_depth != 0].flatten(), bins=256, density=True)
        hist_norm_8bit, bins_norm_8bit = np.histogram(normalized_depth_8bit[normalized_depth_8bit != 0].flatten(), bins=256, density=True)
        hist_equ_8bit, bins_equ_8bit = np.histogram(equalized_depth_8bit[equalized_depth_8bit != 0].flatten(), bins=256, density=True)
        if compressed_depth.ndim == 3 and compressed_depth.shape[2] == 2:
            hist_compressed, bins_compressed = np.histogram(c980N[c980N != 0].flatten(), bins=980, density=True)
        else:
            hist_compressed, bins_compressed = np.histogram(compressed_depth[compressed_depth != 0].flatten(), bins=256, density=True)
        

        # Determine the maximum density value for uniform scale
        max_density = max(hist_raw.max(), hist_equ_8bit.max(), hist_norm_8bit.max(), hist_compressed.max())
        print("Max densities: Raw: {}, Equalized 8bit: {}, Normalized 8bit: {}, Compressed: {}. Overall max: {}"
        .format(hist_raw.max(), hist_equ_8bit.max(), hist_norm_8bit.max(), hist_compressed.max(), max_density))

        # Plot and set y-axis limit for raw depth histogram
        axes[0].hist(raw_depth[raw_depth != 0].flatten(), bins=256, color='c', density=True)
        axes[0].set_title('Histogram of Raw Depth 16bit')
        axes[0].set_xlabel('Pixel Value')
        axes[0].set_ylabel('Density')
        axes[0].set_ylim(0, max_density)

        # Plot and set y-axis limit for normalized depth histogram
        axes[1].hist(normalized_depth_8bit[normalized_depth_8bit != 0].flatten(), bins=256, color='c', density=True)
        axes[1].set_title('Histogram of Normalized Depth 8bit')
        axes[1].set_xlabel('Pixel Value')
        axes[1].set_ylabel('Density')
        axes[1].set_ylim(0, max_density)

        # Plot and set y-axis limit for equalized depth histogram
        axes[2].hist(equalized_depth_8bit[equalized_depth_8bit != 0].flatten(), bins=256, color='c', density=True)
        axes[2].set_title('Histogram of Equalized Depth 8bit')
        axes[2].set_xlabel('Pixel Value')
        axes[2].set_ylabel('Density')
        axes[2].set_ylim(0, max_density)

        if compressed_depth.ndim == 3 and compressed_depth.shape[2] == 2:       
            axes[3].hist(c980N[c980N != 0], bins=980, color='c', density=True)
            axes[3].set_title('Histogram of Compressed Focus Depth (2-channel)')
        else:
            axes[3].hist(compressed_depth[compressed_depth != 0].flatten(), bins=256, color='c', density=True)
            axes[3].set_title('Histogram of Compressed Focus Depth (8-bit)')
        axes[3].set_xlabel('Pixel Value')
        axes[3].set_ylabel('Density')
        axes[3].set_ylim(0, max_density)

        plt.tight_layout()
        plt.show()

def getDepth(raw_depth, index, debug=False, ol_threshold=0,
             min_width=100, max_width=400, min_focus=300, max_focus=3600,
             res_oof_near=15, res_oof_far=20, res_gap=10, res_focus=2,
             num_channels=1, num_peaks=3, num_buckets=40, sigma_value=0.4,
             min_height=0.0002, min_prominence=0, min_distance=1,
             useKDE=True):
    """
    Process the raw 16-bit depth image and return a compressed 8-bit depth image.
    It identifies critical depth intervals via either KDE or a smoothed histogram,
    and uses the top peaks to drive focus correction.

    Parameters:
      raw_depth: numpy array or PIL Image of raw depth values.
      index: Identifier for the image.
      debug: If True, prints debug information.
      ol_threshold: Outlier threshold count.
      min_focus, max_focus: Focus range.
      num_channels: 1 for 8-bit grayscale, 2 for dual-channel RG.
      num_peaks: Maximum number of peaks to consider.
      num_buckets: Number of bins for histogram approximation.
      sigma_value: Sigma for Gaussian smoothing.
      min_height, min_prominence, min_distance: Sensitivity parameters for peak detection.
      useKDE: If True, use KDE to approximate the density; otherwise, use a smoothed histogram.
      
    Returns:
      compressed_depth: Processed depth image compressed to 8-bit.
    """

    # Convert raw_depth to numpy array if it is a PIL Image.
    if isinstance(raw_depth, Image.Image):
        raw_depth = np.array(raw_depth)

    # Flatten and remove invalid data (NaN and zeros)
    raw_depth_flat = raw_depth[~np.isnan(raw_depth) & (raw_depth != 0)].flatten()

    # Filter data to only include values within focus range
    in_focus_data = raw_depth_flat[(raw_depth_flat >= min_focus) & (raw_depth_flat <= max_focus)]
    if len(in_focus_data) == 0:
        if debug:
            print("DEBUG: No depth data in focus range.")
        return (raw_depth / 256).astype(np.uint8)

    # Remove outlier depth values if threshold is specified
    if ol_threshold > 0:
        unique_depths, counts = np.unique(in_focus_data, return_counts=True)
        outliers = unique_depths[counts <= ol_threshold]
        for outlier in outliers:
            idx = np.where(unique_depths == outlier)[0][0]
            replacement_value = unique_depths[idx - 1] if idx == len(unique_depths) - 1 else unique_depths[idx + 1]
            in_focus_data[in_focus_data == outlier] = replacement_value

    # -------------------------------------------------------------------------
    # Density Estimation and Peak Detection
    # -------------------------------------------------------------------------
    if useKDE:
        # Compute KDE from the entire raw depth flat array.
        kde = gaussian_kde(raw_depth_flat, bw_method=0.05)
        x_range = np.linspace(raw_depth_flat.min(), raw_depth_flat.max(), num_buckets)
        # Detect peaks on the KDE curve
        peaks, properties = find_peaks(kde(x_range), height=0, prominence=0)
        peak_centers = x_range[peaks].astype(int)
        peak_values = kde(x_range)[peaks]
        # Filter peaks to only include those within the focus range
        valid_idx = (peak_centers >= min_focus) & (peak_centers <= max_focus)
        peak_centers = peak_centers[valid_idx].astype(int)
        peak_values = peak_values[valid_idx]
        widths, _, _, _ = peak_widths(kde(x_range), peaks, rel_height=0.5)
    else:
        # Compute histogram as a fast approximation to the KDE.
        counts, bin_edges = np.histogram(raw_depth_flat, bins=num_buckets, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        from scipy.ndimage import gaussian_filter1d
        smoothed_counts = gaussian_filter1d(counts, sigma=sigma_value)
        peaks, properties = find_peaks(smoothed_counts, height=min_height, prominence=min_prominence, distance=min_distance)
        peak_centers = bin_centers[peaks].astype(int)
        peak_values = smoothed_counts[peaks]
        widths, _, _, _ = peak_widths(smoothed_counts, peaks, rel_height=0.5)
        valid_idx = (peak_centers >= min_focus) & (peak_centers <= max_focus)
        peak_centers = peak_centers[valid_idx].astype(int)
        peak_values = peak_values[valid_idx]
        widths = widths[valid_idx]

    # -------------------------------------------------------------------------
    # Sort Peaks by Their Heights and Limit the Number of Peaks
    # -------------------------------------------------------------------------
    sorted_indices = np.argsort(-peak_values)
    sorted_peaks = [(peak_centers[i], peak_values[i]) for i in sorted_indices]
    # Scale widths (multiply by 5) and constrain within [min_width, max_width]
    sorted_peak_widths = [min(max(widths[i] * 5, min_width), max_width) for i in sorted_indices]

    limited_sorted_peaks = sorted_peaks[:num_peaks]
    limited_sorted_peak_widths = sorted_peak_widths[:num_peaks]

    if debug:
        print(f"DEBUG: Peak centers: {limited_sorted_peaks}")

    focus_range = (min_focus, max_focus)
    # Compress the depth using enhanced_16bit_to_8bit with the computed peaks and widths.
    compressed_depth, _ = enhanced_16bit_to_8bit(raw_depth, focus_range, limited_sorted_peaks,
                                               limited_sorted_peak_widths, res_oof_near, res_oof_far,
                                               res_gap, res_focus, num_channels, debug)
    compressed_depth_8bit = (compressed_depth / 256).astype(np.uint8)
    return compressed_depth

    
def normals_to_grayscale(depth_map, 
                         sobel_kernel_size=7, 
                         smoothing_kernel_size=(7, 7)):
    """
    Convert a depth map to a grayscale image based on the direction of normals.
    This version applies Gaussian smoothing to the normalized angles for smoother transitions.
    
    Args:
        depth_map (numpy.ndarray): A depth map image.
        sobel_kernel_size (int): Kernel size for Sobel gradients.
        smoothing_kernel_size (tuple): Kernel size for Gaussian smoothing.
    
    Returns:
        numpy.ndarray: A grayscale image where pixel intensities represent normal directions.
    """
    # Compute gradients using a smaller Sobel kernel
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)

    # Calculate normals (using a constant z=1)
    normals = np.dstack((-grad_x, -grad_y, np.ones_like(depth_map)))

    # Compute the norm and avoid division by zero
    norm = np.linalg.norm(normals, axis=2)
    norm[norm == 0] = 1
    normals_normalized = normals / norm[:, :, np.newaxis]

    # Calculate angles (2D projection) from normals
    angles = np.arctan2(normals_normalized[:,:,1], normals_normalized[:,:,0])

    # Normalize angles to [0, 1]
    normalized_angles = (angles + np.pi) / (2 * np.pi)

    # Apply Gaussian smoothing on the normalized angles to achieve smoother transitions.
    # Note: It is important to smooth the floating-point normalized angles,
    # not the final 8-bit image.
    smoothed_normalized_angles = cv2.GaussianBlur(normalized_angles.astype(np.float32), 
                                                   smoothing_kernel_size, 0)

    # Convert the smoothed normalized angles to an 8-bit grayscale image
    grayscale_image = (smoothed_normalized_angles * 255).astype(np.uint8)

    return grayscale_image

