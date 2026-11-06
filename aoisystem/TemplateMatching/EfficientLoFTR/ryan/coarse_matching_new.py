# coarse_matching.py

import cv2
import numpy as np
import torch
import kornia as K
import kornia.feature as KF

import os
import cv2
import random
from copy import deepcopy
import matplotlib.cm as cm
from TemplateMatching.EfficientLoFTR.src.utils.plotting import make_matching_figure
from TemplateMatching.EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from TemplateMatching.EfficientLoFTR.lightglue import LightGlue, SuperPoint
from scipy.spatial import cKDTree
from TemplateMatching.EfficientLoFTR.lightglue.utils import load_image, rbd
from scipy.linalg import polar

from TemplateMatching.EfficientLoFTR.utils import (
    pad_image,
    resize_image,
    convert_to_grayscale,
    visualize_bbox,
    map_bbox_to_high_res,
    enlarge_bbox,
    save_image
)




def recalculate_homography(template_cut, sample_resized, matcher, device):
    # Same as before

     # Ensure images are not grayscale before converting to grayscale
    if len(sample_resized.shape) == 3 and sample_resized.shape[2] == 3:
        # Image is in color (3 channels)
        sample_gray = cv2.cvtColor(sample_resized, cv2.COLOR_BGR2GRAY)
    else:
        # Image is already grayscale
        sample_gray = sample_resized

    if len(template_cut.shape) == 3 and template_cut.shape[2] == 3:
        template_gray = cv2.cvtColor(template_cut, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template_cut

    sample_tensor = torch.from_numpy(sample_gray)[None][None].float().to(device) / 255.0
    template_tensor = torch.from_numpy(template_gray)[None][None].float().to(device) / 255.0

    # Prepare input dictionary
    input_dict = {
        "image0": template_tensor,
        "image1": sample_tensor,
    }

    # Perform inference using EfficientLoFTR
    with torch.no_grad():
        matcher(input_dict)

    mkpts0 = input_dict["mkpts0_f"].cpu().numpy()  # Points from sample image
    mkpts1 = input_dict["mkpts1_f"].cpu().numpy()  # Points from template image

    torch.cuda.empty_cache()

    sample_tensor=None
    template_tensor = None
    input_dict=None
    torch.cuda.empty_cache()


    # Recalculate homography
    H, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
    return H

def coarse_matching(sample_img_path, template_img_path, enlarge=False, margin_percent=0.08,scale_factor=1.6):
    """
    Performs coarse matching between the sample image and the template image.

    Parameters:
    - sample_img_path: Path to the sample image.
    - template_img_path: Path to the template image.
    - enlarge: Whether to enlarge the bounding box.
    - margin_percent: Margin percentage for enlarging the bounding box.

    Returns:
    - object_region_path: Path to the saved cropped template image.
    - corresponding_sample_region_path: Path to the saved cropped sample image.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the images
    sample_img = cv2.imread(sample_img_path)
    template_img = cv2.imread(template_img_path)

    print("template is successfully loaded.")
    if sample_img is None or template_img is None:
        print("Error: Could not load images.")
        return None, None

    full_res_sample_shape = sample_img.shape[:2]
    full_res_template_shape = template_img.shape[:2]
    low_res_shape = (1120, 1120)

    # Scaling factors from low-res to full-res
    scale_y = full_res_template_shape[0] / low_res_shape[1]
    scale_x = full_res_template_shape[1] / low_res_shape[0]
    scale_y_2 = full_res_sample_shape[0] / low_res_shape[1]
    scale_x_2 = full_res_sample_shape[1] / low_res_shape[0]
    print(f"Template scales: ({scale_x}, {scale_y})")
    print(f"Sample scales: ({scale_x_2}, {scale_y_2})")

    # Resize images to low resolution
    scale_factor_padding=scale_factor
    shift_x=1120*scale_factor_padding-1120
    
    sample_resized = cv2.resize(sample_img, low_res_shape, interpolation=cv2.INTER_LINEAR)
    sample_resized_padded = pad_image(sample_resized,scale_factor_padding)
    sample_resized_padded = cv2.resize(sample_resized_padded, low_res_shape, interpolation=cv2.INTER_LINEAR)

    template_resized = cv2.resize(template_img, low_res_shape, interpolation=cv2.INTER_LINEAR)


    
    _default_cfg = deepcopy(full_default_cfg)
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("TemplateMatching/EfficientLoFTR/weights/eloftr_outdoor.ckpt",weights_only=False)['state_dict'])
    matcher_eloftr = reparameter(matcher).eval().to(device)


    # Coarse matching using LoFTR
    # matcher = KF.LoFTR(pretrained="outdoor").to(device)


    if len(sample_resized_padded.shape) == 3 and sample_resized_padded.shape[2] == 3:
            # Image is in color (3 channels)
            sample_gray = cv2.cvtColor(sample_resized_padded, cv2.COLOR_BGR2GRAY)
    else:
        # Image is already grayscale
        sample_gray = sample_resized_padded

    if len(template_resized.shape) == 3 and template_resized.shape[2] == 3:
        template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template_resized

    sample_tensor = torch.from_numpy(sample_gray)[None][None].float().to(device) / 255.0
    template_tensor = torch.from_numpy(template_gray)[None][None].float().to(device) / 255.0

    # img1_tensor = K.utils.image_to_tensor(sample_resized_padded, keepdim=True).float() / 255.0
    # img2_tensor = K.utils.image_to_tensor(template_resized, keepdim=True).float() / 255.0
    # img1_gray = K.color.rgb_to_grayscale(img1_tensor).unsqueeze(0).to(device)
    # img2_gray = K.color.rgb_to_grayscale(img2_tensor).unsqueeze(0).to(device)

    input_dict = {
        "image0": sample_tensor,
        "image1": template_tensor,
    }
    sample_tensor=None
    template_tensor = None
    torch.cuda.empty_cache()
# Perform inference using EfficientLoFTR
    with torch.no_grad():
        matcher(input_dict)

    mkpts0 = input_dict["mkpts0_f"].cpu().numpy()  # Points from sample image
    mkpts1 = input_dict["mkpts1_f"].cpu().numpy()  # Points from template image

    input_dict = None

    # Homography estimation and warping
    H_low_res, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 8, confidence=0.999)

    allocated_memory = torch.cuda.memory_allocated()
    print("已分配的GPU内存：", allocated_memory)


    cached_memory = torch.cuda.memory_cached()
    print("已缓存的GPU内存：", cached_memory)
    

    sample_tensor = None
    template_tensor = None

    torch.cuda.empty_cache()

    allocated_memory = torch.cuda.memory_allocated()
    print("已分配的GPU内存：", allocated_memory)


    cached_memory = torch.cuda.memory_cached()
    print("已缓存的GPU内存：", cached_memory)

    print(H_low_res)
    warped_sample_on_template = cv2.warpPerspective(sample_resized_padded, H_low_res, low_res_shape)


    # Convert warped image to grayscale and calculate the bounding box
    binary_image = convert_to_grayscale(warped_sample_on_template)
    non_zero_coords = cv2.findNonZero(binary_image)
    x_min, y_min = np.min(non_zero_coords[:, 0, :], axis=0)
    x_max, y_max = np.max(non_zero_coords[:, 0, :], axis=0)
    low_res_bbox = {
        'left_upper': (x_min, y_min),
        'right_upper': (x_max, y_min),
        'left_lower': (x_min, y_max),
        'right_lower': (x_max, y_max)
    }

    # Visualize the bounding box on the low-resolution warped image
    visualize_bbox(warped_sample_on_template, low_res_bbox, "Low-Res Bounding Box")

    # Map the low-res bounding box to full-resolution template
    full_res_bbox = map_bbox_to_high_res(low_res_bbox, scale_x, scale_y)

    # Optionally enlarge the bounding box
    if enlarge:
        full_res_bbox = enlarge_bbox(full_res_bbox, margin_percent, full_res_template_shape)

    # Cut the object region from the full-resolution template
    object_region = template_img[full_res_bbox['left_upper'][1]:full_res_bbox['right_lower'][1],
                                 full_res_bbox['left_upper'][0]:full_res_bbox['right_lower'][0]]
    
    # Step 1: Extract the upper-left 2x2 affine part of H
    affine_part = H_low_res[:2, :2]

    # Step 2: Apply polar decomposition to extract the rotation component
    rotation_matrix, _ = polar(affine_part)

    # Step 3: Apply the rotation to img2 to align it with img1
    # Get the center of img2 to apply rotation around it
    center = (object_region.shape[1] // 2, object_region.shape[0] // 2)

    # Convert rotation matrix to angle and scale for OpenCV's warpAffine
    angle = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    scale = np.sqrt(np.linalg.det(rotation_matrix))  # Scaling factor (optional)

    print(angle)
    print(scale)
    # Generate the affine transformation matrix for rotation
    rotation_affine_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Apply rotation to img2
    img2_aligned = cv2.warpAffine(object_region, rotation_affine_matrix, (object_region.shape[1], object_region.shape[0]))
    
    

    object_region_path = 'object_region_full_res.jpg'
    

    # Recalculate H matrix using the cut object region and the sample image (low resolution)
    object_region_resized = cv2.resize(object_region, low_res_shape, interpolation=cv2.INTER_LINEAR)
    cached_memory = torch.cuda.memory_cached()
    print("已缓存的GPU内存：", cached_memory)
    H_fine = recalculate_homography(object_region_resized, sample_resized_padded, matcher, device)
    print(H_fine)

    # Warp the template cut back to the sample image in low resolution using the new H matrix
    warped_object_on_sample = cv2.warpPerspective(object_region_resized, H_fine, low_res_shape)

    # Convert the warped object back to grayscale and find the bounding box
    binary_image_fine = convert_to_grayscale(warped_object_on_sample)
    non_zero_coords_fine = cv2.findNonZero(binary_image_fine)
    x_min_fine, y_min_fine = np.min(non_zero_coords_fine[:, 0, :], axis=0)
    x_max_fine, y_max_fine = np.max(non_zero_coords_fine[:, 0, :], axis=0)

    x_min_fine=x_min_fine*scale_factor_padding-shift_x
    x_max_fine=x_max_fine*scale_factor_padding-shift_x
    low_res_bbox_fine = {
        'left_upper': (x_min_fine, y_min_fine),
        'right_upper': (x_max_fine, y_min_fine),
        'left_lower': (x_min_fine, y_max_fine),
        'right_lower': (x_max_fine, y_max_fine)
    }
    
    # Visualize the bounding box on the low-resolution sample
    # visualize_bbox(warped_object_on_sample, low_res_bbox_fine, "Low-Res Sample Bounding Box")

    # Map the low-res bounding box back to full-resolution sample
    full_res_bbox_fine = map_bbox_to_high_res(low_res_bbox_fine, scale_x_2, scale_y_2)

    # Optionally enlarge the bounding box
    if enlarge:
        full_res_bbox_fine = enlarge_bbox(full_res_bbox_fine, margin_percent, full_res_sample_shape)

    # Cut the corresponding region from the full-resolution sample image
    corresponding_sample_region = sample_img[full_res_bbox_fine['left_upper'][1]:full_res_bbox_fine['right_lower'][1],
                                             full_res_bbox_fine['left_upper'][0]:full_res_bbox_fine['right_lower'][0]]
    corresponding_sample_region_path = 'corresponding_sample_region_full_res.jpg'
    cv2.imwrite(corresponding_sample_region_path, corresponding_sample_region)

    sample_height, sample_width = corresponding_sample_region.shape[:2]
    resized_object_region = cv2.resize(img2_aligned, (sample_width, sample_height))
    cv2.imwrite(object_region_path, resized_object_region)

    print("Coarse matching completed. Object region and corresponding sample region saved.")

    return object_region_path, corresponding_sample_region_path

                                                                                                                                                                                                                      
def coarse_matching_super(sample_img_path, template_img_path, X_total, X_bar, Delta_X_sample, enlarge=False, margin_percent=0.08, scale_factor=1.6):
    """
    Performs coarse matching between the sample image and the template image.
    Returns additional information needed for processing multiple images.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the images
    sample_img = cv2.imread(sample_img_path)
    template_img = cv2.imread(template_img_path)

    if sample_img is None or template_img is None:
        print("Error: Could not load images.")
        return None

    full_res_sample_shape = sample_img.shape[:2]
    full_res_template_shape = template_img.shape[:2]
    low_res_shape = (1440, 1120)


    scale_y = full_res_template_shape[0] / low_res_shape[1]
    scale_x = full_res_template_shape[1] / low_res_shape[0]
    scale_y_2 = full_res_sample_shape[0] / low_res_shape[1]
    scale_x_2 = full_res_sample_shape[1] / low_res_shape[0]
    print(f"Template scales: ({scale_x}, {scale_y})")
    print(f"Sample scales: ({scale_x_2}, {scale_y_2})")


    # Resize images to low resolution
    scale_factor_padding = scale_factor
    shift_x = 1440 * scale_factor_padding - 1440

    sample_resized = cv2.resize(sample_img, low_res_shape, interpolation=cv2.INTER_LINEAR)
    sample_resized_padded = pad_image(sample_resized, scale_factor_padding)
    sample_resized_padded = cv2.resize(sample_resized_padded, low_res_shape, interpolation=cv2.INTER_LINEAR)

    template_resized = cv2.resize(template_img, low_res_shape, interpolation=cv2.INTER_LINEAR)

    # Initialize matcher
    _default_cfg = deepcopy(full_default_cfg)
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("TemplateMatching/EfficientLoFTR/weights/eloftr_outdoor.ckpt",weights_only=False)['state_dict'])
    matcher = reparameter(matcher).eval().to(device)

    # Convert images to grayscale
    sample_gray = convert_to_grayscale(sample_resized_padded)
    template_gray = convert_to_grayscale(template_resized)

    sample_tensor = torch.from_numpy(sample_gray)[None][None].float().to(device) / 255.0
    template_tensor = torch.from_numpy(template_gray)[None][None].float().to(device) / 255.0

    input_dict = {
        "image0": sample_tensor,
        "image1": template_tensor,
    }

    # Perform inference using LoFTR
    with torch.no_grad():
        matcher(input_dict)

    mkpts0 = input_dict["mkpts0_f"].cpu().numpy()  # Points from sample image
    mkpts1 = input_dict["mkpts1_f"].cpu().numpy()  # Points from template image

    # Homography estimation
    H_low_res, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 8, confidence=0.999)

    # Warp sample image onto template image
    warped_sample_on_template = cv2.warpPerspective(sample_resized_padded, H_low_res, low_res_shape)

    # Convert warped image to grayscale and calculate the bounding box
    binary_image = convert_to_grayscale(warped_sample_on_template)
    non_zero_coords = cv2.findNonZero(binary_image)
    x_min, y_min = np.min(non_zero_coords[:, 0, :], axis=0)
    x_max, y_max = np.max(non_zero_coords[:, 0, :], axis=0)
    low_res_bbox_template = {
        'left_upper': (x_min, y_min),
        'right_upper': (x_max, y_min),
        'left_lower': (x_min, y_max),
        'right_lower': (x_max, y_max)
    }

      # Visualize the bounding box on the low-resolution warped image
    visualize_bbox(warped_sample_on_template, low_res_bbox_template , "Low-Res Bounding Box")
    full_res_bbox_template = map_bbox_to_high_res(low_res_bbox_template,  scale_x, scale_y)

    # Optionally enlarge the bounding boxes
    if enlarge:
       
        full_res_bbox_template = enlarge_bbox(full_res_bbox_template, margin_percent, full_res_template_shape)


        # Cut the object region from the full-resolution template
    object_region = template_img[full_res_bbox_template['left_upper'][1]:full_res_bbox_template['right_lower'][1],
                                 full_res_bbox_template['left_upper'][0]:full_res_bbox_template['right_lower'][0]]

    # Recalculate H matrix using the cut object region and the sample image (low resolution)
    object_region_resized = cv2.resize(object_region, low_res_shape, interpolation=cv2.INTER_LINEAR)
    H_fine = recalculate_homography(object_region_resized, sample_resized_padded, matcher, device)

    # Warp the template cut back to the sample image in low resolution using the new H matrix
    warped_object_on_sample = cv2.warpPerspective(object_region_resized, H_fine, low_res_shape)

    # Convert the warped object back to grayscale and find the bounding box
    binary_image_fine = convert_to_grayscale(warped_object_on_sample)
    non_zero_coords_fine = cv2.findNonZero(binary_image_fine)
    x_min_fine, y_min_fine = np.min(non_zero_coords_fine[:, 0, :], axis=0)
    x_max_fine, y_max_fine = np.max(non_zero_coords_fine[:, 0, :], axis=0)

    x_min_fine=x_min_fine*scale_factor_padding-shift_x
    x_max_fine=x_max_fine*scale_factor_padding-shift_x
    low_res_bbox_fine = {
        'left_upper': (x_min_fine, y_min_fine),
        'right_upper': (x_max_fine, y_min_fine),
        'left_lower': (x_min_fine, y_max_fine),
        'right_lower': (x_max_fine, y_max_fine)
    }
    
    # Visualize the bounding box on the low-resolution sample
    # visualize_bbox(warped_object_on_sample, low_res_bbox_fine, "Low-Res Sample Bounding Box")

    # Map the low-res bounding box back to full-resolution sample
    full_res_bbox_fine = map_bbox_to_high_res(low_res_bbox_fine, scale_x_2, scale_y_2)

    # Optionally enlarge the bounding box
    if enlarge:
        full_res_bbox_fine = enlarge_bbox(full_res_bbox_fine, margin_percent, full_res_sample_shape)

    # Cut the corresponding region from the full-resolution sample image
    corresponding_sample_region = sample_img[full_res_bbox_fine['left_upper'][1]:full_res_bbox_fine['right_lower'][1],
                                             full_res_bbox_fine['left_upper'][0]:full_res_bbox_fine['right_lower'][0]]
    
    os.makedirs('./demo/refined_sample_regions', exist_ok=True)
    sample_subregions_refined_path = os.path.join('./demo/refined_sample_regions', f'sample_0.png')
    cv2.imwrite(sample_subregions_refined_path, corresponding_sample_region)

    
    corresponding_sample_region_path = 'demo/sample_regions/sample_0.jpg'


    X_0_in_pixel=full_res_bbox_fine['right_lower'][0]-full_res_bbox_fine['left_upper'][0]

    # Calculate the width of the bounding boxes in pixels
    bbox_width_pixels_template = full_res_bbox_template['right_lower'][0] - full_res_bbox_template['left_upper'][0]
    x_0_prime_in_pixel=full_res_bbox_template['right_lower'][0]

    # Calculate pixels per mm in sample image
    pixel_width_sample = sample_img.shape[1]
    X_bar_in_pixel = pixel_width_sample
    pixels_per_mm_sample = pixel_width_sample / X_bar
    Delta_X_sample_in_pixel=Delta_X_sample *pixels_per_mm_sample
    X_total_sample_in_pixel = X_total * pixels_per_mm_sample

    # Calculate X_bar()
    # X_0_in_mm =X_0_in_pixel / pixels_per_mm_sample

    # Calculate number of object regions needed to cover the PCB
    remaining_width_sample_in_total = X_total_sample_in_pixel - X_0_in_pixel
    num_additional_regions = int(np.ceil(remaining_width_sample_in_total / Delta_X_sample_in_pixel))
    num_object_regions = 1 + num_additional_regions  # Including the first region
    
    delta_x_prime_in_pixel=Delta_X_sample_in_pixel*(x_0_prime_in_pixel/X_0_in_pixel)


    # Extract object region from full-res template
    object_region = template_img[full_res_bbox_template['left_upper'][1]:full_res_bbox_template['right_lower'][1],
                                 full_res_bbox_template['left_upper'][0]:full_res_bbox_template['right_lower'][0]]
    object_region_path = 'datasets/obj_regions/object_region_full_res_0.jpg'
    object_region_refined_path = 'demo/refined_obj_regions/object_region_full_res_0.png'
    cv2.imwrite(object_region_path, object_region)
    cv2.imwrite(object_region_refined_path, object_region)

    # Generate the rest of the object regions from the template image
    object_region_paths = [object_region_path]  # Include the first object region

    for i in range(1, num_object_regions):
        shift = int(i * delta_x_prime_in_pixel)
        left = full_res_bbox_template['left_upper'][0] + shift
        right = full_res_bbox_template['right_lower'][0] + shift

        # Ensure the bbox doesn't exceed image boundaries
        if right > full_res_template_shape[1]:
            right = full_res_template_shape[1]
            left = right - bbox_width_pixels_template
            if left < 0:
                left = 0  # Adjust if beyond the image boundary

        # Extract object region
        object_region_i = template_img[full_res_bbox_template['left_upper'][1]:full_res_bbox_template['right_lower'][1],
                                       int(left):int(right)]
        object_region_i_path = f'datasets/obj_regions/object_region_full_res_{i}.jpg'
        cv2.imwrite(object_region_i_path, object_region_i)
        object_region_paths.append(object_region_i_path)
        cv2.imwrite(corresponding_sample_region_path, corresponding_sample_region)

    # Return corresponding sample region path (for the first image)
    # You may extract the corresponding region from the sample image if needed
    # corresponding_sample_region_path = corresponding_sample_region  # Using the original sample image

    # Return all the required information
    return (object_region_paths[0],
            object_region_paths,
            sample_subregions_refined_path,
            num_object_regions)



def coarse_matching_each(sample_img_path, template_img_path, idx, enlarge=False, margin_percent=0.08,scale_factor=1.6):
    """
    Performs coarse matching between the sample image and the template image.

    Parameters:
    - sample_img_path: Path to the sample image.
    - template_img_path: Path to the template image.
    - enlarge: Whether to enlarge the bounding box.
    - margin_percent: Margin percentage for enlarging the bounding box.

    Returns:
    - object_region_path: Path to the saved cropped template image.
    - corresponding_sample_region_path: Path to the saved cropped sample image.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the images
    sample_img = cv2.imread(sample_img_path)
    template_img = cv2.imread(template_img_path)

    print("template is successfully loaded.")
    if sample_img is None or template_img is None:
        print("Error: Could not load images.")
        return None, None

    full_res_sample_shape = sample_img.shape[:2]
    full_res_template_shape = template_img.shape[:2]
    low_res_shape = (1600, 1120)

    # Scaling factors from low-res to full-res
    scale_y = full_res_template_shape[0] / low_res_shape[1]
    scale_x = full_res_template_shape[1] / low_res_shape[0]
    scale_y_2 = full_res_sample_shape[0] / low_res_shape[1]
    scale_x_2 = full_res_sample_shape[1] / low_res_shape[0]
    print(f"Template scales: ({scale_x}, {scale_y})")
    print(f"Sample scales: ({scale_x_2}, {scale_y_2})")

    # Resize images to low resolution
    scale_factor_padding=scale_factor
    shift_x=1600*scale_factor_padding-1600
    
    sample_resized = cv2.resize(sample_img, low_res_shape, interpolation=cv2.INTER_LINEAR)
    sample_resized_padded = pad_image(sample_resized,scale_factor_padding)
    sample_resized_padded = cv2.resize(sample_resized_padded, low_res_shape, interpolation=cv2.INTER_LINEAR)

    template_resized = cv2.resize(template_img, low_res_shape, interpolation=cv2.INTER_LINEAR)


    
    _default_cfg = deepcopy(full_default_cfg)
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("TemplateMatching/EfficientLoFTR/weights/eloftr_outdoor.ckpt",weights_only=False)['state_dict'])
    matcher_eloftr = reparameter(matcher).eval().to(device)


    # Coarse matching using LoFTR
    # matcher = KF.LoFTR(pretrained="outdoor").to(device)


    if len(sample_resized_padded.shape) == 3 and sample_resized_padded.shape[2] == 3:
            # Image is in color (3 channels)
            sample_gray = cv2.cvtColor(sample_resized_padded, cv2.COLOR_BGR2GRAY)
    else:
        # Image is already grayscale
        sample_gray = sample_resized_padded

    if len(template_resized.shape) == 3 and template_resized.shape[2] == 3:
        template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template_resized

    sample_tensor = torch.from_numpy(sample_gray)[None][None].float().to(device) / 255.0
    template_tensor = torch.from_numpy(template_gray)[None][None].float().to(device) / 255.0

    # img1_tensor = K.utils.image_to_tensor(sample_resized_padded, keepdim=True).float() / 255.0
    # img2_tensor = K.utils.image_to_tensor(template_resized, keepdim=True).float() / 255.0
    # img1_gray = K.color.rgb_to_grayscale(img1_tensor).unsqueeze(0).to(device)
    # img2_gray = K.color.rgb_to_grayscale(img2_tensor).unsqueeze(0).to(device)

    input_dict = {
        "image0": sample_tensor,
        "image1": template_tensor,
    }
# Perform inference using EfficientLoFTR
    with torch.no_grad():
        matcher(input_dict)

    mkpts0 = input_dict["mkpts0_f"].cpu().numpy()  # Points from sample image
    mkpts1 = input_dict["mkpts1_f"].cpu().numpy()  # Points from template image

    # Homography estimation and warping
    H_low_res, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 8, confidence=0.999)
    warped_sample_on_template = cv2.warpPerspective(sample_resized_padded, H_low_res, low_res_shape)


    # Convert warped image to grayscale and calculate the bounding box
    binary_image = convert_to_grayscale(warped_sample_on_template)
    non_zero_coords = cv2.findNonZero(binary_image)
    x_min, y_min = np.min(non_zero_coords[:, 0, :], axis=0)
    x_max, y_max = np.max(non_zero_coords[:, 0, :], axis=0)
    low_res_bbox = {
        'left_upper': (x_min, y_min),
        'right_upper': (x_max, y_min),
        'left_lower': (x_min, y_max),
        'right_lower': (x_max, y_max)
    }

    # Visualize the bounding box on the low-resolution warped image
    visualize_bbox(warped_sample_on_template, low_res_bbox, "Low-Res Bounding Box")

    # Map the low-res bounding box to full-resolution template
    full_res_bbox = map_bbox_to_high_res(low_res_bbox, scale_x, scale_y)

    # Optionally enlarge the bounding box
    if enlarge:
        full_res_bbox = enlarge_bbox(full_res_bbox, margin_percent, full_res_template_shape)

    # Cut the object region from the full-resolution template
    object_region = template_img[full_res_bbox['left_upper'][1]:full_res_bbox['right_lower'][1],
                                 full_res_bbox['left_upper'][0]:full_res_bbox['right_lower'][0]]
    

    os.makedirs('./demo/refined_obj_regions', exist_ok=True)
    obj_subregions_path = os.path.join('./demo/refined_obj_regions', f'object_region_full_res_{idx}.png')
    cv2.imwrite(obj_subregions_path, object_region)
    print(f"Refined object origin image saved to {obj_subregions_path}.")
    # object_region_path = './demo/refined_object_region_full_res.jpg'
    

    # Recalculate H matrix using the cut object region and the sample image (low resolution)
    object_region_resized = cv2.resize(object_region, low_res_shape, interpolation=cv2.INTER_LINEAR)
    H_fine = recalculate_homography(object_region_resized, sample_resized_padded, matcher, device)

    # Warp the template cut back to the sample image in low resolution using the new H matrix
    warped_object_on_sample = cv2.warpPerspective(object_region_resized, H_fine, low_res_shape)

    # Convert the warped object back to grayscale and find the bounding box
    binary_image_fine = convert_to_grayscale(warped_object_on_sample)
    non_zero_coords_fine = cv2.findNonZero(binary_image_fine)
    x_min_fine, y_min_fine = np.min(non_zero_coords_fine[:, 0, :], axis=0)
    x_max_fine, y_max_fine = np.max(non_zero_coords_fine[:, 0, :], axis=0)

    x_min_fine=x_min_fine*scale_factor_padding-shift_x
    x_max_fine=x_max_fine*scale_factor_padding-shift_x
    low_res_bbox_fine = {
        'left_upper': (x_min_fine, y_min_fine),
        'right_upper': (x_max_fine, y_min_fine),
        'left_lower': (x_min_fine, y_max_fine),
        'right_lower': (x_max_fine, y_max_fine)
    }
    
    # Visualize the bounding box on the low-resolution sample
    # visualize_bbox(warped_object_on_sample, low_res_bbox_fine, "Low-Res Sample Bounding Box")

    # Map the low-res bounding box back to full-resolution sample
    full_res_bbox_fine = map_bbox_to_high_res(low_res_bbox_fine, scale_x_2, scale_y_2)

    # Optionally enlarge the bounding box
    if enlarge:
        full_res_bbox_fine = enlarge_bbox(full_res_bbox_fine, margin_percent, full_res_sample_shape)

    # Cut the corresponding region from the full-resolution sample image
    corresponding_sample_region = sample_img[full_res_bbox_fine['left_upper'][1]:full_res_bbox_fine['right_lower'][1],
                                             full_res_bbox_fine['left_upper'][0]:full_res_bbox_fine['right_lower'][0]]
    

    os.makedirs('./demo/refined_sample_regions', exist_ok=True)
    sample_subregions_path = os.path.join('./demo/refined_sample_regions', f'sample_{idx}.png')
    cv2.imwrite(sample_subregions_path, corresponding_sample_region)
    print(f"Refined sample image saved to {sample_subregions_path}.")


    # corresponding_sample_region_path = 'corresponding_sample_region_full_res.jpg'
    # cv2.imwrite(corresponding_sample_region_path, corresponding_sample_region)

    sample_height, sample_width = corresponding_sample_region.shape[:2]
    resized_object_region = cv2.resize(object_region, (sample_width, sample_height))


    print("Coarse matching completed. Object region and corresponding sample region saved.")

    return obj_subregions_path, sample_subregions_path


def coarse_matching_eloftr(sample_img_path, template_img_path, enlarge=False, margin_percent=0.08,scale_factor=1.6):
    """
    Performs coarse matching between the sample image and the template image.

    Parameters:
    - sample_img_path: Path to the sample image.
    - template_img_path: Path to the template image.
    - enlarge: Whether to enlarge the bounding box.
    - margin_percent: Margin percentage for enlarging the bounding box.

    Returns:
    - object_region_path: Path to the saved cropped template image.
    - corresponding_sample_region_path: Path to the saved cropped sample image.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the images
    sample_img = cv2.imread(sample_img_path)
    template_img = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)

    print("template is successfully loaded.")
    if sample_img is None or template_img is None:
        print("Error: Could not load images.")
        return None, None

    full_res_sample_shape = sample_img.shape[:2]
    full_res_template_shape = template_img.shape[:2]
    low_res_shape = (1280, 960)

    # Scaling factors from low-res to full-res
    scale_y = full_res_template_shape[0] / low_res_shape[1]
    scale_x = full_res_template_shape[1] / low_res_shape[0]
    scale_y_2 = full_res_sample_shape[0] / low_res_shape[1]
    scale_x_2 = full_res_sample_shape[1] / low_res_shape[0]
    # print(f"Template scales: ({scale_x}, {scale_y})")
    # print(f"Sample scales: ({scale_x_2}, {scale_y_2})")

    # Resize images to low resolution
    scale_factor_padding=scale_factor
    shift_x=1280*scale_factor_padding-1280
    
    sample_resized = cv2.resize(sample_img, low_res_shape, interpolation=cv2.INTER_LINEAR)
    sample_resized_padded = pad_image(sample_resized,scale_factor_padding)
    sample_resized_padded = cv2.resize(sample_resized_padded, low_res_shape, interpolation=cv2.INTER_LINEAR)

    sample_resized_padded_gray = cv2.cvtColor(sample_resized_padded, cv2.COLOR_BGR2GRAY)

    template_resized = cv2.resize(template_img, low_res_shape, interpolation=cv2.INTER_LINEAR)


    _default_cfg = deepcopy(full_default_cfg)
    print(_default_cfg)
    cfg = _default_cfg
    cfg["match_coarse"]["thr"] = 0.015

    matcher = LoFTR(config=cfg)
    matcher.load_state_dict(torch.load("TemplateMatching/EfficientLoFTR/weights/eloftr_outdoor.ckpt",weights_only=False)['state_dict'])
    # matcher.load_state_dict(torch.load("TemplateMatching/EfficientLoFTR/weights/last.ckpt",weights_only=False)['state_dict'])
    matcher = reparameter(matcher)
    matcher = matcher.eval().cuda()
    
    img0 = torch.from_numpy(sample_resized_padded_gray)[None][None].cuda() / 255.
    img1 = torch.from_numpy(template_resized)[None][None].cuda() / 255.
    # Coarse matching using LoFTR
    input_dict = {
        "image0": img0,
        "image1": img1,
    }
    with torch.no_grad():
        matcher(input_dict)
    
    mkpts0 = input_dict['mkpts0_f'].cpu().numpy()
    mkpts1 = input_dict['mkpts1_f'].cpu().numpy()
    confidences = input_dict['mconf'].cpu().numpy()
    # color = cm.jet(confidences)
    # text = [
    #     'LoFTR',
    #     'Matches: {}'.format(len(mkpts0)),
    # ]
    # fig1 = make_matching_figure(sample_resized_padded_gray, template_resized, mkpts0, mkpts1, color, text=text)
    # plt.show()

    # Homography estimation and warping
    H_low_res, mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 8, 0.999, 10000)
    inliers_mkpts0 = mkpts0[mask.ravel() == 1]
    inliers_mkpts1 = mkpts1[mask.ravel() == 1]
    inlier_mconf = confidences[mask.ravel() == 1]


    # color = cm.jet(inlier_mconf)
    # text = [
    #     'LoFTR',
    #     'Matches: {}'.format(len(inliers_mkpts0)),
    # ]
    # fig1 = make_matching_figure(sample_resized_padded_gray, template_resized, inliers_mkpts0, inliers_mkpts1, color, text=text)

    # plt.show()
    warped_sample_on_template = cv2.warpPerspective(sample_resized_padded, H_low_res, low_res_shape)


    if len(warped_sample_on_template.shape) == 2:  # Grayscale image
            binary_image = warped_sample_on_template
    else:  # Color image
            binary_image = convert_to_grayscale(warped_sample_on_template)

    non_zero_coords = cv2.findNonZero(binary_image)
    x_min, y_min = np.min(non_zero_coords[:, 0, :], axis=0)
    x_max, y_max = np.max(non_zero_coords[:, 0, :], axis=0)
    low_res_bbox = {
        'left_upper': (x_min, y_min),
        'right_upper': (x_max, y_min),
        'left_lower': (x_min, y_max),
        'right_lower': (x_max, y_max)
    }

    # Visualize the bounding box on the low-resolution warped image
    # visualize_bbox(warped_sample_on_template, low_res_bbox, "Low-Res Bounding Box")

    # Map the low-res bounding box to full-resolution template
    full_res_bbox = map_bbox_to_high_res(low_res_bbox, scale_x, scale_y)

    # Optionally enlarge the bounding box
    if enlarge:
        full_res_bbox = enlarge_bbox(full_res_bbox, margin_percent, full_res_template_shape)

    # Cut the object region from the full-resolution template
    object_region = template_img[full_res_bbox['left_upper'][1]:full_res_bbox['right_lower'][1],
                                 full_res_bbox['left_upper'][0]:full_res_bbox['right_lower'][0]]
    object_region_path = 'object_region_full_res.jpg'
    

    # Recalculate H matrix using the cut object region and the sample image (low resolution)
    object_region_resized = cv2.resize(object_region, low_res_shape, interpolation=cv2.INTER_LINEAR)

    torch.cuda.empty_cache()

    img0=None
    img1 = None
    input_dict=None
    torch.cuda.empty_cache()

    H_fine = recalculate_homography(object_region_resized, sample_resized_padded, matcher, device)

    # Warp the template cut back to the sample image in low resolution using the new H matrix
    warped_object_on_sample = cv2.warpPerspective(object_region_resized, H_fine, low_res_shape)

    # Convert the warped object back to grayscale and find the bounding box

    if len(warped_object_on_sample.shape) == 2:  # Grayscale image
        binary_image_fine = warped_object_on_sample
    else:  # Color image
        binary_image_fine = convert_to_grayscale(warped_object_on_sample)


    non_zero_coords_fine = cv2.findNonZero(binary_image_fine)
    x_min_fine, y_min_fine = np.min(non_zero_coords_fine[:, 0, :], axis=0)
    x_max_fine, y_max_fine = np.max(non_zero_coords_fine[:, 0, :], axis=0)

    x_min_fine=x_min_fine*scale_factor_padding-shift_x
    x_max_fine=x_max_fine*scale_factor_padding-shift_x
    low_res_bbox_fine = {
        'left_upper': (x_min_fine, y_min_fine),
        'right_upper': (x_max_fine, y_min_fine),
        'left_lower': (x_min_fine, y_max_fine),
        'right_lower': (x_max_fine, y_max_fine)
    }
    
    # Visualize the bounding box on the low-resolution sample
    # visualize_bbox(warped_object_on_sample, low_res_bbox_fine, "Low-Res Sample Bounding Box")

    # Map the low-res bounding box back to full-resolution sample
    full_res_bbox_fine = map_bbox_to_high_res(low_res_bbox_fine, scale_x_2, scale_y_2)

    # Optionally enlarge the bounding box
    if enlarge:
        full_res_bbox_fine = enlarge_bbox(full_res_bbox_fine, margin_percent, full_res_sample_shape)

    # Cut the corresponding region from the full-resolution sample image
    corresponding_sample_region = sample_img[full_res_bbox_fine['left_upper'][1]:full_res_bbox_fine['right_lower'][1],
                                             full_res_bbox_fine['left_upper'][0]:full_res_bbox_fine['right_lower'][0]]
    corresponding_sample_region_path = 'corresponding_sample_region_full_res.jpg'
    cv2.imwrite(corresponding_sample_region_path, corresponding_sample_region)

    sample_height, sample_width = corresponding_sample_region.shape[:2]
    resized_object_region = cv2.resize(object_region, (sample_width, sample_height))
    cv2.imwrite(object_region_path, resized_object_region)

    print("Coarse matching completed. Object region and corresponding sample region saved.")

    return object_region_path, corresponding_sample_region_path
