# coarse_matching.py

import cv2
import numpy as np
import torch
import kornia as K
import kornia.feature as KF

import os
import cv2
# Set the maximum allowed pixels to a higher value

from utils import (
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
    template_tensor = K.utils.image_to_tensor(template_cut, keepdim=True).float() / 255.0
    sample_tensor = K.utils.image_to_tensor(sample_resized, keepdim=True).float() / 255.0

    template_gray = K.color.rgb_to_grayscale(template_tensor).unsqueeze(0).to(device)
    sample_gray = K.color.rgb_to_grayscale(sample_tensor).unsqueeze(0).to(device)

    input_dict = {
        "image0": template_gray,
        "image1": sample_gray
    }

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()

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
    low_res_shape = (640, 480)

    # Scaling factors from low-res to full-res
    scale_y = full_res_template_shape[0] / low_res_shape[1]
    scale_x = full_res_template_shape[1] / low_res_shape[0]
    scale_y_2 = full_res_sample_shape[0] / low_res_shape[1]
    scale_x_2 = full_res_sample_shape[1] / low_res_shape[0]
    print(f"Template scales: ({scale_x}, {scale_y})")
    print(f"Sample scales: ({scale_x_2}, {scale_y_2})")

    # Resize images to low resolution
    scale_factor_padding=scale_factor
    shift_x=640*scale_factor_padding-640
    
    sample_resized = cv2.resize(sample_img, low_res_shape, interpolation=cv2.INTER_LINEAR)
    sample_resized_padded = pad_image(sample_resized,scale_factor_padding)
    sample_resized_padded = cv2.resize(sample_resized_padded, low_res_shape, interpolation=cv2.INTER_LINEAR)

    template_resized = cv2.resize(template_img, low_res_shape, interpolation=cv2.INTER_LINEAR)

    # Coarse matching using LoFTR
    matcher = KF.LoFTR(pretrained="outdoor").to(device)
    
    img1_tensor = K.utils.image_to_tensor(sample_resized_padded, keepdim=True).float() / 255.0
    img2_tensor = K.utils.image_to_tensor(template_resized, keepdim=True).float() / 255.0
    img1_gray = K.color.rgb_to_grayscale(img1_tensor).unsqueeze(0).to(device)
    img2_gray = K.color.rgb_to_grayscale(img2_tensor).unsqueeze(0).to(device)

    input_dict = {
        "image0": img1_gray,
        "image1": img2_gray,
    }
    with torch.inference_mode():
        correspondences = matcher(input_dict)
    
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()

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
    object_region_path = 'object_region_full_res.jpg'
    

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
    corresponding_sample_region_path = 'corresponding_sample_region_full_res.jpg'
    cv2.imwrite(corresponding_sample_region_path, corresponding_sample_region)

    sample_height, sample_width = corresponding_sample_region.shape[:2]
    resized_object_region = cv2.resize(object_region, (sample_width, sample_height))
    cv2.imwrite(object_region_path, resized_object_region)

    print("Coarse matching completed. Object region and corresponding sample region saved.")

    return object_region_path, corresponding_sample_region_path
