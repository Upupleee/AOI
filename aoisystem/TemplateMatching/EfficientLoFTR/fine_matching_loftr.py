# fine_matching.py

import cv2
import numpy as np
import os
import time
import kornia as K
import kornia.feature as KF
import torch
import random
from copy import deepcopy
import matplotlib.cm as cm
from TemplateMatching.EfficientLoFTR.src.utils.plotting import make_matching_figure
from TemplateMatching.EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from TemplateMatching.EfficientLoFTR.lightglue import LightGlue, SuperPoint
from scipy.spatial import cKDTree
from TemplateMatching.EfficientLoFTR.lightglue.utils import load_image, rbd
from PIL import Image
# 解除大图像素限制
Image.MAX_IMAGE_PIXELS = None

def match_subregions_eloftr_sp_lightglue(sample_subregion, template_subregion, subregion_index, device, model_type='full', precision='fp32'):
    # Resize the subregions
    sample_subregion_resized = resize_image(sample_subregion)
    template_subregion_resized = resize_image(template_subregion)

 # Ensure images are not grayscale before converting to grayscale
    if len(sample_subregion_resized.shape) == 3 and sample_subregion_resized.shape[2] == 3:
        # Image is in color (3 channels)
        sample_gray = cv2.cvtColor(sample_subregion_resized, cv2.COLOR_BGR2GRAY)
    else:
        # Image is already grayscale
        sample_gray = sample_subregion_resized

    if len(template_subregion_resized.shape) == 3 and template_subregion_resized.shape[2] == 3:
        template_gray = cv2.cvtColor(template_subregion_resized, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template_subregion_resized

    sample_tensor = torch.from_numpy(sample_gray)[None][None].float().to(device) / 255.0
    template_tensor = torch.from_numpy(template_gray)[None][None].float().to(device) / 255.0

    # Initialize EfficientLoFTR matcher
    if model_type == 'full':
        _default_cfg = deepcopy(full_default_cfg)
    elif model_type == 'opt':
        _default_cfg = deepcopy(opt_default_cfg)

    if precision == 'mp':
        _default_cfg['mp'] = True
    elif precision == 'fp16':
        _default_cfg['half'] = True

    matcher_eloftr = LoFTR(config=_default_cfg)
    matcher_eloftr.load_state_dict(torch.load("weights/eloftr_outdoor.ckpt")['state_dict'])
    matcher_eloftr = reparameter(matcher_eloftr).eval().to(device)

    # Perform inference using EfficientLoFTR
    input_dict = {
        "image0": sample_tensor,
        "image1": template_tensor,
    }

    with torch.no_grad():
        matcher_eloftr(input_dict)

    mkpts0 = input_dict["mkpts0_f"].cpu().numpy()  # Points from sample image
    mkpts1 = input_dict["mkpts1_f"].cpu().numpy()  # Points from template image

    # Extract SuperPoint features
    extractor = SuperPoint(max_num_keypoints=2048*2).eval().to(device)
    feats0 = extractor.extract(sample_tensor)
    feats1 = extractor.extract(template_tensor)

    # Convert SuperPoint keypoints to NumPy arrays
    sp_keypoints0 = feats0['keypoints'].cpu().numpy()[0]  # SuperPoint keypoints for sample image
    sp_keypoints1 = feats1['keypoints'].cpu().numpy()[0]  # SuperPoint keypoints for template image

    # Define the maximum distance to consider keypoints as overlapping
    max_distance = 4  # Adjust based on your requirements

    # Build KDTree for SuperPoint keypoints
    sp_tree0 = cKDTree(sp_keypoints0)
    sp_tree1 = cKDTree(sp_keypoints1)

    # Query overlapping indices
    overlapping_indices0 = sp_tree0.query_ball_point(mkpts0, r=max_distance)
    overlapping_indices1 = sp_tree1.query_ball_point(mkpts1, r=max_distance)

    # Flatten and convert indices to tensor
    indices0 = np.hstack(overlapping_indices0).astype(int)
    indices1 = np.hstack(overlapping_indices1).astype(int)

    indices0_tensor = torch.from_numpy(indices0).long().to(device)
    indices1_tensor = torch.from_numpy(indices1).long().to(device)

    keypoints_shape = feats0['keypoints'].shape 
    print(f"feats0['keypoints'] shape: {keypoints_shape}")
    print(f"indices0_tensor min: {indices0_tensor.min()}, max: {indices0_tensor.max()}")

    # Extract overlapping features
    overlapping_feats0 = {
        'keypoints': feats0['keypoints'][:, indices0_tensor, :],
        'keypoint_scores': feats0['keypoint_scores'][:, indices0_tensor],
        'descriptors': feats0['descriptors'][:, indices0_tensor, :],
        'image_size': feats0['image_size'],
    }

    overlapping_feats1 = {
        'keypoints': feats1['keypoints'][:, indices1_tensor, :],
        'keypoint_scores': feats1['keypoint_scores'][:, indices1_tensor],
        'descriptors': feats1['descriptors'][:, indices1_tensor, :],
        'image_size': feats1['image_size'],
    }

    # Initialize LightGlue matcher
    matcher_lightglue = LightGlue(features='superpoint').eval().to(device)

    # Use LightGlue as per your pipeline
    with torch.no_grad():
        matches_glue = matcher_lightglue({'image0': overlapping_feats0, 'image1': overlapping_feats1})

    # Remove batch dimension using rbd
    feats0_rbd, feats1_rbd, matches01 = [rbd(x) for x in [overlapping_feats0, overlapping_feats1, matches_glue]]

    matches = matches01['matches']  # Indices with shape (K, 2)

    # Extract matched keypoints
    points0 = feats0_rbd['keypoints'][matches[..., 0]]  # Coordinates in image #0, shape (K, 2)
    points1 = feats1_rbd['keypoints'][matches[..., 1]]  # Coordinates in image #1, shape (K, 2)

    
    print(points0.shape)
    print(points0)
    print(points1)

    # Convert keypoints to NumPy arrays
    mkpts0_matched = points0.cpu().numpy()
    mkpts1_matched = points1.cpu().numpy()

    print(f"Subregion {subregion_index + 1}: {mkpts0_matched.shape[0]} matches found.")

    if mkpts0_matched.shape[0] < 4:
        print(f"Not enough matches to compute homography for subregion {subregion_index + 1}.")
        return None  # Handle insufficient matches appropriately

    src_pts = mkpts1_matched
    dst_pts = mkpts0_matched

    # Compute homography matrix using RANSAC
    homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return homography_matrix




def match_subregions_lightglue(sample_subregion, template_subregion, subregion_index, device, model_type='full', precision='fp32'):
    # Resize the subregions
    sample_subregion_resized = resize_image(sample_subregion)
    template_subregion_resized = resize_image(template_subregion)

 # Ensure images are not grayscale before converting to grayscale
    if len(sample_subregion_resized.shape) == 3 and sample_subregion_resized.shape[2] == 3:
        # Image is in color (3 channels)
        sample_gray = cv2.cvtColor(sample_subregion_resized, cv2.COLOR_BGR2GRAY)
    else:
        # Image is already grayscale
        sample_gray = sample_subregion_resized

    if len(template_subregion_resized.shape) == 3 and template_subregion_resized.shape[2] == 3:
        template_gray = cv2.cvtColor(template_subregion_resized, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template_subregion_resized

    sample_tensor = torch.from_numpy(sample_gray)[None][None].float().to(device) / 255.0
    template_tensor = torch.from_numpy(template_gray)[None][None].float().to(device) / 255.0

    
    # Extract SuperPoint features
    extractor = SuperPoint(max_num_keypoints=2048*2).eval().to(device)
    feats0 = extractor.extract(sample_tensor)
    feats1 = extractor.extract(template_tensor)

    
    keypoints_shape = feats0['keypoints'].shape 
    print(f"feats0['keypoints'] shape: {keypoints_shape}")
  
    # Initialize LightGlue matcher
    matcher_lightglue = LightGlue(features='superpoint').eval().to(device)

    # Use LightGlue as per your pipeline
    with torch.no_grad():
        matches_glue = matcher_lightglue({'image0': feats0, 'image1': feats1})

    # Remove batch dimension using rbd
    feats0_rbd, feats1_rbd, matches01 = [rbd(x) for x in [feats0, feats1, matches_glue]]

    matches = matches01['matches']  # Indices with shape (K, 2)

    # Extract matched keypoints
    points0 = feats0_rbd['keypoints'][matches[..., 0]]  # Coordinates in image #0, shape (K, 2)
    points1 = feats1_rbd['keypoints'][matches[..., 1]]  # Coordinates in image #1, shape (K, 2)

    
    print(points0.shape)

    # Convert keypoints to NumPy arrays
    mkpts0_matched = points0.cpu().numpy()
    mkpts1_matched = points1.cpu().numpy()

    print(f"Subregion {subregion_index + 1}: {mkpts0_matched.shape[0]} matches found.")

    if mkpts0_matched.shape[0] < 4:
        print(f"Not enough matches to compute homography for subregion {subregion_index + 1}.")
        return None  # Handle insufficient matches appropriately

    src_pts = mkpts1_matched
    dst_pts = mkpts0_matched

    # Compute homography matrix using RANSAC
    homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return homography_matrix








def match_subregions_eloftr(sample_subregion, template_subregion, subregion_index, device, model_type='full', precision='fp32'):
    # Resize the subregions

     # Initialize the Efficient LoFTR matcher inside the function
    if model_type == 'full':
        _default_cfg = deepcopy(full_default_cfg)
    elif model_type == 'opt':
        _default_cfg = deepcopy(opt_default_cfg)
    
    if precision == 'mp':
        _default_cfg['mp'] = True
    elif precision == 'fp16':
        _default_cfg['half'] = True

        # Initialize the matcher and load weights
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("weights/eloftr_outdoor.ckpt")['state_dict'])
    matcher = reparameter(matcher).eval().to(device)

    sample_subregion_resized = resize_image(sample_subregion)
    template_subregion_resized = resize_image(template_subregion)

    # Convert the OpenCV images from BGR to grayscale (LoFTR works with grayscale images)
    # sample_gray = cv2.cvtColor(sample_subregion_resized, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_subregion_resized, cv2.COLOR_BGR2GRAY)

    # Convert images to tensors and normalize to [0, 1]
    sample_tensor = torch.from_numpy(sample_subregion_resized)[None][None].float().to(device) / 255.0
    template_tensor = torch.from_numpy(template_gray)[None][None].float().to(device) / 255.0

    # Prepare input dictionary
    input_dict = {
        "image0": sample_tensor,
        "image1": template_tensor,
    }

    # Perform inference using EfficientLoFTR
    with torch.no_grad():
        matcher(input_dict)
    
    mkpts0 = input_dict["mkpts0_f"].cpu().numpy()  # Points from sample image
    mkpts1 = input_dict["mkpts1_f"].cpu().numpy()  # Points from template image

    print(f"Subregion {subregion_index + 1}: {mkpts0.shape[0]} matches found.")
    src_pts = np.float32(mkpts1).reshape(-1, 1, 2)
    dst_pts = np.float32(mkpts0).reshape(-1, 1, 2)
    
    homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,8.0)
    return homography_matrix

def calculate_condition_number(H):
    # Calculate the condition number of the first 2x2 submatrix
    condition_number = np.linalg.cond(H[:2, :2])
    return condition_number

def decompose_homography(H):
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H[:2, :2])
    
    # Extract rotation and scaling matrices
    R = np.dot(U, Vt)
    S_matrix = np.diag(S)
    
    # Ensure that R is a valid rotation matrix (det(R) should be +1)
    if np.linalg.det(R) < 0:
        R *= -1

    # Average the scaling factors to reduce non-uniform scaling
    avg_scale = (S[0] + S[1]) / 2
    S_matrix = np.array([[avg_scale, 0], [0, avg_scale]])

    # Compose the modified homography with balanced scaling and reduced rotation
    modified_H = np.eye(3)
    modified_H[:2, :2] = S_matrix  # Apply balanced scaling
    modified_H[:2, 2] = H[:2, 2]  # Keep translation constant
    modified_H[2, :] = H[2, :]  # Keep perspective constant

    return modified_H


def visualize_rectangle_homography(H, i, j):
    """Visualizes the effect of the homography matrix H using a rectangle transformation."""
    
    # Create a blank white image for visualization (size 800x600 to include both rectangle and matrix)
    vis_image = np.ones((600, 1000, 3), dtype=np.uint8) * 255

    # Define the original rectangle (4 points) as a simple rectangle
    original_rectangle = np.array([[50, 50], [590, 50], [590, 430], [50, 430]], dtype=np.float32)
    
    # Draw the original rectangle on the left side (blue rectangle)
    cv2.polylines(vis_image, [original_rectangle.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Apply homography to the rectangle's points
    original_rectangle_homogeneous = np.hstack([original_rectangle, np.ones((4, 1))])  # Convert to homogeneous coordinates
    warped_rectangle = H @ original_rectangle_homogeneous.T
    warped_rectangle = (warped_rectangle / warped_rectangle[2, :])[:2, :].T  # Normalize and convert back to 2D
    
    # Draw the warped rectangle on the right side (red rectangle)
    cv2.polylines(vis_image, [warped_rectangle.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
    
    # Create a folder for saving H matrix and visualization
    h_matrix_folder = "./demo/H_matrices"
    os.makedirs(h_matrix_folder, exist_ok=True)

    # Save the homography matrix as a text file
    h_matrix_file_path = os.path.join(h_matrix_folder, f'homography_matrix_{i+1}_{j+1}.txt')
    np.savetxt(h_matrix_file_path, H, fmt='%.6f')
    print(f"Homography matrix for subregion ({i+1}, {j+1}) saved to {h_matrix_file_path}.")

    # Add the homography matrix to the image as text
    matrix_text = f"H = [{H[0, 0]:.3f}, {H[0, 1]:.3f}, {H[0, 2]:.3f}]\n     [{H[1, 0]:.3f}, {H[1, 1]:.3f}, {H[1, 2]:.3f}]\n     [{H[2, 0]:.3f}, {H[2, 1]:.3f}, {H[2, 2]:.3f}]"
    y0, dy = 50, 30
    for k, line in enumerate(matrix_text.split("\n")):
        y = y0 + k * dy
        cv2.putText(vis_image, line, (680, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Save the visualization image
    vis_image_path = os.path.join(h_matrix_folder, f'homography_visualization_{i+1}_{j+1}.png')
    cv2.imwrite(vis_image_path, vis_image)
    print(f"Visualization with homography matrix saved to {vis_image_path}.")

def save_condition_number(C, i, j):
    # Create a folder for saving condition numbers
    Condition_num_folder = "./demo/Condition_numbers"
    os.makedirs(Condition_num_folder, exist_ok=True)

    # Save the condition number as a text file
    c_file_path = os.path.join(Condition_num_folder, f'condition_number_{i+1}_{j+1}.txt')
    with open(c_file_path, "w") as f:
        f.write(f"{C}\n")
        print(f"Condition number for subregion ({i+1}, {j+1}) saved to {Condition_num_folder}.")



def compute_grid_size(width, height, min_subregions=9, max_subregions=26):
    aspect_ratio = width / height
    possible_configs = []

    # We can go up to the square root of max_subregions to limit the number of iterations
    max_cols = max_subregions
    max_rows = max_subregions

    for num_cols in range(2, max_cols + 1):
        num_rows = int(height / width * num_cols)
        if num_rows < 2:
            continue
        total_subregions = num_cols * num_rows
        if total_subregions < min_subregions or total_subregions > max_subregions:
            continue
        current_aspect_ratio = num_cols / num_rows
        aspect_diff = abs(current_aspect_ratio - aspect_ratio)
        possible_configs.append((aspect_diff, num_cols, num_rows))

    if not possible_configs:
        # Fallback to default
        return 3, 3
    # Select the configuration with minimal aspect difference
    possible_configs.sort()
    _, best_cols, best_rows = possible_configs[0]
    return best_cols, best_rows


def create_overlay(warped_sample, template_subregion):
    """
    Creates an overlay image by combining the warped sample and the template subregion.

    Args:
        warped_sample: The warped sample image.
        template_subregion: The template subregion image.

    Returns:
        overlay: The overlay image.
    """
    # Convert images to grayscale
    sample_gray = cv2.cvtColor(warped_sample, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_subregion, cv2.COLOR_BGR2GRAY)

    # Create a color image for overlay
    overlay = np.zeros_like(warped_sample)

    # Assign the grayscale images to different channels
    overlay[:, :, 0] = sample_gray  # Blue channel
    overlay[:, :, 1] = template_gray  # Green channel

    return overlay




def resize_image_2(image, width=640, height=480):
    # Load the image from the specified path
    
    image_1=image.copy()
    # Check if image was loaded correctly
    if image is None:
        print(f"Error: Unable to load image")
        return

    # Specify the new width and height
    new_dimensions = (width, height)

    # Resize the image
    resized_image = cv2.resize(image_1, new_dimensions)

    return resized_image



def resize_image(image, width=1600, height=1440):
    # Load the image from the specified path
    
    image_1=image.copy()
    # Check if image was loaded correctly
    if image is None:
        print(f"Error: Unable to load image")
        return

    # Specify the new width and height
    new_dimensions = (width, height)

    # Resize the image
    resized_image = cv2.resize(image_1, new_dimensions)

    return resized_image


# 对样本图像进行不重叠的网格切分
def split_sample_image_grid(image, num_cols, num_rows):
    height, width = image.shape[:2]
    slice_height = height // num_rows
    slice_width = width // num_cols
    subregions = []

    for i in range(num_rows):
        for j in range(num_cols):
            start_y = i * slice_height
            end_y = (i + 1) * slice_height if i < num_rows - 1 else height
            start_x = j * slice_width
            end_x = (j + 1) * slice_width if j < num_cols - 1 else width
            subregion = image[start_y:end_y, start_x:end_x]
            subregions.append(((i, j), subregion))  # 使用(i, j)作为索引
    return subregions

# 对模板图像进行带有重叠的网格切分
def split_template_image_grid_with_overlap(image, num_cols, num_rows, overlap):
    height, width = image.shape[:2]
    slice_height = height // num_rows
    slice_width = width // num_cols
    subregions = []

    for i in range(num_rows):
        for j in range(num_cols):
            start_y = i * slice_height - overlap if i > 0 else 0
            end_y = (i + 1) * slice_height + overlap if i < num_rows - 1 else height
            start_x = j * slice_width - overlap if j > 0 else 0
            end_x = (j + 1) * slice_width + overlap if j < num_cols - 1 else width
            subregion = image[start_y:end_y, start_x:end_x]
            subregions.append(((i, j), subregion))  # 使用(i, j)作为索引
    return subregions

def split_image_grid_with_overlap(image, num_cols, num_rows, overlap):
    
    height, width = image.shape[:2]
    resized_height, resized_width = 1440, 1600  # Target dimensions for each subregion

    # Compute the exact positions for slicing
    positions_y = np.linspace(0, height, num_rows + 1, dtype=int)
    positions_x = np.linspace(0, width, num_cols + 1, dtype=int)

    subregions = []
    original_coords = []
    local_coords = []
    resized_local_coords = []

    for i in range(num_rows):
        for j in range(num_cols):
            # Non-overlapping coordinates
            orig_start_y = positions_y[i]
            orig_end_y = positions_y[i + 1]
            orig_start_x = positions_x[j]
            orig_end_x = positions_x[j + 1]

            # Overlapping coordinates
            start_y = max(0, orig_start_y - overlap)
            end_y = min(height, orig_end_y + overlap)
            start_x = max(0, orig_start_x - overlap)
            end_x = min(width, orig_end_x + overlap)

            # Local coordinates excluding the overlap in the subregion
            local_start_y = orig_start_y - start_y
            local_end_y = orig_end_y - start_y
            local_start_x = orig_start_x - start_x
            local_end_x = orig_end_x - start_x

            # Subregion extraction
            subregion = image[start_y:end_y, start_x:end_x]

            # Subregion size including overlap
            subregion_height = end_y - start_y
            subregion_width = end_x - start_x

            # Scaling factors
            height_scale_factor = resized_height / subregion_height
            width_scale_factor = resized_width / subregion_width

            # Resized local coordinates
            resized_local_start_x = int(local_start_x * width_scale_factor)
            resized_local_end_x = int(local_end_x * width_scale_factor)
            resized_local_start_y = int(local_start_y * height_scale_factor)
            resized_local_end_y = int(local_end_y * height_scale_factor)

            subregions.append(((i, j), subregion))
            original_coords.append((orig_start_x, orig_start_y, orig_end_x, orig_end_y))
            local_coords.append((local_start_x, local_start_y, local_end_x, local_end_y))
            resized_local_coords.append((resized_local_start_x, resized_local_start_y, resized_local_end_x, resized_local_end_y))

    return subregions, original_coords,local_coords, resized_local_coords



def match_subregions_loftr(sample_subregion, template_subregion, subregion_index, device):
    # Resize the subregions
    sample_subregion_resized = resize_image(sample_subregion)
    template_subregion_resized = resize_image(template_subregion)

    # Convert the OpenCV images from BGR to RGB (required by Kornia)
    sample_subregion_rgb = cv2.cvtColor(sample_subregion_resized, cv2.COLOR_BGR2RGB)
    template_subregion_rgb = cv2.cvtColor(template_subregion_resized, cv2.COLOR_BGR2RGB)

    # Convert images to tensors and normalize to [0, 1]
    sample_tensor = K.utils.image_to_tensor(sample_subregion_rgb, keepdim=True).float() / 255.0
    template_tensor = K.utils.image_to_tensor(template_subregion_rgb, keepdim=True).float() / 255.0

    # Add a batch dimension (LoFTR expects a batch size)
    sample_tensor = sample_tensor.unsqueeze(0)  # shape: [1, 3, H, W]
    template_tensor = template_tensor.unsqueeze(0)  # shape: [1, 3, H, W]

    # Convert the images to grayscale (LoFTR works with grayscale images)
    sample_gray = K.color.rgb_to_grayscale(sample_tensor)  # shape: [1, 1, H, W]
    template_gray = K.color.rgb_to_grayscale(template_tensor)  # shape: [1, 1, H, W]

    # Move tensors to GPU
    sample_gray_gpu = sample_gray.to(device)
    template_gray_gpu = template_gray.to(device)

    # Initialize the LoFTR matcher
    matcher = KF.LoFTR(pretrained="outdoor").to(device)

    # Prepare input dictionary
    input_dict = {
        "image0": sample_gray_gpu,
        "image1": template_gray_gpu,
    }

    # Perform inference
    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences["keypoints0"].cpu().numpy()  # Points from sample image
    mkpts1 = correspondences["keypoints1"].cpu().numpy()  # Points from template image

    print(f"子区域 {subregion_index + 1}: 共有 {mkpts0.shape[0]} 个匹配点。")
    src_pts = np.float32(mkpts1).reshape(-1, 1, 2)
    dst_pts = np.float32(mkpts0).reshape(-1, 1, 2)
    homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC,8.0)
    return homography_matrix



# 将样本子区域变换到模板子区域上
def warp_sample_onto_template(sample_subregion, template_subregion, homography_matrix):
    h_t, w_t = template_subregion.shape[:2]
    warped_sample = cv2.warpPerspective(sample_subregion, homography_matrix, (w_t, h_t))
    return warped_sample

def warp_template_onto_sample(sample_subregion, template_subregion, homography_matrix):
    h_t, w_t = sample_subregion.shape[:2]
    warped_sample = cv2.warpPerspective(template_subregion, homography_matrix, (w_t, h_t))
    return warped_sample


def warp_processed_back_to_sample(processed_overlap, homography_matrix, sample_subregion_shape):
    """
    This function warps the processed overlapping subregion back onto the original sample subregion.
    """
    h, w = sample_subregion_shape[:2]
    
    # Warp the processed overlap region back to the sample subregion space using the inverse homography matrix
    homography_inv = np.linalg.inv(homography_matrix)  # Inverse of homography matrix
    warped_back = cv2.warpPerspective(processed_overlap, homography_inv, (w, h))
    
    return warped_back
 

def visualize_and_save_in_middle(warped_sample, template_subregion, row_num,column_num):
    # 创建叠加图像
    overlay = template_subregion.copy()
    # 创建掩码
    mask = cv2.cvtColor(warped_sample, cv2.COLOR_BGR2GRAY)
    #_, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)


   # Convert the images to grayscale
    #img1_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # Grayscale image 1
    img1_gray = mask
    img2_gray = overlay  # Grayscale image 2
    # Ensure both images are the same size, resize if necessary
    if img1_gray.shape != img2_gray.shape:
        img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))
    # Create an empty RGB image (H, W, 3) where H and W are the height and width of the grayscale images
    stacked_rgb = np.zeros((img1_gray.shape[0], img1_gray.shape[1], 3), dtype=np.uint8)

    # os.makedirs('./demo/sample_images_to_overlay',exist_ok=True)
    # img1_gray_path = os.path.join("sample_images_to_overlay",f'sample_subregion_{row_num + 1}_{column_num + 1}.png')
    # cv2.imwrite(img1_gray_path,img1_gray)

    # os.makedirs('./demo/template_images_to_overlay',exist_ok=True)
    # img2_gray_path = os.path.join("template_images_to_overlay",f'template_subregion_{row_num + 1}_{column_num + 1}.png')
    # cv2.imwrite(img2_gray_path,img2_gray)
    # Assign the grayscale images to the red and green channels
    stacked_rgb[:, :, 0] = img1_gray  # Red channel -> first grayscale image
    stacked_rgb[:, :, 1] = img2_gray  # Green channel -> second grayscale image
    # The blue channel (stacked_rgb[:, :, 2]) remains 0, which is already set by np.zeros
    dst = stacked_rgb

    os.makedirs('./results/overlays', exist_ok=True)
    overlay_output_path = os.path.join('./results/overlays', f'overlay_subregion_{row_num + 1}_{column_num + 1}.png')
    cv2.imwrite(overlay_output_path, dst)
    print(f"叠加结果已保存到 {overlay_output_path}")

    return dst

def stack_images(image1, image2_warped):
    stacked_image = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
    stacked_image[:, :, 1] = image1  #
    stacked_image[:, :, 0] = image2_warped  # 
    return stacked_image

def visualize_and_save_overlay_stackimage(warped_sample, template_subregion, subregion_index,non_warped_sample,h_matrix):
    # Step 1: Create an overlap mask
    mask = cv2.cvtColor(warped_sample, cv2.COLOR_BGR2GRAY)
    
    # Threshold the mask to create a binary image (1 for non-zero pixels, 0 for zero pixels)
    _, overlap_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # Invert the mask to get the non-overlapping areas
    mask_inv = cv2.bitwise_not(overlap_mask)

    # Step 2: Remove the non-overlapping areas from the template subregion
    template_overlap = cv2.bitwise_and(template_subregion, template_subregion, mask=overlap_mask)
    
    # Remove the non-overlapping areas from the warped sample
    sample_overlap = cv2.bitwise_and(warped_sample, warped_sample, mask=overlap_mask)

    # Step 3: Convert the images to grayscale
    img1_gray = cv2.cvtColor(sample_overlap, cv2.COLOR_BGR2GRAY)  # Grayscale warped sample
    img2_gray = cv2.cvtColor(template_overlap, cv2.COLOR_BGR2GRAY)  # Grayscale template

    img1_gray_origin=cv2.cvtColor(non_warped_sample, cv2.COLOR_BGR2GRAY)
    img2_gray_back=warp_processed_back_to_sample(img2_gray,h_matrix,non_warped_sample.shape)

    img2_gray_back_new=warp_template_onto_sample(non_warped_sample,template_subregion,h_matrix)
        # Debugging: Show the template_overlap for visualization



    stacked_rgb = np.zeros((img1_gray_origin.shape[0], img1_gray_origin.shape[1], 3), dtype=np.uint8)
    
    # Step 5: Assign the grayscale images to the red and green channels
    stacked_rgb[:, :, 1] = img1_gray_origin  # Red channel -> warped sample (sample_overlap)
    stacked_rgb[:, :, 0] =  img2_gray_back # Green channel -> template (template_overlap)


    
    # Step 6: Save the result
    os.makedirs('overlays', exist_ok=True)
    overlay_output_path = os.path.join('overlays', f'overlay_subregion_{subregion_index + 1}.png')
    cv2.imwrite(overlay_output_path, stacked_rgb)
    print(f"叠加结果已保存到 {overlay_output_path}")

    return stacked_rgb

def random_points_in_lowres(image_shape, num_points=8, margin=400):
    h, w = image_shape[:2]
    points = []
    
    # Ensure that we pick points inside the image, avoiding the margins
    for _ in range(num_points):
        x = random.randint(margin, w - margin)  # Avoid boundary
        y = random.randint(margin, h - margin)  # Avoid boundary
        points.append([x, y])
    
    return np.array(points, dtype=np.float32)

# Function to warp image using the homography matrix
def warp_image(image, homography_matrix, size):
    return cv2.warpPerspective(image, homography_matrix, size)

# Function to warp points using homography matrix
def warp_points(points, H):
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # Convert to homogeneous coordinates
    warped_points_homogeneous = np.dot(H, points_homogeneous.T).T  # Apply homography
    warped_points = warped_points_homogeneous[:, :2] / warped_points_homogeneous[:, 2][:, np.newaxis]  # Convert back to 2D
    return warped_points

# Function to map points back to high resolution based on scale
def map_points_to_highres(points, original_width, original_height, lowres_width=1600, lowres_height=1440):
    scale_x = original_width / lowres_width
    scale_y = original_height / lowres_height
    return np.array([[x * scale_x, y * scale_y] for x, y in points], dtype=np.float32)
 
# 主处理函数
def process_stitched_image(stitched_image, template_image,sample_image_otus, overlap_sample,overlap_template,auto_split=True,default_row=3,default_columns=2, device='cuda'):
    # 记录整个处理的开始时间
    total_start_time = time.perf_counter()

    height, width = template_image.shape[:2]
    if(auto_split):
        num_cols, num_rows = compute_grid_size(width, height)
    else:
        num_cols=default_columns
        num_rows=default_row
    print(f"Number of columns (vertical slices): {num_cols}, rows (horizontal slices): {num_rows}")

    sample_subregions, original_coords,local_coords, resized_local_coords = split_image_grid_with_overlap(stitched_image, num_cols, num_rows, overlap_sample)
    sample_subregions_otus, original_coords, local_coords,resized_local_coords = split_image_grid_with_overlap(sample_image_otus, num_cols, num_rows, overlap_sample)
    template_subregions, _,_, _ = split_image_grid_with_overlap(template_image, num_cols, num_rows, overlap_template)
    print('Images are splited .')
    overlay_subregions = []

    for idx, (grid_idx, sample_subregion_origin) in enumerate(sample_subregions):
        i, j = grid_idx  # 行和列索引
        print(f"正在处理子区域 (行: {i + 1}, 列: {j + 1})/{num_rows}x{num_cols}...")
        # 记录子区域处理的开始时间
        subregion_start_time = time.perf_counter()

        try:
            # 获取对应的模板子区域
            template_grid_idx, template_subregion_origin = template_subregions[idx]

            assert template_grid_idx == grid_idx, "模板子区域索引与样本子区域索引不匹配"

            sample_otus_index,sample_subregion_otus = sample_subregions_otus[idx]

            resized_local_start_x, resized_local_start_y, resized_local_end_x, resized_local_end_y= resized_local_coords[idx]

            local_start_x, local_start_y,local_end_x,local_end_y =local_coords[idx]

            print(resized_local_start_x, resized_local_start_y, resized_local_end_x, resized_local_end_y)


            # 预处理子区域


            # os.makedirs('./demo/sample_subregions_org', exist_ok=True)
            # sample_subregions_path = os.path.join('./demo/sample_subregions_org', f'sample_subregion_{i + 1}_{j + 1}.png')
            # cv2.imwrite(sample_subregions_path, sample_subregion_origin)
            # print(f"Sample subregions origin image saved to {sample_subregions_path}.")

            # os.makedirs('./demo/sample_subregions_org_otus', exist_ok=True)
            # sample_subregions_path_otus = os.path.join('./demo/sample_subregions_org_otus', f'sample_subregion_otus_{i + 1}_{j + 1}.png')
            # cv2.imwrite(sample_subregions_path_otus, sample_subregion_otus)
            # print(f"Sample subregions origin otus image saved to {sample_subregions_path_otus}.")

            # os.makedirs('./demo/template_subregions_org', exist_ok=True)
            # template_subregions_path = os.path.join('./demo/template_subregions_org', f'template_subregion_{i + 1}_{j + 1}.png')
            # cv2.imwrite(template_subregions_path, template_subregion_origin)
            # print(f"Template subregions origin image saved to {template_subregions_path}.")

            sample_subregion_resized = resize_image(sample_subregion_origin)
            sample_subregion_otus_resized =resize_image(sample_subregion_otus)
            template_subregion_resized = resize_image(template_subregion_origin)


            # os.makedirs('./demo/sample_subregions', exist_ok=True)
            # sample_subregions_path = os.path.join('./demo/sample_subregions', f'sample_subregion_{i + 1}_{j + 1}.png')
            # cv2.imwrite(sample_subregions_path, sample_subregion_resized)
            # print(f"Sample subregions image saved to {sample_subregions_path}.")

            # os.makedirs('./demo/sample_subregions_otus', exist_ok=True)
            # sample_subregions_path = os.path.join('./demo/sample_subregions_otus', f'sample_subregion_otus_{i + 1}_{j + 1}.png')
            # cv2.imwrite(sample_subregions_path, sample_subregion_otus_resized)
            # print(f"Sample subregions applied otus thresholding saved to {sample_subregions_path}.")

            # os.makedirs('./demo/template_subregions', exist_ok=True)
            # template_subregions_path = os.path.join('./demo/template_subregions', f'template_subregion_{i + 1}_{j + 1}.png')
            # cv2.imwrite(template_subregions_path, template_subregion_resized)
            # print(f"Template subregions image saved to {template_subregions_path}.")


            lowres_points = random_points_in_lowres(sample_subregion_otus_resized.shape)

            # 匹配子区域并计算单应矩阵
            homography_matrix = match_subregions_lightglue(sample_subregion_otus_resized, template_subregion_resized, idx, device) 

            H_inv =np.linalg.inv(homography_matrix)

            # condition_num_origin= calculate_condition_number(homography_matrix)



            # if(condition_num_origin>1.055):
            #     homography_matrix = decompose_homography(homography_matrix)

            # save_condition_number(condition_num_origin,i,j)
            # visualize_rectangle_homography(homography_matrix, i, j) #save H matrices

            # warped_template = cv2.warpPerspective(template_subregion_resized, homography_matrix, (sample_subregion_resized.shape[1], sample_subregion_resized.shape[0]), borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            warped_template = cv2.warpPerspective(template_subregion_resized, homography_matrix, (sample_subregion_resized.shape[1], sample_subregion_resized.shape[0]), borderMode=cv2.BORDER_CONSTANT,borderValue=0)

            # Convert both sample and warped template to grayscale
            sample_gray=sample_subregion_otus_resized
            # sample_gray = cv2.cvtColor(sample_subregion_resized, cv2.COLOR_BGR2GRAY)


            warped_gray = cv2.cvtColor(warped_template, cv2.COLOR_BGR2GRAY)

            # Ensure both grayscale images have the same size
            if sample_gray.shape != warped_gray.shape:
                warped_gray = cv2.resize(warped_gray, (sample_gray.shape[1], sample_gray.shape[0]))

            # Create a 3-channel stacked image with grayscale sample and warped template
            stacked_rgb = np.zeros((sample_gray.shape[0], sample_gray.shape[1], 3), dtype=np.uint8)
            stacked_rgb[:, :, 1] = sample_gray  # Assign sample subregion to green channel
            stacked_rgb[:, :, 0] = warped_gray  # Assign warped template to red channel
            # Blue channel remains zero

            # os.makedirs('./results/stacked_images_before_cropping', exist_ok=True)
            # stacked_rgb_path = os.path.join('./results/stacked_images_before_cropping', f'stacked_subregion_{i + 1}_{j + 1}.png')
            # cv2.imwrite(stacked_rgb_path, stacked_rgb)
            # print(f"Stacked RGB image saved to {stacked_rgb_path} for debugging.")

            warped_points_in_img2_lowres = warp_points(lowres_points,H_inv)


            # Save the visualization for inspection

            # Step 7: Map points to high-resolution images
            w1, h1 = sample_subregion_origin.shape[1], sample_subregion_origin.shape[0]  # width and height of image1
            w2, h2 = template_subregion_origin.shape[1], template_subregion_origin.shape[0]  # width and height of image2
            highres_points_img1 = map_points_to_highres(lowres_points, w1, h1)
            highres_points_img2 = map_points_to_highres(warped_points_in_img2_lowres, w2, h2)

            # Recalculate H' for high-resolution images using the mapped points
            H_prime = cv2.findHomography(highres_points_img2, highres_points_img1, cv2.RANSAC)[0]

            template_subregion_origin = cv2.cvtColor(template_subregion_origin, cv2.COLOR_BGR2GRAY)

            # Step 10: Warp original high-resolution image2 using H'
            warped_image2_highres = warp_image(template_subregion_origin, H_prime, (w1, h1))
            stacked_highres = stack_images(sample_subregion_otus, warped_image2_highres)

            # os.makedirs('./results/stacked_images_original_before_cropping', exist_ok=True)
            # stacked_rgb_path = os.path.join('./results/stacked_images_original_before_cropping', f'stacked_subregion_{i + 1}_{j + 1}.png')
            # cv2.imwrite(stacked_rgb_path, stacked_highres)
            # print(f"Stacked RGB image saved to {stacked_rgb_path} for debugging.")


            cropped_image = stacked_rgb[resized_local_start_y:resized_local_end_y, resized_local_start_x:resized_local_end_x]


            # os.makedirs('./results/stacked_images_after_cropping', exist_ok=True)
            # stacked_rgb_path = os.path.join('./results/stacked_images_after_cropping', f'stacked_subregion_{i + 1}_{j + 1}.png')
            # cv2.imwrite(stacked_rgb_path, cropped_image)
            # print(f"Cropped Stacked RGB image saved to {stacked_rgb_path} for debugging.")

            cropped_image_origin = stacked_highres[local_start_y:local_end_y,local_start_x:local_end_x]

            # os.makedirs('./results/stacked_images_origin_after_cropping', exist_ok=True)
            # stacked_rgb_path = os.path.join('./results/stacked_images_origin_after_cropping', f'stacked_subregion_{i + 1}_{j + 1}.png')
            # cv2.imwrite(stacked_rgb_path, cropped_image_origin)
            # print(f"Cropped Original Stacked RGB image saved to {stacked_rgb_path} for debugging.")

            overlay=cropped_image_origin


            # 将叠加结果添加到列表中
            overlay_subregions.append(((i, j), overlay))

        except Exception as e:
            print(f"处理子区域 {i + 1} 时出错：{e}")
            # 如果出错，使用原始模板子区域
            overlay_subregions.append(template_subregion_resized)
        finally:
            # 记录子区域处理的结束时间
            subregion_end_time = time.perf_counter()
            subregion_time = subregion_end_time - subregion_start_time
            print(f"子区域 {i + 1} 处理时间: {subregion_time:.4f} 秒")

    # 合并所有叠加后的子区域，生成最终的大图
 # Sort by row and column indices (ensure that the first element in the tuple is being compared)
    overlay_subregions_sorted = sorted(overlay_subregions, key=lambda x: (x[0][0], x[0][1]))

    # Combine sorted subregions into the final image
    overlay_rows = []
    for i in range(num_rows):
        row_subregions = [subregion[1] for subregion in overlay_subregions_sorted if subregion[0][0] == i]
        row_combined = np.hstack(row_subregions)  # Combine subregions horizontally for this row
        overlay_rows.append(row_combined)

    # Combine all rows vertically to form the final stitched overlay image
    final_overlay_image = np.vstack(overlay_rows)
    # 保存最终的大图
    os.makedirs('results', exist_ok=True)
    final_output_path = os.path.join('results', 'final_overlay_image.jpg')
    cv2.imwrite(final_output_path, final_overlay_image)
    print(f"最终叠加图像已保存到 {final_output_path}")

    # 记录整个处理的结束时间
    total_end_time = time.perf_counter()
    total_time = total_end_time - total_start_time
    print(f"整个图像处理时间: {total_time:.4f} 秒")

    return final_output_path


def process_stitched_image_each(stitched_image, template_image,sample_image_otus,index, overlap_sample,overlap_template, device='cuda'):
    # 记录整个处理的开始时间
    total_start_time = time.perf_counter()

    height, width = template_image.shape[:2]
    num_cols, num_rows = compute_grid_size(width, height)
    num_cols=3
    num_rows=6
    print(f"Number of columns (vertical slices): {num_cols}, rows (horizontal slices): {num_rows}")

    sample_subregions, original_coords,local_coords, resized_local_coords = split_image_grid_with_overlap(stitched_image, num_cols, num_rows, overlap_sample)
    sample_subregions_otus, original_coords, local_coords,resized_local_coords = split_image_grid_with_overlap(sample_image_otus, num_cols, num_rows, overlap_sample)
    template_subregions, _,_, _ = split_image_grid_with_overlap(template_image, num_cols, num_rows, overlap_template)
    print('Images are splited .')
    overlay_subregions = []

    for idx, (grid_idx, sample_subregion_origin) in enumerate(sample_subregions):
        i, j = grid_idx  # 行和列索引
        print(f"正在处理子区域 (行: {i + 1}, 列: {j + 1})/{num_rows}x{num_cols}...")
        # 记录子区域处理的开始时间
        subregion_start_time = time.perf_counter()

        try:
            # 获取对应的模板子区域
            template_grid_idx, template_subregion_origin = template_subregions[idx]

            assert template_grid_idx == grid_idx, "模板子区域索引与样本子区域索引不匹配"

            sample_otus_index,sample_subregion_otus = sample_subregions_otus[idx]

            resized_local_start_x, resized_local_start_y, resized_local_end_x, resized_local_end_y= resized_local_coords[idx]

            local_start_x, local_start_y,local_end_x,local_end_y =local_coords[idx]

            print(resized_local_start_x, resized_local_start_y, resized_local_end_x, resized_local_end_y)


            # 预处理子区域

            
            # os.makedirs('./demo/sample_subregions_org', exist_ok=True)
            # sample_subregions_path = os.path.join('./demo/sample_subregions_org', f'sample_{index}_subregion_{i + 1}_{j + 1}.png')
            # cv2.imwrite(sample_subregions_path, sample_subregion_origin)
            # print(f"Sample subregions origin image saved to {sample_subregions_path}.")

            # os.makedirs('./demo/sample_subregions_org_otus', exist_ok=True)
            # sample_subregions_path_otus = os.path.join('./demo/sample_subregions_org_otus', f'sample_{index}_subregion_otus_{i + 1}_{j + 1}.png')
            # cv2.imwrite(sample_subregions_path_otus, sample_subregion_otus)
            # print(f"Sample subregions origin otus image saved to {sample_subregions_path_otus}.")
            
            # os.makedirs('./demo/template_subregions_org', exist_ok=True)
            # template_subregions_path = os.path.join('./demo/template_subregions_org', f'template_{index}_subregion_{i + 1}_{j + 1}.png')
            # cv2.imwrite(template_subregions_path, template_subregion_origin)
            # print(f"Template subregions origin image saved to {template_subregions_path}.")

            sample_subregion_resized = resize_image(sample_subregion_origin)
            sample_subregion_otus_resized =resize_image(sample_subregion_otus)
            template_subregion_resized = resize_image(template_subregion_origin)


            # os.makedirs('./demo/sample_subregions', exist_ok=True)
            # sample_subregions_path = os.path.join('./demo/sample_subregions', f'sample_{index}_subregion_{i + 1}_{j + 1}.png')
            # cv2.imwrite(sample_subregions_path, sample_subregion_resized)
            # print(f"Sample subregions image saved to {sample_subregions_path}.")

            # os.makedirs('./demo/sample_subregions_otus', exist_ok=True)
            # sample_subregions_path = os.path.join('./demo/sample_subregions_otus', f'sample_{index}_subregion_otus_{i + 1}_{j + 1}.png')
            # cv2.imwrite(sample_subregions_path, sample_subregion_otus_resized)
            # print(f"Sample subregions applied otus thresholding saved to {sample_subregions_path}.")

            # os.makedirs('./demo/template_subregions', exist_ok=True)
            # template_subregions_path = os.path.join('./demo/template_subregions', f'template_{index}_subregion_{i + 1}_{j + 1}.png')
            # cv2.imwrite(template_subregions_path, template_subregion_resized)
            # print(f"Template subregions image saved to {template_subregions_path}.")

            
            lowres_points = random_points_in_lowres(sample_subregion_otus_resized.shape)

            # 匹配子区域并计算单应矩阵
            homography_matrix = match_subregions_lightglue(sample_subregion_otus_resized, template_subregion_resized, idx, device) 
            
            H_inv =np.linalg.inv(homography_matrix)

            condition_num_origin= calculate_condition_number(homography_matrix)
        
           
            
            # if(condition_num_origin>1.055):
            #     homography_matrix = decompose_homography(homography_matrix)

            # save_condition_number(condition_num_origin,i,j)
            # visualize_rectangle_homography(homography_matrix, i, j) #save H matrices

            # warped_template = cv2.warpPerspective(template_subregion_resized, homography_matrix, (sample_subregion_resized.shape[1], sample_subregion_resized.shape[0]), borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            warped_template = cv2.warpPerspective(template_subregion_resized, homography_matrix, (sample_subregion_resized.shape[1], sample_subregion_resized.shape[0]), borderMode=cv2.BORDER_CONSTANT,borderValue=0)

            # Convert both sample and warped template to grayscale
            sample_gray=sample_subregion_otus_resized
            # sample_gray = cv2.cvtColor(sample_subregion_resized, cv2.COLOR_BGR2GRAY)


            warped_gray = cv2.cvtColor(warped_template, cv2.COLOR_BGR2GRAY)

            # Ensure both grayscale images have the same size
            if sample_gray.shape != warped_gray.shape:
                warped_gray = cv2.resize(warped_gray, (sample_gray.shape[1], sample_gray.shape[0]))

            # Create a 3-channel stacked image with grayscale sample and warped template
            stacked_rgb = np.zeros((sample_gray.shape[0], sample_gray.shape[1], 3), dtype=np.uint8)
            stacked_rgb[:, :, 1] = sample_gray  # Assign sample subregion to green channel
            stacked_rgb[:, :, 0] = warped_gray  # Assign warped template to red channel
            # Blue channel remains zero

            os.makedirs('./results/stacked_images_before_cropping', exist_ok=True)
            stacked_rgb_path = os.path.join('./results/stacked_images_before_cropping', f'stacked_subregion_{index}_{i + 1}_{j + 1}.png')
            cv2.imwrite(stacked_rgb_path, stacked_rgb)
            print(f"Stacked RGB image saved to {stacked_rgb_path} for debugging.")

            warped_points_in_img2_lowres = warp_points(lowres_points,H_inv)


            # Save the visualization for inspection

            # Step 7: Map points to high-resolution images
            w1, h1 = sample_subregion_origin.shape[1], sample_subregion_origin.shape[0]  # width and height of image1
            w2, h2 = template_subregion_origin.shape[1], template_subregion_origin.shape[0]  # width and height of image2
            highres_points_img1 = map_points_to_highres(lowres_points, w1, h1)
            highres_points_img2 = map_points_to_highres(warped_points_in_img2_lowres, w2, h2)

            # Recalculate H' for high-resolution images using the mapped points
            H_prime = cv2.findHomography(highres_points_img2, highres_points_img1, cv2.RANSAC)[0]

            template_subregion_origin = cv2.cvtColor(template_subregion_origin, cv2.COLOR_BGR2GRAY)

            # Step 10: Warp original high-resolution image2 using H'
            warped_image2_highres = warp_image(template_subregion_origin, H_prime, (w1, h1))
            stacked_highres = stack_images(sample_subregion_otus, warped_image2_highres)

            os.makedirs('./results/stacked_images_original_before_cropping', exist_ok=True)
            stacked_rgb_path = os.path.join('./results/stacked_images_original_before_cropping', f'stacked_subregion_{index}_{i + 1}_{j + 1}.png')
            cv2.imwrite(stacked_rgb_path, stacked_highres)
            print(f"Stacked RGB image saved to {stacked_rgb_path} for debugging.")


            cropped_image = stacked_rgb[resized_local_start_y:resized_local_end_y, resized_local_start_x:resized_local_end_x]
            
            
            # os.makedirs('./results/stacked_images_after_cropping', exist_ok=True)
            # stacked_rgb_path = os.path.join('./results/stacked_images_after_cropping', f'stacked_subregion_{index}_{i + 1}_{j + 1}.png')
            # cv2.imwrite(stacked_rgb_path, cropped_image)
            # print(f"Cropped Stacked RGB image saved to {stacked_rgb_path} for debugging.")

            cropped_image_origin = stacked_highres[local_start_y:local_end_y,local_start_x:local_end_x]

            os.makedirs('./results/stacked_images_origin_after_cropping', exist_ok=True)
            stacked_rgb_path = os.path.join('./results/stacked_images_origin_after_cropping', f'stacked_subregion_{index}_{i + 1}_{j + 1}.png')
            cv2.imwrite(stacked_rgb_path, cropped_image_origin)
            print(f"Cropped Original Stacked RGB image saved to {stacked_rgb_path} for debugging.")

            overlay=cropped_image_origin


            # 将叠加结果添加到列表中
            overlay_subregions.append(((i, j), overlay))
            
        except Exception as e:
            print(f"处理子区域 {i + 1} 时出错：{e}")
            # 如果出错，使用原始模板子区域
            overlay_subregions.append(template_subregion_resized)
        finally:
            # 记录子区域处理的结束时间
            subregion_end_time = time.perf_counter()
            subregion_time = subregion_end_time - subregion_start_time
            print(f"子区域 {i + 1} 处理时间: {subregion_time:.4f} 秒")

    # 合并所有叠加后的子区域，生成最终的大图
 # Sort by row and column indices (ensure that the first element in the tuple is being compared)
    overlay_subregions_sorted = sorted(overlay_subregions, key=lambda x: (x[0][0], x[0][1]))

    # Combine sorted subregions into the final image
    overlay_rows = []
    for i in range(num_rows):
        row_subregions = [subregion[1] for subregion in overlay_subregions_sorted if subregion[0][0] == i]
        row_combined = np.hstack(row_subregions)  # Combine subregions horizontally for this row
        overlay_rows.append(row_combined)

    # Combine all rows vertically to form the final stitched overlay image
    final_overlay_image = np.vstack(overlay_rows)
    # 保存最终的大图
    os.makedirs('results', exist_ok=True)
    final_output_path = os.path.join('results', f'final_overlay_image_{index}.jpg')
    cv2.imwrite(final_output_path, final_overlay_image)
    print(f"最终叠加图像已保存到 {final_output_path}")

    # 记录整个处理的结束时间
    total_end_time = time.perf_counter()
    total_time = total_end_time - total_start_time
    print(f"整个图像处理时间: {total_time:.4f} 秒")

    return final_overlay_image



def fine_matching(sample_image_path, template_image_path, overlap_sample=300,overlap_template=500,Auto_split=True,default_rows=3,default_columns=2, device='cuda'):
    """
    Performs fine matching using the cropped sample image and the cropped template image.

    Parameters:
    - sample_image_path: Path to the cropped sample image.
    - template_image_path: Path to the cropped template image.
    - overlap: The overlap size for template subregions.
    - device: The computation device.
    """
    device = torch.device(device)
    print(f"Using device: {device}")

    # Check if images exist
    if not os.path.exists(sample_image_path):
        print(f"Sample image not found: {sample_image_path}")
        return
    if not os.path.exists(template_image_path):
        print(f"Template image not found: {template_image_path}")
        return

    sample_image = cv2.imread(sample_image_path)
    sample_image_gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    template_image = cv2.imread(template_image_path)

    # _, otsu_threshold_image = cv2.threshold(sample_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))
    clahe_image = clahe.apply(sample_image_gray.copy())

    clahe_image_2 =clahe.apply(clahe_image.copy())

    sample_image=sample_image_gray
    sample_image_otus = clahe_image_2

    if sample_image is None:
        print("Failed to load sample image.")
        return
    if template_image is None:
        print("Failed to load template image.")
        return

    # Process images
    final_overlay_image = process_stitched_image(sample_image, template_image,sample_image_otus, overlap_sample,overlap_template,Auto_split,default_rows,default_columns,device)
    print("Fine matching completed.")
    return final_overlay_image



def fine_matching_each(sample_image_path, template_image_path, idx,overlap_sample=300,overlap_template=500, device='cuda'):
    """
    Performs fine matching using the cropped sample image and the cropped template image.

    Parameters:
    - sample_image_path: Path to the cropped sample image.
    - template_image_path: Path to the cropped template image.
    - overlap: The overlap size for template subregions.
    - device: The computation device.
    """
    device = torch.device(device)
    print(f"Using device: {device}")

    # Check if images exist
    if not os.path.exists(sample_image_path):
        print(f"Sample image not found: {sample_image_path}")
        return
    if not os.path.exists(template_image_path):
        print(f"Template image not found: {template_image_path}")
        return

    sample_image = cv2.imread(sample_image_path)
    sample_image_gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    template_image = cv2.imread(template_image_path)

    # _, otsu_threshold_image = cv2.threshold(sample_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))
    clahe_image = clahe.apply(sample_image_gray.copy())

    clahe_image_2 =clahe.apply(clahe_image.copy())
    
    sample_image=sample_image_gray
    sample_image_otus = clahe_image_2

    if sample_image is None:
        print("Failed to load sample image.")
        return
    if template_image is None:
        print("Failed to load template image.")
        return

    # Process images
    final_overlay_image = process_stitched_image_each(sample_image, template_image,sample_image_otus, idx,overlap_sample,overlap_template, device)
    print("Fine matching completed.")
    return final_overlay_image

def process_stitched_image_pillow(stitched_image, template_image, sample_image_otus, overlap_sample, overlap_template, auto_split=True, default_row=3, default_columns=2, device='cuda'):
    """
    对裁剪后的图像进行网格分割、局部匹配并拼接生成最终叠加图。

    参数:
    - stitched_image: 裁剪后的样本图像（灰度）。
    - template_image: 裁剪后的模板图像（RGB）。
    - sample_image_otus: 经过 CLAHE 二值化处理后的样本图像。
    - overlap_sample: 样本图像子区域的重叠尺寸。
    - overlap_template: 模板图像子区域的重叠尺寸。
    - auto_split: 是否自动分割子区域。
    - default_row: 默认行数。
    - default_columns: 默认列数。
    - device: 计算设备。
    
    返回:
    - 最终叠加图像的保存路径。
    """
    total_start_time = time.perf_counter()

    height, width = template_image.shape[:2]
    if auto_split:
        num_cols, num_rows = compute_grid_size(width, height)
    else:
        num_cols = default_columns
        num_rows = default_row
    print(f"Number of columns (vertical slices): {num_cols}, rows (horizontal slices): {num_rows}")

    # 分割图像为子区域（假设 split_image_grid_with_overlap 返回：子区域列表、原始坐标、局部坐标、重置大小后的局部坐标）
    sample_subregions, original_coords, local_coords, resized_local_coords = split_image_grid_with_overlap(stitched_image, num_cols, num_rows, overlap_sample)
    sample_subregions_otus, _, local_coords, resized_local_coords = split_image_grid_with_overlap(sample_image_otus, num_cols, num_rows, overlap_sample)
    template_subregions, _, _, _ = split_image_grid_with_overlap(template_image, num_cols, num_rows, overlap_template)
    print('Images are split.')
    overlay_subregions = []

    for idx, (grid_idx, sample_subregion_origin) in enumerate(sample_subregions):
        i, j = grid_idx  # 行和列索引
        print(f"Processing subregion (row: {i + 1}, column: {j + 1})/{num_rows}x{num_cols}...")
        subregion_start_time = time.perf_counter()

        try:
            # 获取对应的模板子区域和 CLAHE 后的样本子区域
            template_grid_idx, template_subregion_origin = template_subregions[idx]
            assert template_grid_idx == grid_idx, "Template subregion index mismatch."
            sample_otus_index, sample_subregion_otus = sample_subregions_otus[idx]

            resized_local_start_x, resized_local_start_y, resized_local_end_x, resized_local_end_y = resized_local_coords[idx]
            local_start_x, local_start_y, local_end_x, local_end_y = local_coords[idx]
            print(resized_local_start_x, resized_local_start_y, resized_local_end_x, resized_local_end_y)

            # 对子区域进行预处理，调整尺寸（假设 resize_image 函数已定义，可处理 numpy 数组）
            sample_subregion_resized = resize_image(sample_subregion_origin)
            sample_subregion_otus_resized = resize_image(sample_subregion_otus)
            template_subregion_resized = resize_image(template_subregion_origin)

            # 生成低分辨率下的随机点（用于后续映射）
            lowres_points = random_points_in_lowres(sample_subregion_otus_resized.shape)

            # 使用 lightglue 或其它匹配方法匹配子区域，计算单应矩阵
            homography_matrix = match_subregions_lightglue(sample_subregion_otus_resized, template_subregion_resized, idx, device)
            H_inv = np.linalg.inv(homography_matrix)

            # 利用计算得到的单应矩阵，将模板子区域进行透视变换
            warped_template = cv2.warpPerspective(template_subregion_resized, homography_matrix, (sample_subregion_resized.shape[1], sample_subregion_resized.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # 为避免颜色空间不匹配，采用 Pillow 将 warped_template 转为灰度
            warped_gray = np.array(Image.fromarray(warped_template).convert('L'))
            sample_gray = sample_subregion_otus_resized  # 已经是灰度图

            # 保证两幅灰度图尺寸一致，否则调整 warped_gray 的尺寸
            if sample_gray.shape != warped_gray.shape:
                warped_gray = cv2.resize(warped_gray, (sample_gray.shape[1], sample_gray.shape[0]))

            # 构造 3 通道叠加图：将样本子区域放到绿通道，变换后的模板子区域放到红通道
            stacked_rgb = np.zeros((sample_gray.shape[0], sample_gray.shape[1], 3), dtype=np.uint8)
            stacked_rgb[:, :, 1] = sample_gray
            stacked_rgb[:, :, 0] = warped_gray

            warped_points_in_img2_lowres = warp_points(lowres_points, H_inv)

            # 将低分辨率点映射到高分辨率（假设 map_points_to_highres 已实现）
            w1, h1 = sample_subregion_origin.shape[1], sample_subregion_origin.shape[0]
            w2, h2 = template_subregion_origin.shape[1], template_subregion_origin.shape[0]
            highres_points_img1 = map_points_to_highres(lowres_points, w1, h1)
            highres_points_img2 = map_points_to_highres(warped_points_in_img2_lowres, w2, h2)

            # 重新计算高分辨率图像之间的单应矩阵（使用 RANSAC）
            H_prime, _ = cv2.findHomography(highres_points_img2, highres_points_img1, cv2.RANSAC)

            # 将模板子区域原图转换为灰度（使用 Pillow 转换）
            template_subregion_origin_gray = np.array(Image.fromarray(template_subregion_origin).convert('L'))

            # 利用 H_prime 对高分辨率模板子区域进行透视变换
            warped_image2_highres = warp_image(template_subregion_origin_gray, H_prime, (w1, h1))
            stacked_highres = stack_images(sample_subregion_otus, warped_image2_highres)

            # 根据局部坐标对叠加图进行裁剪
            cropped_image = stacked_rgb[resized_local_start_y:resized_local_end_y, resized_local_start_x:resized_local_end_x]
            cropped_image_origin = stacked_highres[local_start_y:local_end_y, local_start_x:local_end_x]

            overlay = cropped_image_origin
            overlay_subregions.append(((i, j), overlay))
        except Exception as e:
            print(f"Error processing subregion {i + 1}: {e}")
            # 出错时，直接使用调整尺寸后的模板子区域
            overlay_subregions.append(template_subregion_resized)
        finally:
            subregion_end_time = time.perf_counter()
            subregion_time = subregion_end_time - subregion_start_time
            print(f"Subregion {i + 1} processing time: {subregion_time:.4f} seconds")

    # 按行列顺序对子区域排序并拼接
    overlay_subregions_sorted = sorted(overlay_subregions, key=lambda x: (x[0][0], x[0][1]))
    overlay_rows = []
    for i in range(num_rows):
        row_subregions = [subregion[1] for subregion in overlay_subregions_sorted if subregion[0][0] == i]
        row_combined = np.hstack(row_subregions)
        overlay_rows.append(row_combined)
    final_overlay_image = np.vstack(overlay_rows)

    # 使用 Pillow 保存最终叠加图像，避免 cv2.imwrite
    os.makedirs('results', exist_ok=True)
    final_output_path = os.path.join('results', 'final_overlay_image.tiff')

    save_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1] 
    result_test_final=cv2.imwrite(final_output_path, final_overlay_image,save_params)
    print("Write success:", result_test_final)

    print(f"Final overlay image saved to {final_output_path}")

    total_end_time = time.perf_counter()
    total_time = total_end_time - total_start_time
    print(f"Total processing time: {total_time:.4f} seconds")

    return final_output_path






def fine_matching_pillow(sample_image_path, template_image_path, overlap_sample=300, overlap_template=500, Auto_split=True, default_rows=3, default_columns=2, device='cuda'):
    """
    使用裁剪后的样本图像与模板图像进行细匹配。

    参数:
    - sample_image_path: 裁剪后的样本图像路径。
    - template_image_path: 裁剪后的模板图像路径。
    - overlap_sample: 样本图像子区域的重叠尺寸。
    - overlap_template: 模板图像子区域的重叠尺寸。
    - Auto_split: 是否自动分割子区域。
    - default_rows: 默认行数（当不自动分割时）。
    - default_columns: 默认列数（当不自动分割时）。
    - device: 计算设备（如 'cuda'）。
    
    返回:
    - final_overlay_image 的保存路径。
    """
    device = torch.device(device)
    print(f"Using device: {device}")

    # 检查图片是否存在
    if not os.path.exists(sample_image_path):
        print(f"Sample image not found: {sample_image_path}")
        return
    if not os.path.exists(template_image_path):
        print(f"Template image not found: {template_image_path}")
        return

    # 用 Pillow 加载图片，避免 cv2 的读图限制
    # sample_image_gray= np.array(Image.open(sample_image_path).convert('L'))
    sample_image_gray = np.array(Image.open(sample_image_path).convert('L'))
    template_image = np.array(Image.open(template_image_path).convert('L'))

    # 使用 CLAHE 进行图像增强（操作在灰度图上）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))
    clahe_image = clahe.apply(sample_image_gray.copy())
    clahe_image_2 = clahe.apply(clahe_image.copy())

    # 此处使用灰度图作为样本原图，并将 CLAHE 后的结果作为阈值图
    sample_image = sample_image_gray
    sample_image_otus = clahe_image_2

    if sample_image is None:
        print("Failed to load sample image.")
        return
    if template_image is None:
        print("Failed to load template image.")
        return

    # 调用 process_stitched_image 进行细匹配处理
    final_overlay_image = process_stitched_image_pillow(sample_image, template_image, sample_image_otus, overlap_sample, overlap_template, Auto_split, default_rows, default_columns, device)
    print("Fine matching completed.")
    return final_overlay_image
