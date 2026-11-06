from lightglue import LightGlue, SuperPoint
from lightglue.superpoint import sample_descriptors
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import kornia
from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from scipy.spatial import KDTree, cKDTree
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import cv2
import numpy as np
import os

torch.set_grad_enabled(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_torch_image(fname):
    img = kornia.io.load_image(fname, kornia.io.ImageLoadType.RGB32)
    img = img[None]  # 1xCxHxW / fp32 / [0, 1]
    return img


def stack_images(image1, image2):
    stacked_image = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
    stacked_image[:, :, 1] = image1  # Green channel for image1
    stacked_image[:, :, 0] = image2  # Blue channel for warped image2
    return stacked_image

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


_default_cfg = deepcopy(full_default_cfg)
matcher = LoFTR(config=_default_cfg)
matcher.load_state_dict(torch.load("weights/eloftr_outdoor.ckpt")['state_dict'])
matcher_eloftr = reparameter(matcher).eval().to(device)

# Load LoFTR matcher
img1= cv2.imread('demo/sample_subregions_otus/sample_subregion_otus_3_2.png')
img2 = cv2.imread('demo/template_subregions/template_subregion_3_2.png')

sample_resized = resize_image(img1)
template_resized = resize_image(img2)

# Convert the OpenCV images from BGR to grayscale (LoFTR works with grayscale images)
sample_gray = cv2.cvtColor(sample_resized, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)

# Convert images to tensors and normalize to [0, 1]
sample_tensor = torch.from_numpy(sample_gray)[None][None].float().to(device) / 255.0
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

# Extract SuperPoint features
extractor = SuperPoint(max_num_keypoints=2048*2).eval().to(device)
feats0 = extractor.extract(sample_tensor )
feats1 = extractor.extract(template_tensor)

# Convert SuperPoint keypoints to NumPy arrays
sp_keypoints0 = feats0['keypoints'].cpu().numpy()[0]  # SuperPoint keypoints for img1
sp_keypoints1 = feats1['keypoints'].cpu().numpy()[0]  # SuperPoint keypoints for img2

#Convert images to NumPy arrays for plotting
img1_np = sample_resized  # H x W x C
img2_np = template_resized

# Create a figure with two subplots for the image pairs
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Display the images
axes[0].imshow(img1_np)
axes[1].imshow(img2_np)

# Plot SuperPoint keypoints (smaller dots) on the images
axes[0].scatter(sp_keypoints0[:, 0], sp_keypoints0[:, 1], 
                s=3, c='blue', marker='.', label='SuperPoint')
axes[1].scatter(sp_keypoints1[:, 0], sp_keypoints1[:, 1], 
                s=3, c='blue', marker='.', label='SuperPoint')

# Plot LoFTR keypoints (larger circles) on the images
axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], 
                s=6, facecolors='none', edgecolors='red', marker='o', label='ELoFTR')
axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], 
                s=6, facecolors='none', edgecolors='red', marker='o', label='ELoFTR')

# Set titles for the subplots
axes[0].set_title('Image 1 with Keypoints')
axes[1].set_title('Image 2 with Keypoints')

# Add legends to the subplots
axes[0].legend()
axes[1].legend()

# Remove axis ticks and labels for a cleaner look
axes[0].axis('off')
axes[1].axis('off')

# 定义判定为重合的最大距离（像素）
max_distance = 3  # 您可以根据需求调整这个值

# 构建 SuperPoint 特征点的 KDTree
sp_tree0 = cKDTree(sp_keypoints0)
sp_tree1 = cKDTree(sp_keypoints1)

# 查询重合的索引列表
overlapping_indices0 = sp_tree0.query_ball_point(mkpts0, r=max_distance)
overlapping_indices1 = sp_tree1.query_ball_point(mkpts1, r=max_distance)

# 将索引列表展平并转换为整数类型
# indices0 = np.hstack(overlapping_indices0).astype(int)
# indices1 = np.hstack(overlapping_indices1).astype(int)

# 收集所有重合的 SuperPoint 特征点坐标
# overlapping_sp_keypoints0 = sp_keypoints0[np.unique(indices0)]
# overlapping_sp_keypoints1 = sp_keypoints1[np.unique(indices1)]
# print(overlapping_sp_keypoints0)

# 可视化这些重合的 SuperPoint 特征点
# # 在图像上绘制重合的 SuperPoint 特征点（使用绿色叉号）
# axes[0].scatter(overlapping_sp_keypoints0[:, 0], overlapping_sp_keypoints0[:, 1], 
#                 s=6, c='green', marker='x', label='Overlapping SuperPoint')
# axes[1].scatter(overlapping_sp_keypoints1[:, 0], overlapping_sp_keypoints1[:, 1], 
#                 s=6, c='green', marker='x', label='Overlapping SuperPoint')

# # 更新图例
# axes[0].legend()
# axes[1].legend()

# # 显示更新后的图像
# plt.show()

# print(1)

# 将 NumPy 数组转换为 PyTorch 张量，并移动到正确的设备（CPU 或 GPU）
# indices0_tensor = torch.from_numpy(indices0).long().to(device)
# indices1_tensor = torch.from_numpy(indices1).long().to(device)
keypoints_shape = feats0['keypoints'].shape 
print(f"feats0['keypoints'] shape: {keypoints_shape}")
# print(f"indices0_tensor min: {indices0_tensor.min()}, max: {indices0_tensor.max()}")

# 从 feats0 和 feats1 中提取重合的特征
# overlapping_feats0 = {
#     'keypoints': feats0['keypoints'][:, indices0_tensor, :],            # shape: [1, M0, 2]
#     'keypoint_scores': feats0['keypoint_scores'][:, indices0_tensor],   # shape: [1, M0]
#     'descriptors': feats0['descriptors'][:,indices0_tensor, :],        # shape: [1, D, M0]
#     'image_size': feats0['image_size'],                                 # shape: [1, 2]
# }

# overlapping_feats1 = {
#     'keypoints': feats1['keypoints'][:, indices1_tensor, :],            # shape: [1, M1, 2]
#     'keypoint_scores': feats1['keypoint_scores'][:, indices1_tensor],   # shape: [1, M1]
#     'descriptors': feats1['descriptors'][:,indices1_tensor,:],        # shape: [1, D, M1]
#     'image_size': feats1['image_size'],                                 # shape: [1, 2]
# }

# 现在，overlapping_feats0 和 overlapping_feats1 包含了仅对应于重合特征点的所有特征
# # 例如，提取重合的 keypoints 进行可视化
# overlapping_sp_keypoints0 = overlapping_feats0['keypoints'][0].cpu().numpy()
# overlapping_sp_keypoints1 = overlapping_feats1['keypoints'][0].cpu().numpy()

# 在图像上绘制重合的 SuperPoint 特征点（使用绿色叉号）
# axes[0].scatter(overlapping_sp_keypoints0[:, 0], overlapping_sp_keypoints0[:, 1], 
#                 s=6, c='green', marker='x', label='Overlapping SuperPoint')
# axes[1].scatter(overlapping_sp_keypoints1[:, 0], overlapping_sp_keypoints1[:, 1], 
#                 s=6, c='green', marker='x', label='Overlapping SuperPoint')

# # 更新图例
# axes[0].legend()
# axes[1].legend()

# # 显示更新后的图像
# plt.show()
# print(0)

# matcher = LightGlue(features='superpoint').eval().cuda()
# matches_glue = matcher({'image0': overlapping_feats0, 'image1': overlapping_feats1})

# feats0, feats1, matches01 = [rbd(x) for x in [overlapping_feats0, overlapping_feats1, matches_glue]]  # remove batch dimension
# matches = matches01['matches']  # indices with shape (K,2)
# points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
# points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)



matcher = LightGlue(features='superpoint').eval().cuda()
matches_glue = matcher({'image0': feats0, 'image1': feats1})

feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches_glue]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)
points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)


print(points0.shape)

pts0 = points0.cpu().numpy()
pts1 = points1.cpu().numpy()

H, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 5.0)

# Step 2: Warp img2 onto img1
height, width = img1_np.shape[:2]
img2_warped = cv2.warpPerspective(img2_np, H, (width, height))

# Ensure images are grayscale
if len(img1_np.shape) == 3 and img1_np.shape[2] == 3:
    img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_BGR2GRAY)
else:
    img1_gray = img1_np

if len(img2_warped.shape) == 3 and img2_warped.shape[2] == 3:
    img2_gray = cv2.cvtColor(img2_warped, cv2.COLOR_BGR2GRAY)
else:
    img2_gray = img2_warped

stacked_result = stack_images(img1_gray, img2_gray)

os.makedirs('./test/lightglue', exist_ok=True)
stacked_rgb_path = os.path.join('./test/lightglue', f'superpoints_loftr_lightglue.png')
cv2.imwrite(stacked_rgb_path, stacked_result)
print(f"Stacked RGB image saved to {stacked_rgb_path} for debugging.")
