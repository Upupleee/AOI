import cv2
import numpy as np
from scipy.linalg import polar

# Load your images
img1 = cv2.imread("datasets/fake_test/test_10_rotation/test8-1400dpi_augmented_with_rotation_1.jpg")
img2 = cv2.imread("datasets/fake_test/template/test8-1400dpi.png")

# Example homography matrix H (replace this with your actual H matrix)
H = np.array([
    [9.95644367e-01,  1.18444865e-01, -5.97071619e+01],
    [-6.38370565e-02,  9.95219696e-01,  5.08416923e+01],
    [-3.48809598e-07, -3.78266210e-07,  1.00000000e+00]
])

# Step 1: Extract the upper-left 2x2 affine part of H
affine_part = H[:2, :2]

# Step 2: Apply polar decomposition to extract the rotation component
rotation_matrix, _ = polar(affine_part)

# Step 3: Apply the rotation to img2 to align it with img1
# Get the center of img2 to apply rotation around it
center = (img2.shape[1] // 2, img2.shape[0] // 2)

# Convert rotation matrix to angle and scale for OpenCV's warpAffine
angle = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
scale = np.sqrt(np.linalg.det(rotation_matrix))  # Scaling factor (optional)

print(angle)
print(scale)
# Generate the affine transformation matrix for rotation
rotation_affine_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# Apply rotation to img2
img2_aligned = cv2.warpAffine(img2, rotation_affine_matrix, (img2.shape[1], img2.shape[0]))

# Display the result
# Save the images instead of displaying them
cv2.imwrite("img1_aligned.jpg", img1)
cv2.imwrite("img2_aligned.jpg", img2_aligned)
