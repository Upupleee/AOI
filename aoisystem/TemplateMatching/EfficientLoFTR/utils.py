# utils.py

import cv2
import numpy as np
import os
import torch
import kornia as K

def pad_image(image, scale_factor=1.6):
    """
    Pads the image on the left side to make it wider.

    Parameters:
    - image: The input image.
    - scale_factor: The factor by which to increase the width.

    Returns:
    - padded_image: The padded image.
    """
    # This function is directly from your `corse_loftr.py`
    if image is None:
        raise ValueError("Image not found")

    height, width = image.shape
    new_width = int(width * scale_factor)
    padded_image = np.zeros((height, new_width), dtype=np.uint8)
    padded_image[:, new_width - width:] = image
    return padded_image


def pad_image_gray(image, scale_factor=1.6):
    """
    Pads the image on the left side to make it wider.

    Parameters:
    - image: The input image.
    - scale_factor: The factor by which to increase the width.

    Returns:
    - padded_image: The padded image.
    """
    # This function is directly from your `corse_loftr.py`
    if image is None:
        raise ValueError("Image not found")

    height, width = image.shape
    new_width = int(width * scale_factor)
    padded_image = np.zeros((height, new_width), dtype=np.uint8)
    padded_image[:, new_width - width:] = image
    return padded_image


def resize_image(image, width=640, height=480):
    """
    Resizes the image to the specified width and height.

    Parameters:
    - image: The input image.
    - width: The desired width.
    - height: The desired height.

    Returns:
    - resized_image: The resized image.
    """
    # Moved from both `corse_loftr.py` and `fine_loftr.py`
    image_copy = image.copy()
    new_dimensions = (width, height)
    resized_image = cv2.resize(image_copy, new_dimensions)
    return resized_image

def convert_to_grayscale(image):
    """
    Converts the image to grayscale and applies thresholding.

    Parameters:
    - image: The input image.

    Returns:
    - binary_image: The binary (thresholded) grayscale image.
    """
    # This function is directly from your `corse_loftr.py`
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray_image.dtype != np.uint8:
        gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def visualize_bbox(image, bbox, window_name):
    pt1 = (int(bbox['left_upper'][0]), int(bbox['left_upper'][1]))
    pt2 = (int(bbox['right_lower'][0]), int(bbox['right_lower'][1]))
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def map_bbox_to_high_res(low_res_bbox, scale_x, scale_y):
    """
    Maps the bounding box from low-resolution to high-resolution coordinates.

    Parameters:
    - low_res_bbox: The bounding box in low-resolution coordinates.
    - scale_x: The scale factor in the x-direction.
    - scale_y: The scale factor in the y-direction.

    Returns:
    - high_res_bbox: The bounding box in high-resolution coordinates.
    """
    # This function is directly from your `corse_loftr.py`
    high_res_bbox = {
        'left_upper': (int(low_res_bbox['left_upper'][0] * scale_x), int(low_res_bbox['left_upper'][1] * scale_y)),
        'right_upper': (int(low_res_bbox['right_upper'][0] * scale_x), int(low_res_bbox['right_upper'][1] * scale_y)),
        'left_lower': (int(low_res_bbox['left_lower'][0] * scale_x), int(low_res_bbox['left_lower'][1] * scale_y)),
        'right_lower': (int(low_res_bbox['right_lower'][0] * scale_x), int(low_res_bbox['right_lower'][1] * scale_y))
    }
    print("The cutting points on template:", high_res_bbox)
    return high_res_bbox

def enlarge_bbox(bbox, margin_percent, image_shape):
    """
    Enlarges the bounding box by a certain margin.

    Parameters:
    - bbox: The original bounding box.
    - margin_percent: The percentage to enlarge the bounding box.
    - image_shape: The shape of the image to ensure the bbox stays within bounds.

    Returns:
    - enlarged_bbox: The enlarged bounding box.
    """
    # This function is directly from your `corse_loftr.py`
    margin_x = int((bbox['right_lower'][0] - bbox['left_upper'][0]) * margin_percent)
    margin_y = int((bbox['right_lower'][1] - bbox['left_upper'][1]) * margin_percent)

    enlarged_bbox = {
        'left_upper': (max(0, bbox['left_upper'][0] - margin_x), max(0, bbox['left_upper'][1] - margin_y)),
        'right_lower': (
            min(image_shape[1], bbox['right_lower'][0] + margin_x),
            min(image_shape[0], bbox['right_lower'][1] + margin_y)
        )
    }
    return enlarged_bbox

def save_image(image, directory, filename):
    """
    Saves the image to the specified directory with the given filename.

    Parameters:
    - image: The image to save.
    - directory: The directory to save the image in.
    - filename: The name of the file.
    """
    # This is a new utility function to handle image saving
    os.makedirs(directory, exist_ok=True)
    output_path = os.path.join(directory, filename)
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")


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