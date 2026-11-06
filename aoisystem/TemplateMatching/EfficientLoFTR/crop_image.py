from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def crop_image_by_percentage(image_path, output_path, left_percent, top_percent, right_percent, bottom_percent):
    """
    Crop the image based on percentage intervals of width and height.

    Parameters:
    - image_path: str, path to the input image
    - output_path: str, path to save the cropped image
    - left_percent: float, percentage of the width for the left boundary (0 to 1)
    - top_percent: float, percentage of the height for the top boundary (0 to 1)
    - right_percent: float, percentage of the width for the right boundary (0 to 1)
    - bottom_percent: float, percentage of the height for the bottom boundary (0 to 1)
    """
    
    # Open the image
    image = Image.open(image_path)
    width, height = image.size

    # Calculate the pixel boundaries for cropping
    left = int(left_percent * width)
    top = int(top_percent * height)
    right = int(right_percent * width)
    bottom = int(bottom_percent * height)

    # Crop the image using the calculated boundaries
    cropped_image = image.crop((left, top, right, bottom))

    # Save the cropped image
    cropped_image.save(output_path)

    print(f"Image cropped and saved to {output_path}")

# Example usage:
image_path = "datasets/fake_test/test10.png"
output_path = "datasets/fake_test/test10_new.png"

# Crop 10% from the left and top, and 90% on the right and bottom
crop_image_by_percentage(image_path, output_path, 0, 0.04, 1, 0.96)
