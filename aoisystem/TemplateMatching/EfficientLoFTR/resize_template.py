from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2

# Open the image using PIL
image_path = 'datasets/fake_test/test_10_rotation/test10-1800dpi_augmented_with_rotation.jpg'  # Replace with the path to your image
img = Image.open(image_path)


# Resize the image to 0.5x
new_size = (int(img.width * 0.5), int(img.height * 0.5))
resized_img = img.resize(new_size ,Image.LANCZOS)

# Save the resized image in PNG format
resized_img.save('datasets/fake_test/test_10_rotation/test10-1800dpi_augmented_with_rotation.png', format='PNG')

print("Image resized and saved as 'resized_image.png'.")