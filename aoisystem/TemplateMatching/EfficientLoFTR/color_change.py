from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# Open an image file
img = Image.open('datasets/template_1/template2_2000_full.png')

# Ensure the image is in RGBA mode
img = img.convert('RGBA')

# Get the pixel data
pixels = img.getdata()

# Define the lilac color (you can adjust the RGB values as needed)
lilac = (200, 162, 200, 255)  # (R, G, B, A)

# Create a list to hold the new pixel data
new_pixels = []

# Threshold to determine what is considered "white"
white_threshold = 240  # You can adjust this value

# Transform the pixels
for pixel in pixels:
    # Check if the pixel is white (you can adjust the condition as needed)
    if pixel[0] >= white_threshold and pixel[1] >= white_threshold and pixel[2] >= white_threshold:
        new_pixels.append(lilac)
    else:
        new_pixels.append(pixel)

# Update the image with the new pixels
img.putdata(new_pixels)

# Save the transformed image
img.save('datasets/template_1/template2_2000_full_lila.png')
