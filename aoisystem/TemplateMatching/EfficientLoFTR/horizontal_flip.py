from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Open the image
image = Image.open("25_um_template_new.png")

# Rotate the image -90 degrees (counter-clockwise) and adjust width and height
rotated_image = image.transpose(Image.ROTATE_180)

# Save the rotated image
rotated_image.save("25_um_template_new.png")

print("Image rotated by -90 degrees successfully!")
