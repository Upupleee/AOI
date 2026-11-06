from PIL import Image
import os
import math

Image.MAX_IMAGE_PIXELS = None

# === 图像路径 ===
image_path = "/home/aoi/image_stiching/506p2e241354a0-gtl-3000DPI.bmp"
img = Image.open(image_path)
if img.mode != 'L':
    img = img.convert('L')

# === 参数 ===
# Template DPI
dpi_template = 3000
pixels_per_mm_template = dpi_template / 25.4  # ≈118.11

# Sample 参数
sample_physical_width_mm = 125.5   # 相机视野物理宽度
sample_pixels = 16384              # sample的像素宽度

# Step 宽度（template每列的物理宽度）
step_mm = 120
step_pixels = int(round(step_mm * pixels_per_mm_template))  # 每一列template切块的宽度
template_coverage_px = int(round(sample_physical_width_mm * pixels_per_mm_template))  # sample拍摄宽度在template上的像素

# 图像尺寸
W, H = img.size

# 输出路径
save_dir = "splits_fixed_precise"
os.makedirs(save_dir, exist_ok=True)

# === 精准切分（从右往左）===
boxes = []
x_right = W
while x_right - step_pixels > 0:
    x_left = x_right - step_pixels
    boxes.append((x_left, 0, x_right, H))
    x_right = x_left  # 向左推进

# 最后一列：吸收左边剩余部分（不管多宽）
if x_right > 0:
    boxes.append((0, 0, x_right, H))

# === 写入文件 ===
for idx, box in enumerate(boxes, 1):
    slice_img = img.crop(box)
    slice_filename = os.path.join(save_dir, f"slice_{idx:03d}.bmp")
    slice_img.save(slice_filename)
    print(f"✅ Saved {slice_filename}, box={box}")
