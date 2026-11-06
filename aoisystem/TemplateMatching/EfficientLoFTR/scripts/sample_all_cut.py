import os
import glob
from PIL import Image

# 取消 Pillow 的解压炸弹检查
Image.MAX_IMAGE_PIXELS = None

# 1. 配置源文件夹和输出文件夹
source_folder = '/home/aoi/image_stiching/sample_new_2'    # 替换为你的拍摄图片文件夹
output_folder = '/home/aoi/image_stiching/cropped_photos'
os.makedirs(output_folder, exist_ok=True)

# 2. 定义要处理的文件类型
extensions = ('*.bmp', '*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg')

# 3. 遍历所有图片文件
for ext in extensions:
    for filepath in glob.glob(os.path.join(source_folder, ext)):
        # 打开图像
        img = Image.open(filepath)
        w, h = img.size

        # 如果图像高度小于2000，跳过或直接保存原图
        if h <= 2000:
            print(f" 图像高度 {h}px 小于等于 2000px，跳过裁剪：{os.path.basename(filepath)}")
            continue

        # 4. 裁剪掉上方2000像素
        #    box = (left, upper, right, lower)
        crop_box = (0, 1800, w, h)
        cropped = img.crop(crop_box)

        # 5. 构造输出路径并保存
        filename = os.path.basename(filepath)
        out_path = os.path.join(output_folder, filename)
        cropped.save(out_path)
        print(f"Saved cropped image: {out_path}")
