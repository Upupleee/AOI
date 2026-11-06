import os
from PIL import Image
def q3():
    # 裁右上角
    # 解除 Pillow 对大图的像素限制
    Image.MAX_IMAGE_PIXELS = None

    ref_path = r".\output\test_cropped.tif"
    # ref_path = r".\aligned_cropped.tif"

    if not os.path.exists(ref_path):
        print(f"错误：参考图不存在 -> {ref_path}")
    else:
        try:
            # 打开图像（惰性加载）
            img = Image.open(ref_path)
            width, height = img.size

            # 裁剪最右侧 4096x4096（从顶部开始）
            crop_width = 8192
            crop_height = 8192
            if width < crop_width or height < crop_height:
                print(f"警告：图像尺寸 ({width}x{height}) 小于裁剪尺寸，将截取实际大小。")
                crop_width = min(crop_width, width)
                crop_height = min(crop_height, height)

            x = width - crop_width
            y = 0
            crop = img.crop((x, y, x + crop_width, y + crop_height))
            crop.save("output/test_crop_rightmost.png")
        except Exception as e:
            print(f"❌ 处理失败：{e}")