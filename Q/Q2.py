import os
from PIL import Image


def q2(input_path = r"D:\dataset\0407\0407\p016p4e019723a0\27\gbl\test_imgs1.tif"):
    # 处理原测试图
    # 解除 Pillow 对大图的像素限制
    Image.MAX_IMAGE_PIXELS = None

    # 指定输入图片路径（请根据实际情况修改）
    input_path = r"D:\dataset\0407\0407\p016p4e019723a0\27\gbl\test_imgs1.tif"

    # 输出路径
    output_path = r".\output\test_cropped.tif"

    # 上下各裁剪的像素数
    crop_top = 9070
    crop_bottom = 9070

    if not os.path.exists(input_path):
        print(f"错误：文件不存在 -> {input_path}")
    else:
        try:
            # 打开图像（仅读取文件头）
            img = Image.open(input_path)
            width, height = img.size

            # 检查高度是否足够裁剪
            if height <= crop_top + crop_bottom:
                print(f"错误：图像高度 {height} 不足以裁剪上下 {crop_top}+{crop_bottom} 像素")
            else:
                # 裁剪区域：左0，上 crop_top，右 width，下 height - crop_bottom
                crop_box = (0, crop_top, width, height - crop_bottom)
                # 裁剪（只读取该区域数据）
                cropped = img.crop(crop_box)
                # 保存为 TIFF（默认 LZW 压缩，可保持无损）
                cropped.save(output_path, format='TIFF')
                print(f"✅ 裁剪成功！新图片保存为：{output_path}")
                print(f"   原尺寸：{width} x {height}，新尺寸：{width} x {height - crop_top - crop_bottom}")
        except Exception as e:
            print(f"❌ 处理失败：{e}")

