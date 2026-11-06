from PIL import Image
import os
import numpy as np
import tifffile


Image.MAX_IMAGE_PIXELS = None
def combine_images_from_folder(folder_path):
    # 获取文件夹中所有 TIFF 文件（不区分大小写）
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tiff')]
    
    # 按文件名排序，如有需要根据实际情况调整排序方式
    files.sort()
    files.reverse()

    if not files:
        print("文件夹中没有找到 TIFF 文件！")
        return

    # 计算拼接后图像的总宽度和最大高度
    total_width = 0
    max_height = 0
    images = []
    for file_name in files:
        image_path = os.path.join(folder_path, file_name)
        # 使用 Pillow 加载图像并转换为 RGB 模式
        img = Image.open(image_path).convert('RGB')
        images.append(img)
        width, height = img.size
        total_width += width
        # 以最高高度作为合成图像的高度（这里简单采用所有图像中最高者）
        if height > max_height:
            max_height = height

    # 创建一个新的空白 RGB 图像，背景颜色可根据需要修改（此处使用黑色背景）
    combined_image = Image.new('RGB', (total_width, max_height), (0, 0, 0))

    # 将每个图像依次粘贴到新图像中（从左至右，顶部对齐）
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    # 将拼接后的 PIL 图像转换为 NumPy 数组（RGB 顺序）
    combined_array = np.array(combined_image)

    # 使用 tifffile 保存为 BigTIFF 格式的文件
    output_filename = "res_506s4e.tif"
    tifffile.imwrite(output_filename, combined_array, bigtiff=True)
    print(f"拼接完成，输出文件已保存为：{output_filename}")

# 修改 folder_path 为包含待拼接图片的文件夹路径
folder_path = "/home/aoi/AOI/TemplateMatching/EfficientLoFTR/results_overlay"
combine_images_from_folder(folder_path)
