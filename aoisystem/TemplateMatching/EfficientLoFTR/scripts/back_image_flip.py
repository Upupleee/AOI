import numpy as np
from PIL import Image
import time

Image.MAX_IMAGE_PIXELS = None

def flip_image_numpy(image_array):
    """

    如果需要内存连续数组，可以调用 .copy()
    """
    return image_array[:,::-1]

def main():
    input_file = "2652s2g241387a0-gbl-3000DPI.bmp"        # 输入的灰度TIFF文件路径
    output_file = "2652s2g241387a0-gbl-3000DPI_flipped.bmp"       # 翻转后的输出文件路径

    # 打开 TIFF 图像（假设是灰度图像，Pillow会以单通道方式加载）
    img = Image.open(input_file)
    print("原始图像模式:", img.mode)
    print("原始图像尺寸:", img.size)

    # 转换为 NumPy 数组
    img_array = np.array(img)

    # 使用 NumPy 进行同时上下和左右翻转
    # TODO: flip mode has to be changed in order to get correct image
    start_time = time.time()
    flipped_array = flip_image_numpy(img_array)
    elapsed = time.time() - start_time
    print(f"翻转操作耗时: {elapsed:.6f}秒")

    # 如有需要，可调用 copy() 获得数据拷贝（通常用于后续需连续内存的情况）
    # flipped_array = flipped_array.copy()

    # 将翻转后的数组转换回图像，并保存
    flipped_img = Image.fromarray(flipped_array)
    flipped_img.save(output_file)
    print("翻转后的图像已保存为:", output_file)

if __name__ == '__main__':
    main()
