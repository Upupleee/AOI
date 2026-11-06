import os
import glob
from PIL import Image
import re
Image.MAX_IMAGE_PIXELS = None
# 1. 获取模板图片文件列表并排序
template_folder = '/home/aoi/image_stiching/506s4e/template'  # 请替换为你的模板文件夹路径
# 假设模板文件名格式为: xxx_数字.tiff
template_files = glob.glob(os.path.join(template_folder, '*.tif'))

# 定义一个函数，根据文件名提取数字
def extract_number(filename):
    # 匹配末尾数字
    m = re.search(r'_(\d+)\.tif$', filename)
    if m:
        return int(m.group(1))
    return -1  # 如果没有匹配则放在最前面

# 排序文件列表，按照编号从小到大排序
template_files.sort(key=extract_number)

if len(template_files) < 2:
    raise ValueError("模板文件数量不足两张，请检查模板文件夹。")

# 选取倒数第二张模板和最后一张模板
template_second_last = template_files[-2]
template_last = template_files[-1]

# 2. 读取这两张模板图片，并获取它们的宽度
img_second_last = Image.open(template_second_last)
img_last = Image.open(template_last)

width_second_last = img_second_last.width
width_last = img_last.width

print(f"倒数第二张模板宽度: {width_second_last}")
print(f"最后一张模板宽度: {width_last}")

# 计算宽度比例
ratio = width_last / width_second_last
print(f"宽度比例 (最后一张/倒数第二张): {ratio}")

# 3. 裁剪样本图像（相机采集的最后一张图片）
sample_image_path = '/home/aoi/image_stiching/506s4e/sample/test_imgs4.tif'  # 替换为样本图片路径
sample_img = Image.open(sample_image_path)
sample_width, sample_height = sample_img.width, sample_img.height

# 根据比例计算保留宽度，从右侧保留
crop_width = int(sample_width * ratio)

# 处理防止超出边界（即 crop_width 不能大于 sample_width）
if crop_width > sample_width:
    print("警告：计算得到的保留宽度超过了样本图片总宽度，直接保留整个图片。")
    crop_width = sample_width

# 裁剪区域：从右侧开始，即 [左侧起点, 上, 右, 下]
left = sample_width - crop_width
upper = 0
right = sample_width
lower = sample_height

cropped_img = sample_img.crop((left, upper, right, lower))

# 保存裁剪后的图片
output_path = '/home/aoi/image_stiching/506s4e/sample/test_imgs4_cropped.tif'
cropped_img.save(output_path)
print(f"裁剪后的图片已保存到 {output_path}")

# 可选：展示裁剪后的图片
# cropped_img.show()
