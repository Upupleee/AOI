import numpy as np
from PIL import Image
from skimage.exposure import match_histograms
# 禁用解压炸弹检查（或设置为更高的值）
Image.MAX_IMAGE_PIXELS = None

def histogram_match_template(template_path, captured_path, output_path): ##worst
    # 打开图像并转换为灰度图
    template_img = Image.open(template_path).convert('L')
    captured_img = Image.open(captured_path).convert('L')
    
    # 将图像转换为 numpy 数组
    template_array = np.array(template_img)
    captured_array = np.array(captured_img)
    
    # 进行直方图匹配
    matched = match_histograms(template_array, captured_array)
    
    # 转换回 PIL Image，并保存结果
    matched_img = Image.fromarray(np.uint8(matched))
    matched_img.save(output_path)
    print("直方图匹配处理完成，结果保存到：", output_path)



def linear_adjust_template(template_path, captured_path, output_path): #best
    # 打开图像并转换为灰度图
    template_img = Image.open(template_path).convert('L')
    captured_img = Image.open(captured_path).convert('L')
    
    # 转换为 numpy 数组并转为浮点数
    template_array = np.array(template_img).astype(np.float32)
    captured_array = np.array(captured_img).astype(np.float32)
    
    # 计算均值和标准差
    mu_template = template_array.mean()
    sigma_template = template_array.std()
    mu_captured = captured_array.mean()
    sigma_captured = captured_array.std()
    
    # 线性调整公式
    adjusted = (template_array - mu_template) / sigma_template * sigma_captured + mu_captured
    
    # 限制值域在 [0, 255] 内
    adjusted = np.clip(adjusted, 0, 255)
    
    # 转换回图像并保存
    adjusted_img = Image.fromarray(adjusted.astype(np.uint8))
    adjusted_img.save(output_path)
    print("线性亮度/对比度调整处理完成，结果保存到：", output_path)



def replace_high_pixels(image_path, output_path, threshold, new_value):# manual 
    """
    将图像中所有灰度值大于 threshold 的像素替换为 new_value，其它像素保持不变。
    
    参数:
      image_path: 原图路径
      output_path: 输出图像路径
      threshold: 灰度值阈值（例如200）
      new_value: 替换后的值（例如128）
    """
    # 打开图像并转换为灰度图
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # 使用 np.where 替换灰度值大于 threshold 的像素
    replaced_array = np.where(img_array > threshold, new_value, img_array)
    
    # 转换回 PIL Image，并保存结果
    replaced_img = Image.fromarray(replaced_array.astype(np.uint8))
    replaced_img.save(output_path)
    print(f"替换灰度值大于 {threshold} 的像素为 {new_value} 完成，结果保存到：", output_path)



if __name__ == "__main__":
    # 根据你的大图处理逻辑，这里可以加入其他预处理步骤，例如分块或翻转处理
    template_file = "/home/aoi/image_stiching/splits_test/slice_5.bmp"   # 模板图路径
    captured_file = "sample_new/cropped_sample_image_5.tiff"     # 采集图路径
    output_file = "/home/aoi/image_stiching/test_folder_new/template_image_matched_5.tiff"  # 输出路径
    
    linear_adjust_template(template_file, captured_file, output_file)
    # replace_high_pixels(template_file, output_file, threshold=150, new_value=118)