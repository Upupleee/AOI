import torch
from coarse_matching_new import coarse_matching_eloftr_pillow_sample_only
from fine_matching_loftr_new import fine_matching_pillow_on_template
import os
import re
from PIL import Image

def extract_index_from_filename(filename):
    """
    从文件名中提取数字作为索引，例如 "test_imgs2.tif" 提取2，"template_image_matched_1.tiff" 提取1。
    """
    m = re.search(r'(\d+)', filename)
    if m:
        return int(m.group(1))
    return None
def main():
    # 定义采集图像和模板图像所在文件夹
    acquired_folder = "/home/aoi/pretest/images/p2062_GTL_record_2"    # 采集图片，如 test_imgs1.tif, test_imgs2.tif, ...
    template_folder = "/home/aoi/pretest/images/P2062_GTL_template_split"        # 模板图片，如 template_image_matched_1.tiff, template_image_matched_2.tiff, ...
    
    # 获取两个文件夹下支持的图像文件（.tif, .tiff），并按文件名中的数字排序
    acquired_files = sorted(
        [f for f in os.listdir(acquired_folder) if f.lower().endswith(('.tif', '.tiff'))],
        key=lambda x: extract_index_from_filename(x)
    )
    template_files = sorted(
        [f for f in os.listdir(template_folder) if f.lower().endswith(('.tif', '.tiff'))],
        key=lambda x: extract_index_from_filename(x)
    )
    
    if len(acquired_files) != len(template_files):
        print("警告：采集图片和模板图片的数量不一致！")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")


    fine_matching_configs = [
    {"overlap_sample": 200, "default_rows": 4, "default_columns": 3},
    {"overlap_sample": 200, "default_rows": 4, "default_columns": 3},
    {"overlap_sample": 200, "default_rows": 4, "default_columns": 3},
    {"overlap_sample": 200, "default_rows": 4, "default_columns": 3},
    {"overlap_sample": 200, "default_rows": 4, "default_columns": 3}

    # 可以继续添加……
    ]

    
    # 针对每一对图片依次进行配准处理
    for idx, (acq_filename, tpl_filename) in enumerate(zip(acquired_files, template_files)):
        sample_image_path = os.path.join(acquired_folder, acq_filename)
        template_image_path = os.path.join(template_folder, tpl_filename)
        print("\n---------------------------------------")
        print(f"配准序号 {idx}：")
        print(f"采集图片路径: {sample_image_path}")
        print(f"模板图片路径: {template_image_path}")
        
        # Step 1: 粗匹配阶段
        print("开始粗匹配阶段...")
        object_region, corresponding_sample_region = coarse_matching_eloftr_pillow_sample_only(
            sample_image_path, 
            template_image_path,
            enlarge=True,
            margin_percent=0,
            scale_factor=1.0
        )
        if object_region is None or corresponding_sample_region is None:
            print("粗匹配失败，跳过此组配准。")
            continue
        print("粗匹配完成。")

        # 根据索引获得对应的细匹配参数设置，如果数量不足则使用默认设置
        if idx < len(fine_matching_configs):
            cfg = fine_matching_configs[idx]
        else:
            cfg = {"overlap_sample": 200, "default_rows": 4, "default_columns": 2}

        # Step 2: 细匹配阶段（利用粗匹配裁剪结果）
        print("开始细匹配阶段...")
        final_overlay_path = fine_matching_pillow_on_template(
            corresponding_sample_region,   # 裁剪后的采集图像
            object_region,                 # 裁剪后的模板图像
            overlap_sample=cfg["overlap_sample"],
            overlap_template=0,
            Auto_split=False,
            default_rows=cfg["default_rows"],
            default_columns=cfg["default_columns"],
            device=device.type,
            image_index=idx
        )
        print(f"细匹配完成，结果保存为: {final_overlay_path}")
                # 清理 CUDA 缓存，防止内存泄漏和OOM错误
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
        
if __name__ == '__main__':
    main()
