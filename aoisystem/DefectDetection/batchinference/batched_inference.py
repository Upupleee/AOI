import sahi
import os
import glob
import time
import shutil
import torch
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_batched_prediction, get_batched_sliced_prediction, get_sliced_prediction
from sahi.slicing import slice_image
from sahi.utils.cv import read_image
from PIL import Image
import numpy as np

import json
import os
import subprocess

# 添加GPU显存监控
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False
    print("NVML库不可用，无法监控显存")

Image.MAX_IMAGE_PIXELS = None

def get_gpu_memory_usage():
    """获取GPU显存使用情况"""
    if not NVML_AVAILABLE:
        if torch.cuda.is_available():
            # 使用torch.cuda获取
            used = torch.cuda.memory_allocated() / (1024 ** 3)  # 转换为GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return used, total
        return None, None
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 获取第一个GPU的句柄
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = info.used / (1024 ** 3)  # 转换为GB
        total = info.total / (1024 ** 3)
        return used, total
    except:
        return None, None

def clear_gpu_memory(gpu_index=0):
    """
    清除 GPU 中非系统进程的显存占用（仅清理非 Xorg/gnome-shell 等安全进程）
    """
    try:
        result = subprocess.check_output([
            'nvidia-smi',
            '--query-compute-apps=pid,process_name,gpu_uuid',
            '--format=csv,noheader,nounits'
        ])
        lines = result.decode().strip().split('\n')
        for line in lines:
            # 跳过空行
            if not line.strip():
                continue

            parts = [x.strip() for x in line.split(',')]
            if len(parts) != 3:
                print(f"跳过格式异常行: {line}")
                continue

            pid_str, name, uuid = parts
            pid = int(pid_str)

            # 排除常驻进程
            if name in ["Xorg", "gnome-shell", "sunloginclient"]:
                continue

            print(f"尝试终止进程: PID={pid} 名称={name}")
            subprocess.run(["kill", "-9", str(pid)])
        print("非必要GPU进程已清理完成。")
    except subprocess.CalledProcessError as e:
        print("执行 nvidia-smi 查询失败:", e)

# 打印初始GPU状态
initial_gpu_used, gpu_total = get_gpu_memory_usage()
if initial_gpu_used is not None:
    print(f"初始GPU显存: {initial_gpu_used:.2f}GB / {gpu_total:.2f}GB")

# 清理非必要 GPU 占用（以便加载模型）
print("开始清理GPU显存中非必要进程...")
clear_gpu_memory()
time.sleep(2)  # 等待进程彻底释放显存

# 再次打印显存状态
after_clear_gpu, _ = get_gpu_memory_usage()
if after_clear_gpu is not None:
    print(f"清理后GPU显存: {after_clear_gpu:.2f}GB / {gpu_total:.2f}GB")
start_time = time.time()

# 加载检测模型
model_load_start = time.time()
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="/home/dual3080/Workspace/ProjectTest/AOI/Results/aoi-0409-patience-0-1800epoch/weights/best-v12n-0409.pt",
    confidence_threshold=0.1,
    device="0"
)
model_load_time = time.time() - model_load_start
print(f"模型加载时间: {model_load_time:.2f}秒")

# 图像路径和结果目录
image_path = "/home/dual3080/Dataset/AOI_hsx/test_imgs3_matched_20250525_110731.tiff"
result_dir = "/home/dual3080/Workspace/test/results"

# 确保结果目录存在
os.makedirs(result_dir, exist_ok=True)

# 打开图像
image_load_start = time.time()
image = Image.open(image_path)
image_np = np.array(image)
image_size = image.size
image_load_time = time.time() - image_load_start
print(f"原始图像尺寸: {image_size}")
print(f"图像加载时间: {image_load_time:.2f}秒")

# 设置切片参数
slice_height = 1600
slice_width = 1600
overlap_height_ratio = 0.05
overlap_width_ratio = 0.05

# 记录推理前的GPU显存
pre_inference_gpu_used, _ = get_gpu_memory_usage()
if pre_inference_gpu_used is not None:
    print(f"\n推理前GPU显存: {pre_inference_gpu_used:.2f}GB / {gpu_total:.2f}GB")

print("\n开始优化后的批量切片推理流程...")
inference_start_time = time.time()

batch_result = get_batched_sliced_prediction(
    image=image_np,
    detection_model=detection_model,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
    slices_outdir = "/home/dual3080/Workspace/test/results/slices",
    slices_filename="output_slices",
    slices_ext=".jpg",
    visualize_slices=True,                 # 启用切片可视化
    slice_visualization_dir="/home/dual3080/Workspace/test/results/visualized", # 切片可视化输出目录
    visual_bbox_thickness=2,               # 边界框厚度
    visual_text_size=0.6,                  # 文本大小
    visual_hide_labels=False,              # 显示标签
    visual_hide_conf=False,                # 显示置信度
    verbose=1                              # 显示详细信息
)

inference_time = time.time() - inference_start_time

# 记录推理后的GPU显存
post_inference_gpu_used, _ = get_gpu_memory_usage()

print(f"批量推理完成，耗时: {inference_time:.2f}秒")
if post_inference_gpu_used is not None:
    print(f"推理后GPU显存: {post_inference_gpu_used:.2f}GB / {gpu_total:.2f}GB")
    if pre_inference_gpu_used is not None:
        print(f"推理过程显存增加: {post_inference_gpu_used - pre_inference_gpu_used:.2f}GB")

# 可视化结果
print("\n开始可视化结果...")
visualization_start_time = time.time()

# 2. 可视化完整图像上的预测结果
batch_result.visualize_full_predictions(
    original_image=image,                  # 使用原始图像
    export_dir="/home/dual3080/Workspace/test/results/visualized",               # 导出目录
    file_name="full_prediction",           # 输出文件名
    text_size=1.0,                         # 文本大小
    rect_th=2,                             # 边界框厚度
    hide_labels=False,                     # 显示标签
    hide_conf=False                        # 显示置信度
)
# 可视化完整图像上的预测结果
# batch_result.visualize_full_predictions(
#     original_image=image_np,
#     export_dir=result_dir,
#     file_name="full_prediction_result",
#     text_size=1.0,
#     rect_th=2
# )



coco_predictions = batch_result.to_coco_predictions(image_id = 1)
coco_dir = "./coco_outputs/"
# 确保输出目录存在
os.makedirs(coco_dir, exist_ok=True)

with open(os.path.join(coco_dir, "detection_results.json"), "w") as f:
    json.dump(coco_predictions, f, indent=4)
print(f"已将检测结果保存为COCO格式: {os.path.join(coco_dir, 'detection_results.json')}")

visualization_time = time.time() - visualization_start_time
print(f"可视化完成，耗时: {visualization_time:.2f}秒")

# 计算检测到的目标数量
detection_count = len(batch_result.object_prediction_lists)
print(f"\n共检测到 {detection_count} 个目标")

# 总结性能指标
total_time = time.time() - start_time
print("\n性能统计摘要:")
print(f"总处理时间: {total_time:.2f}秒")
print(f"  - 模型加载: {model_load_time:.2f}秒 ({model_load_time/total_time*100:.1f}%)")
print(f"  - 图像加载: {image_load_time:.2f}秒 ({image_load_time/total_time*100:.1f}%)")
print(f"  - 批量推理: {inference_time:.2f}秒 ({inference_time/total_time*100:.1f}%)")
print(f"  - 结果可视化: {visualization_time:.2f}秒 ({visualization_time/total_time*100:.1f}%)")

# 显存使用总结
final_gpu_used, _ = get_gpu_memory_usage()
if initial_gpu_used is not None and final_gpu_used is not None:
    print(f"\nGPU显存使用摘要:")
    print(f"  - 初始显存: {initial_gpu_used:.2f}GB")
    print(f"  - 最终显存: {final_gpu_used:.2f}GB")
    print(f"  - 净增显存: {final_gpu_used - initial_gpu_used:.2f}GB")