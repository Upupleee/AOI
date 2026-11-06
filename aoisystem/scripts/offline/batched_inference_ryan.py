#DIFFICULT_DEFECTS
#RYAN_CUT
#RYAN_TEST
import DefectDetection.sahi.sahi
import os
import glob
import time
import shutil
import torch
from pathlib import Path

from DefectDetection.sahi.sahi import AutoDetectionModel
from DefectDetection.sahi.sahi.predict import get_batched_prediction, get_batched_sliced_prediction, get_sliced_prediction
from DefectDetection.sahi.sahi.slicing import slice_image
from DefectDetection.sahi.sahi.utils.cv import read_image
from PIL import Image
import numpy as np

import json
import os
import subprocess

import cv2

#RYAN_CUT
def split_image_save_to_folder(image_path):
    # Input image path
    #image_path = "/home/aoi/pretest/AOI_V1/results/p503p8e018486a0_test_record/gbl_record_2/gbl_2.tiff"
    # Create tmp directory if it doesn't exist
    tmp_dir = "./tmp/split_images"
    os.makedirs(tmp_dir, exist_ok=True)
    # Open the image
    with Image.open(image_path) as img:
        width, height = img.size
        # Calculate split points (horizontal split)
        half = height // 2
        # Split into top and bottom halves
        top_half = img.crop((0, 0, width, half))
        bottom_half = img.crop((0, half, width, height))
        # Save the split images
        base_name = os.path.basename(image_path).split('.')[0]
        #top_path = os.path.join(tmp_dir, f"{base_name}_top.tiff")
        #bottom_path = os.path.join(tmp_dir, f"{base_name}_bottom.tiff")
        top_path = os.path.join(tmp_dir, f"top.tiff")
        bottom_path = os.path.join(tmp_dir, f"bottom.tiff")
        top_half.save(top_path)
        bottom_half.save(bottom_path)
        print(f"Images saved to:\n{top_path}\n{bottom_path}")


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
    #RYAN_TEST
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/yolo12s_deeppcb_original.pt",
    #model_path='/home/aoi/pretest/AOI_V1/DefectDetection/models/yolo12s_ryan_argumentation.pt',
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0729_deepblue.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/AOI-DATA-0729_5900epoch.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_07311000.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0731_1100.pt",
    # model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0731_1400.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0731_2050_correctdata.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0731_2139_correctdata.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0801_0231.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/yolo12_deeppcb_new.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0801_1428.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0801_1343.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0801_2400.pt",
    #RYAN_TEST
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0804_drew_aug4classes.pt",
    # model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0804_2036.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0807_1527.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0809_1107.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0810_1049_aggressive.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0810_1438_aggressive.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0810_1537_aggressive.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0809_1024by1024.pt",
    #RYAN_TEST
    # model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0812_0856.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0811_1935_aggresive.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0812_1939_1024by1024_aggressive.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0812_2239_1024by1024_aggressive.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0813_2136.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0813_2256.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0814_0911_1024by1024_aggressive.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0814_2121_1024by1024_aggressive.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0815_0903_1024by1024_aggressive.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0819_1010_640by640_aggressive.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0820_0922_640by640_aggressive.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0820_1439_640by640_12n_cleandata.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0821_1253_verify_annotationmethod.pt",

    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0820_1955_v8s_640by640_aggressive.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0820_2208_v12n_640by640_aggressive_annotationtest_deprecated__cleandata.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0818_1900_1024by1024_aggressive.pt",
    #####model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_08160300_1024by1024_aggressive.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0821_2022_v11s_verifiy_hsv.pt",
    model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0822_1753_640_v12n_maskoutring.pt",
    #RYAN_TEST
    #########################################
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0821_2022_v11s_verifiy_hsv.pt",
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/drew_models/yolo11s-p2_aug9.pt",
    #RYAN_TEST
    #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/drew_models/aug12_finetunning.pt",
    # model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/drew_models/yolo12s_aug8.pt",
    # model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/drew_models/yolo12s_aug9.pt",
    # model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/drew_models/yolov8s-p2_aug9.pt",
    #confidence_threshold=0.8,
    #confidence_threshold=0.031,
    #confidence_threshold=0.011,
    #confidence_threshold=0.0061,
    confidence_threshold=0.3261,
    device="0",
    #model_path="/home/aoi/Downloads/yolo11s-p2_aug10.pt"
)
__import__('ipdb').set_trace()
model_load_time = time.time() - model_load_start
print(f"模型加载时间: {model_load_time:.2f}秒")

# 图像路径和结果目录
#image_path = "results/gtl/gtl_2.tiff"
#RYAN_TEST
#image_path = "results/gbl/gbl_2.tiff"
#image_path = " /home/aoi/pretest/images/gbl_disaster/test_imgs2.tif"
image_path = "/home/aoi/pretest/AOI_V1/results/p503p8e018486a0_test_record/gbl_record_1/gbl_3.tiff"
image_path = "/home/aoi/pretest/AOI_V1/results/p503p8e018486a0_test_record/gbl_record_2/gbl_3.tiff"  #fail
image_path = "/home/aoi/pretest/AOI_V1/results/p503p8e018486a0_test_record/gtl_record_1/gtl_3.tiff"  #data is bad
image_path = "/home/aoi/pretest/AOI_V1/results/p503p8e018486a0_test_record/gbl_record_2/gbl_1.tiff"  #bottom fail
image_path = "/home/aoi/pretest/AOI_V1/results/p503p8e018486a0_test_record/gbl_record_2/gbl_2.tiff"  #fail 
image_path = "/home/aoi/pretest/AOI_V1/results/p503p8e018486a0_test_record/gtl_record_2/gtl_2.tiff"  
image_path = "/home/aoi/pretest/AOI_V1/results/p503p8e018486a0_test_record/gbl_record_1/gbl_4.tiff"  #fail  
image_path = "/home/aoi/pretest/AOI_V1/results/p503p8e018486a0_test_record/gbl_record_3/gbl_3.tiff"  # fail
#image_path = "/home/aoi/pretest/AOI_V1/results/gtl/gtl_3.tiff"  # fail
image_path = "/home/aoi/pretest/AOI_V1/results/gtl/gtl_4.tiff"  # fail
image_path = "/home/aoi/pretest/AOI_V1/results/test_0809_night/17021_19_a_ng_upperright.tif"  # fail
image_path = "/home/aoi/pretest/AOI_V1/results/gtl/gtl_3.tiff"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/p533p4w018562a0_12/gbl_500/test_imgs5_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/p533p4w018562a0_12/gtl_500/test_imgs5_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/p533p4w018562a0_11/gbl_500/test_imgs5_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/p533p4w018562a0_12/gbl_500/test_imgs2_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/p533p4w018562a0_12/gbl_500/test_imgs4_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/0_819_p2580p4e013568a1/gtl/test_imgs1_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/819_p1939p4e010480a0/gbl/test_imgs3_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/full_prediction_0820test.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/819_p1939p4e010480a0/gbl/test_imgs5_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/820_p2018p6e01863a0/test_imgs1_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/0_819_p2580p4e013568a1/gbl/test_imgs2_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/0_819_p2580p4e013568a1/gbl/test_imgs2_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/2025_8_20/p961p10e01876a0/500/test_imgs3_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/0_819_p2580p4e013568a1/gbl/test_imgs3_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/2025_8_21/test_imgs3_concat.tif"  # fail   #DIFFICULT_DEFECTS
#image_path = "/media/aoi/BU100/strips_old/0_819_p2580p4e013568a1/gbl/test_imgs3_concat.tif"  # fail   #DIFFICULT_DEFECTS
#image_path = "/media/aoi/BU100/strips_old/0_819_p2580p4e013568a1/test_imgs2_concat.tif"  # fail   #DIFFICULT_DEFECTS
#image_path = "/media/aoi/BU100/strips_old/0_819_p2580p4e013568a1/test_imgs3_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/0822/test_imgs2_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/0822/test_imgs4_concat.tif"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/0822/test_imgs2_concat.tif_0"  # fail   #DIFFICULT_DEFECTS
image_path = "/home/aoi/pretest/AOI_V1/results/strips/0822/test_imgs4_concat.tif_"  # fail   #DIFFICULT_DEFECTS
# image_path = "/home/aoi/pretest/AOI_V1/results/p503p8e018486a0_test_record/gbl_record_3/gbl_3.tiff"  # fail   #DIFFICULT_DEFECTS
# image_path = "/home/aoi/pretest/AOI_V1/results/gtl/gtl_4.tiff"  # fail   #DIFFICULT_DEFECTS
# image_path = "/home/aoi/pretest/AOI_V1/results/gbl/gbl_1.tiff"  # fail   #DIFFICULT_DEFECTS
# image_path = "/home/aoi/pretest/AOI_V1/results/gbl/gbl_5_makeup.tiff"  # fail   #DIFFICULT_DEFECTS
#image_path = "/home/aoi/pretest/AOI_V1/results/gbl/gbl_2.tiff"  # fail   #DIFFICULT_DEFECTS
#image_path = "/home/aoi/Desktop/2025_8_13/2025_8_13_multiple_defects_smallerscale.jpg"  # fail   #DIFFICULT_DEFECTS
#image_path = "/home/aoi/Desktop/2025_8_13/2025_8_13_multiple_defects.jpg"  # fail   #DIFFICULT_DEFECTS
# image_path = "/home/aoi/pretest/AOI_V1/results/gtl/gtl_2.tiff"  # fail   #DIFFICULT_DEFECTS
#image_path = "/home/aoi/pretest/AOI_V1/results/strips/final/test_imgs1_concat_ng.tif"  # fail   #DIFFICULT_DEFECTS
#image_path = "/home/aoi/pretest/AOI_V1/results/gtl/gtl_1.tiff"  # fail   #DIFFICULT_DEFECTS
#image_path = "/home/aoi/pretest/AOI_V1/results/gtl/gtl_2.tiff"  # fail   #DIFFICULT_DEFECTS
#image_path = "/home/aoi/pretest/AOI_V1/results/test_0809_night/15_ng.tif"  # fail
#image_path = "/home/aoi/pretest/AOI_V1/results/full_prediction_for_detection_unittest.jpg"  # fail
result_dir = "./test_speed/results"

###################################################
#RYAN_CUT
split_image_save_to_folder(image_path)
image_path =  "./tmp/split_images/top.tiff"
image_path =  "./tmp/split_images/bottom.tiff"
###################################################

# 确保结果目录存在
os.makedirs(result_dir, exist_ok=True)

# 打开图像
image_load_start = time.time()
image = Image.open(image_path)
##匹配结果就是这个IMAGENP
image_np = np.array(image)
image_size = image.size
image_load_time = time.time() - image_load_start
print(f"原始图像尺寸: {image_size}")
print(f"图像加载时间: {image_load_time:.2f}秒")

# 设置切片参数
slice_height = 1500
slice_width = 1500


slice_height = 1280
slice_width = 1280
slice_height = 1180
slice_width = 1180
#slice_height = 700
#slice_width = 700
slice_height = 600
slice_width = 600
slice_height = 550
slice_width = 550
#slice_height = 400
#slice_width = 400
#slice_height = 1000
#slice_width = 1000
#slice_height = 900
#slice_width = 900
#FOR PANEL 450*390
######################################

overlap_height_ratio = 0.05
overlap_width_ratio = 0.05

overlap_height_ratio = 0.15
overlap_width_ratio = 0.15

# 记录推理前的GPU显存
pre_inference_gpu_used, _ = get_gpu_memory_usage()
if pre_inference_gpu_used is not None:
    print(f"\n推理前GPU显存: {pre_inference_gpu_used:.2f}GB / {gpu_total:.2f}GB")

print("\n开始优化后的批量切片推理流程...")
inference_start_time = time.time()

__import__('ipdb').set_trace()
batch_result = get_batched_sliced_prediction(
    image=image_np,
    detection_model=detection_model,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
    # slices_outdir = "/home/dual3080/Workspace/test/results/slices",
    slices_outdir = None,
    # slices_filename="output_slices",
    slices_filename=None,
    slices_ext=".jpg",
    visualize_slices=True,                 # 启用切片可视化
    # slice_visualization_dir="/home/dual3080/Workspace/test/results/visualized", # 切片可视化输出目录
    slice_visualization_dir=None, # 切片可视化输出目录
    visual_bbox_thickness=2,               # 边界框厚度
    visual_text_size=0.6,                  # 文本大小
    visual_hide_labels=False,              # 显示标签
    visual_hide_conf=False,                # 显示置信度
    verbose=1                              # 显示详细信息
)

#    直接返回带框的numpy数组
full_vis = batch_result.visualize_full_predictions(
    original_image=image_np,   # numpy array, BGR
    export_dir='results',
    file_name="full_prediction",
    rect_th=2,                 # 边框宽度
    text_size=0.6,             # 标签文字大小
    hide_labels=False,         # 不隐藏类别
    hide_conf=False            # 不隐藏置信度
)

# # 3. 保存到文件
# os.makedirs("results", exist_ok=True)
# cv2.imwrite("results/full_prediction.jpg", full_vis)

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
# batch_result.visualize_full_predictions(
#     original_image=image,                  # 使用原始图像
#     export_dir="/home/dual3080/Workspace/test/results/visualized",               # 导出目录
#     file_name="full_prediction",           # 输出文件名
#     text_size=1.0,                         # 文本大小
#     rect_th=2,                             # 边界框厚度
#     hide_labels=False,                     # 显示标签
#     hide_conf=False                        # 显示置信度
# )
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
