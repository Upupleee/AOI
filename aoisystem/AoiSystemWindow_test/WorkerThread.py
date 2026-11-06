import os
import re
import logging
import time
import tifffile
import numpy as np
import torch
import PIL.Image
from PIL import Image
import subprocess
import sys
from copy import deepcopy
from queue import Queue
from PyQt5.QtCore import pyqtSignal, QThread, QObject, QThreadPool, QRunnable, pyqtSlot, QDateTime
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from TemplateMatching.EfficientLoFTR.coarse_matching_new_test import coarse_matching_first, coarse_matching_eloftr_pillow_sample_only, coarse_matching_all
from TemplateMatching.EfficientLoFTR.fine_matching_loftr_new_drew_test import fine_matching_first, fine_matching_pillow_on_template, fine_matching_all, fine_matching_pillow_on_template_drew, fine_matching_pillow_on_template_drew_return_bgr
from .ModelManger import ModelManager
from DefectDetection.sahi.sahi.auto_model import AutoDetectionModel
from DefectDetection.sahi.sahi.predict import predict, get_sliced_prediction,get_batched_sliced_prediction
from DefectDetection.sahi.sahi.utils.cv import  (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    crop_object_predictions,
    cv2,
    get_video_reader,
    read_image_as_pil,
    visualize_object_predictions,
)
from DefectDetection.sahi.sahi.utils.file import Path, increment_path
from DefectDetection.batchinference.vrs_generation import convert_coco_to_vrs,convert_coco_to_vrs_with_varing_area
from DefectDetection.batchinference.vrs_generation import convert_coco_to_vrs_for_merged_result
import numpy as np
import json
import ultralytics
import datetime 
import io

PIL.Image.MAX_IMAGE_PIXELS = None 
MATERIAL_ROOT = "material"
####多图
# from TemplateMatching.EfficientLoFTR.coarse_matching_new_double import coarse_matching_super_double,coarse_matching_each
# from TemplateMatching.EfficientLoFTR.fine_matching_loftr_double import fine_matching_each
# 配置日志记录器
# logging.basicConfig(level=logging.ERROR,
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     filename='./Logger/aoi_system.log',
#                     filemode='a')

class MatchAndDetectThread(QThread):
    match_finished = pyqtSignal(str)  # 用于发出匹配完成信号
    detect_finished = pyqtSignal(str)  # 用于发出检测完成信号
    match_error_occurred = pyqtSignal(str)


    def __init__(self, image_path, template_path, Auto_split, default_rows, default_columns, flag, slice_win, conf_thres, overlap_samp, overlap_temp):
        super().__init__()
        self.image_path = image_path
        self.template_path = template_path
        self.Auto_split = Auto_split
        self.default_rows = default_rows
        self.default_columns = default_columns
        self.is_running = False
        self.flag = flag
        self.slice_window = slice_win
        self.conf_threshold = conf_thres
        self.overlap_samp = overlap_samp
        self.overlap_temp = overlap_temp

    def extract_number_from_path(self, path):
        pattern = r'_?(\d+)\.\w+$'
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
        return None
    
    def get_tif_filenames_from_list(self, image_path_list, number):
        if 1 <= number <= len(image_path_list):
        # 获取第 number 个路径（转换为0-based索引）
            path = image_path_list[number - 1].strip()
                
            if path and path.lower().endswith(('.tif', '.tiff')):
                # 提取文件名部分
                filename = path.split("/")[-1]
                
                # 去掉后缀名（兼容 .tif 和 .tiff）
                if filename.lower().endswith(".tiff"):
                    filename_without_ext = filename[:-5]  # 移除 .tiff
                else:
                    filename_without_ext = filename[:-4]  # 移除 .tif
        
        return filename_without_ext
    
    def process_images(self, template_files, sample_image_path):
        Image.MAX_IMAGE_PIXELS = None

        if len(template_files) < 2:
            raise ValueError("模板文件数量不足两张，请检查模板文件夹。")

        template_second_last = template_files[-2]
        template_last = template_files[-1]

        img_second_last = Image.open(template_second_last).convert('L')
        img_last = Image.open(template_last).convert('L')

        width_second_last = img_second_last.width
        width_last = img_last.width

        ratio = width_last / width_second_last

        sample_img = Image.open(sample_image_path)
        sample_width, sample_height = sample_img.width, sample_img.height

        crop_width = int(sample_width * ratio)

        if crop_width > sample_width:
            print("警告：计算得到的保留宽度超过了样本图片总宽度，直接保留整个图片。")
            crop_width = sample_width

        left = sample_width - crop_width
        upper = 0
        right = sample_width
        lower = sample_height
        self.default_columns = 1

        cropped_img = sample_img.crop((left, upper, right, lower))

        return cropped_img, img_last
    
    def linear_adjust_template(self, template_img, captured_img):

        w,h = captured_img.size
        captured_img_crop = captured_img.crop((0, 1671, w, h))
        template_array = np.array(template_img).astype(np.float32)
        captured_array = np.array(captured_img_crop).astype(np.float32)

        mu_template = template_array.mean()
        sigma_template = template_array.std()
        mu_captured = captured_array.mean()
        sigma_captured = captured_array.std()

        adjusted = (template_array - mu_template) / sigma_template * sigma_captured + mu_captured

        adjusted = np.clip(adjusted, 0, 255)

        adjusted = adjusted.astype(np.uint8)
        captured_array = captured_array.astype(np.uint8)

        return adjusted, captured_array

    def template_matching(self, sample_img_path, template_img_list, Auto_split, default_rows,default_columns):
        number=self.extract_number_from_path(sample_img_path)
        if len(template_img_list) ==number:
            sample_img, template_img=self.process_images(template_files=template_img_list, sample_image_path=sample_img_path)
            
            print("最后一个采样路径:",{sample_img_path})
            print("列表路径:",{template_img_list[number-1]})
        else:
            sample_img = Image.open(sample_img_path).convert('L')
            template_img = Image.open(template_img_list[number-1]).convert('L')
            print("采样路径:",{sample_img_path})
            print("列表路径:",{template_img_list[number-1]})

        file_name = self.get_tif_filenames_from_list(template_img_list, number)
        print(f"文件名字：{file_name}")
        template_arry, sample_arry=self.linear_adjust_template(template_img, sample_img)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        object_region,corresponding_sample_region = coarse_matching_eloftr_pillow_sample_only(
            sample_arry, 
            template_arry,
            enlarge=True,
            margin_percent=0,
            scale_factor=1.0
        )
        if object_region is None or corresponding_sample_region is None:

            print("Coarse matching failed.")
            return 
        print("Coarse Matching Completed.")

        # Step 2: Fine Matching
        print("Starting Fine Matching Stage...")
        # Provide the paths to the cropped images from coarse matching
        
        # TODO: adding template automatic shifting procedure
        
        result_overlay_image = fine_matching_pillow_on_template(  # Cropped sample image
            corresponding_sample_region,
            object_region,
            overlap_sample=self.overlap_samp,
            overlap_template=self.overlap_temp,# Cropped template image
            Auto_split= Auto_split,
            default_rows=default_rows,
            default_columns=default_columns
            
        )

        return result_overlay_image, file_name

    def extract_prefix(self, filename, delimiter='_'):
        
        last_delimiter_index = filename.rfind(delimiter)
        
        if last_delimiter_index != -1:
            # 如果找到分隔符，返回分割符之前的所有内容
            return filename[:last_delimiter_index]
        else:
            # 如果没有找到分隔符，返回原文件名
            return filename



    def defect_detection(self, matching_image, file_name, model_flag, slice_window, conf_threshold):
        #__import__('ipdb').set_trace()
        if model_flag == 0:#初始模型
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/weights1031/best-1031.engine"
        elif model_flag ==1:
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1207/best-30-95um-1207.engine"
        elif model_flag ==2:
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_lowpass/best-lowpass-1206.engine"
        elif model_flag ==3:
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_1/best-008f-1203.engine"
        elif model_flag ==4:
            model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_2/best-008f-bg1.engine"
        elif model_flag ==5 :
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30um-ryan_1/best-30um-1.engine"
        elif model_flag ==6:
            #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1207_pm/best-30-95um-1207-2.engine"
            model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/models/drew_models/aug12_finetunning.pt"
        elif model_flag ==7:
            #model_path ="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_3/last_008f_1205.engine"
            model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/models/drew_models/aug12_finetunning_nofreeze.pt"
        elif model_flag == 8:
            model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1206_lowpass/best-1206-30um.engine"
        # elif model_flag == 9:
        #     model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1206/best-30-95um-1206-17_15.engine"
        #yolov12 model added on 250330
        elif model_flag == 9:
            # model_path = "/home/aoi/Downloads/best-0331.pt" 
            #model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/best_demo_0524.pt"
            #model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/0524_demo_0.pt"
            #model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/best_2025_5_26.pt"
            #UPDATE_LASTEST_WEIGHTS
            model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0804_2036.pt",
            #model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0807_1527.pt",

        elif model_flag == 10:
            model_path = "/home/aoi/Downloads/best-v12n-0408.pt"
        elif model_flag == 11:
            model_path = "/home/aoi/Downloads/best-v12n-0409.pt"

        elif model_flag == 12:
            model_path = "/home/aoi/Downloads/best_0410.pt"

        print(f"置信度：{conf_threshold}")
        detector = AutoDetectionModel.from_pretrained(model_type="ultralytics",
                                                        model_path=model_path, 
                                                        confidence_threshold=conf_threshold,    # ← drop <0.3 
                                                        device="cuda:0")
        
        detector.load_model()
        
        # prediction_result = get_sliced_prediction(
        #         image = matching_image,
        #         detection_model = detector,
        #         slice_height = slice_window,
        #         slice_width = slice_window,
        #         overlap_height_ratio = 0.2,
        #         overlap_width_ratio = 0.2,
        #         verbose = 2
        #     )
        # object_prediction_list = prediction_result.object_prediction_list

        #batch_inference
        # __import__('ipdb').set_trace()
        batch_prediction = get_batched_sliced_prediction(
        image=matching_image,
        detection_model=detector,
        slice_height=slice_window,
        slice_width=slice_window,
        overlap_height_ratio=0.05,
        overlap_width_ratio=0.05,    
        verbose=0
            )
        
        object_batch_prediction_list=batch_prediction.object_prediction_lists


        # ===================================
        # ===================================
        #SEND_TO_VRS_DEBUG_HERE
        # 5. 保存 COCO JSON
        #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #coco_dir = os.path.join(output_root, "coco")
        #os.makedirs(coco_dir, exist_ok=True)
        #coco_json_path = os.path.join(coco_dir, f"detection_{timestamp}.json")
        #coco_predictions = prediction.to_coco_predictions(image_id=1)
        #with open(coco_json_path, "w") as f:
        #    json.dump(coco_predictions, f, indent=4)
        #print(f"[batch_detect] COCO 结果已保存: {coco_json_path}")

        ## 6. 转 VRS 并保存
        #vrs_dir = os.path.join(output_root, "vrs")
        #os.makedirs(vrs_dir, exist_ok=True)
        #vrs_path = os.path.join(vrs_dir, f"detection_{timestamp}.vrs")
        #convert_coco_to_vrs(
        #    coco_predictions,
        #    vrs_path,
        #    image_height=img_np.shape[0]
        #)
        #print(f"[batch_detect] VRS 文件已保存: {vrs_path}")

        #return {
        #    "coco_json": coco_json_path,
        #    "vrs_file": vrs_path
        #}
        # ===================================
        # ===================================
        project_name= self.extract_prefix(file_name)
        print(f"文件名：{project_name}")
        project=f"./prediction_visual/{project_name}"
        print(f"路径：{project}")
        save_dir = Path(project)
        # name = "exp"
        # save_dir = Path(increment_path(Path(project) / name, exist_ok=False))
        save_dir.mkdir(parents=True, exist_ok=True)

        # result = visualize_object_predictions(
        #         matching_image,
        #         object_prediction_list=object_batch_prediction_list,
        #         rect_th=1,
        #         text_size=0.5,
        #         text_th=1,
        #         hide_labels=False, #True
        #         hide_conf=False, #True
        #         output_dir= save_dir
        #     )
        
        # batch_prediction.visualize_full_predictions(
        #         matching_image,
        #         export_dir=str(save_dir),
        #         file_name=file_name,
        #         text_size=0.5,     # 文字大小
        #         rect_th=1,         # 边框粗细
        #         hide_labels=False, # 显示标签
        #         hide_conf=False    # 显示置信度
        #     )
        return save_dir, project_name
    

    def run(self):
        self.is_running = True
        try:

            matching_result, file_name = self.template_matching(self.image_path, self.template_path, self.Auto_split, self.default_rows, self.default_columns)
            

            #RYAN
            # self.match_finished.emit(matching_result_path)
            if matching_result is not None:
                # matching_result = Image.fromarray(matching_result)

                            ##modified 0523:saving matching result
                # matching_np = np.array(matching_result)                          # RGB or L
                # if matching_np.ndim == 2:                                        # 单通道灰度
                #     matching_cv = matching_np
                # else:                                                            # 3 通道 RGB → BGR
                #     matching_cv = cv2.cvtColor(matching_np, cv2.COLOR_RGB2BGR)

                # # 确保目录存在
                # os.makedirs('results', exist_ok=True)
                # base_name = Path(self.image_path).stem
                # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # final_output_path = os.path.join(
                #     'results',
                #     f"{base_name}_matched_{timestamp}.tiff"
                # )
                # save_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
                # result_test_final = cv2.imwrite(final_output_path, matching_cv, save_params)
                # print("Write success:", result_test_final)
                # print(f"Final overlay image saved to {final_output_path}")


            ##Jinshu
                detection_result, project_name = self.defect_detection(matching_result, file_name ,model_flag = self.flag, slice_window= self.slice_window, conf_threshold=self.conf_threshold)
                self.detect_finished.emit(str(project_name))
        except Exception as e:
            error_message = f"在匹配和检测线程中发生异常: {e}"
            self.match_error_occurred.emit(error_message)
            logging.error(error_message)
        finally:
            self.is_running = False

    def stop(self):
        self.is_running = False
        self.quit()
        self.wait()


class ImageCombinerThread(QThread):
    # 定义信号
    progress_signal = pyqtSignal(int)   # 进度信号（百分比）
    status_signal = pyqtSignal(str)     # 状态提示信号
    error_signal = pyqtSignal(str)      # 错误信号
    finished_signal = pyqtSignal(str)      # 完成信号

    def __init__(self, folder_path, output_filename):
        super().__init__()
        self.folder_path = folder_path
        self.output_filename = output_filename
        self.is_running = True

    def run(self):
        try:
            # self.status_signal.emit("开始处理...")
            print(f"文件夹：{self.folder_path}")
            
            # 1. 检查文件夹是否存在
            if not os.path.isdir(self.folder_path):
                raise FileNotFoundError(f"文件夹不存在: {self.folder_path}")
            
            # 2. 获取所有TIFF文件并排序
            files = [f for f in os.listdir(self.folder_path)
                     if f.lower().endswith(('.tif', '.tiff'))]
            if not files:
                raise ValueError("目录中没有TIFF文件")
            
            files.sort(reverse=True)
            total_files = len(files)
            print(f"找到 {total_files} 个文件，开始拼接...")

            # 3. 初始化拼接参数
            images = []
            total_width = 0
            max_height = 0
            
            # 4. 读取文件并计算总尺寸（带进度更新）
            for i, fn in enumerate(files):
                if not self.is_running:
                    self.status_signal.emit("操作已取消")
                    return
                
                file_path = os.path.join(self.folder_path, fn)
                try:
                    img = Image.open(file_path).convert('RGB')
                except Exception as e:
                    self.error_signal.emit(f"读取文件 {fn} 失败: {str(e)}")
                    return
                
                images.append(img)
                total_width += img.width
                max_height = max(max_height, img.height)
                
                # # 更新进度（每处理一个文件更新一次）
                progress = int((i + 1) / total_files * 100)
                print(progress)

            # 5. 创建拼接图像
            combined = Image.new('RGB', (total_width, max_height), (0, 0, 0))
            x = 0
            for img in images:
                if not self.is_running:
                    self.status_signal.emit("操作已取消")
                    return
                
                combined.paste(img, (x, 0))
                x += img.width

            # 6. 保存为BigTIFF
            output_path = os.path.join(self.folder_path, self.output_filename)
            arr = np.array(combined)
            tifffile.imwrite(output_path, arr, bigtiff=True)
            
            print(f"拼接完成，保存至: {output_path}")
            self.finished_signal.emit(output_path)

        except Exception as e:
            self.error_signal.emit(f"拼接失败: {str(e)}")

        finally:
            self.is_running = False

    def stop(self):
        """停止线程"""
        self.is_running = False
        self.wait()


#########################
class MatchThread(QThread):
    match_finished = pyqtSignal(str)  # 用于发出匹配完成信号
    match_error_occurred = pyqtSignal(str)


    def __init__(self, image_path, template_path, Auto_split, default_rows, default_columns,overlap_samp, overlap_temp):
        super().__init__()
        self.image_path = image_path
        self.template_path = template_path
        self.Auto_split = Auto_split
        self.default_rows = default_rows
        self.default_columns = default_columns
        self.is_running = False
        self.overlap_samp = overlap_samp
        self.overlap_temp = overlap_temp


    def extract_number_from_path(self, path):
        pattern = r'_?(\d+)\.\w+$'
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
        return None
    
    def get_tif_filenames_from_list(self, image_path_list, number):
        if 1 <= number <= len(image_path_list):
        # 获取第 number 个路径（转换为0-based索引）
            path = image_path_list[number - 1].strip()
                
            if path and path.lower().endswith(('.tif', '.tiff')):
                # 提取文件名部分
                filename = path.split("/")[-1]
                
                # 去掉后缀名（兼容 .tif 和 .tiff）
                if filename.lower().endswith(".tiff"):
                    filename_without_ext = filename[:-5]  # 移除 .tiff
                else:
                    filename_without_ext = filename[:-4]  # 移除 .tif
        
        return filename_without_ext
    
    def process_images(self, template_files, sample_image_path):
        Image.MAX_IMAGE_PIXELS = None

        if len(template_files) < 2:
            raise ValueError("模板文件数量不足两张，请检查模板文件夹。")

        template_second_last = template_files[-2]
        template_last = template_files[-1]

        img_second_last = Image.open(template_second_last).convert('L')
        img_last = Image.open(template_last).convert('L')

        width_second_last = img_second_last.width
        width_last = img_last.width

        ratio = width_last / width_second_last

        sample_img = Image.open(sample_image_path)
        sample_width, sample_height = sample_img.width, sample_img.height

        crop_width = int(sample_width * ratio)

        if crop_width > sample_width:
            print("警告：计算得到的保留宽度超过了样本图片总宽度，直接保留整个图片。")
            crop_width = sample_width

        left = sample_width - crop_width
        upper = 0
        right = sample_width
        lower = sample_height
        self.default_columns = 1

        cropped_img = sample_img.crop((left, upper, right, lower))

        return cropped_img, img_last
    
    def linear_adjust_template(self, template_img, captured_img):

        w,h = captured_img.size
        captured_img_crop = captured_img.crop((0, 1671, w, h))
        template_array = np.array(template_img).astype(np.float32)
        captured_array = np.array(captured_img_crop).astype(np.float32)

        mu_template = template_array.mean()
        sigma_template = template_array.std()
        mu_captured = captured_array.mean()
        sigma_captured = captured_array.std()

        adjusted = (template_array - mu_template) / sigma_template * sigma_captured + mu_captured

        adjusted = np.clip(adjusted, 0, 255)

        adjusted = adjusted.astype(np.uint8)
        captured_array = captured_array.astype(np.uint8)

        return adjusted, captured_array

    def template_matching(self, sample_img_path, template_img_list, Auto_split, default_rows,default_columns):
        number=self.extract_number_from_path(sample_img_path)
        if len(template_img_list) ==number:
            sample_img, template_img=self.process_images(template_files=template_img_list, sample_image_path=sample_img_path)
            
            print("最后一个采样路径:",{sample_img_path})
            print("列表路径:",{template_img_list[number-1]})
        else:
            sample_img = Image.open(sample_img_path).convert('L')
            template_img = Image.open(template_img_list[number-1]).convert('L')
            print("采样路径:",{sample_img_path})
            print("列表路径:",{template_img_list[number-1]})

        img_name = self.get_tif_filenames_from_list(template_img_list, number)
        print(f"文件名字：{img_name}")
        template_arry, sample_arry=self.linear_adjust_template(template_img, sample_img)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        object_region,corresponding_sample_region = coarse_matching_eloftr_pillow_sample_only(
            sample_arry, 
            template_arry,
            enlarge=True,
            margin_percent=0,
            scale_factor=1.0
        )
        if object_region is None or corresponding_sample_region is None:

            print("Coarse matching failed.")
            return 
        print("Coarse Matching Completed.")

        # Step 2: Fine Matching
        print("Starting Fine Matching Stage...")
        # Provide the paths to the cropped images from coarse matching
        
        # TODO: adding template automatic shifting procedure
        file_name = self.extract_prefix(img_name)
        result_overlay_image = fine_matching_pillow_on_template(  # Cropped sample image
            corresponding_sample_region,
            object_region,
            overlap_sample=self.overlap_samp,
            overlap_template=self.overlap_temp,# Cropped template image
            Auto_split= False,
            default_rows=default_rows,
            default_columns=default_columns
            
        )

        return result_overlay_image, file_name, img_name

    def extract_prefix(self, filename, delimiter='_'):
        
        last_delimiter_index = filename.rfind(delimiter)
        
        if last_delimiter_index != -1:
            # 如果找到分隔符，返回分割符之前的所有内容
            return filename[:last_delimiter_index]
        else:
            # 如果没有找到分隔符，返回原文件名
            return filename


    def run(self):
        self.is_running = True
        try:

            matching_result, file_name, img_name = self.template_matching(self.image_path, self.template_path, self.Auto_split, self.default_rows, self.default_columns)
            #RYAN
            if matching_result is not None:
                            ##modified 0523:saving matching result
                matching_np = np.array(matching_result)                          # RGB or L
                if matching_np.ndim == 2:                                        # 单通道灰度
                    matching_cv = matching_np
                else:                                                            # 3 通道 RGB → BGR
                    matching_cv = cv2.cvtColor(matching_np, cv2.COLOR_RGB2BGR)

                # 确保目录存在
                base_name = Path(file_name).stem
                results_dir= os.path.join('results',base_name)
                os.makedirs(results_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                final_output_path = os.path.join(
                    results_dir,
                    f"{img_name}.tiff"
                )
                save_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
                result_test_final = cv2.imwrite(final_output_path, matching_cv, save_params)
                print("Write success:", result_test_final)
                print(f"Final overlay image saved to {final_output_path}")
                self.match_finished.emit(file_name)

        except Exception as e:
            error_message = f"在匹配线程中发生异常: {e}"
            self.match_error_occurred.emit(error_message)
            logging.error(error_message)
        finally:
            self.is_running = False

    def stop(self):
        self.is_running = False
        self.quit()
        self.wait()



class DetectionThread(QThread):
    detect_finished = pyqtSignal(str)
    detect_error_occurred = pyqtSignal(str)

    def __init__(self, image_str, file_name, flag, slice_win, conf_thres, save_image):
        super().__init__()
        self.image_str = image_str
        self.file_name = file_name
        self.is_running = True
        self.flag = flag
        self.slice_window = slice_win
        self.conf_threshold = conf_thres

    def extract_prefix(self, filename, delimiter='_'):
        
        last_delimiter_index = filename.rfind(delimiter)
        
        if last_delimiter_index != -1:
            # 如果找到分隔符，返回分割符之前的所有内容
            return filename[:last_delimiter_index]
        else:
            # 如果没有找到分隔符，返回原文件名
            return filename

    def defect_detection(self, matching_image, file_name, model_flag, slice_window, conf_threshold):
        #__import__('ipdb').set_trace()
        if model_flag == 0:#初始模型
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/weights1031/best-1031.engine"
        elif model_flag ==1:
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1207/best-30-95um-1207.engine"
        elif model_flag ==2:
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_lowpass/best-lowpass-1206.engine"
        elif model_flag ==3:
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_1/best-008f-1203.engine"
        elif model_flag ==4:
            model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_2/best-008f-bg1.engine"
        elif model_flag ==5 :
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30um-ryan_1/best-30um-1.engine"
        elif model_flag ==6:
            #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1207_pm/best-30-95um-1207-2.engine"
            model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/models/drew_models/aug12_finetunning.pt"

            
        elif model_flag ==7:
            model_path ="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_3/last_008f_1205.engine"
        elif model_flag == 8:
            model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1206_lowpass/best-1206-30um.engine"
        # elif model_flag == 9:
        #     model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1206/best-30-95um-1206-17_15.engine"
        #yolov12 model added on 250330
        elif model_flag == 9:
            # model_path = "/home/aoi/Downloads/best-0331.pt" 
            #model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/best_demo_0524.pt"
            #model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/0524_demo_0.pt"
            model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/best_2025_5_26.pt"
        elif model_flag == 10:
            model_path = "/home/aoi/Downloads/best-v12n-0408.pt"
        elif model_flag == 11:
            model_path = "/home/aoi/Downloads/best-v12n-0409.pt"

        elif model_flag == 12:
            model_path = "/home/aoi/Downloads/best_0410.pt"

        print(f"置信度：{conf_threshold}")
        detector = AutoDetectionModel.from_pretrained(model_type="ultralytics",
                                                        model_path=model_path, 
                                                        confidence_threshold=conf_threshold,    # ← drop <0.3 
                                                        device="cuda:0")
        
        detector.load_model()

        #batch_inference
        # __import__('ipdb').set_trace()
        batch_prediction = get_batched_sliced_prediction(
        image=matching_image,
        detection_model=detector,
        slice_height=slice_window,
        slice_width=slice_window,
        overlap_height_ratio=0.05,
        overlap_width_ratio=0.05,    
        verbose=0
            )
        
        project_name= self.extract_prefix(file_name)
        print(f"文件名：{project_name}")
        project=f"./prediction_visual/{project_name}"
        print(f"路径：{project}")
        save_dir = Path(project)
        save_dir.mkdir(parents=True, exist_ok=True)
        img = Image.open(self.image_str)
        img_np = np.array(img)
        
        object_batch_prediction_list=batch_prediction.object_prediction_lists

        #SEND_TO_VRS_DEBUG_HERE
        # 5. 保存 COCO JSON
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        coco_dir = os.path.join(project, "coco")
        os.makedirs(coco_dir, exist_ok=True)
        coco_json_path = os.path.join(coco_dir, f"detection_{timestamp}.json")
        coco_predictions = batch_prediction.to_coco_predictions(image_id=1)
        with open(coco_json_path, "w") as f:
           json.dump(coco_predictions, f, indent=4)
        print(f"[batch_detect] COCO 结果已保存: {coco_json_path}")

        # 6. 转 VRS 并保存
        vrs_dir = os.path.join(project, "vrs")
        os.makedirs(vrs_dir, exist_ok=True)
        vrs_path = os.path.join(vrs_dir, f"detection_{timestamp}.vrs")
        convert_coco_to_vrs(
           coco_predictions,
           vrs_path,
           image_height=img_np.shape[0]
        )
        print(f"[batch_detect] VRS 文件已保存: {vrs_path}")

        return save_dir, project_name,coco_json_path
    
    def run(self):
        self.is_running = True
        try:
            #UPDATE_LASTEST_WEIGHTS
            detection_result, project_name,coco_json_path = self.defect_detection(self.image_str, self.file_name ,model_flag = self.flag, slice_window= self.slice_window, conf_threshold=self.conf_threshold)
            self.detect_finished.emit(str(project_name))
        except Exception as e:
            error_message = f"在检测线程中发生异常: {e}"
            self.detect_error_occurred.emit(error_message)
            logging.error(error_message)
        finally:
            self.is_running = False

    def stop(self):
        self.is_running = False
        self.quit()
        self.wait()


class CombinerAndDetectionThread(QThread):
    cdfinished = pyqtSignal(str)

    def __init__(self, folder_path, output_filename, file_name, flag, slice_win, conf_thres, save_image):
        super().__init__()
        self.folder_path = folder_path
        self.output_filename = output_filename
        self.is_running = True
        self.flag = flag
        self.file_name = file_name
        self.saveimage = save_image
        self.slice_window = slice_win
        self.conf_threshold = conf_thres
    def combiner(self):
        if not os.path.isdir(self.folder_path):
            raise FileNotFoundError(f"文件夹不存在: {self.folder_path}")
        
        # 获取并排序TIFF文件
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.tif', '.tiff'))]
        if not files:
            raise ValueError("目录中没有TIFF文件")
        files.sort(reverse=True)
        total_files = len(files)
        print(f"找到 {total_files} 个文件，开始拼接...")

        # 初始化拼接参数（不保存所有图像，而是逐个处理）
        combined = None  # 拼接结果
        total_width = 0
        max_height = 0

        for i, fn in enumerate(files):
            if not self.is_running:
                return None
            
            file_path = os.path.join(self.folder_path, fn)
            try:
                # 读取单张图像（使用with语句自动关闭）
                with Image.open(file_path).convert('RGB') as img:
                    img_width, img_height = img.size
                    total_width += img_width
                    max_height = max(max_height, img_height)

                    # 第一次处理：初始化拼接画布
                    if combined is None:
                        combined = Image.new('RGB', (total_width, max_height), (0, 0, 0))
                        x_offset = 0  # 当前拼接位置
                    else:
                        # 非第一次：重新创建更大的画布（避免原地修改的内存开销）
                        new_combined = Image.new('RGB', (total_width, max_height), (0, 0, 0))
                        # 复制已有内容
                        new_combined.paste(combined, (0, 0))
                        combined.close()  # 释放旧画布
                        combined = new_combined
                        x_offset = total_width - img_width  # 计算新图像的位置

                    # 拼接当前图像
                    combined.paste(img, (x_offset, 0))
                    progress = int((i + 1) / total_files * 100)
                    print(f"拼接进度: {progress}%")

            except Exception as e:
                logging.error(f"处理图像 {fn} 失败: {e}")
                continue

        # 保存拼接结果（若需要）
        if self.saveimage and combined is not None:
            output_path = os.path.join(self.folder_path, self.output_filename)
            # 转换为数组并保存（保存后立即释放数组）
            with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
                arr = np.array(combined)
                tif.write(arr)
                del arr  # 立即删除大数组

        return combined

    # def combiner(self):
    #     if not os.path.isdir(self.folder_path):
    #             raise FileNotFoundError(f"文件夹不存在: {self.folder_path}")
            
    #     # 2. 获取所有TIFF文件并排序
    #     files = [f for f in os.listdir(self.folder_path)
    #                 if f.lower().endswith(('.tif', '.tiff'))]
    #     if not files:
    #         raise ValueError("目录中没有TIFF文件")
        
    #     files.sort(reverse=True)
    #     total_files = len(files)
    #     print(f"找到 {total_files} 个文件，开始拼接...")

    #     # 3. 初始化拼接参数
    #     images = []
    #     total_width = 0
    #     max_height = 0
        
    #     # 4. 读取文件并计算总尺寸（带进度更新）
    #     for i, fn in enumerate(files):
    #         if not self.is_running:
    #             return
            
    #         file_path = os.path.join(self.folder_path, fn)
    #         try:
    #             img = Image.open(file_path).convert('RGB')
    #         except Exception as e:
    #             return
            
    #         images.append(img)
    #         total_width += img.width
    #         max_height = max(max_height, img.height)
            
    #         # # 更新进度（每处理一个文件更新一次）
    #         progress = int((i + 1) / total_files * 100)
    #         print(progress)

    #     # 5. 创建拼接图像
    #     combined = Image.new('RGB', (total_width, max_height), (0, 0, 0))
    #     x = 0
    #     for img in images:
    #         if not self.is_running:
    #             return
            
    #         combined.paste(img, (x, 0))
    #         x += img.width
    #         img.close()  # 释放单张图像内存
    #     del images 
        
    #     if self.saveimage:
    #         output_path = os.path.join(self.folder_path, self.output_filename)
    #         arr = np.array(combined)
    #         tifffile.imwrite(output_path, arr, bigtiff=True)

    #     return combined
    
    def extract_prefix(self, filename, delimiter='_'):
        
        last_delimiter_index = filename.rfind(delimiter)
        
        if last_delimiter_index != -1:
            # 如果找到分隔符，返回分割符之前的所有内容
            return filename[:last_delimiter_index]
        else:
            # 如果没有找到分隔符，返回原文件名
            return filename

    def detection(self, matching_image, file_name, model_flag, slice_window, conf_threshold):
        #__import__('ipdb').set_trace()
        if model_flag == 0:#初始模型
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/weights1031/best-1031.engine"
        elif model_flag ==1:
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1207/best-30-95um-1207.engine"
        elif model_flag ==2:
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_lowpass/best-lowpass-1206.engine"
        elif model_flag ==3:
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_1/best-008f-1203.engine"
        elif model_flag ==4:
            model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_2/best-008f-bg1.engine"
        elif model_flag ==5 :
            model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30um-ryan_1/best-30um-1.engine"
        elif model_flag ==6:
            #model_path="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1207_pm/best-30-95um-1207-2.engine"
            model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/models/drew_models/aug12_finetunning.pt"

        elif model_flag ==7:
            model_path ="/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_3/last_008f_1205.engine"
        elif model_flag == 8:
            model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1206_lowpass/best-1206-30um.engine"
        # elif model_flag == 9:
        #     model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1206/best-30-95um-1206-17_15.engine"
        #yolov12 model added on 250330
        elif model_flag == 9:
            # model_path = "/home/aoi/Downloads/best-0331.pt" 
            #model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/best_demo_0524.pt"
            #model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/0524_demo_0.pt"
            model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/best_2025_5_26.pt"
        elif model_flag == 10:
            model_path = "/home/aoi/Downloads/best-v12n-0408.pt"
        elif model_flag == 11:
            model_path = "/home/aoi/Downloads/best-v12n-0409.pt"

        elif model_flag == 12:
            model_path = "/home/aoi/Downloads/best_0410.pt"

        print(f"置信度：{conf_threshold}")
        detector = AutoDetectionModel.from_pretrained(model_type="ultralytics",
                                                        model_path=model_path, 
                                                        confidence_threshold=conf_threshold,    # ← drop <0.3 
                                                        device="cuda:0")
        
        detector.load_model()
        try:
            #batch_inference
            # __import__('ipdb').set_trace()
            batch_prediction = get_batched_sliced_prediction(
            image=matching_image,
            detection_model=detector,
            slice_height=slice_window,
            slice_width=slice_window,
            overlap_height_ratio=0.05,
            overlap_width_ratio=0.05,    
            verbose=0
                )
        
            object_batch_prediction_list=batch_prediction.object_prediction_lists

            project_name= self.extract_prefix(file_name)
            print(f"文件名：{project_name}")
            project=f"./prediction_visual/{project_name}"
            print(f"路径：{project}")
            save_dir = Path(project)
            save_dir.mkdir(parents=True, exist_ok=True)

            #SEND_TO_VRS_DEBUG_HERE
            # 5. 保存 COCO JSON
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            coco_dir = os.path.join(project, "coco")
            os.makedirs(coco_dir, exist_ok=True)
            coco_json_path = os.path.join(coco_dir, f"detection_{timestamp}.json")
            coco_predictions = batch_prediction.to_coco_predictions(image_id=1)
            with open(coco_json_path, "w") as f:
                json.dump(coco_predictions, f, indent=4)
            print(f"[batch_detect] COCO 结果已保存: {coco_json_path}")

            # 6. 转 VRS 并保存
            vrs_dir = os.path.join(project, "vrs")
            os.makedirs(vrs_dir, exist_ok=True)
            vrs_path = os.path.join(vrs_dir, f"detection_{timestamp}.vrs")
            convert_coco_to_vrs(
            coco_predictions,
            vrs_path,
            image_height=matching_image.size[1]
            )
            print(f"[batch_detect] VRS 文件已保存: {vrs_path}")

            return save_dir, project_name, coco_json_path
        
        finally:
            # 强制释放模型和图像资源
            del detector, batch_prediction, coco_predictions
            # 释放输入图像（检测完成后不再需要）
            matching_image.close()
            del matching_image
            # 清理GPU缓存（包括PyTorch和CUDA的残留）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # 清理进程间通信的残留
            # 手动触发垃圾回收
            import gc
            gc.collect()


    def run(self):
        self.is_running = True
        try:
            combinerImage = self.combiner()
            if combinerImage is not None:
                detection_result, project_name , coco_json_path = self.detection(combinerImage, self.file_name, self.flag, self.slice_window, self.conf_threshold)
                self.cdfinished.emit(str(coco_json_path))
        except Exception as e:
            error_message = f"在检测线程中发生异常: {e}"
            logging.error(error_message)
        finally:
            if combinerImage:
                combinerImage.close()
                del combinerImage
        # 清理GPU缓存（如果使用了GPU）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.is_running = False
            
    def stop(self):
        self.is_running = False
        self.quit()
        self.wait()

class ImageProcessingThread(QThread):
    """图片处理线程（优化中转文件夹）"""
    progress_updated = pyqtSignal(int, str)  # 进度值, 处理中的图片名称
    processing_completed = pyqtSignal(dict)
    processing_failed = pyqtSignal(str, str)  # 错误信息, 图片名称
    image_loading = pyqtSignal(str)  # 图片加载进度信号
    
    def __init__(self, images_and_sides, material_id, generate_thumbnails=True):
        super().__init__()
        self.images_and_sides = images_and_sides  # 格式: [(path, side), ...]
        # 修改：料号文件夹路径改为 material/料号
        self.material_id = os.path.join(MATERIAL_ROOT, material_id)
        self.generate_thumbnails = generate_thumbnails  # 是否生成缩略图
        # 切分参数
        self.segment_mm = 127
        self.step_mm = 113.3
        self.start_mm = 104.7
        self.offset_mm = 0
        
    def run(self):
        try:
            results = {"success": [], "failed": []}
            total = len(self.images_and_sides)
            
            # 创建缩略图目录（路径已包含material根目录）
            thumbnails_dir = os.path.join(self.material_id, "thumbnails")
            os.makedirs(thumbnails_dir, exist_ok=True)
            os.chmod(thumbnails_dir, 0o777)
            
            for i, (image_path, side) in enumerate(self.images_and_sides):
                try:
                    filename = os.path.basename(image_path)
                    self.image_loading.emit(f"开始加载图片: {filename}（可能需要几秒）")
                    
                    # 创建目标文件夹（直接保存切分结果，路径已包含material根目录）
                    target_folder = os.path.join(self.material_id, side)
                    os.makedirs(target_folder, exist_ok=True)
                    os.chmod(target_folder, 0o777)
                    
                    # 大图片加载优化
                    with Image.open(image_path) as img:
                        self.image_loading.emit(f"图片加载完成，开始处理: {filename}")
                        
                        # 生成缩略图
                        thumbnail_path = None
                        if self.generate_thumbnails:
                            thumbnail_path = self.generate_thumbnail(img, filename, thumbnails_dir)
                        
                        # 直接在目标文件夹切分并保存（无临时文件夹）
                        saved_paths = self.slice_image(
                            img=img,
                            img_name=os.path.splitext(filename)[0],
                            target_folder=target_folder,  # 传入目标文件夹
                            segment_mm=self.segment_mm,
                            step_mm=self.step_mm,
                            start_mm=self.start_mm,
                            offset_mm=self.offset_mm
                        )
                    
                    results["success"].append({
                        "original": image_path,
                        "processed": saved_paths,
                        "side": side,
                        "thumbnail": thumbnail_path
                    })
                except Exception as e:
                    results["failed"].append({
                        "path": image_path,
                        "error": str(e),
                        "side": side
                    })
                finally:
                    # 发送进度更新
                    progress = int((i + 1) / total * 100)
                    self.progress_updated.emit(progress, filename)
            
            self.processing_completed.emit(results)
        except Exception as e:
            self.processing_failed.emit(str(e), "批量处理")
    
    def slice_image(self, img, img_name, target_folder, segment_mm=127, step_mm=113.3, start_mm=104.7, offset_mm=0):
        """切分图片（保持原始逻辑，优化保存路径）"""
        # 禁用解压炸弹检查
        Image.MAX_IMAGE_PIXELS = None
        saved_file_paths = []

        try:
            if img.mode != 'L':
                img = img.convert('L')
            pixels_per_mm = 16384 / segment_mm  # 每mm对应的像素数
            W, H = img.size
            x_left = W
            i = -1

            while x_left != 0:
                right_mm = (start_mm - offset_mm) + i * step_mm
                left_mm = start_mm + (i + 1) * step_mm

                if right_mm < 0:
                    right_mm = 0
                x_right = int(W - right_mm * pixels_per_mm)
                x_left = int(W - left_mm * pixels_per_mm)
                if x_left < 0:
                    x_left = 0
                box = (x_left, 0, x_right, H)
                slice_img = img.crop(box)
                
                # 直接保存到目标文件夹（路径已包含material根目录）
                slice_filename = f"{img_name}_{i+1}.tif"  # 文件名保持不变
                slice_path = os.path.join(target_folder, slice_filename)  # 保存到目标文件夹
                slice_img.save(slice_path)
                print(f"Saved {slice_path} with box {box}")
                saved_file_paths.append(slice_path)

                i += 1
        except FileNotFoundError:
            print(f"错误: 文件 {img} 未找到。")
        except Exception as e:
            print(f"错误: 发生了一个未知错误: {e}")

        return saved_file_paths
    
    def generate_thumbnail(self, img, original_filename, thumbnails_dir):
        """生成图片缩略图"""
        try:
            # 获取原始尺寸
            width, height = img.size
            
            # 计算缩略图尺寸
            new_width = max(1, width // 8)
            new_height = max(1, height // 8)
            
            # 调整图片分辨率
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 构建缩略图路径（已包含material根目录）
            base_name, ext = os.path.splitext(original_filename)
            thumbnail_filename = f"{base_name}_thumb.jpg"
            thumbnail_path = os.path.join(thumbnails_dir, thumbnail_filename)
            
            # 保存缩略图
            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')
            resized_img.save(thumbnail_path, "JPEG", quality=80)
            
            return thumbnail_path
        except Exception as e:
            print(f"生成缩略图失败: {str(e)}")
            return None
        
# class ModelManager:
#     _instance = None
#     _model_cache = {}  # 缓存不同flag对应的模型

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance

#     def get_model(self, flag, conf_threshold):
#         """获取模型实例，相同flag仅加载一次"""
#         if flag not in self._model_cache:
#             # 模型路径映射（复用原逻辑）
#             model_paths = {
#                 0: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/weights1031/best-1031.engine",
#                 1: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1207/best-30-95um-1207.engine",
#                 2: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_lowpass/best-lowpass-1206.engine",
#                 3: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_1/best-008f-1203.engine",
#                 4: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_2/best-008f-bg1.engine",
#                 5: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30um-ryan_1/best-30um-1.engine",
#                 6: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1207_pm/best-30-95um-1207-2.engine",
#                 7: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_3/last_008f_1205.engine",
#                 8: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1206_lowpass/best-1206-30um.engine",
#                 9: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/best_2025_5_26.pt",
#                 10: "/home/aoi/Downloads/best-v12n-0408.pt",
#                 11: "/home/aoi/Downloads/best-v12n-0409.pt",
#                 12: "/home/aoi/Downloads/best_0410.pt"
#             }
#             model_path = model_paths.get(flag, model_paths[0])
#             # 加载模型（仅首次调用时执行）
#             detector = AutoDetectionModel.from_pretrained(
#                 model_type="ultralytics",
#                 model_path=model_path,
#                 confidence_threshold=conf_threshold,
#                 device="cuda:0"
#             )
#             detector.load_model()
#             self._model_cache[flag] = detector
#         return self._model_cache[flag]


class TaskSignals(QObject):
    match_finished = pyqtSignal(str)
    detect_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

# 全局JSON文件路径（基础名，实际会拼接资料名）
BASE_TASK_INFO_JSON = "_match_info.json"

# --- 模拟耗时任务的 QRunnable ---
class MatchAndDetectRunnable(QRunnable):
        # 类变量：收集方法1的任务信息（多线程共享）
    match_info = {"part_id": "", "samples": []}
    _file_read = False  # 方法2是否已读取过JSON文件
    
    def __init__(self, sample_path, template_list, detection_params, callback_done, ab_cycle_count, material_id):
        super().__init__()
        self.signals = TaskSignals()
        self.sample_path = sample_path
        self.template_list = template_list
        self.ab_cycle_count = ab_cycle_count
        self.params = detection_params
        # 模型管理器（单例）
        self.model_manager = ModelManager()
        self.callback_done = callback_done

        self.is_method1 = False  # 标记当前任务是否使用方法1
        self.material_id = material_id  # 资料ID

    def extract_number_from_path(self, path):
        pattern = r'_?(\d+)\.\w+$'
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
        return None
    
    def get_tif_filenames_from_list(self, image_path_list, number):
        if 1 <= number <= len(image_path_list):
        # 获取第 number 个路径（转换为0-based索引）
            path = image_path_list[number - 1].strip()
                
            if path and path.lower().endswith(('.tif', '.tiff')):
                # 提取文件名部分
                filename = path.split("/")[-1]
                
                # 去掉后缀名（兼容 .tif 和 .tiff）
                if filename.lower().endswith(".tiff"):
                    filename_without_ext = filename[:-5]  # 移除 .tiff
                else:
                    filename_without_ext = filename[:-4]  # 移除 .tif
        
        return filename_without_ext
    
    def process_images(self, template_files, sample_image_path):
        Image.MAX_IMAGE_PIXELS = None

        if len(template_files) < 2:
            raise ValueError("模板文件数量不足两张，请检查模板文件夹。")

        template_second_last = template_files[-2]
        template_last = template_files[-1]

        img_second_last = Image.open(template_second_last).convert('L')
        img_last = Image.open(template_last).convert('L')

        width_second_last = img_second_last.width
        width_last = img_last.width

        ratio = width_last / width_second_last

        sample_img = Image.open(sample_image_path)
        sample_width, sample_height = sample_img.width, sample_img.height

        crop_width = int(sample_width * ratio)

        if crop_width > sample_width:
            print("警告：计算得到的保留宽度超过了样本图片总宽度，直接保留整个图片。")
            crop_width = sample_width

        left = sample_width - crop_width
        upper = 0
        right = sample_width
        lower = sample_height
        self.default_columns = 1

        cropped_img = sample_img.crop((left, upper, right, lower))

        return cropped_img, img_last
    
    def linear_adjust_template(self, template_img, captured_img):

        w,h = captured_img.size
        captured_img_crop = captured_img.crop((0, 9070, w, h)) #wrong value,should be 9000!before is 1671


        start = time.time()
        template_array = np.array(template_img).astype(np.float32)
        captured_array = np.array(captured_img_crop).astype(np.float32)

        mu_template = template_array.mean()
        sigma_template = template_array.std()
        mu_captured = captured_array.mean()
        sigma_captured = captured_array.std()

        adjusted = (template_array - mu_template) / sigma_template * sigma_captured + mu_captured

        adjusted = np.clip(adjusted, 0, 255)

        adjusted = adjusted.astype(np.uint8)
        captured_array = captured_array.astype(np.uint8)

        end = time.time()
        print(f"耗时: {end - start:.4f} 秒")
        return adjusted, captured_array

        #Drew image argumentation
        # template_array = np.array(template_img).astype(np.uint8)
        # captured_array = np.array(captured_img_crop).astype(np.uint8)
        # return template_array, captured_array
        #
 
    def template_matching(self, match_info):

        fname = Path(self.sample_path).name
        entry = next((e for e in match_info['samples']
                      if e['file_name'] == fname), None)
        if entry is None:
            print(f"No match info for {fname}")
            return None
        bbox    = entry['bbox']
        H_list  = entry['H_matrices']
        number=self.extract_number_from_path(self.sample_path)
        if len(self.template_list) ==number:
            sample_img, template_img=self.process_images(template_files=self.template_list, sample_image_path=self.sample_path)
            
            print("最后一个采样路径:",{self.sample_path})
            print("列表路径:",{self.template_list[number-1]})
        else:
            sample_img = Image.open(self.sample_path).convert('L')
            template_img = Image.open(self.template_list[number-1]).convert('L')
            print("采样路径:",{self.sample_path})
            print("列表路径:",{self.template_list[number-1]})

        img_name = self.get_tif_filenames_from_list(self.template_list, number)
        print(f"文件名字：{img_name}")
        template_arry, sample_arry=self.linear_adjust_template(template_img, sample_img)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        object_region,corresponding_sample_region = coarse_matching_all(
            sample_arry, 
            template_arry,
            bbox
        )
        if object_region is None or corresponding_sample_region is None:

            print("Coarse matching failed.")
            return 
        print("Coarse Matching Completed.")

        # Step 2: Fine Matching
        print("Starting Fine Matching Stage...")
        # Provide the paths to the cropped images from coarse matching
        
        # TODO: adding template automatic shifting procedure
        file_name = self.extract_prefix(img_name)
        result_overlay_image = fine_matching_all(  # Cropped sample image
            corresponding_sample_region,
            object_region,
            H_list
        )

        return result_overlay_image, file_name, img_name

    def extract_prefix(self, filename, delimiter='_'):
        
        last_delimiter_index = filename.rfind(delimiter)
        
        if last_delimiter_index != -1:
            # 如果找到分隔符，返回分割符之前的所有内容
            return filename[:last_delimiter_index]
        else:
            # 如果没有找到分隔符，返回原文件名
            return filename

    def _read_match_info_json(self):
        """读取JSON文件（仅执行一次）"""
        json_file = f"{self.material_id}{self.params.get('mode')}{BASE_TASK_INFO_JSON}"
        json_dir = "./material"  # 替换为你的JSON文件目录
        
        # 构建JSON文件路径
        json_path = os.path.join(json_dir, self.material_id, self.params.get('mode'),json_file)
        
        # 检查文件是否存在
        if not os.path.exists(json_path):
            print(f"❌ Match info file not found: {json_path}")
            return
        
        # 读取JSON内容
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                MatchAndDetectRunnable.match_info = json.load(jf)  # 存储到类变量供全局使用
                MatchAndDetectRunnable._file_read = True  # 标记为已读取
                print(f"✅ Successfully read match info (once): {json_path}")
                # 可在此处添加JSON数据的验证逻辑
        except json.JSONDecodeError:
            print(f"❌ Failed to parse JSON: {json_path}")
        except Exception as e:
            print(f"❌ Error reading JSON: {str(e)}")

    #WITH_JPG_SPEEDUP
    def image_jpg_compression_to_speedup(self,matching_result, quality: int = 85) -> np.ndarray:
        """
        Compress a NumPy array to JPEG in memory and return as NumPy array.
        Parameters:
            arr: Input array (uint8, shape (H,W,3) for RGB or (H,W) for grayscale)
            quality: JPEG quality (1-100, higher=better)
        Returns:
            Decompressed array (may have slight quality loss)
        Example:
            >>> rgb_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            >>> compressed = compress_to_jpg(rgb_array)
        """
        # Validate input
        if not isinstance(matching_result, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        if matching_result.dtype != np.uint8:
            raise ValueError("Array must be uint8 dtype")
        # Convert array to PIL Image
        img = Image.fromarray(matching_result)
        # Create in-memory buffer
        with io.BytesIO() as buffer:
            # Save as JPEG to memory buffer
            img.save(buffer, format='JPEG', quality=quality)
            # Rewind buffer and load back as PIL Image
            buffer.seek(0)
            compressed_img = Image.open(buffer)
            # Convert back to NumPy array
            return np.array(compressed_img)


    def run(self):
        self.is_running = True
        try:
            # self.is_method1 = (self.params.get('mode') in ['A', 'B'] and self.ab_cycle_count<2)
            # ====================
            self.is_method1 = False
            #DETECT_SMALLER_DEFECTS

            #MATCH_WITH_DREW
            #self.is_method1 = True
            # ====================

            #__import__('ipdb').set_trace()

            # 初始化或更新资料ID
            if not MatchAndDetectRunnable.match_info["part_id"]:
                MatchAndDetectRunnable.match_info["part_id"] = self.material_id
            elif MatchAndDetectRunnable.match_info["part_id"] != self.material_id:
                # 资料切换时，清空历史任务
                MatchAndDetectRunnable.match_info["part_id"] = self.material_id
                MatchAndDetectRunnable.match_info["samples"] = []
            if self.is_method1:
                matching_result, file_name, img_name, bbox, H_list= self.template_matching_first(self.model_manager)
                sample_info = {
                    "file_name": Path(self.sample_path).name,
                    "bbox": {
                        "x_min": int(bbox['x_min']),
                        "y_min": int(bbox['y_min']),
                        "x_max": int(bbox['x_max']),
                        "y_max": int(bbox['y_max'])
                    },
                    "H_matrices": []
                }
                for h_item in H_list:
                    grid = h_item.get("grid", [])
                    H_mat = h_item.get("H", [])
                    sample_info["H_matrices"].append({
                        "grid": [int(grid[0]), int(grid[1])] if grid else grid,
                        "H": [[float(val) for val in row] for row in H_mat] if H_mat else H_mat
                    })
                MatchAndDetectRunnable.match_info["samples"].append(sample_info)
                # json_path = os.path.join('results', f"{Path(file_name).stem}_match_info.json")
                # with open(json_path, 'w', encoding='utf-8') as jf:
                #     json.dump(match_info, jf, indent=4, ensure_ascii=False)
                # print(f"Match info JSON saved: {json_path}")
            else:
                # if not MatchAndDetectRunnable._file_read:
                #     self._read_match_info_json()  # 读取文件
                #     MatchAndDetectRunnable._file_read = True  # 标记为已读取
                # json_path = os.path.join('results', f"{Path(file_name).stem}_match_info.json")
                # if not os.path.exists(json_path):
                #     print("Match info file not found:", json_path)
                #     return
                # with open(json_path, 'r', encoding='utf-8') as jf:
                #     match_info = json.load(jf)
                # matching_result, file_name, img_name = self.template_matching(match_info)
                # ===================================
                matching_result, file_name, img_name = self.template_matching_drew(self.model_manager)
                # __import__('ipdb').set_trace()
                
                #MATCH_WITH_DREW
                #matching_result, file_name, img_name = self.template_matching_drew_corrected(self.model_manager)
                # ===================================

            # ===================================
            #SAVE_IMAGE_FOR_DEBUG
            # path = '/home/aoi/pretest/AOI_V1/results/full_prediction_original_justmatched.jpg'
            # cv2.imwrite(path,matching_result)
            # ===================================

            # ================
            detector = self.model_manager.get_model(ModelManager.MODEL_TYPE_DETECTION, flag=self.params.get('flag'), conf_threshold=self.params.get('conf_thres'))
            #PRELOAD_FIXED_DETECTION_MODEL
            # detector = AutoDetectionModel.from_pretrained(
            #     model_type="ultralytics",
            #     #ALIGN_DETECTION_WITH_OFFLINE_RUNNER
            #     model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0807_1527.pt",
            #     confidence_threshold=0.011,
            #     device="0"
            # )
            # ================
            # ====================
            #WITH_JPG_SPEEDUP
            #matching_result = self.image_jpg_compression_to_speedup(matching_result,85) 
            # ====================

            if matching_result is not None:
                if self.params.get('save_image'):
                    matching_np = np.array(matching_result)
                    # __import__('ipdb').set_trace()
                    if matching_np.ndim == 2:
                        matching_cv = matching_np
                    else:
                        # ============================
                        #RGB_BGR_BUG
                        #matching_cv = cv2.cvtColor(matching_np, cv2.COLOR_RGB2BGR)
                        # matching_cv_bgr = cv2.cvtColor(matching_np, cv2.COLOR_RGB2BGR)
                        matching_cv = matching_np
                        # ============================

                    base_name = Path(file_name).stem
                    results_dir = os.path.join('results', base_name)

                    os.makedirs(results_dir, exist_ok=True)
                    output_path = os.path.join(results_dir, f"{img_name}.tiff")
                    cv2.imwrite(output_path, matching_cv, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
                    print(f"Overlay saved: {output_path}")
                #__import__('ipdb').set_trace()
                # 2. 缺陷检测（使用预加载的模型）

                # ================
                ##detector = self.model_manager.get_model(ModelManager.MODEL_TYPE_DETECTION, flag=self.params.get('flag'), conf_threshold=self.params.get('conf_thres'))
                ##PRELOAD_FIXED_DETECTION_MODEL
                #detector = AutoDetectionModel.from_pretrained(
                #    model_type="ultralytics",
                #    #ALIGN_DETECTION_WITH_OFFLINE_RUNNER
                #    model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0807_1527.pt",
                #    confidence_threshold=0.011,
                #    device="0"
                #)
                # ================


                # ===================================
                #SAVE_IMAGE_FOR_DEBUG COMMENT OUT
                # path = '/home/aoi/pretest/AOI_V1/results/full_prediction_for_detection.jpg'
                # cv2.imwrite(path,matching_result)
                # ===================================
                # ===================================
                #project, fn, coco_path = self._defect_detection(matching_result, file_name, detector)
                #DETECT_SMALLER_DEFECTS
                #project, fn, coco_path = self._defect_detection_smaller(matching_result, file_name, detector)
                #DETECT_SMALLER_DEFECTS_CUT_WITH_NP
                project, fn, coco_path = self._defect_detection_smaller_cutwithnp(matching_result, file_name, detector)
                # ===================================
                self.signals.detect_finished.emit(str(coco_path))
        except Exception as e:
            error_message = f"在匹配和检测线程中发生异常: {e}"
            self.signals.error_occurred.emit(error_message)
            logging.error(error_message)
        finally:
            self.is_running = False
        print(f"处理: {self.sample_path}\n参数: {self.params}\n当前模式: {self.params.get('mode', '未知')}")
        self.callback_done(self.sample_path)

    
    def template_matching_first(self, model_manger):
        number = self.extract_number_from_path(self.sample_path)
        if not number:
            raise ValueError(f"无法从路径提取编号: {self.sample_path}")

        if len(self.template_list) == number:
            sample_img, template_img = self.process_images(
                template_files=self.template_list, 
                sample_image_path=self.sample_path
            )
        else:
            sample_img = Image.open(self.sample_path).convert('L')
            template_img = Image.open(self.template_list[number-1]).convert('L')

        img_name = self.get_tif_filenames_from_list(self.template_list, number)
        template_arry, sample_arry = self.linear_adjust_template(template_img, sample_img)

        # 执行粗匹配和精匹配（复用原逻辑）
        object_region, corresponding_sample_region, cut_box_sample = coarse_matching_first(
            sample_arry, template_arry, enlarge=True, margin_percent=0, scale_factor=1.0
        )
        if  object_region is None or  corresponding_sample_region is None:
            raise ValueError("粗匹配失败")
        file_name = self.extract_prefix(img_name)
        result_overlay_image, H_matrices= fine_matching_first(
            corresponding_sample_region, object_region,
            overlap_sample=self.params.get('overlap_samp'),
            overlap_template=self.params.get('overlap_temp'),
            Auto_split=self.params.get('auto_split'),
            default_rows=self.params.get('rows'),
            default_columns=self.params.get('cols')
        )
        return result_overlay_image, file_name, img_name, cut_box_sample, H_matrices
    
    def _template_matching(self):
        number = self.extract_number_from_path(self.sample_path)
        if not number:
            raise ValueError(f"无法从路径提取编号: {self.sample_path}")

        if len(self.template_list) == number:
            sample_img, template_img = self.process_images(
                template_files=self.template_list, 
                sample_image_path=self.sample_path
            )
        else:
            sample_img = Image.open(self.sample_path).convert('L')
            template_img = Image.open(self.template_list[number-1]).convert('L')

        img_name = self.get_tif_filenames_from_list(self.template_list, number)
        template_arry, sample_arry = self.linear_adjust_template(template_img, sample_img)

        # 执行粗匹配和精匹配（复用原逻辑）
        object_region, corresponding_sample_region = coarse_matching_eloftr_pillow_sample_only(
            sample_arry, template_arry, enlarge=True, margin_percent=0, scale_factor=1.0
        )
        if  object_region is None or  corresponding_sample_region is None:
            raise ValueError("粗匹配失败")
        file_name = self.extract_prefix(img_name)
        result_overlay_image = fine_matching_pillow_on_template(
            corresponding_sample_region, object_region,
            overlap_sample=self.params.get('overlap_samp'),
            overlap_template=self.params.get('overlap_temp'),
            Auto_split=self.params.get('auto_split'),
            default_rows=self.params.get('rows'),
            default_columns=self.params.get('cols')
        )
        return result_overlay_image, file_name, img_name


    #MATCH_WITH_DREW
    def template_matching_drew_corrected(self, model_manger):
        #__import__('ipdb').set_trace()
        number = self.extract_number_from_path(self.sample_path)
        if not number:
            raise ValueError(f"无法从路径提取编号: {self.sample_path}")

        if len(self.template_list) == number:
            sample_img, template_img = self.process_images(
                template_files=self.template_list, 
                sample_image_path=self.sample_path
            )
        else:
            sample_img = Image.open(self.sample_path).convert('L')
            template_img = Image.open(self.template_list[number-1]).convert('L')

        img_name = self.get_tif_filenames_from_list(self.template_list, number)
        template_arry, sample_arry = self.linear_adjust_template(template_img, sample_img)

        # 执行粗匹配和精匹配（复用原逻辑）
        object_region, corresponding_sample_region = coarse_matching_eloftr_pillow_sample_only(
            sample_arry, template_arry, enlarge=True, margin_percent=0, scale_factor=1.0, model=model_manger
        )
        if  object_region is None or  corresponding_sample_region is None:
            raise ValueError("粗匹配失败")
        file_name = self.extract_prefix(img_name)

        #result_overlay_image = fine_matching_pillow_on_template_drew(
        #SAVE_IMAGE_FOR_DEBUG  
        result_overlay_image = fine_matching_pillow_on_template_drew_return_bgr(
            corresponding_sample_region, object_region,
            grid_size=self.params.get('grid_size'),
            max_sampling_per_grid=self.params.get('max_sampling_per_grid'),
            enhance_algorithms=self.params.get('enhance_algorithms'),
            enhance_param=self.params.get('enhance_param'),
            extractor_name=self.params.get('extractor_name'),
            model = model_manger

        )
        return result_overlay_image, file_name, img_name

    
    def template_matching_drew(self, model_manger):
        number = self.extract_number_from_path(self.sample_path)
        if not number:
            raise ValueError(f"无法从路径提取编号: {self.sample_path}")

        if len(self.template_list) == number:
            sample_img, template_img = self.process_images(
                template_files=self.template_list, 
                sample_image_path=self.sample_path
            )
        else:
            sample_img = Image.open(self.sample_path).convert('L')
            template_img = Image.open(self.template_list[number-1]).convert('L')

        img_name = self.get_tif_filenames_from_list(self.template_list, number)
        template_arry, sample_arry = self.linear_adjust_template(template_img, sample_img)

        print("SAMPLE IMAGE PATH AND TEMPLATE IMAGE PATH IS :" +  self.sample_path)
        print("TEMPLATE IMAGE PATH IS: " + self.template_list[number-1])
        # ====================
        #SAVE_IMAGE_FOR_DEBUG
        #__import__('ipdb').set_trace()
        # path = '/home/aoi/pretest/AOI_V1/results/full_prediction_original_justmatched0.jpg'
        # cv2.imwrite(path,template_arry)
        # path = '/home/aoi/pretest/AOI_V1/results/full_prediction_original_justmatched1.jpg'
        # cv2.imwrite(path,sample_arry)
        # ====================
#0826粗匹
        # # 执行粗匹配和精匹配（复用原逻辑）
        # object_region, corresponding_sample_region = coarse_matching_eloftr_pillow_sample_only(
        #     sample_arry, template_arry, enlarge=True, margin_percent=0, scale_factor=1.0, model=model_manger
        # )
        # if  object_region is None or  corresponding_sample_region is None:
        #     raise ValueError("粗匹配失败")

        object_region=template_arry
        corresponding_sample_region=sample_arry
        file_name = self.extract_prefix(img_name)

        #SAVE_IMAGE_FOR_DEBUG  
        #result_overlay_image = fine_matching_pillow_on_template_drew(
        result_overlay_image = fine_matching_pillow_on_template_drew_return_bgr(
            corresponding_sample_region, object_region,
            grid_size=self.params.get('grid_size'),
            max_sampling_per_grid=self.params.get('max_sampling_per_grid'),
            enhance_algorithms=self.params.get('enhance_algorithms'),
            enhance_param=self.params.get('enhance_param'),
            extractor_name=self.params.get('extractor_name'),
            model = model_manger

        )
        return result_overlay_image, file_name, img_name
    


    #DETECT_SMALLER_DEFECTS
    def split_npimage_save_to_folder(self,npimage,image_path):
        # Input image path
        #image_path = "/home/aoi/pretest/AOI_V1/results/p503p8e018486a0_test_record/gbl_record_2/gbl_2.tiff"
        # Create tmp directory if it doesn't exist
        tmp_dir = "./tmp/split_images"
        os.makedirs(tmp_dir, exist_ok=True)
        # Open the image

        #with Image.open(image_path) as img:
        #width, height = npimage.size
        height, width, _ = npimage.shape

        # Calculate split points (horizontal split)
        half = height // 2

        # Split into top and bottom halves
        #top_half = npimage.crop((0, 0, width, half))
        #bottom_half = npimage.crop((0, half, width, height))
        top_half = npimage[:half, :]      # Rows 0 to half, all columns
        bottom_half = npimage[half:, :]   # Rows half to end, all columns

       # # Save the split images
       # base_name = os.path.basename(image_path).split('.')[0]
       # #top_path = os.path.join(tmp_dir, f"{base_name}_top.tiff")
       # #bottom_path = os.path.join(tmp_dir, f"{base_name}_bottom.tiff")
        top_path = os.path.join(tmp_dir, f"top.tiff")
        bottom_path = os.path.join(tmp_dir, f"bottom.tiff")

        # Convert numpy arrays to PIL Images and save
        Image.fromarray(top_half).save(top_path)
        Image.fromarray(bottom_half).save(bottom_path)
        print(f"Images saved to:\n{top_path}\n{bottom_path}")

    #DETECT_SMALLER_DEFECTS
    def split_image_save_to_folder(self,image_path):
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
            # base_name = os.path.basename(image_path).split('.')[0]
            #top_path = os.path.join(tmp_dir, f"{base_name}_top.tiff")
            #bottom_path = os.path.join(tmp_dir, f"{base_name}_bottom.tiff")
            top_path = os.path.join(tmp_dir, f"top.tiff")
            bottom_path = os.path.join(tmp_dir, f"bottom.tiff")
            top_half.save(top_path)
            bottom_half.save(bottom_path)
            print(f"Images saved to:\n{top_path}\n{bottom_path}")
    


    #DETECT_SMALLER_DEFECTS
    def merge_top_bottom_json_result(self,batch_prediction_top,batch_prediction_bottom,height_offset): 
        # Convert both predictions to COCO format (lists of dicts)
        coco_preds1 = batch_prediction_top.to_coco_predictions(image_id=1)
        coco_preds2 = batch_prediction_bottom.to_coco_predictions(image_id=2)
        
        # Define the height offset (e.g., 100 pixels)
        height_offset = 100  # Adjust this value as needed
        
        # Apply offset to all bounding boxes in batch_prediction1
        for pred in coco_preds1:
            # COCO format: [x_min, y_min, width, height]
            pred["bbox"][1] += height_offset  # Shift y-coordinate (height) by offset
        
        # Merge predictions
        merged_predictions = coco_preds1 + coco_preds2
        coco_json ="coco_json.json"
        
        # Save to JSON
        with open(coco_json, 'w') as f:
            json.dump(merged_predictions, f, indent=4)



    #DETECT_SMALLER_MERGE_TWO_RESULTS
    def predition_result_add_offset_of_height(self, predictions,height_offset):
        """
        Adds an offset to the height of all bounding boxes in COCO-format predictions.
        
        Args:
            predictions: List of dicts in COCO prediction format
            height_offset: Value to add to each bbox's height (can be positive or negative)
        
        Returns:
            List of predictions with updated heights
            return is a dictionary 
        """
        for pred in predictions:
            # bbox format: [x, y, w, h]
            if "bbox" in pred and len(pred["bbox"]) == 4:
                pred["bbox"][3] += height_offset  # Update height (4th element)
                
                # Update area if it exists (optional but recommended)
                if "area" in pred:
                    pred["area"] = pred["bbox"][2] * pred["bbox"][3]
        return predictions 


    #DETECT_SMALLER_MERGE_TWO_RESULTS
    def merge_topdictionary_and_bottomdictionary(self, top_dict,bottom_dict):
        """
        Simply combines two lists of COCO-format predictions without modifications.
        
        Args:
            preds1: First list of prediction dictionaries
            preds2: Second list of prediction dictionaries
            
        Returns:
            New list containing all predictions from both inputs
        """

        #return top_dict + bottom_dict
        #DETECT_SMALLER_MERGE_TWO_RESULTS
        return (top_dict[0] + bottom_dict,)


    #DETECT_SMALLER_DEFECTS_CUT_WITH_NP
    def split_image_horizontally(self,np_image):
        height = np_image.shape[0]
        half = height // 2  # Integer division for even split
        top = np_image[:half, :, :]
        bottom = np_image[half:, :, :]
        return top, bottom

    #SAVE_SMALLER_MERGED_RESULT
    # def mergeTopAndBottomAnnotatedImagesSaveToResult(self,img1,img2):

    #     path1 = '/home/aoi/pretest/AOI_V1/results/full_prediction1.jpg'
    #     cv2.imwrite(path1,img1)
    #     path2 = '/home/aoi/pretest/AOI_V1/results/full_prediction2.jpg'
    #     cv2.imwrite(path2,img2)

    #     path = '/home/aoi/pretest/AOI_V1/results/full_prediction_annotation.jpg'
    #     result = cv2.vconcat([img1, img2]) 
    #     cv2.imwrite(path,result)
    #     return result 

    #SAVE_SMALLER_MERGED_RESULT
    # def mergeTopAndBottomAnnotatedImagesSaveToResult(self,):

    #     path1 = '/home/aoi/pretest/AOI_V1/results/full_prediction_vis.tif'
    #     img1 = cv2.imread(path1)
    #     path2 = '/home/aoi/pretest/AOI_V1/results/full_prediction_vis1.tif'
    #     img2 = cv2.imread(path2)
    #     path = '/home/aoi/pretest/AOI_V1/results/full_prediction_annotation.jpg'
    #     result = cv2.vconcat([img1, img2]) 
    #     cv2.imwrite(path,result)
    #     return result 
    
    def mergeTopAndBottomAnnotatedImagesSaveToResult(self):
        # read the two halves
        img1 = cv2.imread('results/full_prediction_vis.tif')
        img2 = cv2.imread('results/full_prediction_vis1.tif')
        result = cv2.vconcat([img1, img2])

        # make output directory
        out_dir = os.path.join('results', 'strips')
        os.makedirs(out_dir, exist_ok=True)

        # use the sample's own name as the prefix
        sample_stem = Path(self.sample_path).stem  # e.g. "stripA", "stripB", etc.
        out_path = os.path.join(out_dir, f'{sample_stem}_concat.tif')

        cv2.imwrite(out_path, result)
        return out_path
      


    #DETECT_SMALLER_DEFECTS_CUT_WITH_NP
    def _defect_detection_smaller_cutwithnp(self, matching_image, file_name, detector):

        #image_path =  "./tmp/split_images/top.tiff"
        #self.split_npimage_save_to_folder(matching_image,image_path)
        #image_load_start = time.time()
        #image = Image.open(image_path)
        ###匹配结果就是这个IMAGENP
        #image_np = np.array(image)
        #image_size = image.size


        height,width,_ = matching_image.shape
        #DETECT_SMALLER_DEFECTS_CUT_WITH_NP
        topimg_np,bottomimg_np = self.split_image_horizontally(matching_image)
        
        # ===================================
        #SAVE_SMALLER_MERGED_RESULT
        # path1 = '/home/aoi/pretest/AOI_V1/results/full_prediction_original_1.jpg'
        # cv2.imwrite(path1,topimg_np)
        # path2 = '/home/aoi/pretest/AOI_V1/results/full_prediction_original_2.jpg'
        # cv2.imwrite(path2,bottomimg_np)
        # ===================================

        # ===================================
        #RGB_BGR_BUG
        topimg_np_rgb = topimg_np[..., [2, 1, 0]]
        bottomimg_np_rgb = bottomimg_np[..., [2, 1, 0]]
        # ===================================

        # ===================================
        #SAVE_SMALLER_MERGED_RESULT
        # path1 = '/home/aoi/pretest/AOI_V1/results/full_prediction_original_3.jpg'
        # cv2.imwrite(path1,topimg_np_rgb)
        # path2 = '/home/aoi/pretest/AOI_V1/results/full_prediction_original_4.jpg'
        # cv2.imwrite(path2,bottomimg_np_rgb)
        # ===================================

        # ===================================
        #ALIGN_DETECTION_WITH_OFFLINE_RUNNER
        ## 批量切片推理（复用原逻辑，但使用预加载的detector）
        #batch_prediction = get_batched_sliced_prediction(                               
        #    #DETECT_SMALLER_DEFECTS_CUT_WITH_NP                                         
        #    #image=topimg_np,                                                           
        #    #RGB_BGR_BUG
        #    image=topimg_np_rgb,                                                        
        #    detection_model=detector,                                                   
        #    slice_height=self.params.get('slice_win'),                                  
        #    slice_width=self.params.get('slice_win'),                                   
        #    overlap_height_ratio=0.05,                                                  
        #    overlap_width_ratio=0.05,                                                   
        #    verbose=0                                                                   
        #)                                                                               
        #__import__('ipdb').set_trace()                                                 
        # ===================================
                                                                                        
                                                                                        
        # ===================================
        #ALIGN_DETECTION_WITH_OFFLINE_RUNNER
        # ================
        #detection_model = AutoDetectionModel.from_pretrained(
        #    model_type="ultralytics",
        #    #ALIGN_DETECTION_WITH_OFFLINE_RUNNER
        #    model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0807_1527.pt",
        #    confidence_threshold=0.011,
        #    device="0"
        #)
        #PRELOAD_FIXED_DETECTION_MODEL
        detection_model = detector
        # ================

        batch_prediction = get_batched_sliced_prediction(                                                                                                                                                 
            image=topimg_np_rgb,
            detection_model=detection_model,
            #ALIGN_DETECTION_WITH_OFFLINE_RUNNER
            slice_height=1200,
            slice_width=1200,
            overlap_height_ratio=0.05,
            overlap_width_ratio=0.05,
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
            verbose=0                              # 显示详细信息                                                                                       
        )                                                                                                                                                                                             
        # ===================================
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
        # ===================
        #DETECT_SMALLER_DEFECTS                                                         
        #    直接返回带框的numpy数组                                                    
        #RGB_BGR_BUG                                                                    
        full_vis = batch_prediction.visualize_full_predictions(                         
            #DETECT_SMALLER_DEFECTS_CUT_WITH_NP
            #original_image=topimg_np,   # numpy array, BGR
            #RGB_BGR_BUG
            original_image=topimg_np_rgb,   # numpy array, BGR
            export_dir='results',
            #RGB_BGR_BUG
            file_name="full_prediction_vis",
            rect_th=2,                 # 边框宽度
            text_size=0.6,             # 标签文字大小
            hide_labels=False,         # 不隐藏类别
            hide_conf=False            # 不隐藏置信度
        )
        # ====================

        #SAVE_SMALLER_MERGED_RESULT
        # __import__('ipdb').set_trace()
        # __import__('ipdb').set_trace()


        # ===================
        #ALIGN_DETECTION_WITH_OFFLINE_RUNNER
        ###DETECT_SMALLER_DEFECTS
        ##image_path =  "./tmp/split_images/bottom.tiff"
        ##image = Image.open(image_path)
        ##image_np = np.array(image)
        ##image_size = image.size
        ## 批量切片推理（复用原逻辑，但使用预加载的detector）
        #batch_prediction_second = get_batched_sliced_prediction(
        #    #DETECT_SMALLER_DEFECTS_CUT_WITH_NP
        #    #image=bottomimg_np,
        #    #RGB_BGR_BUG
        #    image= bottomimg_np_rgb,
        #    detection_model=detector,
        #    slice_height=self.params.get('slice_win'),
        #    slice_width=self.params.get('slice_win'),
        #    overlap_height_ratio=0.05,
        #    overlap_width_ratio=0.05,
        #    verbose=0
        #)
        # ===================

        # ===================
        #ALIGN_DETECTION_WITH_OFFLINE_RUNNER
        #detection_model = AutoDetectionModel.from_pretrained(
        #    model_type="ultralytics",
        #    model_path="/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0807_1527.pt",
        #    confidence_threshold=0.011,
        #    device="0"
        #)
        batch_prediction_second = get_batched_sliced_prediction(                                                                                                                                                 
            image=bottomimg_np_rgb,
            detection_model=detection_model,
            #ALIGN_DETECTION_WITH_OFFLINE_RUNNER
            slice_height=1200,
            slice_width=1200,
            overlap_height_ratio=0.05,
            overlap_width_ratio=0.05,
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
            verbose=0                              # 显示详细信息                                                                                       
        )                                                                                                                                                                                             
        # ===================


        # ===================
        #DETECT_SMALLER_DEFECTS
        #    直接返回带框的numpy数组
        full_vis = batch_prediction_second.visualize_full_predictions(
            #DETECT_SMALLER_DEFECTS_CUT_WITH_NP
            #original_image=bottomimg_np,   # numpy array, BGR
            #RGB_BGR_BUG
            original_image=bottomimg_np_rgb,   # numpy array, BGR
            export_dir='results',
            #RGB_BGR_BUG
            file_name="full_prediction_vis1",
            rect_th=2,                 # 边框宽度
            text_size=0.6,             # 标签文字大小
            hide_labels=False,         # 不隐藏类别
            hide_conf=False            # 不隐藏置信度
        )
        # ===================

        #SAVE_SMALLER_MERGED_RESULT
        #SAVE_IMAGE_FOR_DEBUG
        # __import__('ipdb').set_trace()
        #self.mergeTopAndBottomAnnotatedImagesSaveToResult(batch_prediction.image,batch_prediction_second.image)
        #RGB_BGR_BUG
        # ===================
        #WITHOUT_SAVE_SPEEDUP
        self.mergeTopAndBottomAnnotatedImagesSaveToResult()
        # ===================

        # 保存结果
        project = f"./prediction_visual/{self.extract_prefix(file_name)}"
        os.makedirs(project, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        coco_dir = os.path.join(project, 'coco')
        os.makedirs(coco_dir, exist_ok=True)
        coco_json = os.path.join(coco_dir, f"detection_{ts}.json")

        # ====================
        #DETECT_SMALLER_MERGE_TWO_RESULTS
        with open(coco_json, 'w') as f:
            json.dump(batch_prediction.to_coco_predictions(image_id=1), f, indent=4)
        # ====================

        # ====================
        #DETECT_SMALLER_MERGE_TWO_RESULTS
        top_dict =  batch_prediction.to_coco_predictions(image_id=1),
        offset = height 
        bottom_dict =  self.predition_result_add_offset_of_height(batch_prediction_second.to_coco_predictions(image_id=1),offset)
        merged_dictionary = self.merge_topdictionary_and_bottomdictionary(top_dict,bottom_dict)
        # ====================


        vrs_dir = os.path.join(project, 'vrs')
        os.makedirs(vrs_dir, exist_ok=True)
        vrs_path = os.path.join(vrs_dir, f"detection_{ts}.vrs")
        #DETECT_SMALLER_MERGE_TWO_RESULTS

        # __import__('ipdb').set_trace()

        # #DETECT_SMALLER_CONVERT_TO_VRS_NEW
        # convert_coco_to_vrs_for_merged_result(
        #     # ==============
        #     #batch_prediction.to_coco_predictions(image_id=1),
        #     #DETECT_SMALLER_MERGE_TWO_RESULTS
        #     merged_dictionary,
        #     # ==============
        #     vrs_path, image_height=matching_image.shape[0]
        # )

        convert_coco_to_vrs_with_varing_area(merged_dictionary, vrs_path, self.params.get('temp_height'), self.params.get('targets'))
        #DETECT_SMALLER_CONVERT_TO_VRS_NEW
        ##convert_coco_to_vrs(
        ##    ##############
        ##    #batch_prediction.to_coco_predictions(image_id=1),
        ##    #DETECT_SMALLER_MERGE_TWO_RESULTS
        ##    merged_dictionary,
        ##    ##############
        ##    vrs_path, image_height=matching_image.shape[0]
        ##)

        return  project, self.extract_prefix(file_name), coco_json
        

    #DETECT_SMALLER_DEFECTS
    def _defect_detection_smaller(self, matching_image, file_name, detector):
        #DETECT_SMALLER_DEFECTS
        # __import__('ipdb').set_trace()
        # ========================
        #DETECT_SMALLER_DEFECTS
        image_path =  "./tmp/split_images/top.tiff"
        # ============
        #self.split_image_save_to_folder(image_path)
        self.split_npimage_save_to_folder(matching_image,image_path)
        # ============

        #image_path =  "./tmp/split_images/bottom.tiff"
        # 打开图像
        image_load_start = time.time()
        image = Image.open(image_path)
        ##匹配结果就是这个IMAGENP
        image_np = np.array(image)
        image_size = image.size

        height,width,_ = matching_image.shape

        # ========================
        # 批量切片推理（复用原逻辑，但使用预加载的detector）
        batch_prediction = get_batched_sliced_prediction(
            #DETECT_SMALLER_DEFECTS
            #image=matching_image,
            image=image_np,
            detection_model=detector,
            slice_height=self.params.get('slice_win'),
            slice_width=self.params.get('slice_win'),
            overlap_height_ratio=0.05,
            overlap_width_ratio=0.05,
            verbose=0
        )
        #__import__('ipdb').set_trace()
        # ========================

        # ========================
        #DETECT_SMALLER_DEFECTS
        #    直接返回带框的numpy数组
        full_vis = batch_prediction.visualize_full_predictions(
            original_image=image_np,   # numpy array, BGR
            export_dir='results',
            file_name="full_prediction_vis",
            rect_th=2,                 # 边框宽度
            text_size=0.6,             # 标签文字大小
            hide_labels=False,         # 不隐藏类别
            hide_conf=False            # 不隐藏置信度
        )
        # ========================



        # ========================
        # __import__('ipdb').set_trace()
        # __import__('ipdb').set_trace()
        # __import__('ipdb').set_trace()
        #DETECT_SMALLER_DEFECTS
        image_path =  "./tmp/split_images/bottom.tiff"
        image = Image.open(image_path)
        image_np = np.array(image)
        image_size = image.size
        # 批量切片推理（复用原逻辑，但使用预加载的detector）
        batch_prediction_second = get_batched_sliced_prediction(
            #DETECT_SMALLER_DEFECTS
            #image=matching_image,
            image=image_np,
            detection_model=detector,
            slice_height=self.params.get('slice_win'),
            slice_width=self.params.get('slice_win'),
            overlap_height_ratio=0.05,
            overlap_width_ratio=0.05,
            verbose=0
        )
        # ========================

        

        # 保存结果
        project = f"./prediction_visual/{self.extract_prefix(file_name)}"
        os.makedirs(project, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        coco_dir = os.path.join(project, 'coco')
        os.makedirs(coco_dir, exist_ok=True)
        coco_json = os.path.join(coco_dir, f"detection_{ts}.json")

        # ========================
        #DETECT_SMALLER_MERGE_TWO_RESULTS
        # __import__('ipdb').set_trace()
        with open(coco_json, 'w') as f:
            json.dump(batch_prediction.to_coco_predictions(image_id=1), f, indent=4)
        # ========================

        # ========================
        #DETECT_SMALLER_MERGE_TWO_RESULTS
        top_dict =  batch_prediction.to_coco_predictions(image_id=1),
        offset = height 
        bottom_dict =  self.predition_result_add_offset_of_height(batch_prediction_second.to_coco_predictions(image_id=1),offset)
        merged_dictionary = self.merge_topdictionary_and_bottomdictionary(top_dict,bottom_dict)
        # ========================


        vrs_dir = os.path.join(project, 'vrs')
        os.makedirs(vrs_dir, exist_ok=True)
        vrs_path = os.path.join(vrs_dir, f"detection_{ts}.vrs")
        #DETECT_SMALLER_MERGE_TWO_RESULTS

        #__import__('ipdb').set_trace()

        #DETECT_SMALLER_CONVERT_TO_VRS_NEW
        convert_coco_to_vrs_for_merged_result(
            ##############
            #batch_prediction.to_coco_predictions(image_id=1),
            #DETECT_SMALLER_MERGE_TWO_RESULTS
            merged_dictionary,
            ##############
            vrs_path, image_height=matching_image.shape[0]
        )

        #DETECT_SMALLER_CONVERT_TO_VRS_NEW
        ##convert_coco_to_vrs(
        ##    ##############
        ##    #batch_prediction.to_coco_predictions(image_id=1),
        ##    #DETECT_SMALLER_MERGE_TWO_RESULTS
        ##    merged_dictionary,
        ##    ##############
        ##    vrs_path, image_height=matching_image.shape[0]
        ##)

        return  project, self.extract_prefix(file_name), coco_json

    def _defect_detection(self, matching_image, file_name, detector):
        #DETECT_SMALLER_DEFECTS
        #__import__('ipdb').set_trace()
        # 批量切片推理（复用原逻辑，但使用预加载的detector）
        batch_prediction = get_batched_sliced_prediction(
            image=matching_image,
            detection_model=detector,
            slice_height=self.params.get('slice_win'),
            slice_width=self.params.get('slice_win'),
            overlap_height_ratio=0.05,
            overlap_width_ratio=0.05,
            verbose=0
        )

        # 保存结果
        project = f"./prediction_visual/{self.extract_prefix(file_name)}"
        os.makedirs(project, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        coco_dir = os.path.join(project, 'coco')
        os.makedirs(coco_dir, exist_ok=True)
        coco_json = os.path.join(coco_dir, f"detection_{ts}.json")
        with open(coco_json, 'w') as f:
            json.dump(batch_prediction.to_coco_predictions(image_id=1), f, indent=4)
        vrs_dir = os.path.join(project, 'vrs')
        os.makedirs(vrs_dir, exist_ok=True)
        vrs_path = os.path.join(vrs_dir, f"detection_{ts}.vrs")
        convert_coco_to_vrs(
            batch_prediction.to_coco_predictions(image_id=1),
            vrs_path, image_height=matching_image.shape[0]
        )

        return  project, self.extract_prefix(file_name), coco_json

# --- 控制器，管理任务、队列、线程池 ---
class DetectionController(QObject):
    detection_round_finished = pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.thread_pool = QThreadPool()
        self.file_queue = Queue()
        self.expected_image_count = 0
        self.finished_count = 0
        self.processing = False
        self.slice_paths = []
        self.detection_params = {}

    def reset(self, expected_count):
        self.file_queue = Queue()
        self.expected_image_count = expected_count
        self.finished_count = 0
        self.processing = False
        # self.slice_paths.clear()
        self.detection_params.clear()

    def enqueue_file(self, file_path):
        self.file_queue.put(file_path)
        self.try_process_next()

    def extract_number_from_path(self, path):
        pattern = r'_?(\d+)\.\w+$'
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
        return None

    def try_process_next(self):
        if self.processing:
            return
        if self.file_queue.empty():
            return
        self.processing = True
        sample_path = self.file_queue.get()
        is_last = self.extract_number_from_path(sample_path) == self.expected_image_count
        
        
        params = self.last_image_params if is_last else self.detection_params
        print(f"当前AB面数量：{self.parent.ab_cycle_count}")
        runnable = MatchAndDetectRunnable(sample_path, self.slice_paths, params, self.matching_finished, self.parent.ab_cycle_count, self.parent.current_material)
        self.thread_pool.start(runnable)
        # self.current_round_is_method1 = (params.get('mode') in ['A','B'] and self.parent.ab_cycle_count<2)


    def matching_finished(self, file_name):
        self.finished_count += 1
        print(f"\n✅ 检测完成: {file_name}  ({self.finished_count}/{self.expected_image_count})")
        self.processing = False

        if self.finished_count == self.expected_image_count:
            print("🎉 当前轮全部检测完成！")
            # if self.current_round_is_method1:
            #     self.write_method1_match_info(json_path)
            self.detection_round_finished.emit()
        else:
            self.try_process_next()

    def write_method1_match_info(self, json_file):
        if not MatchAndDetectRunnable.match_info["samples"]:
            print("[方法1] 警告：未收集到任务信息，不写入JSON")
            return
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(MatchAndDetectRunnable.match_info, f, ensure_ascii=False, indent=4)
                print(f"\n===== 方法1任务信息已写入 {json_file}（共{len(MatchAndDetectRunnable.match_info['samples'])}条，资料: {self.parent.current_material}） =====")
            # 清空类变量（避免下一轮方法1任务重复写入）
            MatchAndDetectRunnable.match_info["samples"].clear()
        except Exception as e:
            print(f"[方法1] 写入JSON失败: {e}")
            
    def prepare_detection(self, mode):
        if mode == "A":
            image_path = self.parent.get_current_image_path("A")
            if not image_path or not os.path.exists(image_path):
                self.parent.show_message("警告", "请先设置有效的A面图片!")
                return
                
            # 获取A面所有切分后图片路径并输出
            self.slice_paths = self.parent.get_slice_image_paths("A")
            print("\n===== 开始A面检测 =====")
            print(f"A面切分后图片路径列表 ({len(self.slice_paths)} 个):")
            for path in self.slice_paths:
                print(path)
                
            current_time = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            self.current_scan_id = f"{self.parent.current_material}_{current_time}"

        elif mode == "B":
            image_path = self.parent.get_current_image_path("B")
            if not image_path or not os.path.exists(image_path):
                self.parent.show_message("警告", "请先设置有效的B面图片!")
                return
                
            # 获取B面所有切分后图片路径并输出
            self.slice_paths = self.parent.get_slice_image_paths("B")
            print("\n===== 开始B面检测 =====")
            print(f"B面切分后图片路径列表 ({len(self.slice_paths)} 个):")
            for path in self.slice_paths:
                print(path)
                
            current_time = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            self.current_scan_id = f"{self.parent.current_material}_{current_time}"

            
        elif mode == "AB":
            a_image_path = self.parent.get_current_image_path("A")
            b_image_path = self.parent.get_current_image_path("B")
            
            if not a_image_path or not os.path.exists(a_image_path):
                self.parent.show_message("警告", "请先设置有效的A面资料!")
                return
                
            if not b_image_path or not os.path.exists(b_image_path):
                self.parent.show_message("警告", "请先设置有效的B面资料!")
                return
                
            current_side = "A" if self.parent.ab_cycle_count % 2 == 0 else "B"
            image_path = a_image_path if current_side == "A" else b_image_path
            
            # 获取当前面所有切分后图片路径并输出
            self.slice_paths = self.parent.get_slice_image_paths(current_side)
                
            current_time = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            self.current_scan_id = f"{self.parent.current_material}_{self.parent.current_ab_cycle}_{current_time}"

            # self.parent.ab_cycle_count += 1
            mode = current_side  # 用于传递当前轮真实模式

        ini = self.parent.inipath
        self.detection_params = {
            'rows': self.parent.read_ini_param(ini, 'row', int),
            'cols': self.parent.read_ini_param(ini, 'col', int),
            'auto_split': self.parent.read_ini_param(ini, 'Auto_split', bool),
            'overlap_temp': self.parent.read_ini_param(ini, 'overlap_temp', int),
            'overlap_samp': self.parent.read_ini_param(ini, 'overlap_samp', int),
            'save_image': self.parent.read_ini_param(ini, 'SaveImage', bool),
            'flag': self.parent.read_ini_param(ini, 'flag', int),
            'slice_win': self.parent.read_ini_param(ini, 'slice', int),
            'conf_thres': self.parent.read_ini_param(ini, 'thres', float),
            'mode': mode,
            'part_infer':self.parent.read_ini_param(ini, 'PartInfer', bool),
            'temp_slice_widths': self.parent.temp_slice_widths,
            'temp_height': self.parent.temp_height,
            'targets':self.parent.targets,
            'max_scale_size': 30000,
            'grid_size': 7,
            'max_sampling_per_grid': 100,
            'enhance_algorithms': 'clahe',
            # 'enhance_algorithms': 'none',
            # 'enhance_param':[50,100,'Linear'],
            'enhance_param': [50, 100, 'None'],
            'extractor_name': 'aliked',
            'base_name': 'final_over_layer',
            'device': 'cuda'
        }

        # 为最后一张图设置特殊参数
        self.last_image_params = self.detection_params.copy()
        self.last_image_params['rows'] = 3
        self.last_image_params['cols'] = 2


