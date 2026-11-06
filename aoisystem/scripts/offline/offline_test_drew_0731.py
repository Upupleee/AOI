#RYAN_TEST
#!/usr/bin/env python3
"""
Standalone AOI algorithm test script:
- Embeds MatchThread and CombinerAndDetectionWorker classes
- Default parameters in DEFAULT_PARAMS
- Modify template_dir and sample_dir directly in main()

Usage:
  python test_detection.py
"""
import os
import re
import json
import logging
import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import tifffile
import torch

# Algorithmic functions (ensure these modules are in PYTHONPATH)
from TemplateMatching.EfficientLoFTR.coarse_matching_new import coarse_matching_eloftr_pillow_sample_only
from TemplateMatching.EfficientLoFTR.fine_matching_loftr_new_drew_0731 import fine_matching_pillow_on_template_drew
from DefectDetection.sahi.sahi.auto_model import AutoDetectionModel
from DefectDetection.sahi.sahi.predict import get_sliced_prediction,get_batched_sliced_prediction
from DefectDetection.sahi.sahi.utils.cv import  visualize_object_predictions
from DefectDetection.sahi.sahi.utils.file import Path, increment_path
from DefectDetection.batchinference.vrs_generation import convert_coco_to_vrs

import time
# --------------------
# Default parameters
# --------------------
DEFAULT_PARAMS = {
    'row': 3,
    'col': 2,
    'auto_split': False,
    'overlap_temp': 0,
    'overlap_samp': 100,
    'flag': 12,
    'slice': 1500,
    'thres': 0.5,
    'save_image': True,

    # 7.30 new
    'grid_size': 6,
    'max_sampling_per_grid': 100,
    'use_clahe': False,
    'enhance_algorithms': [50, 100, 'none'],  
    # [lower_bound, upper_bound, interpolation_method]
    # interpolation_method: linear, sigmoid, clip, none
    'extractor_name': 'superpoint', 
    # 'superpoint' or 'aliked'
    'base_name': 'final_over_layer',
    'device': 'cuda'

}

# --------------------
# Class definitions
# --------------------
class MatchThread:
    """Template matching logic."""
    def __init__(self, image_path, template_paths, auto_split, default_rows, default_columns, overlap_samp, overlap_temp):
        self.image_path = image_path
        self.template_paths = template_paths
        self.auto_split = auto_split
        self.default_rows = default_rows
        self.default_columns = default_columns
        self.overlap_samp = overlap_samp
        self.overlap_temp = overlap_temp

    def extract_number_from_path(self, path):
        match = re.search(r'_?(\d+)\.\w+$', path)
        return int(match.group(1)) if match else None

    def get_tif_filenames_from_list(self, paths, number):
        return Path(paths[number-1]).stem

    def process_images(self, templates, sample):
        Image.MAX_IMAGE_PIXELS = None
        if len(templates) < 2:
            raise ValueError("Need at least two templates.")
        prev_path, last_path = templates[-2], templates[-1]
        img_prev = Image.open(prev_path).convert('L')
        img_last = Image.open(last_path).convert('L')
        ratio = img_last.width / img_prev.width
        samp_img = Image.open(sample)
        w, h = samp_img.size
        crop_w = min(w, int(w * ratio))
        cropped = samp_img.crop((w - crop_w, 0, w, h))
        self.default_columns = 1
        return cropped.convert('L'), img_last

    def linear_adjust_template(self, tmpl_img, sample_img):
        w, h = sample_img.size
        sample_crop = sample_img.crop((0, 9000, w, h))
        ##modified 0726 jingshu
        sample_crop =sample_crop.resize(tmpl_img.size,resample=Image.BILINEAR)
        t_arr = np.array(tmpl_img, dtype=np.float32)
        s_arr = np.array(sample_crop, dtype=np.float32)
        t_norm = (t_arr - t_arr.mean()) / (t_arr.std() + 1e-6) * s_arr.std() + s_arr.mean()
        t_norm = np.clip(t_norm, 0, 255).astype(np.uint8)
        return t_norm, s_arr.astype(np.uint8)
    


    def extract_prefix(self, filename):
        idx = filename.rfind('_')
        return filename[:idx] if idx >= 0 else filename

    #RYAN_TEST
    def template_matching(self):
        num = self.extract_number_from_path(self.image_path)
        if num == len(self.template_paths):
            samp_img, tmpl_img = self.process_images(self.template_paths, self.image_path)
        else:
            samp_img = Image.open(self.image_path).convert('L')
            tmpl_img = Image.open(self.template_paths[num-1]).convert('L')
        name = self.get_tif_filenames_from_list(self.template_paths, num)
        t_arr, s_arr = self.linear_adjust_template(tmpl_img, samp_img)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {dev}")

        ##加入resize，提升匹配
        # s_arr = s_arr.resize(t_arr.size, resample=Image.BILINEAR)
        start = time.time()
        obj_reg, samp_reg = coarse_matching_eloftr_pillow_sample_only(
            s_arr, t_arr,
            enlarge=True,
            margin_percent=0,
            scale_factor=1.0
        )
        end = time.time()
        print("coarse time", end-start)

        if obj_reg is None or samp_reg is None :
            print("Coarse matching failed.")
            return 
        print("Coarse matching done.")
    
        overlay = fine_matching_pillow_on_template_drew(
            samp_reg,
            obj_reg,
            grid_size=DEFAULT_PARAMS['grid_size'],
            max_sampling_per_grid=DEFAULT_PARAMS['max_sampling_per_grid'],
            use_clahe=DEFAULT_PARAMS['use_clahe'],
            enhance_algorithms=DEFAULT_PARAMS['enhance_algorithms'],
            extractor_name=DEFAULT_PARAMS['extractor_name'],
            base_name=DEFAULT_PARAMS['base_name'],
            device=DEFAULT_PARAMS['device']
        )
        return overlay, self.extract_prefix(name), name

class CombinerAndDetectionWorker:
    """Combines TIFFs then runs defect detection."""
    def __init__(self, folder, output_filename, file_name, flag, slice_win, conf_thres, save_image, mode):
        self.folder = folder
        self.output_filename = output_filename
        self.file_name = file_name
        self.flag = flag
        self.slice_win = slice_win
        self.conf_thres = conf_thres
        self.save_image = save_image
        self.mode = mode
        self._running = True

    def combiner(self):
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(f"No such folder: {self.folder}")
        tifs = sorted(
            [f for f in os.listdir(self.folder) if f.lower().endswith(('.tif', '.tiff'))],
            reverse=True
        )
        imgs = []
        total_w = 0
        max_h = 0
        for fn in tifs:
            path = os.path.join(self.folder, fn)
            with Image.open(path) as im:
                imgs.append(im.copy())
                total_w += im.width
                max_h = max(max_h, im.height)
        combined = Image.new('RGB', (total_w, max_h), (0, 0, 0))
        x = 0
        for im in imgs:
            combined.paste(im, (x, 0))
            x += im.width
        arr = np.array(combined)
        if self.save_image:
            output_path = os.path.join(self.folder,self.output_filename)
            tifffile.imwrite(output_path,arr,bigtiff=True)
        return arr

    def extract_prefix(self, filename):
        idx = filename.find('_')
        return filename[:idx] if idx >= 0 else filename

    def detection(self, img_arr):
        model_map = {
            0: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/weights1031/best-1031.engine",
            1: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1207/best-30-95um-1207.engine",
            2: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_lowpass/best-lowpass-1206.engine",
            3: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_1/best-008f-1203.engine",
            4: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_2/best-008f-bg1.engine",
            5: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30um-ryan_1/best-30um-1.engine",
            6: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1207_pm/best-30-95um-1207-2.engine",
            7: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_3/last_008f_1205.engine",
            8: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1206_lowpass/best-1206-30um.engine",
            9: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/best_2025_5_26.pt",
            10:"/home/aoi/Downloads/best-v12n-0408.pt",
            11:"/home/aoi/Downloads/best-v12n-0409.pt",
            12: "/home/aoi/Downloads/yolo11s-p2_aug10.pt"
        }
        model_path = model_map.get(self.flag, model_map[0])
        detector = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=self.conf_thres,
            device="cuda:0"
        )
        detector.load_model()
        if self.mode == "整图批量推理":
            batch_pred = get_batched_sliced_prediction(
                image=img_arr,
                detection_model=detector,
                slice_height=self.slice_win,
                slice_width=self.slice_win,
                overlap_height_ratio=0.05,
                overlap_width_ratio=0.05,
                verbose=0
            )
            project = f"./prediction_visual/{self.extract_prefix(self.file_name)}"
            os.makedirs(project, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            coco_dir = os.path.join(project, 'coco')
            os.makedirs(coco_dir, exist_ok=True)
            coco_json = os.path.join(coco_dir, f"detection_{ts}.json")
            with open(coco_json, 'w') as f:
                json.dump(batch_pred.to_coco_predictions(image_id=1), f, indent=4)
            vrs_dir = os.path.join(project, 'vrs')
            os.makedirs(vrs_dir, exist_ok=True)
            vrs_path = os.path.join(vrs_dir, f"detection_{ts}.vrs")
            convert_coco_to_vrs(
                batch_pred.to_coco_predictions(image_id=1),
                vrs_path, image_height=img_arr.shape[0]
            )
            return project, self.extract_prefix(self.file_name), coco_json
        else:
            slice_pred = get_sliced_prediction(
                image=img_arr,
                detection_model=detector,
                slice_height=self.slice_win,
                slice_width=self.slice_win,
                overlap_height_ratio=0.05,
                overlap_width_ratio=0.05,
                verbose=2
            )
            base = "./prediction_visual"
            save_dir = Path(increment_path(Path(base) / "exp", exist_ok=False))
            save_dir.mkdir(parents=True, exist_ok=True)
            visualize_object_predictions(
                img_arr,
                slice_pred.object_prediction_list,
                rect_th=1,
                text_size=0.5,
                text_th=1,
                hide_labels=False,
                hide_conf=False,
                output_dir=str(save_dir)
            )
            return base, base, str(save_dir)

# --------------------
# Main testing logic
# --------------------
def main():
    # # Modify these paths as needed:
    # template_dir = '/home/aoi/pretest/test/slices'
    # sample_dir = '/home/aoi/pretest/images/'
    #__import__('ipdb').set_trace()

    #template_dir = '/home/aoi/pretest/AOI_V1/material/p1243p4e014492a0_20250724_184555/A'
    template_dir = '/home/aoi/pretest/images/new_data/0821/0821_gerber_split/p003p2e008003a0_20250822_112124/A'

    #sample_dir = '/home/aoi/pretest/images/p1243p4e_gtl_record/'
    #sample_dir = '/home/aoi/pretest/images/gold/gtl_1'
    #sample_dir = '/home/aoi/pretest/images/gold/gtl_2'
    sample_dir = '/home/aoi/pretest/images/new_data/0821/0821_data_record/p003p2e008003a0/1/gtl_1'



    params = DEFAULT_PARAMS
    # Collect file lists
    template_list = sorted(
        [str(p) for p in Path(template_dir).glob('*.tif')] +
        [str(p) for p in Path(template_dir).glob('*.tiff')]
    )
    sample_list = sorted(
        [str(p) for p in Path(sample_dir).glob('*.tif')] +
        [str(p) for p in Path(sample_dir).glob('*.tiff')]
    )

    # __import__('ipdb').set_trace()
    if not template_list or not sample_list:
        print("请确认 template_dir 和 sample_dir 下存在 TIFF 文件。")
        return

    #RYAN_TEST
    # 模板匹配测试
    for sp in sample_list:
        print(f"Processing sample: {sp}")
        mt = MatchThread(
            image_path=sp,
            template_paths=template_list,
            auto_split=params['auto_split'],
            default_rows=params['row'],
            default_columns=params['col'],
            overlap_samp=params['overlap_samp'],
            overlap_temp=params['overlap_temp']
        )
        res = mt.template_matching()
        if res:
            matching_result, fn, nm = res

            if matching_result is not None:
                        ##modified 0523:saving matching result
                matching_np = np.array(matching_result)                          # RGB or L
                if matching_np.ndim == 2:                                        # 单通道灰度
                    matching_cv = matching_np
                else:                                                            # 3 通道 RGB → BGR
                    matching_cv = cv2.cvtColor(matching_np, cv2.COLOR_RGB2BGR)
                
                 # 确保目录存在
                base_name = Path(fn).stem
                results_dir= os.path.join('results',base_name)
                os.makedirs(results_dir, exist_ok=True)
                # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                final_output_path = os.path.join(
                    results_dir,
                    f"{nm}.tiff"
                )
                save_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
                result_test_final = cv2.imwrite(final_output_path, matching_cv, save_params)
                print("Write success:", result_test_final)
                print(f"the overlay blit saved to {final_output_path}")
            
    # 拼接与检测测试
    #__import__('ipdb').set_trace()
    # torch.cuda.empty_cache()
    # print("Starting combine and detection...")
    # worker = CombinerAndDetectionWorker(
    #     folder=results_dir,
    #     output_filename='combined.tif',
    #     file_name=os.path.basename(sample_dir),
    #     flag=params['flag'],
    #     slice_win=params['slice'],
    #     conf_thres=params['thres'],
    #     save_image=params['save_image'],
    #     mode='整图批量推理'
    # )
    # try:
    #     combined_arr = worker.combiner()
    #     print("Image combining done.")
    #     torch.cuda.empty_cache()
    #     proj, name, coco_json = worker.detection(combined_arr)
    #     print(f"Detection JSON at: {coco_json}")
    # except Exception as e:
    #     logging.error(f"Error during combine/detect: {e}")

if __name__ == '__main__':
    main()
