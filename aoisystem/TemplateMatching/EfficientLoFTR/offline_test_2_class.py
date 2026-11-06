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
from TemplateMatching.EfficientLoFTR.coarse_matching_new import coarse_matching_eloftr_pillow_sample_only,coarse_matching_first,coarse_matching_all
from TemplateMatching.EfficientLoFTR.fine_matching_loftr_new import fine_matching_pillow_on_template,fine_matching_first,fine_matching_all
from DefectDetection.sahi.sahi.auto_model import AutoDetectionModel
from DefectDetection.sahi.sahi.predict import get_sliced_prediction,get_batched_sliced_prediction
from DefectDetection.sahi.sahi.utils.cv import  visualize_object_predictions
from DefectDetection.sahi.sahi.utils.file import Path, increment_path
from DefectDetection.batchinference.vrs_generation import convert_coco_to_vrs
# --------------------
# Default parameters
# --------------------
DEFAULT_PARAMS = {
    'row': 4,
    'col': 3,
    'auto_split': False,
    'overlap_temp': 0,
    'overlap_samp': 800,
    'flag': 12,
    'slice': 2400,
    'thres': 0.5,
    'save_image': True,
    'stage':0
}

# --------------------
# Class definitions
# --------------------
class MatchThreadFirst:
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
        offset=w-crop_w
        cropped = samp_img.crop((w - crop_w, 0, w, h))
        self.default_columns = 1
        return cropped.convert('L'), img_last,offset

    def linear_adjust_template(self, tmpl_img, sample_img):
        w, h = sample_img.size
        sample_crop = sample_img.crop((0, 9000, w, h))
        t_arr = np.array(tmpl_img, dtype=np.float32)
        s_arr = np.array(sample_crop, dtype=np.float32)
        t_norm = (t_arr - t_arr.mean()) / (t_arr.std() + 1e-6) * s_arr.std() + s_arr.mean()
        t_norm = np.clip(t_norm, 0, 255).astype(np.uint8)
        return t_norm, s_arr.astype(np.uint8)

    def extract_prefix(self, filename):
        idx = filename.rfind('_')
        return filename[:idx] if idx >= 0 else filename

    def template_matching(self):
        num = self.extract_number_from_path(self.image_path)
        if num == len(self.template_paths):
            samp_img, tmpl_img, offset = self.process_images(self.template_paths, self.image_path)
        else:
            samp_img = Image.open(self.image_path).convert('L')
            tmpl_img = Image.open(self.template_paths[num-1]).convert('L')
            offset = 0
        name = self.get_tif_filenames_from_list(self.template_paths, num)
        t_arr, s_arr = self.linear_adjust_template(tmpl_img, samp_img)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {dev}")
        obj_reg, samp_reg,cut_box_sample = coarse_matching_first(
            s_arr, t_arr,
            enlarge=True,
            margin_percent=0,
            scale_factor=1.0
        )
        if obj_reg is None or samp_reg is None :
            print("Coarse matching failed.")
            return 
        print("Coarse matching done.")

                # 如果是最后条带，加入 offset
        # if offset:
        #     cut_box_sample['x_min'] += offset
        #     cut_box_sample['x_max'] += offset

        overlay,H_matrices= fine_matching_first(
            samp_reg,
            obj_reg,
            overlap_sample=self.overlap_samp,
            overlap_template=self.overlap_temp,
            Auto_split=False,
            default_rows=self.default_rows,
            default_columns=self.default_columns
        )
        return overlay, self.extract_prefix(name), name,cut_box_sample,H_matrices
    


class MatchThreadRemain:
    """Template matching logic."""
    def __init__(self, image_path, template_paths, match_info, auto_split, default_rows, default_columns, overlap_samp, overlap_temp):
        self.image_path = image_path
        self.template_paths = template_paths
        self.match_info = match_info,
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
        t_arr = np.array(tmpl_img, dtype=np.float32)
        s_arr = np.array(sample_crop, dtype=np.float32)
        t_norm = (t_arr - t_arr.mean()) / (t_arr.std() + 1e-6) * s_arr.std() + s_arr.mean()
        t_norm = np.clip(t_norm, 0, 255).astype(np.uint8)
        return t_norm, s_arr.astype(np.uint8)

    def extract_prefix(self, filename):
        idx = filename.rfind('_')
        return filename[:idx] if idx >= 0 else filename

    def template_matching(self):

        fname = Path(self.image_path).name
        entry = next((e for e in self.match_info['samples']
                      if e['file_name'] == fname), None)
        if entry is None:
            print(f"No match info for {fname}")
            return None

        bbox    = entry['bbox']
        H_list  = entry['H_matrices']


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
        obj_reg, samp_reg = coarse_matching_all(
            s_arr, t_arr,bbox
        )
        if obj_reg is None or samp_reg is None :
            print("Coarse matching failed.")
            return 
        print("Coarse matching done.")
        overlay = fine_matching_all(
            samp_reg,
            obj_reg,
            H_list,
            overlap_sample=self.overlap_samp,
            overlap_template=self.overlap_temp,
            Auto_split=False,
            default_rows=self.default_rows,
            default_columns=self.default_columns
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
            12:"/home/aoi/Downloads/best_0410.pt"  
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
    # Modify these paths as needed:
    template_dir = '/home/aoi/pretest/AOI_V1/material/P2018P6W013805A1_20250721_150909/A'
    sample_dir = '/home/aoi/pretest/images/p2018_gtl_record'

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

    if not template_list or not sample_list:
        print("请确认 template_dir 和 sample_dir 下存在 TIFF 文件。")
        return
    
    part_id = Path(sample_dir).stem
    stage = params.get('stage', 0)

    if stage == 0:
            print("Executing first detection (match and save info)...")
            match_info = {"part_id": part_id, "samples": []}
            results_dir = None
            for sp in sample_list:
                print(f"Processing sample (first stage): {sp}")
                mt_first = MatchThreadFirst(
                    image_path=sp,
                    template_paths=template_list,
                    auto_split=params['auto_split'],
                    default_rows=params['row'],
                    default_columns=params['col'],
                    overlap_samp=params['overlap_samp'],
                    overlap_temp=params['overlap_temp']
                )
                result = mt_first.template_matching()
                if result:
                    matching_result, fn, nm, bbox, H_list = result
                
                if matching_result is not None:
                    matching_np = np.array(matching_result)
                    if matching_np.ndim == 2:
                        matching_cv = matching_np
                    else:
                        matching_cv = cv2.cvtColor(matching_np, cv2.COLOR_GRAY2BGR)

                    base_name = Path(fn).stem
                    results_dir = os.path.join('results', base_name)

                    os.makedirs(results_dir, exist_ok=True)
                    output_path = os.path.join(results_dir, f"{nm}.tiff")
                    cv2.imwrite(output_path, matching_cv, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
                    print(f"Overlay saved: {output_path}")

                sample_info = {
                    "file_name": Path(sp).name,
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
                match_info["samples"].append(sample_info)
            if results_dir is None:
                print("No overlay results to save.")
                return
            json_path = os.path.join(results_dir, f"{fn}_match_info.json")
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(match_info, jf, indent=4, ensure_ascii=False)
            print(f"Match info JSON saved: {json_path}")
            torch.cuda.empty_cache()
            print("Starting combine and detection (first stage)...")
            worker = CombinerAndDetectionWorker(
                folder=results_dir,
                output_filename='combined.tif',
                file_name=os.path.basename(sample_dir),
                flag=params['flag'],
                slice_win=params['slice'],
                conf_thres=params['thres'],
                save_image=params['save_image'],
                mode='整图批量推理'
            )
            try:
                combined_arr = worker.combiner()
                print("Image combining done.")
                proj, name, coco_json = worker.detection(combined_arr)
                print(f"Detection results saved. COCO JSON path: {coco_json}")
            except Exception as e:
                logging.error(f"Error during combine/detect (first stage): {e}")

   
   
    else:
        print("Executing production detection (using stored match info)...")

        results_dir = os.path.join('results', part_id)
        json_path = os.path.join(results_dir, f"{part_id}_match_info.json")
        if not os.path.exists(json_path):
            print("Match info file not found:", json_path)
            return
        with open(json_path, 'r', encoding='utf-8') as jf:
            match_info = json.load(jf)

        # 对每个样本条带执行生产阶段匹配
        for sp in sample_list:
            print("Production-stage sample:", sp)
            mt = MatchThreadRemain(
                image_path=sp,
                template_paths=template_list,
                match_info=match_info,
                auto_split=params['auto_split'],
                default_rows=params['row'],
                default_columns=params['col'],
                overlap_samp=params['overlap_samp'],
                overlap_temp=params['overlap_temp']
            )
            res = mt.template_matching()
            if not res:
                continue
            overlay, prefix, name = res

            # 保存 overlay 图
            out_path = os.path.join(results_dir, f"{name}.tiff")
            ov_np = np.array(overlay)
            ov_bgr = cv2.cvtColor(ov_np, cv2.COLOR_GRAY2BGR) if ov_np.ndim == 2 else ov_np
            cv2.imwrite(out_path, ov_bgr, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            print("Overlay saved:", out_path)

        # 复用 CombinerAndDetectionWorker 完成拼接与检测
        torch.cuda.empty_cache()
        print("Starting combine and detection (production stage)...")
        worker = CombinerAndDetectionWorker(
            folder=results_dir,
            output_filename='combined.tif',
            file_name=os.path.basename(sample_dir),
            flag=params['flag'],
            slice_win=params['slice'],
            conf_thres=params['thres'],
            save_image=params['save_image'],
            mode='整图批量推理'
        )
        try:
            combined_arr = worker.combiner()
            print("Image combining done.")
            proj, name, coco_json = worker.detection(combined_arr)
            print(f"Detection results saved. COCO JSON path: {coco_json}")
        except Exception as e:
            logging.error(f"Error during combine/detect (first stage): {e}")


        # first_template_name = Path(template_list[0]).stem if template_list else "result"
        # prefix = first_template_name
        # if '_' in first_template_name:
        #     prefix = first_template_name[:first_template_name.rfind('_')]
        # results_dir = os.path.join('results', prefix)
        # os.makedirs(results_dir, exist_ok=True)
        # json_path = os.path.join(results_dir, f"{prefix}_match_info.json")
        # if not os.path.exists(json_path):
        #     print(f"Match info file not found: {json_path}")
        #     return
        # with open(json_path, 'r', encoding='utf-8') as jf:
        #     match_info_data = json.load(jf)
        # for sp in sample_list:
        #     print(f"Processing sample (production stage): {sp}")
        #     mt_remain = MatchThreadRemain(
        #         image_path=sp,
        #         template_paths=template_list,
        #         match_info=match_info_data,
        #         auto_split=params['auto_split'],
        #         default_rows=params['row'],
        #         default_columns=params['col'],
        #         overlap_samp=params['overlap_samp'],
        #         overlap_temp=params['overlap_temp']
        #     )
        #     res = mt_remain.template_matching_remain()
        #     if res:
        #         overlay_img, prefix, name = res
        #     else:
        #         continue
        #     if overlay_img is not None:
        #         overlay_np = np.array(overlay_img)
        #         if overlay_np.ndim == 2:
        #             overlay_bgr = cv2.cvtColor(overlay_np, cv2.COLOR_GRAY2BGR)
        #         else:
        #             overlay_bgr = overlay_np
        #         output_path = os.path.join(results_dir, f"{name}.tiff")
        #         cv2.imwrite(output_path, overlay_bgr, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        #         print(f"Overlay saved: {output_path}")
        # torch.cuda.empty_cache()
        # print("Starting combine and detection (production stage)...")
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
        #     proj, name, coco_json = worker.detection(combined_arr)
        #     print(f"Detection results saved. COCO JSON path: {coco_json}")
        # except Exception as e:
        #     logging.error(f"Error during combine/detect (production stage): {e}")


if __name__ == '__main__':
    main()
