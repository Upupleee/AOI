#!/usr/bin/env python3
"""
AOI pipeline: separate template matching and defect detection on single images,
then optionally combine the overlays.
Usage:
    pipeline = Pipeline(template_dir, sample_dir, DEFAULT_PARAMS)
    pipeline.templatematching(idx)
    pipeline.detection(idx)
    pipeline.combine_images('combined.tiff')
"""
import os
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
import tifffile
import torch
import datetime

Image.MAX_IMAGE_PIXELS = None


# Template matching imports
from TemplateMatching.EfficientLoFTR.coarse_matching_new import coarse_matching_eloftr_pillow_sample_only
from TemplateMatching.EfficientLoFTR.fine_matching_loftr_new_drew_0731 import fine_matching_pillow_on_template_drew

# Detection imports
from DefectDetection.sahi.sahi.auto_model import AutoDetectionModel
from DefectDetection.sahi.sahi.predict import get_sliced_prediction, get_batched_sliced_prediction
from DefectDetection.sahi.sahi.utils.cv import visualize_object_predictions
from DefectDetection.sahi.sahi.utils.file import Path as SahiPath, increment_path
from DefectDetection.batchinference.vrs_generation import convert_coco_to_vrs

from PIL import Image, ImageDraw, ImageFont

class Samplecut:
    def __init__(self, sample_path, template_paths, params, max_size):
        self.sample = Image.open(sample_path).convert('L')
        self.template = Image.open(template_paths).convert('L')
        self.sample = np.array(self.sample)
        self.template = np.array(self.template)

        h, w = self.template.shape
        # 计算缩放比例，最长边缩放到max_size以内
        if max(h, w) > max_size:
            scale = max_size / h
            new_w = int(w * scale)
            self.template = cv2.resize(self.template, (new_w, max_size), interpolation=cv2.INTER_AREA)

        h, w = self.sample.shape
        # 计算缩放比例，最长边缩放到max_size以内
        self.sample = self.sample[9060:h, 0:w]

        if max(h, w) > max_size:
            scale = max_size / h
            new_w = int(w * scale)
            self.sample = cv2.resize(self.sample, (new_w, max_size), interpolation=cv2.INTER_AREA)


        self.params = params
        self.overlay = None
        self.batch_pred = None
        self.x_offsite = 0
        self.y_offsite = 0

    def add_offsite(self, offsite):
        self.x_offsite, self.y_offsite = offsite

    
class MatchThread:
    """Template matching logic."""
    def __init__(self, sample, template, grid_size, max_sampling_per_grid, extractor_name):
        self.sample = sample
        self.template = template

        self.grid_size = grid_size
        self.max_sampling_per_grid = max_sampling_per_grid
        self.extractor_name = extractor_name


    def linear_adjust_template(self, tmpl_img, sample_img):
        # h, w = sample_img.shape[:2]
        # sample_crop = sample_img[4500:h, 0:w]

        t_arr = np.array(tmpl_img, dtype=np.float32)
        s_arr = np.array(sample_img, dtype=np.float32)
        t_norm = (t_arr - t_arr.mean()) / (t_arr.std() + 1e-6) * s_arr.std() + s_arr.mean()
        t_norm = np.clip(t_norm, 0, 255).astype(np.uint8)
        return t_norm, s_arr.astype(np.uint8)

    def template_matching(self):
        samp_img = self.sample
        tmpl_img = self.template
        t_arr, s_arr = self.linear_adjust_template(tmpl_img, samp_img)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {dev}")
        obj_reg=t_arr
        samp_reg=s_arr
        # obj_reg, samp_reg = coarse_matching_eloftr_pillow_sample_only(
        #     s_arr, t_arr,
        #     enlarge=True,
        #     margin_percent=0,
        #     scale_factor=1.0
        # )

        torch.cuda.empty_cache()

        if obj_reg is None or samp_reg is None:
            print("Coarse matching failed.")
            return None
        print("Coarse matching done.")
        overlay = fine_matching_pillow_on_template_drew(
            samp_reg,
            obj_reg,
            grid_size=self.grid_size,
            max_sampling_per_grid=self.max_sampling_per_grid,
            extractor_name=self.extractor_name,
        )
        return overlay


class DetectionWorker:
    """Defect detection logic on single image arrays."""
    def __init__(self, flag, slice_win, conf_thres, mode):
        self.flag = flag
        self.slice_win = slice_win
        self.conf_thres = conf_thres
        self.mode = mode

    def extract_prefix(self, filename):
        idx = filename.find('_')
        return filename[:idx] if idx >= 0 else filename

    def detect(self, img_arr):
        # Map flags to model paths
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
            10: "/home/aoi/Downloads/best-v12n-0408.pt",
            11: "/home/aoi/Downloads/best-v12n-0409.pt",
            12: "results/yolo12p2.pt"
        }
        model_path = model_map.get(self.flag, model_map[0])
        detector = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=self.conf_thres,
            device="cuda:0"
        )
        detector.load_model()


        batch_pred = get_batched_sliced_prediction(
            image=img_arr,
            detection_model=detector,
            slice_height=self.slice_win,
            slice_width=self.slice_win,
            overlap_height_ratio=0.05,
            overlap_width_ratio=0.05,
            verbose=0
        )
        # project = Path(output_root) / prefix
        # project.mkdir(parents=True, exist_ok=True)
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # # Save COCO JSON
        # coco_dir = project / 'coco'
        # coco_dir.mkdir(exist_ok=True)
        # coco_json = coco_dir / f"detection_{timestamp}.json"
        # with open(coco_json, 'w') as f:
        #     json.dump(batch_pred.to_coco_predictions(image_id=1), f, indent=4)
        # # Save VRS
        # vrs_dir = project / 'vrs'
        # vrs_dir.mkdir(exist_ok=True)
        # vrs_path = vrs_dir / f"detection_{timestamp}.vrs"
        # convert_coco_to_vrs(
        #     batch_pred.to_coco_predictions(image_id=1),
        #     str(vrs_path), image_height=img_arr.shape[0]
        # )
        return batch_pred
        

class Pipeline:
    def __init__(self, template_dir, sample_dir, params):
        tpl = sorted(Path(template_dir).glob('*.tif')) + sorted(Path(template_dir).glob('*.tiff'))
        samp = sorted(Path(sample_dir).glob('*.tif')) + sorted(Path(sample_dir).glob('*.tiff'))
        # samp = samp[::-1] 
        # tpl = tpl[::-1]

        assert len(tpl) == len(samp), "template and sample counts must match"
        self.params = params
        # # compute horizontal offsets based on sample widths
        # widths = [Image.open(p).width for p in samp]
        # print(widths)
        # x_offsets = list(itertools.accumulate([0] + widths))
        # print(x_offsets)
        # # build image cut objects
        self.imgs = []
        for i in range(len(samp)):
            self.imgs.append(Samplecut(samp[i], tpl[i], params, self.params['max_imgsz']))
            print("load img", i)

        widths = [sc.template.shape[1] for sc in self.imgs]
        x_offsets = list(itertools.accumulate([0] + widths))
        for i in range(len(self.imgs)):
            self.imgs[i].add_offsite((x_offsets[i], 0))


    def templatematching(self, idx, if_save=False, save_pth=""):
        sc = self.imgs[idx]
        # run template matching
        mt = MatchThread(
            sc.sample,
            sc.template,
            self.params['grid_size'],
            self.params['max_sampling_per_grid'],
            self.params['extractor_name'],
        )
        overlay_img = mt.template_matching()
        if overlay_img is not None:
            sc.overlay = overlay_img
            # # save overlay
            if if_save:   
                if isinstance(overlay_img, np.ndarray):
                    overlay_pil = Image.fromarray(overlay_img)
                else:
                    overlay_pil = overlay_img
                overlay_pil = overlay_pil.convert('RGB')
                arr = np.array(overlay_pil)
                # BGR for OpenCV
                cv2.imwrite(str(save_pth), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_TIFF_COMPRESSION, 1])
                
                print(f"Template matching overlay saved to {save_pth}")

    def detection(self, idx):
        sc = self.imgs[idx]

        img_arr = sc.overlay
        dw = DetectionWorker(
            self.params['flag'],
            self.params['slice'],
            self.params['thres'],
            self.params['mode']
        )
        batch_pred = dw.detect(img_arr)
        sc.batch_pred = batch_pred

    def combine_images(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        overlays = [Image.fromarray(sc.overlay) for sc in self.imgs]
        total_w = sum(im.width for im in overlays)
        max_h = max(im.height for im in overlays)
        x_offsets = [sc.x_offsite for sc in self.imgs]

        mosaic = Image.new('RGB', (total_w, max_h), (0, 0, 0))
        for idx, im in enumerate(overlays):
            mosaic.paste(im, (x_offsets[idx], 0))
        draw = ImageDraw.Draw(mosaic)
        summary = []
        for idx, sc in enumerate(self.imgs):
            preds = sc.batch_pred.to_coco_predictions(image_id=1)

            for pred in preds:
                x, y, w, h = pred.get('bbox', [])
                x0 = x + sc.x_offsite
                y0 = y + sc.y_offsite
                score = pred.get('score')
                category = pred.get('category_name')
                draw.rectangle([x0, y0, x0+w, y0+h], outline='red', width=2)
                summary.append({'bbox': [x0, y0, w, h], 'score': score, 'category': category})

        max_side = max(mosaic.width, mosaic.height)
        if max_side > 2048*2:
            scale = 2048*2 / max_side
            new_size = (int(mosaic.width * scale), int(mosaic.height * scale))
            mini_mosaic = mosaic.resize(new_size, Image.Resampling.LANCZOS)
        mini_mosaic_path = os.path.join(output_path, "combined.tiff")
        mini_mosaic.save(mini_mosaic_path)
        print(f"Combined image saved to {mini_mosaic_path}")
        
        json_path = os.path.join(output_path, "combined.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Summary JSON saved to {json_path}")

        vrs_path = os.path.join(output_path, "combined.vrs")
        # 将 summary（融合后的所有检测）转换为 COCO 结构
        coco_preds = []
        for det in summary:
            coco_preds.append({
                "bbox": det["bbox"],
                "score": det.get("score", 1.0),
                "category_name": det.get("category", "green"),  # 如果没有类别名，默认"green"
            })
        convert_coco_to_vrs(coco_preds, vrs_path, image_height=mosaic.height)
        print(f"VRS file saved to {vrs_path}")

        # ===== 使用 summary 里的 bbox、score、class 在最终图上打框 =====
        class_colors = {
            'open': 'red',        # Class 0 -> red
            'short': 'blue',       # Class 1 -> blue
            'mousebite': 'green',      # Class 2 -> green
            'spur': 'yellow',     # Class 3 -> yellow
            'spurious_copper': 'purple',     # Class 4 -> purple
            'pin-hole': 'orange'      # Class 5 -> orange
        }
        draw_sum = ImageDraw.Draw(mosaic)
        font = ImageFont.load_default()
        for det in summary:
            x, y, w, h = det['bbox']
            cls = det['category']
            score = det['score']
            x0, y0 = x, y
            x1, y1 = x + w, y + h
            box_color = class_colors[cls]
            draw_sum.rectangle([x0, y0, x1, y1], outline=box_color, width=20)

        max_side = max(mosaic.width, mosaic.height)
        if max_side > 2048*2:
            scale = 2048*2 / max_side
            new_size = (int(mosaic.width * scale), int(mosaic.height * scale))
            mini_mosaic = mosaic.resize(new_size, Image.Resampling.LANCZOS)
        combined_with_box_path = os.path.join(output_path, "combined_with_box.tiff")
        mini_mosaic.save(combined_with_box_path)
        print(f"Combined visualization saved to {combined_with_box_path}")


# Default parameters
DEFAULT_PARAMS = {
    'grid_size' : 8,
    'max_sampling_per_grid': 80,
    'extractor_name': 'aliked', 
    'max_imgsz': 30000,

    'flag': 12,
    'slice': 1500,
    'thres': 0.5,
    'mode': '整图批量推理'
    }

if __name__ == '__main__':
    tpl_dir = '/home/aoi/pretest/images/0825/0825_gerber_split/p961p10e018765a0h/l2'

    samp_dir = '/home/aoi/pretest/images/0825/0825_data_record/P961p10e018765a0h/L2_record_1'

    pipeline = Pipeline(tpl_dir, samp_dir, DEFAULT_PARAMS)
    for i in range(len(pipeline.imgs)):
        folder = "results/new_data"
        os.makedirs(folder, exist_ok=True)
        pipeline.templatematching(i, True, folder + f"/{i+1}_concat.tif")
        # pipeline.detection(i)
    