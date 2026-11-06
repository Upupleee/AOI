import os
import json
import datetime
import itertools
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch

# Detection imports
from DefectDetection.sahi.sahi.auto_model import AutoDetectionModel
from DefectDetection.sahi.sahi.predict import get_sliced_prediction, get_batched_sliced_prediction
from DefectDetection.sahi.sahi.utils.cv import visualize_object_predictions
from DefectDetection.sahi.sahi.utils.file import Path as SahiPath, increment_path
from DefectDetection.batchinference.vrs_generation import convert_coco_to_vrs

class OverlayImage:
    """Holds information for each overlay image and its offset."""
    def __init__(self, path: Path, x_offsite: int, y_offsite: int = 0):
        self.path = Path(path)
        self.x_offsite = x_offsite
        self.y_offsite = y_offsite
        self.coco_json = None
        self.batch_pred = None

class DetectionWorker:
    """Runs defect detection on a single image."""
    def __init__(self, flag, slice_win, conf_thres, mode):
        self.flag = flag
        self.slice_win = slice_win
        self.conf_thres = conf_thres
        self.mode = mode

    def extract_prefix(self, filename: str) -> str:
        idx = filename.find('_')
        return filename[:idx] if idx >= 0 else Path(filename).stem

    def detect(self, img_arr: np.ndarray, file_name: str, output_root: Path):
        # Map flags to model paths
        model_map = {
            0: '/home/aoi/pretest/AOI_V1/DefectDetection/models/yolo12s_deeppcb_original.pt',
            1: '/home/aoi/pretest/AOI_V1/DefectDetection/models/yolo12s_ryan_argumentation.pt'
        }
        model_path = model_map.get(self.flag, model_map[0])
        #model_path = model_map.get(self.flag, model_map[1])
        detector = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=model_path,
            confidence_threshold=self.conf_thres,
            device='cuda:0'
        )
        detector.load_model()

        # Ensure output directory
        os.makedirs(output_root, exist_ok=True)
        prefix = self.extract_prefix(file_name)

        if self.mode == '整图批量推理':
            batch_pred = get_batched_sliced_prediction(
                image=img_arr,
                detection_model=detector,
                slice_height=self.slice_win,
                slice_width=self.slice_win,
                overlap_height_ratio=0.05,
                overlap_width_ratio=0.05,
                verbose=0
            )
            project = output_root / prefix
            project.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            # Save COCO JSON
            coco_dir = project / 'coco'
            coco_dir.mkdir(exist_ok=True)
            coco_json = coco_dir / f'detection_{timestamp}.json'
            with open(coco_json, 'w') as f:
                json.dump(batch_pred.to_coco_predictions(image_id=1), f, indent=4)
            # Save VRS
            vrs_dir = project / 'vrs'
            vrs_dir.mkdir(exist_ok=True)
            vrs_path = vrs_dir / f'detection_{timestamp}.vrs'
            convert_coco_to_vrs(
                batch_pred.to_coco_predictions(image_id=1),
                str(vrs_path),
                image_height=img_arr.shape[0]
            )
            return batch_pred, project, coco_json
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
            base = output_root / 'prediction_visual'
            save_dir = SahiPath(increment_path(SahiPath(base) / prefix, exist_ok=False))
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
            # Dump COCO JSON
            coco_pred = slice_pred.to_coco_predictions(image_id=1)
            coco_json = save_dir / f'{prefix}.json'
            with open(coco_json, 'w') as f:
                json.dump(coco_pred, f, indent=4)
            return save_dir, coco_json

class SimplePipeline:
    """Pipeline for defect detection and final stitching of existing overlays."""
    def __init__(self, overlay_dir: str, params: dict):
        self.params = params
        self.overlay_dir = Path(overlay_dir)
        # Gather overlay images
        overlays = sorted(self.overlay_dir.glob('*.tiff')) + sorted(self.overlay_dir.glob('*.tif'))
        # Compute horizontal offsets based on widths
        widths = [Image.open(p).width for p in overlays]
        x_offsets = list(itertools.accumulate([0] + widths))[:-1]
        # Initialize overlay image objects
        self.imgs = [OverlayImage(p, x_offsets[i], 0) for i,p in enumerate(overlays)]

    def detection(self, idx: int):
        sc = self.imgs[idx]
        img_arr = np.array(Image.open(sc.path))
        results_dir = sc.path.parent
        dw = DetectionWorker(
            self.params['flag'],
            self.params['slice'],
            self.params['thres'],
            self.params['mode']
        )
        batch_pred, detect_dir, coco_json = dw.detect(img_arr, sc.path.name, results_dir)
        sc.batch_pred = batch_pred
        sc.coco_json = coco_json
        print(f"Detection saved in {detect_dir}, JSON at {coco_json}")

    def combine_images(self, output_path: str):
        # Load all overlays
        overlays = [Image.open(sc.path) for sc in self.imgs]
        total_w = sum(im.width for im in overlays)
        max_h = max(im.height for im in overlays)
        # Create mosaic
        mosaic = Image.new('RGB', (total_w, max_h), (0,0,0))
        for sc, im in zip(self.imgs, overlays):
            mosaic.paste(im, (sc.x_offsite, sc.y_offsite))
        # Draw detections
        draw = ImageDraw.Draw(mosaic)
        for sc in self.imgs:
            if sc.coco_json:
                preds = json.load(open(sc.coco_json))
                for pred in preds:
                    x, y, w, h = pred.get('bbox', [])
                    x0 = x + sc.x_offsite
                    y0 = y + sc.y_offsite
                    x1, y1 = x0 + w, y0 + h
                    draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
        # Save combined image
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        mosaic.save(output_path)
        # Save summary JSON
        summary = []
        for sc in self.imgs:
            if sc.coco_json:
                summary.extend(json.load(open(sc.coco_json)))
        summary_path = out_p.with_suffix('.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Combined image saved to {output_path}")
        print(f"Summary JSON saved to {summary_path}")

# Default parameters
DEFAULT_PARAMS = {
    'flag': 1,
    'slice': 2400,
    'thres': 0.5,
    'mode': '整图批量推理'
}

if __name__ == '__main__':
    # Directory containing pre-computed overlay images
    overlay_dir = '/home/aoi/pretest/AOI_V1/results/gtl_3250'
    pipeline = SimplePipeline(overlay_dir, DEFAULT_PARAMS)
    # Run detection on each overlay
    for i in range(len(pipeline.imgs)):
        pipeline.detection(i)
    # Combine all results into a final stitched image
    pipeline.combine_images('results/combined.tiff')
