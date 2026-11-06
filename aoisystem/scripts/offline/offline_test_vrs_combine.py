#!/usr/bin/env python3
# test.py

import os
import re
import time
import json
import numpy as np
from PIL import Image
from DefectDetection.sahi.sahi.auto_model import AutoDetectionModel
from DefectDetection.sahi.sahi.predict import get_batched_sliced_prediction

Image.MAX_IMAGE_PIXELS = None

def find_strip_files(images_dir):
    """扫描目录，返回按后缀数字升序排序的 TIFF 文件列表"""
    files = [f for f in os.listdir(images_dir) if f.lower().endswith((".tif", ".tiff"))]
    def suffix(f):
        m = re.search(r'(\d+)(?=\.(tif|tiff)$)', f)
        return int(m.group(1)) if m else float('inf')
    return sorted(files, key=suffix)

def main():
    # ====== 在这里直接修改默认路径和参数 ======
    images_dir = "/home/aoi/pretest/AOI_V1/results/gtl/test_gerber"
    model_path = "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0801_2400.pt"
    conf_threshold = 0.1
    slice_height = 1000
    slice_width = 1000
    overlap_height_ratio = 0.15
    overlap_width_ratio = 0.15
    output_json = "coco_results.json"
    # ===========================================

    # 初始化检测模型
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device="0"
    )

    # 获取条带文件，suffix 1 = 最右侧
    strip_files = find_strip_files(images_dir)
    if not strip_files:
        raise RuntimeError(f"在 {images_dir} 未找到任何 TIFF 文件")

    # 读取每条带宽高并计算全图偏移
    widths, heights = {}, {}
    for fname in strip_files:
        with Image.open(os.path.join(images_dir, fname)) as img:
            widths[fname], heights[fname] = img.size
    if len(set(heights.values())) != 1:
        raise RuntimeError(f"条带图像高度不一致: {set(heights.values())}")
    full_h = heights[strip_files[0]]

    offsets = {}
    cum_x = 0
    for fname in reversed(strip_files):  # 从最左到最右累加
        offsets[fname] = cum_x
        cum_x += widths[fname]
    full_w = cum_x

    coco_results = []
    image_id = 1

    # 按从右到左（suffix 1→n）顺序处理每个条带
    for fname in strip_files:
        path = os.path.join(images_dir, fname)
        with Image.open(path) as img:
            w, h = img.size
            half = h // 2

            # 内存中分割上下两半
            top_np = np.array(img.crop((0, 0, w, half)))
            bot_np = np.array(img.crop((0, half, w, h)))

        # 批量切片推理函数
        def infer(arr, y_offset):
            res = get_batched_sliced_prediction(
                image=arr,
                detection_model=detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                postprocess_type="GREEDYNMM",
                postprocess_match_metric="IOS",
                postprocess_match_threshold=0.5,
                postprocess_class_agnostic=False,
                perform_standard_pred=False,
                slices_outdir=None,
                slices_filename=None,
                slices_ext=None,
                visualize_slices=False,
                verbose=0
            )
            preds = []
            for pred in res.object_prediction_lists:
                x, y, bw, bh = pred.bbox.to_xywh()
                preds.append((pred.category.id, x, y + y_offset, bw, bh, float(pred.score.value)))
            return preds

        # 上半区推理（y_offset=0），下半区推理（y_offset=half）
        top_preds = infer(top_np, 0)
        bot_preds = infer(bot_np, half)

        # 合并上下 & 转全图坐标系
        x_off = offsets[fname]
        for cat_id, x, y, bw, bh, score in top_preds + bot_preds:
            coco_results.append({
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [x + x_off, y, bw, bh],
                "score": score
            })

        image_id += 1

    # 输出 COCO JSON
    with open(output_json, "w") as f:
        json.dump(coco_results, f, indent=2, ensure_ascii=False)

    print(f"已生成 COCO 检测结果: {output_json}")
    print(f"整图总尺寸: {full_w}×{full_h}")

if __name__ == "__main__":
    main()
