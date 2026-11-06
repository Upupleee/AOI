#!/usr/bin/env python3
# visualize_coco.py

import json
from PIL import Image, ImageDraw, ImageFont,ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    # ===== 在这里修改你的路径 =====
    original_image_path = "/home/aoi/pretest/AOI_V1/hsx_experiment/Ver_gerber/p2830p4h018545a0/gtl.bmp"
    coco_json_path      = "/home/aoi/pretest/AOI_V1/prediction_visual/gtl/detection_20250813_193404.json"
    output_image_path   = "results/visualized_full_compare.png"
    # ============================

    # 打开原图
    img = Image.open(original_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 载入 COCO 结果
    with open(coco_json_path, "r") as f:
        detections = json.load(f)

    # 尝试加载字体（可选）
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except:
        font = ImageFont.load_default()

    # 绘制检测框和标签
    for det in detections:
        x, y, w, h = det["bbox"]
        cat_id      = det["category_id"]
        score       = det.get("score", 0.0)

        # 画框
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline="red",
            width=2
        )

        # 准备标签文本
        text = f"{cat_id}:{score:.2f}"
        # 计算文本宽高
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 在框上方绘制背景
        text_bg = [(x, y - th), (x + tw, y)]
        draw.rectangle(text_bg, fill="red")

        # 绘制文字
        draw.text((x, y - th), text, fill="white", font=font)

    # 保存结果
    img.save(output_image_path)
    print(f"可视化结果已保存到 {output_image_path}")

if __name__ == "__main__":
    main()
