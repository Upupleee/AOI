import os
import json
from PIL import Image, ImageFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 解除 Pillow 的图像像素大小限制（用于读取超大图像）
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def visualize_coco_detections(json_path, image_path, output_path, score_thresh=0.3, zoomable=False):
    # 检查路径
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # 使用 Pillow 读取图像（支持大图）
    image = Image.open(image_path)
    image_np = np.array(image)

    # 加载检测结果
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    # 创建画布
    fig, ax = plt.subplots(figsize=(15, 10), dpi=100)
    ax.imshow(image_np)

    # 类别颜色映射
    category_colors = {
        'green': 'lime',
        'blue': 'deepskyblue',
    }

    # 绘制每个检测框
    for ann in annotations:
        x, y, w, h = ann['bbox']
        score = ann.get('score', 1.0)
        category = ann.get('category_name', str(ann['category_id']))
        
        if score < score_thresh:
            continue

        color = category_colors.get(category, 'red')
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, f"{category} {score:.2f}", color='white',
                fontsize=8, bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    ax.set_axis_off()
    plt.tight_layout()

    # 可选：支持交互缩放查看（适合 notebook 或 GUI 调试）
    if zoomable:
        plt.show()
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Visualization saved to: {output_path}")

# 示例使用
if __name__ == "__main__":
    json_file = "/home/dual3080/Workspace/test/coco_outputs/detection_results.json"
    image_file = "/home/dual3080/Dataset/AOI_hsx/test_imgs3_matched_20250525_110731.tiff"
    output_file = "output/visualized_result.jpg"

    # 设置 zoomable=True 可用于调试时可缩放显示
    visualize_coco_detections(json_file, image_file, output_file, zoomable=True)
