import os
import re
import cv2

def stitch_strips(folder, output_path):
    # 1. 列出所有 *_concat.tif 文件
    files = [
        f for f in os.listdir(folder)
        if re.search(r'(\d+)_concat\.tif$', f)
    ]
    if not files:
        raise RuntimeError(f"No '_concat.tif' files in {folder}")

    # 2. 按数字降序排序（数字大的排在前面→左边）
    def extract_num(fn):
        return int(re.search(r'(\d+)_concat\.tif$', fn).group(1))
    files_sorted = sorted(files, key=extract_num, reverse=True)

    # 3. 读取图片
    imgs = [cv2.imread(os.path.join(folder, fn)) for fn in files_sorted]

    # 4. 统一高度（可选，但推荐以防尺寸不一）
    heights = [im.shape[0] for im in imgs]
    min_h = min(heights)
    imgs_resized = []
    for im in imgs:
        h, w = im.shape[:2]
        if h != min_h:
            new_w = int(w * (min_h / h))
            im = cv2.resize(im, (new_w, min_h))
        imgs_resized.append(im)

    # 5. 从左到右水平拼接
    panorama = cv2.hconcat(imgs_resized)

    # 6. 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, panorama)
    print(f"拼接完成，保存为: {output_path}")

if __name__ == "__main__":
    folder = "results/new_data"              # 你的四张 _concat.tif 所在目录
    output = "results/full_panorama.tif"   # 最终拼接图的保存路径
    stitch_strips(folder, output)
