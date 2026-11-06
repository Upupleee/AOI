
# ==================== 全局配置 ====================
INPUT_DIR = r"C:\Users\sangforsslvdi\Desktop\codetest\python\q1\output\final"       # 图片路径
OUTPUT_DIR = r"C:\Users\sangforsslvdi\Desktop\codetest\python\q1\output\2"     # 输出目录
import os
import random
import cv2
import numpy as np
num_defects = random.randint(4, 8)
# ================= 核心配置参数 =================

# 缺陷类别ID映射定义 (如需修改，请直接调整此处的字典值)
# 1: 开路 (Open)        - 缺铜 (黑色)
# 2: 短路 (Short)       - 多铜 (白色)
# 3: 缺口 (Mouse-bite)  - 缺铜 (黑色)
# 4: 凸铜 (Protrusion)  - 多铜 (白色)
# 5: 残铜 (Residual)    - 多铜 (白色)
# 6: 针孔 (Pinhole)     - 缺铜 (黑色)
DEFECT_IDS = {
    'Open': 1,
    'Short': 2,
    'Mouse_bite': 3,
    'Protrusion': 4,
    'Residual': 5,
    'Pinhole': 6
}

def generate_random_polygon(center_x, center_y, base_radius):
    """生成不规则多边形（用于非矩形缺陷）"""
    points = []
    num_points = random.randint(6, 10)
    for i in range(num_points):
        angle = i * (2 * np.pi / num_points) + random.uniform(-0.3, 0.3)
        r = base_radius * random.uniform(0.7, 1.3)
        pt_x = int(center_x + r * np.cos(angle))
        pt_y = int(center_y + r * np.sin(angle))
        points.append([pt_x, pt_y])
    return np.array(points, dtype=np.int32)

def process_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(os.path.join(OUTPUT_DIR,'pic')):
        os.makedirs(os.path.join(OUTPUT_DIR,'pic'))
    if not os.path.exists(os.path.join(OUTPUT_DIR,'txt')):
        os.makedirs(os.path.join(OUTPUT_DIR,'txt'))
    # 动态匹配常见图片格式
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    all_files = os.listdir(INPUT_DIR) if os.path.exists(INPUT_DIR) else []
    image_files = [f for f in all_files if f.startswith('t') and f.lower().endswith(valid_exts)]

    if not image_files:
        print(f"在目录 {INPUT_DIR} 下未找到匹配的以t开头的图片。")
        return

    for filename in image_files:
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if img.shape[:2] != (768, 768):
            img = cv2.resize(img, (768, 768), interpolation=cv2.INTER_NEAREST)

        out_img = img.copy()
        
        # 提前提取出白色、黑色以及边缘的坐标候选点，格式为 [y, x]
        white_pts = np.argwhere(img == 255)
        black_pts = np.argwhere(img == 0)
        edges = cv2.Canny(img, 50, 150)
        edge_pts = np.argwhere(edges > 0)

        def get_random_pt(pts_array):
            if len(pts_array) == 0:
                return None
            idx = random.randint(0, len(pts_array) - 1)
            # 返回 x, y
            return pts_array[idx][1], pts_array[idx][0]

        
        annotations = []

        # 随机打乱缺陷类别列表用于随机生成
        defect_types = list(DEFECT_IDS.keys())

        for _ in range(num_defects):
            defect_type = random.choice(defect_types)
            defect_id = DEFECT_IDS[defect_type]
            
            center = None
            defect_color = 0
            shape_type = 'poly'
            width, height, angle, radius = 0, 0, 0, 0

            # 根据形态逻辑设置参数
            if defect_type == 'Open':  # 矩形，切断铜线
                center = get_random_pt(white_pts)
                defect_color = 0
                shape_type = 'rect'
                width = random.randint(4, 12)
                height = random.randint(25, 60)
                angle = random.randint(0, 180)
            
            elif defect_type == 'Short':  # 矩形，连接铜线
                center = get_random_pt(black_pts)
                defect_color = 255
                shape_type = 'rect'
                width = random.randint(4, 12)
                height = random.randint(25, 60)
                angle = random.randint(0, 180)
            
            elif defect_type == 'Mouse_bite':  # 隐秘边缘缺铜
                center = get_random_pt(edge_pts)
                defect_color = 0
                radius = random.randint(3, 7)
            
            elif defect_type == 'Protrusion':  # 隐秘边缘凸铜
                center = get_random_pt(edge_pts)
                defect_color = 255
                radius = random.randint(3, 7)
            
            elif defect_type == 'Pinhole':  # 白区黑点，遵循概率大小
                center = get_random_pt(white_pts)
                defect_color = 0
                # 85%概率生成5-8像素的小点，15%概率生成10-25的大点
                radius = random.randint(5, 8) if random.random() < 0.85 else random.randint(10, 25)
            
            elif defect_type == 'Residual':  # 黑区白点，遵循概率大小
                center = get_random_pt(black_pts)
                defect_color = 255
                radius = random.randint(5, 8) if random.random() < 0.85 else random.randint(10, 25)

            if center is None:
                continue

            # 使用空白掩膜来绘制当前缺陷，这样能精准获取Bounding Box而不受图像边界干扰
            defect_mask = np.zeros((768, 768), dtype=np.uint8)
            
            if shape_type == 'rect':
                rect = ((float(center[0]), float(center[1])), (float(width), float(height)), float(angle))
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.fillPoly(defect_mask, [box], 255)
            else:
                pts = generate_random_polygon(center[0], center[1], radius)
                cv2.fillPoly(defect_mask, [pts], 255)

            # 寻找掩膜中的非零点以获取精准边框
            coords = cv2.findNonZero(defect_mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                xmin, ymin = x, y
                xmax, ymax = x + w, y + h
                
                # 将缺陷真实施加到输出图像上
                out_img[defect_mask == 255] = defect_color
                
                # 记录结果，格式: xmin ymin xmax ymax class_id
                annotations.append(f"{xmin} {ymin} {xmax} {ymax} {defect_id}")

        # 仅在实际生成了缺陷的情况下保存文件
        if annotations:
            base_name = os.path.splitext(filename)[0]
            out_img_path = os.path.join(os.path.join(OUTPUT_DIR,'pic'), filename)
            txt_path = os.path.join(os.path.join(OUTPUT_DIR,'txt'), f"{base_name}.txt")
            
            cv2.imwrite(out_img_path, out_img)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(annotations))
    import shutil

    # 将同一目录下以r开头的文件复制到图片保存目录
    for filename in os.listdir(INPUT_DIR):
        if filename.startswith('r'):
            src_file = os.path.join(INPUT_DIR, filename)
            if os.path.isfile(src_file):
                dst_file = os.path.join(os.path.join(OUTPUT_DIR,'pic'), filename)
                shutil.copy(src_file, dst_file)
    print("处理完毕。")

if __name__ == '__main__':
    process_images()