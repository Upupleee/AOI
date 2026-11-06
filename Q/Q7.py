import os
import cv2
import numpy as np
def q7():
    # ==================== 配置 ====================
    INPUT_DIR = r".\output\patches"
    OUTPUT_DIR = r".\output\bpatches"
    AREA_THRESHOLD = 150       # 面积筛选阈值（像素）
    THRESHOLD_OFFSET = 20      # 全局均值减去的偏移量
    CLOSING_RADIUS = 2.5       # 圆形闭运算半径

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def estimate_noise_variance(gray_img, win_size=20):
        """
        估计图像噪声方差（类似 Halcon 的 estimate_noise('foerstner')）。
        使用局部方差的中位数作为噪声方差估计。
        """
        h, w = gray_img.shape
        # 计算局部标准差（使用窗口 win_size）
        kernel = np.ones((win_size, win_size), dtype=np.float32) / (win_size * win_size)
        mean = cv2.filter2D(gray_img.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray_img.astype(np.float32) ** 2), -1, kernel)
        var = sqr_mean - mean ** 2
        var = np.clip(var, 0, None)  # 避免负值
        # 取所有局部方差的中位数作为全局噪声方差
        noise_var = np.median(var)
        return noise_var

    def adaptive_wiener_filter(gray_img, filter_size=5, noise_var=None):
        """
        自适应维纳滤波（对应 Halcon 的局部均值/方差 + 噪声方差修正）
        """
        gray = gray_img.astype(np.float32)
        # 局部均值
        kernel = np.ones((filter_size, filter_size), dtype=np.float32) / (filter_size * filter_size)
        mean = cv2.filter2D(gray, -1, kernel)
        # 局部方差
        sqr_mean = cv2.filter2D(gray ** 2, -1, kernel)
        var = sqr_mean - mean ** 2
        var = np.clip(var, 0, None)
        # 若未提供噪声方差，则估计
        if noise_var is None:
            noise_var = np.median(var)
        # 维纳滤波：输出 = mean + (var - noise_var) / var * (gray - mean)
        # 但为了防止除零，使用 var + eps
        eps = 1e-6
        ratio = (var - noise_var) / (var + eps)
        ratio = np.clip(ratio, 0, 1)  # 确保在 [0,1]
        filtered = mean + ratio * (gray - mean)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    def process_image(image_path, output_path):
        """
        处理单张图像，保存二值掩膜。
        """
        # 读取图像并灰度化
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取 {image_path}，跳过")
            return
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        h, w = gray.shape

        # ---------- 自适应降噪 ----------
        # 计算滤波器尺寸（对应 Halcon 的 FilterSize）
        filter_size = int(w * 0.005)
        if filter_size % 2 == 0:
            filter_size += 1
        filter_size = max(5, min(15, filter_size))

        # 估计噪声方差
        noise_var = estimate_noise_variance(gray, win_size=20)

        # 应用维纳滤波
        denoised = adaptive_wiener_filter(gray, filter_size, noise_var)

        # ---------- 阈值分割 ----------
        mean_val = np.mean(denoised)
        thresh_val = max(mean_val - THRESHOLD_OFFSET, 0)
        _, raw_copper = cv2.threshold(denoised, thresh_val, 255, cv2.THRESH_BINARY)

        # ---------- 连通域分析 + 面积筛选 ----------
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_copper, connectivity=8)
        filtered_mask = np.zeros_like(raw_copper, dtype=np.uint8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= AREA_THRESHOLD:
                filtered_mask[labels == i] = 255

        # ---------- 闭运算（圆形结构元素） ----------
        radius = int(CLOSING_RADIUS)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        closed = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)

        # ---------- 保存结果 ----------
        cv2.imwrite(output_path, closed)
        print(f"已处理: {os.path.basename(image_path)} -> {os.path.basename(output_path)}")

    # ==================== 主循环 ====================
    # 支持的图片扩展名
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith(extensions):
            continue
        if not 'trans' in filename:
            continue
        input_path = os.path.join(INPUT_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, base_name + '_mask.png')
        process_image(input_path, output_path)
    print("全部处理完成！")