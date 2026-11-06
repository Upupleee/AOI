import cv2
import numpy as np
import os

def q4():
    # 参数定义区
    REF_IMAGE_PATH = r".\output/ref_crop_rightmost.png"
    TEST_IMAGE_PATH = r".\output/test_crop_rightmost.png"
    OUTPUT_IMAGE_PATH = r".\output/overlay_result.png"
    NPY_SAVE_PATH=r'.\output\warp_matrix.npy'
    import cv2
    import numpy as np
    import time
    # 得到变换矩阵

    # 粗配准参数
    DOWNSAMPLE_RATIO = 0.25               # 降采样比例 (缩小到 1/4)
    RANSAC_REPROJ_THRESHOLD = 3.0         # 粗配准 RANSAC 阈值

    def coarse_to_fine_align():
        # 1. 读取图像
        img_ref = cv2.imread(REF_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        img_test = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

        if img_ref is None or img_test is None:
            print("无法读取图像，请检查文件路径。")
            return
        h, w = img_test.shape
        start_time = time.perf_counter()

        # ==========================================
        # 降采样粗配准 (获取包含“拉伸”的初始矩阵)
        # ==========================================
        small_ref = cv2.resize(img_ref, (0, 0), fx=DOWNSAMPLE_RATIO, fy=DOWNSAMPLE_RATIO, interpolation=cv2.INTER_AREA)
        small_test = cv2.resize(img_test, (0, 0), fx=DOWNSAMPLE_RATIO, fy=DOWNSAMPLE_RATIO, interpolation=cv2.INTER_AREA)

        sift = cv2.SIFT_create()
        kp_ref, des_ref = sift.detectAndCompute(small_ref, None)
        kp_test, des_test = sift.detectAndCompute(small_test, None)

        if des_ref is None or des_test is None:
            print("特征提取失败。")
            return

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des_test, des_ref, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 3:
            print("匹配点过少，无法计算初始矩阵。")
            return

        src_pts = np.float32([kp_test[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 使用 estimateAffine2D 代替 Partial2D，以允许 X/Y 轴独立缩放（拉伸）和剪切
        matrix_small, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD)

        if matrix_small is None:
            print("无法计算粗略仿射变换矩阵。")
            return

        # 放大矩阵回原图尺寸
        # 线性部分（旋转、拉伸、剪切）保持不变，仅平移量需要除以降采样系数放大
        warp_matrix = matrix_small.copy()
        warp_matrix[0, 2] /= DOWNSAMPLE_RATIO
        warp_matrix[1, 2] /= DOWNSAMPLE_RATIO

        # ==========================================
        # 应用最终矩阵并合成
        # ==========================================
        
        img_test_aligned = cv2.warpAffine(img_test, warp_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        composite_img = np.zeros((h, w, 3), dtype=np.uint8)
        composite_img[:, :, 0] = img_ref          # 蓝：参考图
        composite_img[:, :, 1] = img_test_aligned # 绿：对齐后的测试图

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000

        cv2.imwrite(OUTPUT_IMAGE_PATH, composite_img)
        # 在应用最终矩阵前，保存矩阵到文件
        np.save(NPY_SAVE_PATH, warp_matrix)
        print("变换矩阵已保存至 warp_matrix.npy")
        print(f"粗细结合配准完毕，耗时: {elapsed_ms:.2f} ms")
        print(f"结果已保存至: {OUTPUT_IMAGE_PATH}")

    coarse_to_fine_align()