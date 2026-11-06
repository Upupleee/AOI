import cv2
import numpy as np
import os
from PIL import Image
def q5():
    # 解除 Pillow 对大图的像素限制
    Image.MAX_IMAGE_PIXELS = None
    # ==================== 路径配置 ====================
    TEST_CROPPED_PATH = r".\output\test_cropped.tif"   # 2.py 输出
    REF_IMAGE_PATH   = r"D:\dataset\0407\0407\p016p4e019723a0\gbl.bmp"           # 原始参考图
    WARP_MATRIX_PATH = r".\output\warp_matrix.npy"                                         # 4.py 生成的矩阵

    OUTPUT_TRANSFORMED = r".\output\transformed_test.tif"   # 变换后的整张测试图
    OUTPUT_CROPPED     = r".\output\figure3.png"            # 最终裁剪出的图3

    CROP_SIZE = 8192  # 图1/图2的宽高（固定）

    # ==================== 读取数据 ====================
    # 1. 变换矩阵
    warp_matrix = np.load(WARP_MATRIX_PATH)  # shape (2, 3)


    # 若 OpenCV 无法读取，用 PIL 兜底
    pil_img = Image.open(TEST_CROPPED_PATH).convert('L')
    test_img = np.array(pil_img, dtype=np.uint8)

    # 3. 参考图（仅用于获取尺寸）

    pil_ref = Image.open(REF_IMAGE_PATH)
    ref_w, ref_h = pil_ref.size


    test_h, test_w = test_img.shape

    # ==================== 计算偏移 ====================
    offset_test = np.array([test_w - CROP_SIZE, 0], dtype=np.float32)   # 图1在测试图中的左上角坐标
    offset_ref  = np.array([ref_w  - CROP_SIZE, 0], dtype=np.float32)   # 图2在参考图中的左上角坐标

    # 线性部分 (2x2) 和平移部分 (2x1)
    M_linear = warp_matrix[:, :2]
    M_trans = warp_matrix[:, 2]

    # 新平移量 = M_trans - M_linear·offset_test + offset_ref
    new_trans = M_trans - M_linear @ offset_test + offset_ref

    # 构建新的 2x3 仿射矩阵（应用于整张测试图）
    new_M = np.hstack((M_linear, new_trans.reshape(-1, 1)))

    # ==================== 应用变换 ====================
    transformed = cv2.warpAffine(
        test_img,
        new_M,
        (ref_w, ref_h),                  # 输出尺寸与参考图一致
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    cv2.imwrite(OUTPUT_TRANSFORMED, transformed)
    print(f"变换后的整图已保存至: {OUTPUT_TRANSFORMED}")

    # ==================== 裁剪右上角区域 ====================
    crop_x = ref_w - CROP_SIZE
    crop_y = 0
    cropped = transformed[crop_y:crop_y + CROP_SIZE, crop_x:crop_x + CROP_SIZE]

    cv2.imwrite(OUTPUT_CROPPED, cropped)
    print(f"图3（对齐后的右上角小块）已保存至: {OUTPUT_CROPPED}")
    print(f"图3尺寸: {cropped.shape[1]} x {cropped.shape[0]}")