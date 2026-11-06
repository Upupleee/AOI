import os
import numpy as np
from PIL import Image
def q6():
    Image.MAX_IMAGE_PIXELS = None   # 解除大图限制

    # ========== 路径配置（请按实际情况修改） ==========
    TEST_CROPPED_PATH = r".\output\test_cropped.tif"   # 2.py 输出的测试图（已裁上下）
    REF_IMAGE_PATH   = r"D:\dataset\0407\0407\p016p4e019723a0\gbl.bmp"           # 原始参考图
    WARP_MATRIX_PATH = r".\output\warp_matrix.npy"                                         # 4.py 生成的仿射矩阵
    TRANSFORMED_PATH = r".\output\transformed_test.tif"   # 由前面脚本生成的变换后的整张测试图

    OUTPUT_DIR = r".\output\patches"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    PATCH_SIZE = 6400
    MAX_PATCHES = 1e12   # 快速验证只处理前10个

    # ========== 读取尺寸与矩阵 ==========
    with Image.open(TEST_CROPPED_PATH) as im:
        w_test, h_test = im.size

    with Image.open(REF_IMAGE_PATH) as im:
        w_ref, h_ref = im.size

    warp = np.load(WARP_MATRIX_PATH)   # shape (2,3)
    M_lin = warp[:, :2]
    M_trans = warp[:, 2]

    # 图1（测试图小块）在测试图中的位置（右上角）
    offset_test = np.array([w_test - PATCH_SIZE, 0], dtype=np.float32)
    # 图2（参考图小块）在参考图中的位置（右上角）
    offset_ref = np.array([w_ref  - PATCH_SIZE, 0], dtype=np.float32)

    # 修正平移量，得到将整张测试图映射到参考图坐标系的仿射矩阵
    new_trans = M_trans - M_lin @ offset_test + offset_ref
    full_M = np.hstack((M_lin, new_trans.reshape(-1, 1)))   # 2x3

    # ========== 计算网格数量 ==========
    rows = h_test // PATCH_SIZE
    cols = w_test // PATCH_SIZE
    print(f"测试图尺寸: {w_test} x {h_test}  --> 可完整划分 {rows} 行 × {cols} 列")

    if rows == 0 or cols == 0:
        print("没有完整的 4096 块，退出。")
        exit()

    # ========== 加载两张大图 ==========
    print("加载变换后的测试图...")
    trans_img = Image.open(TRANSFORMED_PATH)
    print("加载参考图...")
    ref_img = Image.open(REF_IMAGE_PATH)

    # ========== 开始裁剪 ==========
    saved = 0
    for i in range(rows):
        for j in range(cols):
            if saved >= MAX_PATCHES:
                break
            # 测试图中网格的左上角坐标
            x_test = j * PATCH_SIZE
            y_test = i * PATCH_SIZE

            # 映射到参考图坐标
            pt = np.array([x_test, y_test, 1.0], dtype=np.float32)
            x_ref, y_ref = (full_M @ pt).astype(int)
            margin = 64
            x_start = x_ref - margin
            y_start = y_ref - margin
            x_end = x_ref + PATCH_SIZE + margin
            y_end = y_ref + PATCH_SIZE + margin
            box = (x_start, y_start, x_end, y_end)
            # 检查裁剪框是否完全在参考图内
            if (x_start < 0 or y_start < 0 or
                x_end > w_ref or
                y_end > h_ref):
                print(f"网格 ({i},{j}) 映射到 ({x_ref},{y_ref}) 超出参考图，跳过")
                continue

            # box = (x_ref, y_ref, x_ref + PATCH_SIZE, y_ref + PATCH_SIZE)

            # 从变换后的测试图中裁剪
            patch_trans = trans_img.crop(box)
            patch_trans.save(os.path.join(OUTPUT_DIR, f"trans_patch_{i}_{j}.png"))

            # 从参考图中裁剪（相同位置）
            patch_ref = ref_img.crop(box)
            patch_ref.save(os.path.join(OUTPUT_DIR, f"ref_patch_{i}_{j}.png"))

            print(f"已保存第 {saved+1} 对: 网格 ({i},{j}) -> 参考坐标 ({x_ref}, {y_ref})")
            saved += 1
        if saved >= MAX_PATCHES:
            break

    # ========== 释放内存 ==========
    del trans_img, ref_img
    print(f"完成！共裁剪 {saved} 对子图，保存在 {OUTPUT_DIR} 目录下。")