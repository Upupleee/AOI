import os
from PIL import Image
def q9(out_path):
    # ==================== 配置 ====================
    INPUT_DIR = r".\output\bpatches"      # 输入图片所在文件夹
    # OUTPUT_DIR = r".\output\final\test"    # 输出子图的根目录（会自动创建子文件夹）
    OUTPUT_DIR = out_path

    STEP = 640                     # 相邻块起始点的间隔（步长）
    PATCH_SIZE = 640 + 64 * 2      # 实际裁剪的块大小（含掩码）
    GRID = 10                      # 每边划分的块数（10×10）

    # ==================== 主函数 ====================
    def split_images(input_dir, output_dir, step=STEP, patch_size=PATCH_SIZE, grid=GRID):
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        # 支持的图片扩展名（根据实际情况可增减）
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        # 遍历输入目录
        for filename in os.listdir(input_dir):
            if not filename.lower().endswith(extensions):
                continue

            filepath = os.path.join(input_dir, filename)
            try:
                img = Image.open(filepath)
            except Exception as e:
                print(f"无法读取 {filepath}，跳过。错误：{e}")
                continue

            width, height = img.size
            # 检查尺寸是否至少能容纳 grid 个块（每个块起始间隔为 step，最后一个块结束于起始+patch_size）
            required_width = step * (grid - 1) + patch_size
            required_height = step * (grid - 1) + patch_size
            if width < required_width or height < required_height:
                print(f"警告：{filename} 尺寸为 {width}x{height}，不足以拆分为 {grid}x{grid} 个 {patch_size}x{patch_size} 的块（步长{step}），跳过。")
                continue

            # 去掉扩展名，用于命名
            base_name = os.path.splitext(filename)[0]

            # 循环切割
            for row in range(grid):
                for col in range(grid):
                    left = col * step
                    top = row * step
                    right = left + patch_size
                    bottom = top + patch_size

                    # 裁剪并保存
                    box = (left, top, right, bottom)
                    patch = img.crop(box)
                    patch_name = os.path.join(output_dir, f"{base_name}_{row:02d}_{col:02d}.png")
                    patch.save(patch_name)

            print(f"已处理 {filename}，拆分为 {grid*grid} 个子图。")

        print("所有图片拆分完成！")

    # ==================== 执行 ====================
    split_images(INPUT_DIR, OUTPUT_DIR)