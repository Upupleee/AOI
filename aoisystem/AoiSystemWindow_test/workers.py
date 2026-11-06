"""图片处理工作线程。

从 AoiSystemMain.py 拆分而来，负责批量切分图片、保存靶点数据等后台处理。
"""
import os
import gc
import json

from PIL import Image
from PyQt5.QtCore import QObject, pyqtSignal, QDateTime, Qt

from common.config import config
from common.utils import ensure_dir


class ImageProcessingWorker(QObject):
    """图片处理工作对象（使用 QObject 实现，配合 moveToThread）。"""
    progress_updated = pyqtSignal(int, str)  # 进度值, 处理中的图片名称
    processing_completed = pyqtSignal(dict)
    processing_failed = pyqtSignal(str, str)  # 错误信息, 图片名称
    image_loading = pyqtSignal(str)  # 图片加载进度信号

    def __init__(self, images_and_sides, material_id, image_data):
        super().__init__()
        self.images_and_sides = images_and_sides  # 格式: [(path, side), ...]
        self.material_path = os.path.join(config.material_root, material_id)  # 完整料号路径
        self.image_data = image_data
        self.running = False  # 运行状态标志（用于安全停止）

    def start_processing(self):
        """批量处理入口（确保所有任务完成后再发送信号）。"""
        self.running = True
        results = {"success": [], "failed": []}
        total = len(self.images_and_sides)

        try:
            self._create_necessary_dirs()

            # 循环处理所有图片（确保全部完成后再结束）
            for i, (image_path, side) in enumerate(self.images_and_sides):
                if not self.running:
                    break  # 若被停止，直接退出

                try:
                    filename = os.path.basename(image_path)
                    self.image_loading.emit(f"开始加载图片: {filename}")

                    with Image.open(image_path) as img:
                        self.image_loading.emit(f"图片加载完成，开始处理: {filename}")
                        saved_paths, slice_widths, Height = self._slice_image(img, filename, side)
                        # 保存靶点数据到处理结果（只处理A面）
                        target_data = self._get_target_data(image_path) if side == "A" else []
                        if side == "A" and target_data:
                            self._save_targets_to_file(filename, side, target_data)
                    del img
                    results["success"].append({
                        "original": image_path, "processed": saved_paths,
                        "side": side, "slice_widths": slice_widths,
                        "temp_height": Height,
                        "targets": target_data  # 包含靶点数据（仅A面）
                    })

                except Exception as e:
                    results["failed"].append({"path": image_path, "error": str(e), "side": side})

                finally:
                    # 发送进度更新（确保线程仍在运行）
                    if self.running:
                        progress = int((i + 1) / total * 100)
                        self.progress_updated.emit(progress, filename)

                    gc.collect(generation=0)

            # 所有任务完成后才发送完成信号
            if self.running:
                self.processing_completed.emit(results)

        except Exception as e:
            self.processing_failed.emit(str(e), "批量处理")

    def stop_processing(self):
        """安全停止处理过程。"""
        self.running = False

    def _get_target_data(self, image_path):
        """获取图片的靶点数据。"""
        for info in self.image_data:
            if info["path"] == image_path:
                return info["targets"]
        return []

    def _save_targets_to_file(self, filename, side, targets):
        """将靶点数据保存到文件。"""
        if not targets:
            return

        # 创建靶点数据目录
        targets_dir = os.path.join(self.material_path, "targets")
        ensure_dir(targets_dir)

        # 保存靶点数据
        base_name = os.path.splitext(filename)[0]
        target_path = os.path.join(targets_dir, f"{base_name}_{side}_targets.json")

        try:
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "filename": filename,
                    "side": side,
                    "targets": targets,
                    "timestamp": QDateTime.currentDateTime().toString(Qt.ISODate)
                }, f, ensure_ascii=False, indent=2)
            os.chmod(target_path, 0o777)
        except Exception as e:
            print(f"保存靶点数据失败: {str(e)}")

    def _create_necessary_dirs(self):
        """创建必要的目录结构。"""
        # 主目录
        ensure_dir(self.material_path)

        # 缩略图目录
        thumb_dir = os.path.join(self.material_path, "thumbnails")
        ensure_dir(thumb_dir)

    def _slice_image(self, img, img_filename, side):
        """切分图片为条带并保存。"""
        # 禁用解压炸弹检查
        Image.MAX_IMAGE_PIXELS = None
        saved_paths = []
        slice_widths = []

        # 转灰度
        if img.mode != 'L':
            img = img.convert('L')

        # 读取参数（从配置加载）
        dpi = config.slicing_dpi
        step_mm = config.slicing_step_mm
        margin_mm = config.slicing_margin_mm
        seg_mm = config.slicing_segment_mm

        # 计算像素／毫米
        px_per_mm = dpi / 25.4
        first_px = int(round((seg_mm - margin_mm) * px_per_mm))
        step_px = int(round(step_mm * px_per_mm))

        # 图像尺寸
        W, H = img.size

        # 创建侧面目录
        side_dir = os.path.join(self.material_path, side)
        ensure_dir(side_dir)

        base_name = os.path.splitext(img_filename)[0]
        x_right = W
        x_left = max(0, W - first_px)
        i = 0

        # 第一个条带（右侧首段）
        while (x_right > 0) and getattr(self, 'running', True):
            current_width = x_right - x_left
            slice_widths.append(current_width)
            # 裁剪并保存
            box = (x_left, 0, x_right, H)
            slice_img = img.crop(box)
            slice_path = os.path.join(side_dir, f"{base_name}_{i+1}.tif")
            slice_img.save(slice_path)
            saved_paths.append(slice_path)
            del slice_img
            gc.collect(generation=0)

            # 更新边界，之后按 step_px 切分
            x_right = x_left
            x_left = max(0, x_left - step_px)
            i += 1

        return saved_paths, slice_widths, H
