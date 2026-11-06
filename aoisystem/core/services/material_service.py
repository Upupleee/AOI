"""料号业务服务。

将 MyWindow 中料号相关的业务逻辑(CRUD、图片导入、切分)抽出为纯 Python 服务,
不依赖 PyQt5。服务通过 IMaterialRepository 接口访问数据,
通过 config 单例访问配置。
"""
import os
import shutil
import logging
from typing import Optional

from PIL import Image

from common.utils import ensure_dir, generate_thumbnail, list_image_files
from core.interfaces.i_repository import IMaterialRepository
from core.models.material import Material

logger = logging.getLogger(__name__)


class MaterialService:
    """料号业务服务,封装料号 CRUD、图片导入、切分等业务逻辑。"""

    def __init__(self, material_repo: IMaterialRepository, config):
        """构造函数注入仓储与配置。

        Args:
            material_repo: 料号数据仓储接口。
            config: 配置单例,提供路径与切分参数。
        """
        self.material_repo = material_repo
        self.config = config

    def get_all_materials(self) -> dict:
        """返回所有料号(字典格式,兼容旧 UI)。

        返回: {material_id: material.to_dict(), ...}
        """
        return self.material_repo.get_all()

    def get_material(self, material_id: str) -> Optional[Material]:
        """按 ID 获取料号。"""
        return self.material_repo.get_by_id(material_id)

    def save_material(self, material: Material):
        """保存料号。"""
        self.material_repo.save(material)

    def delete_material(self, material_id: str):
        """删除料号(同时删除实际文件夹)。"""
        # 删除实际文件夹
        full_material_path = os.path.join(self.config.material_root, material_id)
        try:
            if os.path.exists(full_material_path):
                shutil.rmtree(full_material_path)
                logger.info("已删除料号文件夹: %s", full_material_path)
        except OSError as e:
            logger.warning("删除料号文件夹失败 %s: %s", full_material_path, e)
        # 从仓储中删除记录
        self.material_repo.delete(material_id)

    def import_material_from_images(self, images_and_sides: list,
                                    material_id: str) -> Material:
        """导入图片到料号目录。

        Args:
            images_and_sides: 格式 [(path, side), ...],side 为 "A" 或 "B"。
            material_id: 料号名称(不含 material 根目录)。

        Returns:
            创建的 Material 对象。
        """
        # 创建料号根目录与缩略图目录
        full_material_path = os.path.join(self.config.material_root, material_id)
        ensure_dir(full_material_path)
        thumbnails_dir = os.path.join(full_material_path, "thumbnails")
        ensure_dir(thumbnails_dir)

        material = Material(material_id=material_id)
        a_thumbnail = None
        b_thumbnail = None

        for image_path, side in images_and_sides:
            if not os.path.exists(image_path):
                logger.warning("图片不存在,跳过: %s", image_path)
                continue

            # 创建侧面目录
            side_dir = os.path.join(full_material_path, side)
            ensure_dir(side_dir)

            # 复制原图到料号目录
            filename = os.path.basename(image_path)
            dest_path = os.path.join(side_dir, filename)
            try:
                shutil.copy2(image_path, dest_path)
            except OSError as e:
                logger.warning("复制图片失败 %s: %s", image_path, e)
                continue

            # 设置料号的图片路径(只记录第一张)
            if side == "A" and not material.a_image_path:
                material.a_image_path = dest_path
            elif side == "B" and not material.b_image_path:
                material.b_image_path = dest_path

            # 生成缩略图
            try:
                with Image.open(image_path) as img:
                    thumbnail = generate_thumbnail(img, filename, thumbnails_dir)
                    if side == "A" and not a_thumbnail:
                        a_thumbnail = thumbnail
                    elif side == "B" and not b_thumbnail:
                        b_thumbnail = thumbnail
            except Exception as e:
                logger.warning("生成缩略图失败 %s: %s", image_path, e)

        material.a_thumbnail = a_thumbnail
        material.b_thumbnail = b_thumbnail

        # 保存到仓储
        self.material_repo.save(material)
        logger.info("料号导入成功: %s", material_id)
        return material

    def get_slice_paths(self, material_id: str, side: str) -> list:
        """获取指定面的切分后图片路径列表。

        如果 material_data 中有缓存直接返回,否则扫描文件系统。

        Args:
            material_id: 料号名称。
            side: "A" 或 "B"。

        Returns:
            切片图片路径列表(按文件名排序)。
        """
        material = self.material_repo.get_by_id(material_id)
        if not material:
            logger.warning("料号不存在: %s", material_id)
            return []

        # 检查缓存
        if side == "A" and material.a_slices:
            return material.a_slices
        if side == "B" and material.b_slices:
            return material.b_slices

        # 扫描文件系统
        full_material_path = os.path.join(self.config.material_root, material_id)
        side_folder = os.path.join(full_material_path, side)
        slice_paths = list_image_files(side_folder)

        if not slice_paths:
            logger.info("未找到切分图片 %s/%s", material_id, side)
            return []

        # 更新料号缓存并保存
        if side == "A":
            material.a_slices = slice_paths
        elif side == "B":
            material.b_slices = slice_paths
        self.material_repo.save(material)

        return slice_paths

    def slice_image(self, img_path: str, target_folder: str,
                    segment_mm: float = 127, step_mm: float = 113.3,
                    start_mm: float = 104.7, offset_mm: float = 0) -> list:
        """切分图片(按 mm 计算像素位置裁剪)。

        参考 WorkerThread.py 中 ImageProcessingThread.slice_image 的逻辑。

        Args:
            img_path: 原始图片路径。
            target_folder: 切片保存目录。
            segment_mm: 段长度(mm),用于计算像素/毫米。
            step_mm: 步进长度(mm)。
            start_mm: 起始位置(mm)。
            offset_mm: 偏移量(mm)。

        Returns:
            保存的切片路径列表。
        """
        # 禁用解压炸弹检查
        Image.MAX_IMAGE_PIXELS = None
        ensure_dir(target_folder)
        saved_file_paths = []

        # 从路径提取文件名(不含扩展名)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        try:
            with Image.open(img_path) as img:
                if img.mode != 'L':
                    img = img.convert('L')

                pixels_per_mm = 16384 / segment_mm  # 每mm对应的像素数
                W, H = img.size
                x_left = W
                i = -1

                while x_left != 0:
                    right_mm = (start_mm - offset_mm) + i * step_mm
                    left_mm = start_mm + (i + 1) * step_mm

                    if right_mm < 0:
                        right_mm = 0
                    x_right = int(W - right_mm * pixels_per_mm)
                    x_left = int(W - left_mm * pixels_per_mm)
                    if x_left < 0:
                        x_left = 0

                    box = (x_left, 0, x_right, H)
                    slice_img = img.crop(box)

                    # 文件名保持与原逻辑一致
                    slice_filename = f"{img_name}_{i+1}.tif"
                    slice_path = os.path.join(target_folder, slice_filename)
                    slice_img.save(slice_path)
                    saved_file_paths.append(slice_path)

                    i += 1
        except FileNotFoundError:
            logger.error("切分图片失败,文件未找到: %s", img_path)
        except Exception as e:
            logger.error("切分图片发生错误 %s: %s", img_path, e)

        return saved_file_paths
