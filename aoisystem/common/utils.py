"""共享工具函数。

消除项目中重复实现的工具函数：路径解析、缩略图生成、JSON 读写、目录创建等。
"""
import os
import re
import json
import logging

logger = logging.getLogger(__name__)

# 图片扩展名集合
IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def extract_number_from_path(path):
    """从文件路径中提取末尾的数字。

    例如 "images/test_imgs3.tif" -> 3, "data/abc_12.png" -> 12。
    无匹配时返回 None。
    """
    pattern = r"_?(\d+)\.\w+$"
    match = re.search(pattern, path)
    if match:
        return int(match.group(1))
    return None


def generate_thumbnail(img, original_filename, thumbnails_dir,
                       scale_factor=None, jpeg_quality=None):
    """生成图片缩略图并保存到指定目录。

    参数:
        img: PIL.Image 已打开的图片对象
        original_filename: 原始文件名
        thumbnails_dir: 缩略图保存目录
        scale_factor: 缩放因子（默认读配置 8）
        jpeg_quality: JPEG 质量（默认读配置 80）

    返回:
        缩略图文件路径，失败返回 None
    """
    try:
        from PIL import Image as PILImage
        from .config import config

        if scale_factor is None:
            scale_factor = config.thumbnail_scale_factor
        if jpeg_quality is None:
            jpeg_quality = config.thumbnail_jpeg_quality

        width, height = img.size
        new_width = max(1, width // scale_factor)
        new_height = max(1, height // scale_factor)

        resized_img = img.resize((new_width, new_height), PILImage.LANCZOS)

        base_name, _ = os.path.splitext(original_filename)
        thumbnail_filename = f"{base_name}_thumb.jpg"
        thumbnail_path = os.path.join(thumbnails_dir, thumbnail_filename)

        if resized_img.mode != "RGB":
            resized_img = resized_img.convert("RGB")
        resized_img.save(thumbnail_path, "JPEG", quality=jpeg_quality)

        return thumbnail_path
    except Exception as e:
        logger.error("生成缩略图失败: %s", e)
        return None


def load_json_file(file_path, default=None):
    """加载 JSON 文件，失败时返回 default。"""
    if default is None:
        default = {}
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("加载 JSON 失败 %s: %s", file_path, e)
    return default


def save_json_file(file_path, data):
    """保存数据到 JSON 文件。"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except OSError as e:
        logger.warning("保存 JSON 失败 %s: %s", file_path, e)
        return False


def ensure_dir(path, mode=0o777):
    """确保目录存在，不存在则创建。"""
    try:
        os.makedirs(path, exist_ok=True)
        if mode is not None:
            os.chmod(path, mode)
    except OSError as e:
        logger.warning("创建目录失败 %s: %s", path, e)
    return path


def list_image_files(folder_path):
    """列出目录下所有图片文件（按文件名排序）。"""
    if not os.path.exists(folder_path):
        return []
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(IMAGE_EXTENSIONS)]
    files.sort()
    return [os.path.join(folder_path, f) for f in files]
