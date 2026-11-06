"""基于 LoFTR + LightGlue 的模板匹配实现。"""
import logging
import re

import numpy as np
from PIL import Image

from AoiSystemWindow_test.ModelManger import ModelManager
from TemplateMatching.EfficientLoFTR.fine_matching_loftr_new_drew_test import (
    fine_matching_pillow_on_template_drew_return_bgr,
)
from core.interfaces.i_template_matcher import ITemplateMatcher, MatchResult

logger = logging.getLogger(__name__)


class LoFtrTemplateMatcher(ITemplateMatcher):
    """封装 LoFTR + LightGlue 的模板匹配实现。

    参考 WorkerThread.py 中 MatchAndDetectRunnable.template_matching_drew 的逻辑:
    根据样本路径编号选择对应模板,做线性灰度调整后调用
    fine_matching_pillow_on_template_drew_return_bgr 完成精匹配,
    返回叠加结果图像及文件名信息。
    """

    def __init__(self):
        """初始化模板匹配器,持有 ModelManager 单例。"""
        self._model_manager = ModelManager()

    def match(self, sample_image_path: str,
              template_image_paths: list,
              params: dict):
        """执行模板匹配,返回匹配结果。

        Args:
            sample_image_path: 样本图像路径。
            template_image_paths: 模板图像路径列表。
            params: 匹配参数字典,需包含 grid_size、max_sampling_per_grid、
                enhance_algorithms、enhance_param、extractor_name 等字段。

        Returns:
            MatchResult,失败返回 None。
        """
        logger.info("开始模板匹配,样本=%s", sample_image_path)

        # 1. 从样本路径提取编号
        number = self._extract_number_from_path(sample_image_path)
        if not number:
            logger.error("无法从路径提取编号: %s", sample_image_path)
            return None

        # 2. 根据编号选择样本与模板图像
        if len(template_image_paths) == number:
            # 编号等于模板数量:处理最后一张,需基于倒数第二张做比例裁剪
            sample_img, template_img = self._process_images(
                template_files=template_image_paths,
                sample_image_path=sample_image_path,
            )
            logger.info("处理最后一张样本,采样路径=%s", sample_image_path)
        else:
            # 直接打开对应模板图
            sample_img = Image.open(sample_image_path).convert('L')
            template_img = Image.open(template_image_paths[number - 1]).convert('L')
            logger.info("采样路径=%s,模板路径=%s",
                        sample_image_path, template_image_paths[number - 1])

        # 3. 提取文件名(不含扩展名)
        img_name = self._get_tif_filename(template_image_paths, number)
        logger.info("文件名=%s", img_name)

        # 4. 线性灰度调整
        template_arry, sample_arry = self._linear_adjust_template(template_img, sample_img)

        # 5. 提取文件名前缀作为 file_name
        file_name = self._extract_prefix(img_name)

        # 6. 调用精匹配函数
        logger.info("开始精匹配,file_name=%s", file_name)
        result_overlay_image = fine_matching_pillow_on_template_drew_return_bgr(
            sample_arry,
            template_arry,
            grid_size=params.get('grid_size'),
            max_sampling_per_grid=params.get('max_sampling_per_grid'),
            enhance_algorithms=params.get('enhance_algorithms'),
            enhance_param=params.get('enhance_param'),
            extractor_name=params.get('extractor_name'),
            model=self._model_manager,
        )
        logger.info("模板匹配完成,file_name=%s", file_name)

        return MatchResult(
            overlay_image=result_overlay_image,
            file_name=file_name,
            img_name=img_name,
        )

    def _extract_number_from_path(self, path):
        """从文件路径中提取末尾数字编号。

        例如 "sample_3.tif" 提取出 3。
        """
        pattern = r'_?(\d+)\.\w+$'
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
        return None

    def _get_tif_filename(self, image_path_list, number):
        """从模板路径列表中取出第 number 个路径的文件名(不含扩展名)。

        Args:
            image_path_list: 模板图像路径列表。
            number: 1-based 编号。

        Returns:
            不含扩展名的文件名。
        """
        if 1 <= number <= len(image_path_list):
            path = image_path_list[number - 1].strip()
            if path and path.lower().endswith(('.tif', '.tiff')):
                filename = path.split("/")[-1]
                if filename.lower().endswith(".tiff"):
                    filename_without_ext = filename[:-5]
                else:
                    filename_without_ext = filename[:-4]
                return filename_without_ext
        return ""

    def _process_images(self, template_files, sample_image_path):
        """处理最后一张样本:基于倒数第二张模板宽度比例裁剪样本。

        Args:
            template_files: 模板图像路径列表。
            sample_image_path: 样本图像路径。

        Returns:
            (裁剪后的样本 PIL 图像, 最后一张模板 PIL 灰度图像)
        """
        Image.MAX_IMAGE_PIXELS = None

        if len(template_files) < 2:
            raise ValueError("模板文件数量不足两张,请检查模板文件夹。")

        template_second_last = template_files[-2]
        template_last = template_files[-1]

        img_second_last = Image.open(template_second_last).convert('L')
        img_last = Image.open(template_last).convert('L')

        # 计算最后两张模板的宽度比例
        ratio = img_last.width / img_second_last.width

        sample_img = Image.open(sample_image_path)
        sample_width, sample_height = sample_img.width, sample_img.height

        crop_width = int(sample_width * ratio)
        if crop_width > sample_width:
            logger.warning("计算得到的保留宽度超过样本总宽度,直接保留整个图片")
            crop_width = sample_width

        # 从右侧裁剪保留指定宽度
        left = sample_width - crop_width
        cropped_img = sample_img.crop((left, 0, sample_width, sample_height))

        return cropped_img, img_last

    def _linear_adjust_template(self, template_img, captured_img):
        """对模板图像做线性灰度调整,使其与采集图像的亮度分布一致。

        调整公式: adjusted = (template - mu_t) / sigma_t * sigma_c + mu_c
        并对采集图像裁掉顶部 9070 像素(参考 WorkerThread 实现)。

        Args:
            template_img: 模板图像(PIL 灰度)。
            captured_img: 采集图像(PIL)。

        Returns:
            (调整后的模板 numpy 数组 uint8, 裁剪后的采集图像 numpy 数组 uint8)
        """
        w, h = captured_img.size
        # 裁掉采集图像顶部区域,与现有实现保持一致
        captured_img_crop = captured_img.crop((0, 9070, w, h))

        template_array = np.array(template_img).astype(np.float32)
        captured_array = np.array(captured_img_crop).astype(np.float32)

        mu_template = template_array.mean()
        sigma_template = template_array.std()
        mu_captured = captured_array.mean()
        sigma_captured = captured_array.std()

        adjusted = (template_array - mu_template) / sigma_template * sigma_captured + mu_captured
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        captured_array = captured_array.astype(np.uint8)

        return adjusted, captured_array

    def _extract_prefix(self, filename, delimiter='_'):
        """提取文件名中最后一个分隔符之前的部分。

        例如 "p503_2" 提取出 "p503"。
        """
        last_delimiter_index = filename.rfind(delimiter)
        if last_delimiter_index != -1:
            return filename[:last_delimiter_index]
        return filename
