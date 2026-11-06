"""缺陷检测业务服务。

将 WorkerThread.py 中 MatchAndDetectRunnable 的协调逻辑抽出为纯 Python 服务,
不依赖 PyQt5。服务通过 IDetector 与 ITemplateMatcher 接口访问硬件能力,
通过 config 单例访问配置。
"""
import os
import logging
from pathlib import Path

import cv2
import numpy as np

from core.interfaces.i_detector import IDetector
from core.interfaces.i_template_matcher import ITemplateMatcher
from core.models.detection_params import DetectionParams

logger = logging.getLogger(__name__)


class DetectionService:
    """缺陷检测业务服务,封装模板匹配 + 缺陷检测的协调流程。"""

    def __init__(self, detector: IDetector,
                 template_matcher: ITemplateMatcher, config):
        """构造函数注入检测器、模板匹配器与配置。

        Args:
            detector: 缺陷检测器接口。
            template_matcher: 模板匹配器接口。
            config: 配置单例。
        """
        self.detector = detector
        self.template_matcher = template_matcher
        self.config = config

    def detect_single(self, sample_path: str, template_paths: list,
                      params: DetectionParams) -> dict:
        """执行单张图片的模板匹配 + 缺陷检测流程。

        流程:
            1. 调用 template_matcher.match() 获取匹配结果
            2. 如果匹配成功,调用 detector.load_model() 加载模型
            3. 调用 detector.detect() 执行检测
            4. 保存结果到 prediction_visual 目录
            5. 返回结果字典

        Args:
            sample_path: 待检测样本图片路径。
            template_paths: 模板图片路径列表。
            params: 检测参数。

        Returns:
            {'coco_json_path': ..., 'project_name': ..., 'success': True/False}
        """
        result = {
            "coco_json_path": "",
            "project_name": "",
            "success": False,
        }

        try:
            # 1. 模板匹配
            match_params = params.to_dict()
            match_result = self.template_matcher.match(
                sample_path, template_paths, match_params
            )
            if match_result is None or match_result.overlay_image is None:
                logger.warning("模板匹配失败: %s", sample_path)
                return result

            # 推导项目名(用于结果目录命名)
            project_name = match_result.file_name or Path(sample_path).stem
            result["project_name"] = project_name

            # 4. 保存匹配结果到 prediction_visual 目录
            save_dir = os.path.join("prediction_visual", Path(project_name).stem)
            self.save_detection_result(
                match_result.overlay_image, project_name, save_dir
            )

            # 2. 加载检测模型
            self.detector.load_model(params.flag, params.conf_thres)

            # 3. 执行缺陷检测
            detection_result = self.detector.detect(
                match_result.overlay_image,
                slice_size=params.slice_win,
                overlap_ratio=0.05,
            )
            result["coco_json_path"] = detection_result.coco_json_path
            result["success"] = True
            logger.info("检测完成: %s -> %s",
                        sample_path, detection_result.coco_json_path)
        except Exception as e:
            logger.error("检测流程异常 %s: %s", sample_path, e)

        return result

    def save_detection_result(self, image, file_name: str,
                              save_dir: str) -> str:
        """保存匹配结果图片。

        参考 WorkerThread.py 中 MatchAndDetectRunnable.run() 的保存逻辑。

        Args:
            image: 匹配结果图像(PIL.Image 或 ndarray)。
            file_name: 文件名(用于派生输出文件名)。
            save_dir: 保存目录。

        Returns:
            保存的文件路径。
        """
        os.makedirs(save_dir, exist_ok=True)

        base_name = Path(file_name).stem
        output_path = os.path.join(save_dir, f"{base_name}.tiff")

        try:
            # 兼容 PIL.Image 与 ndarray 输入
            if hasattr(image, "mode"):
                # PIL Image 对象
                img_np = np.array(image)
            else:
                img_np = np.asarray(image)

            # 保持原逻辑:二维直接使用,多维也直接使用(原代码注释掉了 BGR 转换)
            if img_np.ndim == 2:
                img_cv = img_np
            else:
                img_cv = img_np

            cv2.imwrite(output_path, img_cv, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            logger.info("匹配结果已保存: %s", output_path)
        except Exception as e:
            logger.error("保存匹配结果失败: %s", e)

        return output_path
