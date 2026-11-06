"""基于 SAHI + Ultralytics 的缺陷检测器实现。"""
import logging

from AoiSystemWindow_test.ModelManger import ModelManager
from DefectDetection.sahi.sahi.predict import get_batched_sliced_prediction
from core.interfaces.i_detector import DetectionResult, IDetector

logger = logging.getLogger(__name__)


class SahiDetector(IDetector):
    """封装 SAHI + Ultralytics 的切片检测实现。

    通过 ModelManager 单例获取检测模型,使用 get_batched_sliced_prediction
    执行批量切片推理并返回 COCO 格式检测结果。
    """

    def __init__(self):
        """初始化检测器,模型延迟到 load_model 调用时加载。"""
        # 检测模型实例(AutoDetectionModel)
        self._detector = None
        # 当前已加载模型的 flag 标识
        self._model_flag = None

    def load_model(self, model_flag: int, conf_threshold: float):
        """通过 ModelManager 单例加载指定 flag 对应的检测模型。

        Args:
            model_flag: 模型标识,对应 ModelManager 中的模型路径映射。
            conf_threshold: 置信度阈值。
        """
        logger.info("加载检测模型,flag=%s,置信度阈值=%s", model_flag, conf_threshold)
        self._detector = ModelManager().get_model(
            ModelManager.MODEL_TYPE_DETECTION,
            flag=model_flag,
            conf_threshold=conf_threshold,
        )
        self._model_flag = model_flag
        logger.info("检测模型加载完成,flag=%s", model_flag)

    def detect(self, image, slice_size: int = 1200,
               overlap_ratio: float = 0.05) -> DetectionResult:
        """对图像执行批量切片检测,返回 COCO 格式检测结果。

        Args:
            image: 待检测图像(numpy 数组或 PIL 图像)。
            slice_size: 切片边长(像素),默认 1200。
            overlap_ratio: 切片重叠比例,默认 0.05。

        Returns:
            DetectionResult,其中 coco_predictions 为 COCO 格式预测列表。
        """
        if self._detector is None:
            logger.error("检测模型未加载,请先调用 load_model()")
            raise RuntimeError("检测模型未加载,请先调用 load_model()")

        logger.info("开始切片检测,切片尺寸=%s,重叠比例=%s", slice_size, overlap_ratio)
        batch_prediction = get_batched_sliced_prediction(
            image=image,
            detection_model=self._detector,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=0,
        )
        coco_predictions = batch_prediction.to_coco_predictions(image_id=1)
        logger.info("切片检测完成,预测数量=%s", len(coco_predictions))
        return DetectionResult(coco_predictions=coco_predictions)
