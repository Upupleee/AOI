"""缺陷检测器抽象接口。"""
from abc import ABC, abstractmethod
from typing import Any, List


class DetectionResult:
    """单次检测结果。"""

    def __init__(self, coco_predictions: list, coco_json_path: str = "",
                 vrs_path: str = "", save_dir: str = ""):
        self.coco_predictions = coco_predictions
        self.coco_json_path = coco_json_path
        self.vrs_path = vrs_path
        self.save_dir = save_dir


class IDetector(ABC):
    """缺陷检测器接口,封装 SAHI + YOLO/TensorRT 推理。"""

    @abstractmethod
    def load_model(self, model_flag: int, conf_threshold: float):
        """加载指定 flag 对应的检测模型。"""

    @abstractmethod
    def detect(self, image, slice_size: int = 1200,
               overlap_ratio: float = 0.05) -> DetectionResult:
        """对图像执行切片检测,返回检测结果。"""
