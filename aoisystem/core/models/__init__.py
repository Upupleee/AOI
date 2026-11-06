"""数据模型层:纯数据类,不依赖 Qt。"""
from .material import Material
from .scan_record import ScanRecord, ScanStatus
from .detection_params import DetectionParams

__all__ = ["Material", "ScanRecord", "ScanStatus", "DetectionParams"]
