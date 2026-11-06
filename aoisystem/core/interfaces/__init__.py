"""抽象接口定义。

所有硬件访问和数据持久化通过接口抽象,业务层只依赖接口,
不依赖具体实现,符合依赖倒置原则(DIP)。
"""
from .i_plc_client import IPlcClient
from .i_acs_comm import IAcsComm
from .i_camera import ICamera
from .i_detector import IDetector
from .i_template_matcher import ITemplateMatcher
from .i_repository import IMaterialRepository, IScanRecordRepository

__all__ = [
    "IPlcClient",
    "IAcsComm",
    "ICamera",
    "IDetector",
    "ITemplateMatcher",
    "IMaterialRepository",
    "IScanRecordRepository",
]
