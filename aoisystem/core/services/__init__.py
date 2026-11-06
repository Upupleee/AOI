"""业务服务层。

将 MyWindow / WorkerThread / LoginWindow 中的业务逻辑抽出为纯 Python 服务,
不依赖 PyQt5(除 VacuumService 使用标准库 threading.Timer)。
服务类通过 core.interfaces 中的抽象接口访问硬件与数据,
通过构造函数注入实现依赖倒置,便于单元测试与具体实现替换。

服务列表:
    MaterialService:              料号 CRUD、图片导入、切分
    ScanService:                  扫描记录生命周期管理
    DetectionService:             模板匹配 + 缺陷检测协调
    VacuumService:                真空/风机/报警控制
    HardwareInitializationService: PLC 与 ACS 连接/断开
"""
from .material_service import MaterialService
from .scan_service import ScanService
from .detection_service import DetectionService
from .vacuum_service import VacuumService
from .hardware_init_service import HardwareInitializationService

__all__ = [
    "MaterialService",
    "ScanService",
    "DetectionService",
    "VacuumService",
    "HardwareInitializationService",
]
