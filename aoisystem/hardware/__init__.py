"""硬件实现层。

为 PLC、ACS、相机、检测器、模板匹配器提供具体实现,
所有实现均继承自 core.interfaces 中的抽象接口,
业务层通过接口依赖,具体实现可在此层替换。

子模块:
    plc:     基于 pymodbus 的 PLC 客户端(ModbusPlcClient)
    acs:     基于 SPiiPlusPython 的 ACS 运动控制器(SpiiPlusAcsComm)
    camera:  基于 QProcess 的相机触发进程(ProcessCamera)
    detection: SAHI 检测器(SahiDetector)与 LoFTR 模板匹配器(LoFtrTemplateMatcher)
"""
from .acs.spiiplus_acs_comm import SpiiPlusAcsComm
from .camera.process_camera import ProcessCamera
from .detection.loftr_template_matcher import LoFtrTemplateMatcher
from .detection.sahi_detector import SahiDetector
from .plc.modbus_plc_client import ModbusPlcClient

__all__ = [
    "ModbusPlcClient",
    "SpiiPlusAcsComm",
    "ProcessCamera",
    "SahiDetector",
    "LoFtrTemplateMatcher",
]
