"""依赖注入容器。

集中组装所有依赖,遵循组合根(Composition Root)模式。
UI 层和入口点只从此容器获取已组装好的服务实例,
不直接 new 具体实现类,实现控制反转(IoC)。

使用方式:
    from core.container import container
    material_service = container.material_service
    vacuum_service = container.vacuum_service
"""
import logging

from common.config import config
from core.interfaces import (
    IPlcClient, IAcsComm, ICamera, IDetector, ITemplateMatcher,
    IMaterialRepository, IScanRecordRepository,
)
from core.services import (
    MaterialService, ScanService, DetectionService,
    VacuumService, HardwareInitializationService,
)
from core.state_machine import ScanStateMachine

logger = logging.getLogger(__name__)


class Container:
    """依赖注入容器,持有所有单例服务实例。

    首次访问时延迟创建,后续返回缓存实例。
    硬件实例在 initialize_hardware() 后才可用。
    """

    def __init__(self):
        self._config = config
        self._state_machine = ScanStateMachine()

        # 仓储层(不依赖硬件,可立即初始化)
        self._material_repo: IMaterialRepository = None
        self._scan_repo: IScanRecordRepository = None

        # 硬件层(延迟初始化,需要显式调用 initialize_hardware)
        self._plc_client: IPlcClient = None
        self._acs_comm: IAcsComm = None
        self._camera: ICamera = None
        self._detector: IDetector = None
        self._template_matcher: ITemplateMatcher = None

        # 服务层
        self._material_service: MaterialService = None
        self._scan_service: ScanService = None
        self._detection_service: DetectionService = None
        self._vacuum_service: VacuumService = None
        self._hardware_init_service: HardwareInitializationService = None

    # ---- 仓储层 ----
    @property
    def material_repository(self) -> IMaterialRepository:
        if self._material_repo is None:
            from data.repositories.material_repository import SqliteMaterialRepository
            self._material_repo = SqliteMaterialRepository(self._config)
            logger.info("料号仓储已初始化")
        return self._material_repo

    @property
    def scan_record_repository(self) -> IScanRecordRepository:
        if self._scan_repo is None:
            from data.repositories.scan_record_repository import SqliteScanRecordRepository
            self._scan_repo = SqliteScanRecordRepository(self._config)
            logger.info("扫描记录仓储已初始化")
        return self._scan_repo

    # ---- 硬件层(延迟创建) ----
    @property
    def plc_client(self) -> IPlcClient:
        if self._plc_client is None:
            raise RuntimeError("PLC 客户端未初始化,请先调用 container.initialize_hardware()")
        return self._plc_client

    @property
    def acs_comm(self) -> IAcsComm:
        if self._acs_comm is None:
            raise RuntimeError("ACS 通信未初始化,请先调用 container.initialize_hardware()")
        return self._acs_comm

    @property
    def camera(self) -> ICamera:
        if self._camera is None:
            from hardware.camera.process_camera import ProcessCamera
            self._camera = ProcessCamera()
            logger.info("相机控制器已初始化")
        return self._camera

    @property
    def detector(self) -> IDetector:
        if self._detector is None:
            from hardware.detection.sahi_detector import SahiDetector
            self._detector = SahiDetector()
            logger.info("缺陷检测器已初始化")
        return self._detector

    @property
    def template_matcher(self) -> ITemplateMatcher:
        if self._template_matcher is None:
            from hardware.detection.loftr_template_matcher import LoFtrTemplateMatcher
            self._template_matcher = LoFtrTemplateMatcher()
            logger.info("模板匹配器已初始化")
        return self._template_matcher

    # ---- 服务层 ----
    @property
    def material_service(self) -> MaterialService:
        if self._material_service is None:
            self._material_service = MaterialService(
                material_repo=self.material_repository,
                config=self._config,
            )
            logger.info("料号服务已初始化")
        return self._material_service

    @property
    def scan_service(self) -> ScanService:
        if self._scan_service is None:
            self._scan_service = ScanService(
                scan_repo=self.scan_record_repository,
                material_repo=self.material_repository,
                config=self._config,
            )
            logger.info("扫描服务已初始化")
        return self._scan_service

    @property
    def detection_service(self) -> DetectionService:
        if self._detection_service is None:
            self._detection_service = DetectionService(
                detector=self.detector,
                template_matcher=self.template_matcher,
                config=self._config,
            )
            logger.info("检测服务已初始化")
        return self._detection_service

    @property
    def vacuum_service(self) -> VacuumService:
        if self._vacuum_service is None:
            self._vacuum_service = VacuumService(
                plc_client=self.plc_client,
                config=self._config,
            )
            logger.info("真空控制服务已初始化")
        return self._vacuum_service

    @property
    def hardware_init_service(self) -> HardwareInitializationService:
        if self._hardware_init_service is None:
            self._hardware_init_service = HardwareInitializationService(
                plc_client=self.plc_client,
                acs_comm=self.acs_comm,
                config=self._config,
            )
            logger.info("硬件初始化服务已初始化")
        return self._hardware_init_service

    @property
    def state_machine(self) -> ScanStateMachine:
        return self._state_machine

    @property
    def app_config(self):
        return self._config

    # ---- 生命周期管理 ----
    def initialize_hardware(self, plc_ip: str = None, plc_port: int = None,
                            acs_ip: str = None, acs_port: int = None) -> bool:
        """初始化硬件连接。

        创建 PLC 和 ACS 实例并尝试连接。
        成功返回 True,失败返回 False。
        """
        try:
            from hardware.plc.modbus_plc_client import ModbusPlcClient
            from hardware.acs.spiiplus_acs_comm import SpiiPlusAcsComm

            ip = plc_ip or self._config.plc_ip
            port = plc_port or self._config.plc_port
            self._plc_client = ModbusPlcClient(ip, port)
            if not self._plc_client.connect():
                logger.error("PLC 连接失败: %s:%s", ip, port)
                return False
            logger.info("PLC 连接成功: %s:%s", ip, port)

            acs_ip_val = acs_ip or self._config.acs_ip
            acs_port_val = acs_port or self._config.acs_port
            self._acs_comm = SpiiPlusAcsComm()
            conn_handle = self._acs_comm.connect(acs_ip_val, acs_port_val)
            if conn_handle == -1:
                logger.error("ACS 连接失败: %s:%s", acs_ip_val, acs_port_val)
                return False
            logger.info("ACS 连接成功: %s:%s (handle=%s)", acs_ip_val, acs_port_val, conn_handle)

            return True
        except Exception as e:
            logger.error("硬件初始化异常: %s", e)
            return False

    def shutdown(self):
        """关闭所有硬件连接,释放资源。"""
        if self._plc_client is not None:
            try:
                self._plc_client.disconnect()
                logger.info("PLC 已断开")
            except Exception as e:
                logger.error("PLC 断开异常: %s", e)

        if self._acs_comm is not None:
            try:
                self._acs_comm.disconnect()
                logger.info("ACS 已断开")
            except Exception as e:
                logger.error("ACS 断开异常: %s", e)

        self._state_machine.reset()
        logger.info("容器已关闭,状态机已重置")


# 全局容器实例
container = Container()
