"""主视图模型。

MyWindow(视图)的业务逻辑代理。视图只负责 UI 显示和事件转发,
所有业务操作委托给 MainViewModel,MainViewModel 再委托给各 Service。

通过 PyQt 信号通知视图更新,通过 pyqtSlot 暴露可被视图连接的操作。
"""
import math
import logging
import configparser

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer

from core.container import container
from core.models.material import Material
from core.models.scan_record import ScanRecord, ScanStatus
from core.models.detection_params import DetectionParams
from core.state_machine import ScanState, ScanStateMachine

logger = logging.getLogger(__name__)


class MainViewModel(QObject):
    """主视图模型,持有全局容器引用,协调各 Service 完成业务操作。"""

    # 信号(通知视图更新)
    material_data_changed = pyqtSignal()           # 料号数据变化
    scan_records_changed = pyqtSignal()            # 扫描记录变化
    sensor_data_updated = pyqtSignal(dict)         # 传感器数据更新 {temperature, wetness, pressure}
    vacuum_state_changed = pyqtSignal(bool)        # 真空状态变化
    detection_progress = pyqtSignal(int, int)      # 检测进度 (finished_count, total_count)
    detection_round_finished = pyqtSignal()        # 一轮检测完成
    error_occurred = pyqtSignal(str)               # 错误发生
    status_message = pyqtSignal(str)               # 状态消息
    state_changed = pyqtSignal(str)                # 状态机状态变化

    def __init__(self):
        super().__init__()
        self._container = container
        self._config = container.app_config
        self._state_machine = container.state_machine
        self._state_machine.add_listener(self._on_state_changed)

        # 当前状态
        self._current_material_id = None
        self._detection_mode = "A"
        self._ab_cycle_count = 0
        self._auto_confirm = False

        # 检测参数
        self._detection_params = None
        self._last_image_params = None

        # 定时器(传感器轮询)
        self._sensor_timer = QTimer()
        self._sensor_timer.timeout.connect(self._poll_sensors)
        self._sensor_timer.start(100)

    @property
    def state_machine(self) -> ScanStateMachine:
        return self._state_machine

    @property
    def current_material_id(self):
        return self._current_material_id

    @property
    def detection_mode(self):
        return self._detection_mode

    # ---- 料号管理 ----
    @pyqtSlot(result=dict)
    def get_all_materials(self):
        return self._container.material_service.get_all_materials()

    @pyqtSlot(str)
    def select_material(self, material_id: str):
        self._current_material_id = material_id
        self._state_machine.transition(ScanState.MATERIAL_SELECTED)
        self.status_message.emit(f"已选料号: {material_id}")

    @pyqtSlot(str)
    def delete_material(self, material_id: str):
        self._container.material_service.delete_material(material_id)
        self.material_data_changed.emit()

    @pyqtSlot(list, str)
    def import_material(self, images_and_sides: list, material_id: str):
        material = self._container.material_service.import_material_from_images(
            images_and_sides, material_id
        )
        self.material_data_changed.emit()
        self.status_message.emit(f"料号 {material_id} 导入成功")

    @pyqtSlot(str, str, result=list)
    def get_slice_paths(self, material_id: str, side: str):
        return self._container.material_service.get_slice_paths(material_id, side)

    # ---- 检测模式 ----
    @pyqtSlot(str)
    def set_detection_mode(self, mode: str):
        self._detection_mode = mode
        self.status_message.emit(f"检测模式: {mode}")

    @pyqtSlot(bool)
    def set_auto_confirm(self, auto: bool):
        self._auto_confirm = auto

    # ---- 扫描流程 ----
    @pyqtSlot(result=bool)
    def start_sample_detection(self) -> bool:
        """开始样本检测流程。"""
        if not self._current_material_id:
            self.error_occurred.emit("请先选择一个料号!")
            return False
        if self._state_machine.is_running():
            self.error_occurred.emit("正在运行,请等待完成!")
            return False

        try:
            # 1. 启动相机
            self._state_machine.transition(ScanState.CAMERA_STARTING)
            camera = self._container.camera
            if camera.is_running():
                camera.send_trigger_command()
            else:
                camera.started_process.connect(camera.send_trigger_command)
                if not camera.start():
                    self.error_occurred.emit("相机启动失败")
                    self._state_machine.transition(ScanState.ERROR)
                    return False

            # 2. PLC software_ready
            self._state_machine.transition(ScanState.PLC_READY)
            self._container.plc_client.write_bit(self._config.coil('software_ready'), 1)

            # 3. ACS 运动
            self._state_machine.transition(ScanState.ACS_MOVING)
            self.status_message.emit("开始采图!")
            return True
        except Exception as e:
            logger.error("开始检测失败: %s", e)
            self.error_occurred.emit(str(e))
            self._state_machine.transition(ScanState.ERROR)
            return False

    @pyqtSlot()
    def on_detection_round_complete(self):
        """一轮检测完成。"""
        if self._detection_mode == "AB":
            self._ab_cycle_count += 1
            self._state_machine.transition(ScanState.AB_CYCLE)
        else:
            self._state_machine.transition(ScanState.FINISHED)
        self.detection_round_finished.emit()

    @pyqtSlot(str)
    def on_new_file(self, file_path: str):
        """新图片到达。"""
        self._state_machine.transition(ScanState.IMAGE_QUEUEING)
        self._state_machine.transition(ScanState.DETECTING)

    @pyqtSlot()
    def on_acs_finished(self):
        """ACS 运动完成。"""
        self._container.plc_client.write_bit(self._config.coil('software_ready'), 0)
        self._container.camera.send_exit_command()
        self.status_message.emit("采图结束")

    # ---- 真空控制 ----
    @pyqtSlot(result=bool)
    def toggle_vacuum(self) -> bool:
        """切换真空状态。"""
        try:
            new_state = self._container.vacuum_service.toggle_vacuum()
            self.vacuum_state_changed.emit(new_state)
            return new_state
        except Exception as e:
            logger.error("真空切换失败: %s", e)
            self.error_occurred.emit(str(e))
            return False

    @pyqtSlot()
    def reset_alarm(self):
        """报警复位。"""
        try:
            self._container.vacuum_service.reset_alarm()
            self.status_message.emit("报警已复位")
        except Exception as e:
            logger.error("报警复位失败: %s", e)
            self.error_occurred.emit(str(e))

    @pyqtSlot()
    def reset_system(self):
        """系统重置。"""
        try:
            self._container.vacuum_service.reset_alarm()
            self._state_machine.reset()
            self.status_message.emit("系统已重置")
        except Exception as e:
            logger.error("系统重置失败: %s", e)
            self.error_occurred.emit(str(e))

    @pyqtSlot(int)
    def set_light_source(self, value: int):
        """设置光源亮度。"""
        try:
            self._container.plc_client.write_register(self._config.register('light_source'), value)
        except Exception as e:
            logger.error("设置光源失败: %s", e)

    # ---- 检测参数 ----
    @pyqtSlot(str, result='object')
    def load_detection_params(self, ini_path: str) -> DetectionParams:
        """从 INI 文件加载检测参数。"""
        parser = configparser.ConfigParser()
        parser.read(ini_path)
        self._detection_params = DetectionParams.from_ini(parser)
        self._detection_params.mode = self._detection_mode

        # 最后一张图使用特殊参数
        self._last_image_params = DetectionParams.from_ini(parser)
        self._last_image_params.mode = self._detection_mode
        self._last_image_params.rows = 3
        self._last_image_params.cols = 2

        return self._detection_params

    @property
    def detection_params(self):
        return self._detection_params

    @property
    def last_image_params(self):
        return self._last_image_params

    @pyqtSlot(float, result=int)
    def calculate_iteration_count(self, board_width_mm: float) -> int:
        """计算运动循环次数。"""
        divisor = self._config.board_width_divisor
        return math.ceil(board_width_mm / divisor)

    # ---- 传感器轮询 ----
    def _poll_sensors(self):
        """定时轮询传感器数据。"""
        try:
            data = self._container.vacuum_service.read_sensors()
            self.sensor_data_updated.emit(data)

            # 检查安全
            if self._container.vacuum_service.check_safety():
                logger.warning("光幕触发!")
                self.error_occurred.emit("光幕安全触发!")
        except Exception as e:
            logger.debug("传感器轮询异常: %s", e)

    # ---- 扫描记录 ----
    @pyqtSlot(str, result=list)
    def get_scan_records(self, material_id: str):
        return self._container.scan_service.get_scan_records(material_id)

    # ---- 状态机回调 ----
    def _on_state_changed(self, new_state: ScanState):
        self.state_changed.emit(new_state.name)
        logger.info("状态变更: %s", new_state.name)
