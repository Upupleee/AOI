"""全局配置加载。

从项目根目录的 config.yaml 读取配置，提供统一的访问入口。
所有硬编码的 IP、路径、PLC 地址、运动参数等集中在此管理。
"""
import os
import logging
import yaml

# 项目根目录（common/ 的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


class AppConfig:
    """应用配置，从 config.yaml 加载并提供属性式访问。"""

    def __init__(self, config_path=None):
        self._config_path = config_path or CONFIG_PATH
        self._data = {}
        self.load()

    def load(self):
        """加载（或重新加载）配置文件。"""
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logging.warning("配置文件不存在: %s，使用默认空配置", self._config_path)
            self._data = {}
        except yaml.YAMLError as e:
            logging.error("配置文件解析失败: %s", e)
            self._data = {}

    # ---- 硬件连接 ----
    @property
    def plc_ip(self):
        return self._data.get("hardware", {}).get("plc", {}).get("ip", "20.0.0.200")

    @property
    def plc_port(self):
        return self._data.get("hardware", {}).get("plc", {}).get("port", 502)

    @property
    def acs_ip(self):
        return self._data.get("hardware", {}).get("acs", {}).get("ip", "10.0.0.100")

    @property
    def acs_port(self):
        return self._data.get("hardware", {}).get("acs", {}).get("port", 701)

    # ---- 路径 ----
    @property
    def material_root(self):
        return self._data.get("paths", {}).get("material_root", "material")

    @property
    def temp_images_dir(self):
        return self._data.get("paths", {}).get("temp_images", "/home/aoi/pretest/images")

    @property
    def ccf_path(self):
        return self._data.get("paths", {}).get("ccf_file", "/home/aoi/Documents/best_0429.ccf")

    @property
    def motion_settings_ini(self):
        return self._data.get("paths", {}).get("motion_settings_ini",
            "/home/aoi/pretest/AOI_V1/SystemSettingWindow/motion_settings.ini")

    @property
    def acs_config_path(self):
        return self._data.get("paths", {}).get("acs_config", "./Acs/config_1104.yaml")

    @property
    def acs_buffers_dir(self):
        return self._data.get("paths", {}).get("acs_buffers_dir",
            "/home/aoi/pretest/acs_test/Buffer_single_new_aoi")

    @property
    def camera_working_dir(self):
        return self._data.get("paths", {}).get("camera_working_dir",
            "/home/aoi/pretest/camera_test/ex_trigger/ex_trigger_command")

    @property
    def camera_executable(self):
        return self._data.get("paths", {}).get("camera_executable",
            "/home/aoi/pretest/camera_test/ex_trigger/ex_trigger_command/external_trigger")

    @property
    def loftr_weights_path(self):
        return self._data.get("paths", {}).get("loftr_weights",
            "TemplateMatching/EfficientLoFTR/weights/eloftr_outdoor.ckpt")

    @property
    def log_file(self):
        return self._data.get("paths", {}).get("log_file", "./Logger/aoi_system.log")

    @property
    def settings_ini_path(self):
        return os.path.join(os.path.expanduser("~"),
            self._data.get("paths", {}).get("settings_ini", ".aoi_software_settings.ini"))

    @property
    def material_data_json(self):
        return self._data.get("paths", {}).get("material_data_json", "material_data.json")

    @property
    def scan_records_json(self):
        return self._data.get("paths", {}).get("scan_records_json", "scan_records.json")

    @property
    def remembered_credentials_json(self):
        return self._data.get("paths", {}).get("remembered_credentials_json",
            "remembered_credentials.json")

    @property
    def database_path(self):
        return self._data.get("paths", {}).get("database", "aoi_system.db")

    # ---- ACS buffer 文件名 ----
    def acs_buffer(self, name):
        """根据名称获取 ACS buffer 文件名。"""
        return self._data.get("acs_buffers", {}).get(name, "")

    def acs_buffer_path(self, name):
        """根据名称获取 ACS buffer 文件完整路径。"""
        return os.path.join(self.acs_buffers_dir, self.acs_buffer(name))

    # ---- PLC 地址 ----
    @property
    def plc_coils(self):
        return self._data.get("plc", {}).get("coils", {})

    @property
    def plc_holding_registers(self):
        return self._data.get("plc", {}).get("holding_registers", {})

    def coil(self, name):
        """获取线圈地址。"""
        return self.plc_coils.get(name, 0)

    def register(self, name):
        """获取保持寄存器地址。"""
        return self.plc_holding_registers.get(name, 0)

    # ---- 图像切分参数 ----
    @property
    def slicing_dpi(self):
        return self._data.get("slicing", {}).get("dpi", 3250)

    @property
    def slicing_step_mm(self):
        return self._data.get("slicing", {}).get("step_mm", 120)

    @property
    def slicing_margin_mm(self):
        return self._data.get("slicing", {}).get("margin_mm", 5.0)

    @property
    def slicing_segment_mm(self):
        return self._data.get("slicing", {}).get("segment_mm", 128)

    # ---- 缩略图 ----
    @property
    def thumbnail_scale_factor(self):
        return self._data.get("thumbnail", {}).get("scale_factor", 8)

    @property
    def thumbnail_jpeg_quality(self):
        return self._data.get("thumbnail", {}).get("jpeg_quality", 80)

    # ---- 真空/风机 ----
    @property
    def blower_delay_seconds(self):
        return self._data.get("vacuum", {}).get("blower_delay_seconds", 10)

    @property
    def vacuum_time_interval_threshold(self):
        return self._data.get("vacuum", {}).get("time_interval_threshold", 300)

    # ---- 运动参数 ----
    @property
    def board_width_divisor(self):
        return self._data.get("motion", {}).get("board_width_divisor", 127)

    # ---- 日志 ----
    @property
    def log_level(self):
        level_str = self._data.get("logging", {}).get("level", "INFO")
        return getattr(logging, level_str, logging.INFO)

    @property
    def log_format(self):
        return self._data.get("logging", {}).get("format",
            "%(asctime)s - %(levelname)s - %(message)s")


# 全局单例配置实例
config = AppConfig()
