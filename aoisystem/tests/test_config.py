"""common.config 配置加载单元测试。"""
import os
import tempfile
import unittest

from common.config import AppConfig


SAMPLE_CONFIG = """
hardware:
  plc:
    ip: "192.168.1.100"
    port: 502
  acs:
    ip: "192.168.1.200"
    port: 701

paths:
  material_root: "test_material"
  temp_images: "/tmp/test_images"
  ccf_file: "/tmp/test.ccf"

acs_buffers:
  x_home: "x_home_1.txt"
  forward: "forward_0.txt"

plc:
  coils:
    ready_status: 100
    vacuum: 255
  holding_registers:
    temperature: 202

slicing:
  dpi: 3250
  step_mm: 120

thumbnail:
  scale_factor: 8
  jpeg_quality: 80

vacuum:
  blower_delay_seconds: 10
  time_interval_threshold: 300

motion:
  board_width_divisor: 127

logging:
  level: "DEBUG"
  format: "%(message)s"
"""


class TestAppConfig(unittest.TestCase):
    """AppConfig 配置加载测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "test_config.yaml")
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CONFIG)

    def test_load_from_file(self):
        cfg = AppConfig(self.config_path)
        self.assertEqual(cfg.plc_ip, "192.168.1.100")
        self.assertEqual(cfg.plc_port, 502)
        self.assertEqual(cfg.acs_ip, "192.168.1.200")
        self.assertEqual(cfg.acs_port, 701)

    def test_paths(self):
        cfg = AppConfig(self.config_path)
        self.assertEqual(cfg.material_root, "test_material")
        self.assertEqual(cfg.temp_images_dir, "/tmp/test_images")
        self.assertEqual(cfg.ccf_path, "/tmp/test.ccf")

    def test_acs_buffers(self):
        cfg = AppConfig(self.config_path)
        self.assertEqual(cfg.acs_buffer("x_home"), "x_home_1.txt")
        self.assertEqual(cfg.acs_buffer("forward"), "forward_0.txt")
        self.assertEqual(cfg.acs_buffer("nonexistent"), "")

    def test_acs_buffer_path(self):
        cfg = AppConfig(self.config_path)
        path = cfg.acs_buffer_path("x_home")
        self.assertTrue(path.endswith("x_home_1.txt"))

    def test_plc_addresses(self):
        cfg = AppConfig(self.config_path)
        self.assertEqual(cfg.coil("ready_status"), 100)
        self.assertEqual(cfg.coil("vacuum"), 255)
        self.assertEqual(cfg.register("temperature"), 202)

    def test_slicing_params(self):
        cfg = AppConfig(self.config_path)
        self.assertEqual(cfg.slicing_dpi, 3250)
        self.assertEqual(cfg.slicing_step_mm, 120)

    def test_thumbnail_params(self):
        cfg = AppConfig(self.config_path)
        self.assertEqual(cfg.thumbnail_scale_factor, 8)
        self.assertEqual(cfg.thumbnail_jpeg_quality, 80)

    def test_vacuum_params(self):
        cfg = AppConfig(self.config_path)
        self.assertEqual(cfg.blower_delay_seconds, 10)
        self.assertEqual(cfg.vacuum_time_interval_threshold, 300)

    def test_motion_params(self):
        cfg = AppConfig(self.config_path)
        self.assertEqual(cfg.board_width_divisor, 127)

    def test_logging_params(self):
        cfg = AppConfig(self.config_path)
        import logging
        self.assertEqual(cfg.log_level, logging.DEBUG)
        self.assertEqual(cfg.log_format, "%(message)s")

    def test_nonexistent_file_uses_defaults(self):
        cfg = AppConfig("/nonexistent/path/config.yaml")
        # 应使用默认值，不抛异常
        self.assertEqual(cfg.plc_ip, "20.0.0.200")
        self.assertEqual(cfg.material_root, "material")

    def test_reload(self):
        cfg = AppConfig(self.config_path)
        self.assertEqual(cfg.plc_ip, "192.168.1.100")
        # 修改文件后重新加载
        with open(self.config_path, "w") as f:
            f.write("hardware:\n  plc:\n    ip: \"10.0.0.50\"\n")
        cfg.load()
        self.assertEqual(cfg.plc_ip, "10.0.0.50")


if __name__ == "__main__":
    unittest.main()
