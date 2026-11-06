"""核心数据模型单元测试。

覆盖 Material / ScanRecord / DetectionParams 三个数据类的
创建、字段映射、字典序列化与反序列化。
"""
from configparser import ConfigParser
from datetime import datetime

import pytest

from core.models.material import Material
from core.models.scan_record import ScanRecord, ScanStatus
from core.models.detection_params import DetectionParams


# =====================================================================
# Material
# =====================================================================

class TestMaterial:
    """料号数据模型测试。"""

    def test_material_creation(self):
        """验证 Material 创建后字段赋值正确。"""
        material = Material(
            material_id="M001",
            a_image_path="/imgs/A.tif",
            b_image_path="/imgs/B.tif",
            a_slices=["/imgs/a1.tif", "/imgs/a2.tif"],
            b_slices=["/imgs/b1.tif"],
            a_thumbnail="/thumb/a.jpg",
            b_thumbnail="/thumb/b.jpg",
            template_type="高精度",
            material_type="外层",
            board_thickness=1.6,
            etch_value=25,
            board_polarity="负极",
        )
        assert material.material_id == "M001"
        assert material.a_image_path == "/imgs/A.tif"
        assert material.b_image_path == "/imgs/B.tif"
        assert material.a_slices == ["/imgs/a1.tif", "/imgs/a2.tif"]
        assert material.b_slices == ["/imgs/b1.tif"]
        assert material.a_thumbnail == "/thumb/a.jpg"
        assert material.b_thumbnail == "/thumb/b.jpg"
        assert material.template_type == "高精度"
        assert material.material_type == "外层"
        assert material.board_thickness == pytest.approx(1.6)
        assert material.etch_value == 25
        assert material.board_polarity == "负极"

    def test_material_defaults(self):
        """验证 Material 默认值。"""
        material = Material(material_id="M002")
        assert material.material_id == "M002"
        assert material.a_image_path == ""
        assert material.b_image_path == ""
        assert material.a_slices == []
        assert material.b_slices == []
        assert material.a_thumbnail is None
        assert material.b_thumbnail is None
        assert material.template_type == "默认"
        assert material.material_type == "内层"
        assert material.board_thickness == pytest.approx(1.0)
        assert material.etch_value == 20
        assert material.board_polarity == "正极"

    def test_material_to_dict(self):
        """验证 to_dict 返回的中文字段键及值映射正确。"""
        material = Material(
            material_id="M003",
            a_image_path="/A.tif",
            b_image_path="/B.tif",
            a_slices=["a1", "a2"],
            b_slices=["b1"],
            a_thumbnail="/ta.jpg",
            b_thumbnail="/tb.jpg",
            template_type="高精度",
            material_type="外层",
            board_thickness=2.0,
            etch_value=30,
            board_polarity="负极",
        )
        d = material.to_dict()
        # 中文字段键
        assert d["A面"] == "/A.tif"
        assert d["B面"] == "/B.tif"
        assert d["A面切分后"] == ["a1", "a2"]
        assert d["B面切分后"] == ["b1"]
        assert d["A面缩略图"] == "/ta.jpg"
        assert d["B面缩略图"] == "/tb.jpg"
        # 英文字段键
        assert d["templatevalue"] == "高精度"
        assert d["materialvalue"] == "外层"
        assert d["boardthicknessvalue"] == pytest.approx(2.0)
        assert d["etchvalue"] == 30
        assert d["boardpolarityvalue"] == "负极"

    def test_material_from_dict(self):
        """验证 from_dict 字段映射与类型转换。"""
        data = {
            "A面": "/data/A.tif",
            "B面": "/data/B.tif",
            "A面切分后": ["/s/a1", "/s/a2"],
            "B面切分后": ["/s/b1"],
            "A面缩略图": "/t/a.jpg",
            "B面缩略图": "/t/b.jpg",
            "templatevalue": "高精度",
            "materialvalue": "外层",
            "boardthicknessvalue": "1.6",  # 字符串应被转为 float
            "etchvalue": "25",  # 字符串应被转为 int
            "boardpolarityvalue": "负极",
        }
        material = Material.from_dict("M004", data)
        assert material.material_id == "M004"
        assert material.a_image_path == "/data/A.tif"
        assert material.b_image_path == "/data/B.tif"
        assert material.a_slices == ["/s/a1", "/s/a2"]
        assert material.b_slices == ["/s/b1"]
        assert material.a_thumbnail == "/t/a.jpg"
        assert material.b_thumbnail == "/t/b.jpg"
        assert material.template_type == "高精度"
        assert material.material_type == "外层"
        assert material.board_thickness == pytest.approx(1.6)
        assert material.etch_value == 25
        assert material.board_polarity == "负极"

    def test_material_from_dict_defaults(self):
        """空字典应回退到默认值。"""
        material = Material.from_dict("M005", {})
        assert material.material_id == "M005"
        assert material.a_image_path == ""
        assert material.template_type == "默认"
        assert material.material_type == "内层"
        assert material.board_thickness == pytest.approx(1.0)
        assert material.etch_value == 20
        assert material.board_polarity == "正极"

    def test_material_roundtrip(self, sample_material):
        """to_dict -> from_dict -> to_dict 序列化往返一致。"""
        first_dict = sample_material.to_dict()
        rebuilt = Material.from_dict(sample_material.material_id, first_dict)
        second_dict = rebuilt.to_dict()
        assert second_dict == first_dict
        # 关键字段也应一致
        assert rebuilt.material_id == sample_material.material_id
        assert rebuilt.a_slices == sample_material.a_slices
        assert rebuilt.board_thickness == sample_material.board_thickness
        assert rebuilt.etch_value == sample_material.etch_value


# =====================================================================
# ScanRecord
# =====================================================================

class TestScanRecord:
    """扫描记录数据模型测试。"""

    def test_scan_record_creation(self):
        """验证 ScanRecord 创建后字段赋值正确。"""
        start = datetime(2026, 1, 1, 12, 0, 0)
        end = datetime(2026, 1, 1, 12, 5, 30)
        record = ScanRecord(
            scan_id="S001",
            material_id="M001",
            mode="AB",
            ab_cycle=2,
            status=ScanStatus.COMPLETED,
            start_time=start,
            end_time=end,
            coco_json_path="/r/result.json",
            vrs_path="/r/result.vrs",
            result_dir="/r/out",
            defect_count=5,
            notes="ok",
        )
        assert record.scan_id == "S001"
        assert record.material_id == "M001"
        assert record.mode == "AB"
        assert record.ab_cycle == 2
        assert record.status is ScanStatus.COMPLETED
        assert record.start_time == start
        assert record.end_time == end
        assert record.coco_json_path == "/r/result.json"
        assert record.vrs_path == "/r/result.vrs"
        assert record.result_dir == "/r/out"
        assert record.defect_count == 5
        assert record.notes == "ok"

    def test_scan_record_defaults(self):
        """验证 ScanRecord 默认值。"""
        record = ScanRecord(scan_id="S002", material_id="M002")
        assert record.scan_id == "S002"
        assert record.material_id == "M002"
        assert record.mode == "A"
        assert record.ab_cycle == 0
        assert record.status is ScanStatus.PENDING
        assert record.start_time is None
        assert record.end_time is None
        assert record.coco_json_path == ""
        assert record.vrs_path == ""
        assert record.result_dir == ""
        assert record.defect_count == 0
        assert record.notes == ""

    def test_scan_record_to_dict(self):
        """验证 to_dict 返回的键值正确,datetime 转为 isoformat 字符串。"""
        start = datetime(2026, 1, 1, 12, 0, 0)
        end = datetime(2026, 1, 1, 12, 5, 30)
        record = ScanRecord(
            scan_id="S003",
            material_id="M003",
            mode="B",
            ab_cycle=1,
            status=ScanStatus.FAILED,
            start_time=start,
            end_time=end,
            coco_json_path="/r/r.json",
            vrs_path="/r/r.vrs",
            result_dir="/r/out",
            defect_count=2,
            notes="失败案例",
        )
        d = record.to_dict()
        assert d["scan_id"] == "S003"
        assert d["material_id"] == "M003"
        assert d["mode"] == "B"
        assert d["ab_cycle"] == 1
        assert d["status"] == "failed"
        assert d["start_time"] == start.isoformat()
        assert d["end_time"] == end.isoformat()
        assert d["coco_json_path"] == "/r/r.json"
        assert d["vrs_path"] == "/r/r.vrs"
        assert d["result_dir"] == "/r/out"
        assert d["defect_count"] == 2
        assert d["notes"] == "失败案例"

    def test_scan_record_to_dict_none_times(self):
        """start_time / end_time 为 None 时字典中也应为 None。"""
        record = ScanRecord(scan_id="S004", material_id="M004")
        d = record.to_dict()
        assert d["start_time"] is None
        assert d["end_time"] is None
        assert d["status"] == "pending"

    def test_scan_record_from_dict(self):
        """验证 from_dict 字段映射,字符串时间还原为 datetime。"""
        start = datetime(2026, 1, 1, 12, 0, 0)
        end = datetime(2026, 1, 1, 12, 5, 30)
        data = {
            "scan_id": "S005",
            "material_id": "M005",
            "mode": "AB",
            "ab_cycle": 3,
            "status": "running",
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "coco_json_path": "/r/c.json",
            "vrs_path": "/r/c.vrs",
            "result_dir": "/r/dir",
            "defect_count": 7,
            "notes": "从字典恢复",
        }
        record = ScanRecord.from_dict(data)
        assert record.scan_id == "S005"
        assert record.material_id == "M005"
        assert record.mode == "AB"
        assert record.ab_cycle == 3
        assert record.status is ScanStatus.RUNNING
        assert record.start_time == start
        assert record.end_time == end
        assert record.coco_json_path == "/r/c.json"
        assert record.vrs_path == "/r/c.vrs"
        assert record.result_dir == "/r/dir"
        assert record.defect_count == 7
        assert record.notes == "从字典恢复"

    def test_scan_record_from_dict_none_times(self):
        """start_time / end_time 缺失或为 None 时字段应为 None。"""
        data = {
            "scan_id": "S006",
            "material_id": "M006",
        }
        record = ScanRecord.from_dict(data)
        assert record.start_time is None
        assert record.end_time is None
        assert record.status is ScanStatus.PENDING

    def test_scan_record_roundtrip(self, sample_scan_record):
        """to_dict -> from_dict -> to_dict 序列化往返一致。"""
        first_dict = sample_scan_record.to_dict()
        rebuilt = ScanRecord.from_dict(first_dict)
        second_dict = rebuilt.to_dict()
        assert second_dict == first_dict
        assert rebuilt.scan_id == sample_scan_record.scan_id
        assert rebuilt.material_id == sample_scan_record.material_id
        assert rebuilt.status is sample_scan_record.status

    def test_scan_status_enum(self):
        """验证 ScanStatus 枚举值与字符串映射。"""
        assert ScanStatus.PENDING.value == "pending"
        assert ScanStatus.RUNNING.value == "running"
        assert ScanStatus.COMPLETED.value == "completed"
        assert ScanStatus.FAILED.value == "failed"
        # 字符串反向构造
        assert ScanStatus("pending") is ScanStatus.PENDING
        assert ScanStatus("running") is ScanStatus.RUNNING
        assert ScanStatus("completed") is ScanStatus.COMPLETED
        assert ScanStatus("failed") is ScanStatus.FAILED
        # 未知状态应抛 ValueError
        with pytest.raises(ValueError):
            ScanStatus("unknown_status")


# =====================================================================
# DetectionParams
# =====================================================================

class TestDetectionParams:
    """检测参数数据模型测试。"""

    def test_detection_params_defaults(self):
        """验证 DetectionParams 默认值。"""
        params = DetectionParams()
        assert params.rows == 6
        assert params.cols == 6
        assert params.auto_split is True
        assert params.overlap_temp == 0
        assert params.overlap_samp == 800
        assert params.save_image is True
        assert params.flag == 6
        assert params.slice_win == 650
        assert params.conf_thres == pytest.approx(0.2)
        assert params.mode == "A"
        assert params.part_infer is False
        assert params.grid_size == 7
        assert params.max_sampling_per_grid == 100
        assert params.enhance_algorithms == "clahe"
        assert params.extractor_name == "aliked"

    def test_detection_params_to_dict(self):
        """验证 to_dict 返回的字典与字段一致。"""
        params = DetectionParams(
            rows=8,
            cols=4,
            auto_split=False,
            overlap_temp=10,
            overlap_samp=500,
            save_image=False,
            flag=12,
            slice_win=800,
            conf_thres=0.35,
            mode="B",
            part_infer=True,
            grid_size=9,
            max_sampling_per_grid=200,
            enhance_algorithms="histogram",
            enhance_param=[40, 80, "CLAHE"],
            extractor_name="lightglue",
        )
        d = params.to_dict()
        assert d["rows"] == 8
        assert d["cols"] == 4
        assert d["auto_split"] is False
        assert d["overlap_temp"] == 10
        assert d["overlap_samp"] == 500
        assert d["save_image"] is False
        assert d["flag"] == 12
        assert d["slice_win"] == 800
        assert d["conf_thres"] == pytest.approx(0.35)
        assert d["mode"] == "B"
        assert d["part_infer"] is True
        assert d["grid_size"] == 9
        assert d["max_sampling_per_grid"] == 200
        assert d["enhance_algorithms"] == "histogram"
        assert d["enhance_param"] == [40, 80, "CLAHE"]
        assert d["extractor_name"] == "lightglue"

    def test_detection_params_to_dict_default_enhance_param(self):
        """enhance_param 为 None 时,to_dict 应返回默认列表。"""
        params = DetectionParams()
        d = params.to_dict()
        assert d["enhance_param"] == [50, 100, "None"]

    def test_detection_params_from_ini(self):
        """使用 ConfigParser 模拟 INI 文件加载,验证字段映射与类型转换。"""
        parser = ConfigParser()
        parser["General"] = {
            "row": "8",
            "col": "4",
            "Auto_split": "False",
            "overlap_temp": "15",
            "overlap_samp": "600",
            "SaveImage": "False",
            "flag": "12",
            "slice": "900",
            "thres": "0.35",
            "PartInfer": "True",
        }
        params = DetectionParams.from_ini(parser, section="General")
        assert params.rows == 8
        assert params.cols == 4
        assert params.auto_split is False
        assert params.overlap_temp == 15
        assert params.overlap_samp == 600
        assert params.save_image is False
        assert params.flag == 12
        assert params.slice_win == 900
        assert params.conf_thres == pytest.approx(0.35)
        assert params.part_infer is True

    def test_detection_params_from_ini_missing_keys(self):
        """INI 中缺少键时使用默认值。"""
        parser = ConfigParser()
        parser["General"] = {"row": "10"}
        params = DetectionParams.from_ini(parser, section="General")
        assert params.rows == 10
        assert params.cols == 6  # 默认
        assert params.auto_split is True  # 默认
        assert params.save_image is True  # 默认
        assert params.flag == 6  # 默认
        assert params.slice_win == 650  # 默认
        assert params.conf_thres == pytest.approx(0.2)  # 默认
        assert params.part_infer is False  # 默认

    def test_detection_params_from_ini_invalid_value(self):
        """INI 中数值字段非法时回退到默认值。"""
        parser = ConfigParser()
        parser["General"] = {
            "row": "not_an_int",     # 非法整数 -> 默认 6
            "col": "8",
            "thres": "not_a_float",  # 非法浮点 -> 默认 0.2
        }
        params = DetectionParams.from_ini(parser, section="General")
        assert params.rows == 6  # 回退默认
        assert params.cols == 8  # 正常解析
        assert params.conf_thres == pytest.approx(0.2)  # 回退默认

    def test_detection_params_from_ini_empty_section(self):
        """section 不存在时应使用全部默认值,不抛异常。"""
        parser = ConfigParser()
        parser["Other"] = {"foo": "bar"}
        params = DetectionParams.from_ini(parser, section="General")
        assert params.rows == 6
        assert params.cols == 6
        assert params.auto_split is True
        assert params.flag == 6
