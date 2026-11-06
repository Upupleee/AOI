"""pytest 公共 fixtures。

为各测试模块提供统一的 Mock 依赖与样例数据,
避免测试连接真实硬件或 PyQt5。
"""
from unittest.mock import Mock

import pytest

from core.interfaces import (
    IPlcClient, IAcsComm, ICamera, IDetector, ITemplateMatcher,
    IMaterialRepository, IScanRecordRepository,
)
from core.models.material import Material
from core.models.scan_record import ScanRecord, ScanStatus


# ---- 硬件 / 仓储 Mock ----

@pytest.fixture
def mock_plc_client():
    """返回遵循 IPlcClient 接口规范的 Mock 实例。"""
    return Mock(spec=IPlcClient)


@pytest.fixture
def mock_acs_comm():
    """返回遵循 IAcsComm 接口规范的 Mock 实例。"""
    return Mock(spec=IAcsComm)


@pytest.fixture
def mock_camera():
    """返回遵循 ICamera 接口规范的 Mock 实例。"""
    return Mock(spec=ICamera)


@pytest.fixture
def mock_material_repo():
    """返回遵循 IMaterialRepository 接口规范的 Mock 实例。"""
    return Mock(spec=IMaterialRepository)


@pytest.fixture
def mock_scan_repo():
    """返回遵循 IScanRecordRepository 接口规范的 Mock 实例。"""
    return Mock(spec=IScanRecordRepository)


@pytest.fixture
def mock_detector():
    """返回遵循 IDetector 接口规范的 Mock 实例。"""
    return Mock(spec=IDetector)


@pytest.fixture
def mock_template_matcher():
    """返回遵循 ITemplateMatcher 接口规范的 Mock 实例。"""
    return Mock(spec=ITemplateMatcher)


# ---- 样例数据 ----

@pytest.fixture
def sample_material():
    """返回一个料号 Material 实例,包含 A/B 面路径与切片信息。"""
    return Material(
        material_id="TEST_MATERIAL_001",
        a_image_path="/data/test/A.tif",
        b_image_path="/data/test/B.tif",
        a_slices=["/data/test/slices/a_1.tif", "/data/test/slices/a_2.tif"],
        b_slices=["/data/test/slices/b_1.tif", "/data/test/slices/b_2.tif"],
        a_thumbnail="/data/test/thumb_a.jpg",
        b_thumbnail="/data/test/thumb_b.jpg",
        template_type="高精度",
        material_type="外层",
        board_thickness=1.6,
        etch_value=25,
        board_polarity="负极",
    )


@pytest.fixture
def sample_scan_record():
    """返回一条已完成的扫描记录 ScanRecord 实例。"""
    return ScanRecord(
        scan_id="TEST_MATERIAL_001_20260101_120000",
        material_id="TEST_MATERIAL_001",
        mode="AB",
        ab_cycle=2,
        status=ScanStatus.COMPLETED,
        coco_json_path="/data/test/result.json",
        vrs_path="/data/test/result.vrs",
        result_dir="/data/test/result",
        defect_count=3,
        notes="测试扫描记录",
    )
