"""扫描记录业务服务。

将 MyWindow 中扫描记录相关的业务逻辑抽出为纯 Python 服务,
不依赖 PyQt5。服务通过 IScanRecordRepository 与 IMaterialRepository
接口访问数据,通过 config 单例访问配置。
"""
import math
import logging
from datetime import datetime
from typing import Optional

from core.interfaces.i_repository import (
    IMaterialRepository,
    IScanRecordRepository,
)
from core.models.scan_record import ScanRecord, ScanStatus

logger = logging.getLogger(__name__)


class ScanService:
    """扫描记录业务服务,封装扫描记录的生命周期管理。"""

    def __init__(self, scan_repo: IScanRecordRepository,
                 material_repo: IMaterialRepository, config):
        """构造函数注入仓储与配置。

        Args:
            scan_repo: 扫描记录仓储接口。
            material_repo: 料号仓储接口(用于聚合查询)。
            config: 配置单例,提供运动参数。
        """
        self.scan_repo = scan_repo
        self.material_repo = material_repo
        self.config = config

    def create_scan(self, material_id: str, mode: str,
                    ab_cycle: int = 0) -> ScanRecord:
        """创建扫描记录。

        scan_id 格式: 料号_模式_时间戳

        Args:
            material_id: 关联料号 ID。
            mode: 检测模式(A / B / AB)。
            ab_cycle: AB 面循环次数。

        Returns:
            创建的 ScanRecord 对象。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scan_id = f"{material_id}_{mode}_{timestamp}"

        record = ScanRecord(
            scan_id=scan_id,
            material_id=material_id,
            mode=mode,
            ab_cycle=ab_cycle,
            status=ScanStatus.RUNNING,
            start_time=datetime.now(),
        )
        self.scan_repo.save(record)
        logger.info("创建扫描记录: %s", scan_id)
        return record

    def complete_scan(self, scan_id: str, coco_json_path: str,
                      vrs_path: str, result_dir: str,
                      defect_count: int = 0):
        """标记扫描完成并保存结果路径。

        Args:
            scan_id: 扫描记录 ID。
            coco_json_path: 检测结果 COCO JSON 路径。
            vrs_path: VRS 文件路径。
            result_dir: 结果目录。
            defect_count: 缺陷数量。
        """
        record = self._get_scan_by_id(scan_id)
        if not record:
            logger.warning("扫描记录不存在: %s", scan_id)
            return

        record.status = ScanStatus.COMPLETED
        record.end_time = datetime.now()
        record.coco_json_path = coco_json_path
        record.vrs_path = vrs_path
        record.result_dir = result_dir
        record.defect_count = defect_count
        self.scan_repo.save(record)
        logger.info("扫描完成: %s,缺陷数=%d", scan_id, defect_count)

    def fail_scan(self, scan_id: str, notes: str = ""):
        """标记扫描失败。

        Args:
            scan_id: 扫描记录 ID。
            notes: 失败备注。
        """
        record = self._get_scan_by_id(scan_id)
        if not record:
            logger.warning("扫描记录不存在: %s", scan_id)
            return

        record.status = ScanStatus.FAILED
        record.end_time = datetime.now()
        record.notes = notes
        self.scan_repo.save(record)
        logger.info("扫描失败: %s,备注=%s", scan_id, notes)

    def get_scan_records(self, material_id: str) -> list:
        """获取指定料号的扫描记录。"""
        return self.scan_repo.get_by_material(material_id)

    def get_all_scans(self) -> list:
        """获取所有扫描记录。

        优先调用仓储的 get_all 扩展方法(若实现类提供),
        否则遍历所有料号聚合扫描记录。
        """
        # 仓储实现类提供了 get_all 扩展方法,优先使用
        if hasattr(self.scan_repo, "get_all"):
            return self.scan_repo.get_all()

        # 回退:遍历所有料号聚合扫描记录
        records = []
        for material_id in self.material_repo.get_all().keys():
            records.extend(self.scan_repo.get_by_material(material_id))
        return records

    def delete_scan(self, scan_id: str):
        """删除扫描记录。"""
        self.scan_repo.delete(scan_id)
        logger.info("已删除扫描记录: %s", scan_id)

    def calculate_iteration_count(self, board_width_mm: float,
                                  divisor: int = None) -> int:
        """计算运动循环次数 = ceil(board_width / divisor)。

        参考 main_window.py 中 acs_thread 方法的逻辑。

        Args:
            board_width_mm: 板宽(mm)。
            divisor: 切分除数,默认取 config.board_width_divisor。

        Returns:
            循环次数。
        """
        if divisor is None:
            divisor = self.config.board_width_divisor
        if not divisor:
            logger.warning("board_width_divisor 为 0,返回 0")
            return 0
        return math.ceil(board_width_mm / divisor)

    def _get_scan_by_id(self, scan_id: str) -> Optional[ScanRecord]:
        """按 scan_id 查询扫描记录。

        仓储接口未提供 get_by_id,通过聚合查询实现。
        """
        # 优先使用 get_all 扩展方法
        if hasattr(self.scan_repo, "get_all"):
            for record in self.scan_repo.get_all():
                if record.scan_id == scan_id:
                    return record
            return None

        # 回退:遍历所有料号查找
        for material_id in self.material_repo.get_all().keys():
            for record in self.scan_repo.get_by_material(material_id):
                if record.scan_id == scan_id:
                    return record
        return None
