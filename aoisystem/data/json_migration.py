"""JSON 到 SQLite 的数据迁移工具。

将旧的 material_data.json 和 scan_records.json 中的数据导入到 SQLite 仓储,
实现从文件存储到数据库存储的平滑过渡。
"""
import logging

from common.utils import load_json_file
from core.models.material import Material
from core.models.scan_record import ScanRecord

logger = logging.getLogger(__name__)


class JsonToSqliteMigrator:
    """从旧 JSON 文件迁移数据到 SQLite 仓储。

    支持迁移 material_data.json 和 scan_records.json 两种文件。
    """

    def migrate_materials(self, json_path: str, repo) -> int:
        """从 material_data.json 导入料号数据到仓储。

        参数:
            json_path: 旧 JSON 文件路径
            repo: IMaterialRepository 实现(需提供 save 方法)

        返回:
            成功迁移的记录数
        """
        # 旧 JSON 格式:{material_id: {A面: ..., B面: ..., ...}}
        data = load_json_file(json_path, default={})
        if not data:
            logger.warning("料号 JSON 文件为空或不存在: %s", json_path)
            return 0

        count = 0
        for material_id, material_data in data.items():
            try:
                material = Material.from_dict(material_id, material_data)
                repo.save(material)
                count += 1
            except Exception as e:
                logger.error("迁移料号失败 %s: %s", material_id, e)
        logger.info("料号迁移完成,共迁移 %d 条记录", count)
        return count

    def migrate_scan_records(self, json_path: str, repo) -> int:
        """从 scan_records.json 导入扫描记录数据到仓储。

        参数:
            json_path: 旧 JSON 文件路径
            repo: IScanRecordRepository 实现(需提供 save 方法)

        返回:
            成功迁移的记录数
        """
        # 旧 JSON 格式可能为 {scan_id: {scan_id: ..., material_id: ...}}
        # 或 {scan_id: {...(不含 scan_id)}}
        data = load_json_file(json_path, default={})
        if not data:
            logger.warning("扫描记录 JSON 文件为空或不存在: %s", json_path)
            return 0

        count = 0
        for scan_id, record_data in data.items():
            try:
                # 兼容旧 JSON 中可能缺失 scan_id 字段的情况
                if isinstance(record_data, dict):
                    if "scan_id" not in record_data:
                        record_data["scan_id"] = scan_id
                    record = ScanRecord.from_dict(record_data)
                    repo.save(record)
                    count += 1
                else:
                    logger.warning("扫描记录格式异常,跳过 %s", scan_id)
            except Exception as e:
                logger.error("迁移扫描记录失败 %s: %s", scan_id, e)
        logger.info("扫描记录迁移完成,共迁移 %d 条记录", count)
        return count
