"""扫描记录仓储的 SQLite 实现。"""
import logging
import sqlite3
from datetime import datetime
from typing import List, Optional

from core.interfaces.i_repository import IScanRecordRepository
from core.models.scan_record import ScanRecord, ScanStatus
from data.database import DatabaseManager

logger = logging.getLogger(__name__)


class SqliteScanRecordRepository(IScanRecordRepository):
    """基于 SQLite 的扫描记录仓储实现。"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def get_by_material(self, material_id: str) -> List[ScanRecord]:
        """获取指定料号的扫描记录(按开始时间倒序)。"""
        records = []
        conn = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM scan_records WHERE material_id = ? ORDER BY start_time DESC",
                (material_id,),
            )
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                record = self._row_to_record(row, columns)
                if record:
                    records.append(record)
        except sqlite3.Error as e:
            logger.error("查询扫描记录失败 %s: %s", material_id, e)
        finally:
            if conn:
                conn.close()
        return records

    def get_all(self) -> List[ScanRecord]:
        """获取所有扫描记录(按开始时间倒序)。"""
        records = []
        conn = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scan_records ORDER BY start_time DESC")
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                record = self._row_to_record(row, columns)
                if record:
                    records.append(record)
        except sqlite3.Error as e:
            logger.error("查询所有扫描记录失败: %s", e)
        finally:
            if conn:
                conn.close()
        return records

    def save(self, record: ScanRecord):
        """保存扫描记录(基于 scan_id 主键,插入或更新)。"""
        conn = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO scan_records (
                    scan_id, material_id, mode, ab_cycle, status,
                    start_time, end_time, coco_json_path, vrs_path,
                    result_dir, defect_count, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.scan_id,
                record.material_id,
                record.mode,
                record.ab_cycle,
                record.status.value,
                record.start_time.isoformat() if record.start_time else None,
                record.end_time.isoformat() if record.end_time else None,
                record.coco_json_path,
                record.vrs_path,
                record.result_dir,
                record.defect_count,
                record.notes,
            ))
            conn.commit()
            logger.info("保存扫描记录成功: %s", record.scan_id)
        except sqlite3.Error as e:
            logger.error("保存扫描记录失败 %s: %s", record.scan_id, e)
            raise
        finally:
            if conn:
                conn.close()

    def delete(self, scan_id: str):
        """删除扫描记录。"""
        conn = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM scan_records WHERE scan_id = ?",
                (scan_id,),
            )
            conn.commit()
            logger.info("删除扫描记录成功: %s", scan_id)
        except sqlite3.Error as e:
            logger.error("删除扫描记录失败 %s: %s", scan_id, e)
            raise
        finally:
            if conn:
                conn.close()

    def _row_to_record(self, row, columns) -> Optional[ScanRecord]:
        """将数据库行转换为 ScanRecord 对象。"""
        try:
            data = dict(zip(columns, row))
            # 时间字段以 ISO 格式字符串存储,读取时反序列化
            start_time = None
            end_time = None
            if data.get("start_time"):
                start_time = datetime.fromisoformat(data["start_time"])
            if data.get("end_time"):
                end_time = datetime.fromisoformat(data["end_time"])
            return ScanRecord(
                scan_id=data["scan_id"],
                material_id=data["material_id"],
                mode=data.get("mode") or "A",
                ab_cycle=int(data.get("ab_cycle") or 0),
                status=ScanStatus(data.get("status") or "pending"),
                start_time=start_time,
                end_time=end_time,
                coco_json_path=data.get("coco_json_path") or "",
                vrs_path=data.get("vrs_path") or "",
                result_dir=data.get("result_dir") or "",
                defect_count=int(data.get("defect_count") or 0),
                notes=data.get("notes") or "",
            )
        except (ValueError, KeyError) as e:
            logger.error("解析扫描记录失败: %s", e)
            return None
