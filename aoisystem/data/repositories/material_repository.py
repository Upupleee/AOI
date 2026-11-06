"""料号数据仓储的 SQLite 实现。"""
import json
import logging
import sqlite3
from typing import Optional

from core.interfaces.i_repository import IMaterialRepository
from core.models.material import Material
from data.database import DatabaseManager

logger = logging.getLogger(__name__)


class SqliteMaterialRepository(IMaterialRepository):
    """基于 SQLite 的料号仓储实现。"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def get_all(self) -> dict:
        """获取所有料号。

        返回字典:键为 material_id,值为 Material 对象的 to_dict()。
        """
        result = {}
        conn = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM materials")
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                material = self._row_to_material(row, columns)
                if material:
                    result[material.material_id] = material.to_dict()
        except sqlite3.Error as e:
            logger.error("查询所有料号失败: %s", e)
        finally:
            if conn:
                conn.close()
        return result

    def get_all_raw(self) -> dict:
        """获取所有料号的原始字典格式。

        兼容旧的 material_data.json 读取方式,
        返回 {material_id: {A面: ..., B面: ..., ...}} 形式。
        """
        result = {}
        conn = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM materials")
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                material = self._row_to_material(row, columns)
                if material:
                    result[material.material_id] = material.to_dict()
        except sqlite3.Error as e:
            logger.error("查询所有料号(原始格式)失败: %s", e)
        finally:
            if conn:
                conn.close()
        return result

    def get_by_id(self, material_id: str) -> Optional[Material]:
        """按 ID 获取料号。"""
        conn = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM materials WHERE material_id = ?",
                (material_id,),
            )
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            if row:
                return self._row_to_material(row, columns)
        except sqlite3.Error as e:
            logger.error("查询料号失败 %s: %s", material_id, e)
        finally:
            if conn:
                conn.close()
        return None

    def save(self, material: Material):
        """插入或更新料号(基于 material_id 主键)。"""
        conn = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO materials (
                    material_id, a_image_path, b_image_path,
                    a_slices, b_slices, a_thumbnail, b_thumbnail,
                    template_type, material_type, board_thickness,
                    etch_value, board_polarity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                material.material_id,
                material.a_image_path,
                material.b_image_path,
                json.dumps(material.a_slices, ensure_ascii=False),
                json.dumps(material.b_slices, ensure_ascii=False),
                material.a_thumbnail,
                material.b_thumbnail,
                material.template_type,
                material.material_type,
                material.board_thickness,
                material.etch_value,
                material.board_polarity,
            ))
            conn.commit()
            logger.info("保存料号成功: %s", material.material_id)
        except sqlite3.Error as e:
            logger.error("保存料号失败 %s: %s", material.material_id, e)
            raise
        finally:
            if conn:
                conn.close()

    def delete(self, material_id: str):
        """删除料号。"""
        conn = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM materials WHERE material_id = ?",
                (material_id,),
            )
            conn.commit()
            logger.info("删除料号成功: %s", material_id)
        except sqlite3.Error as e:
            logger.error("删除料号失败 %s: %s", material_id, e)
            raise
        finally:
            if conn:
                conn.close()

    def _row_to_material(self, row, columns) -> Optional[Material]:
        """将数据库行转换为 Material 对象。"""
        try:
            data = dict(zip(columns, row))
            # a_slices / b_slices 存为 JSON 文本,读取时反序列化
            return Material(
                material_id=data["material_id"],
                a_image_path=data.get("a_image_path") or "",
                b_image_path=data.get("b_image_path") or "",
                a_slices=json.loads(data.get("a_slices") or "[]"),
                b_slices=json.loads(data.get("b_slices") or "[]"),
                a_thumbnail=data.get("a_thumbnail"),
                b_thumbnail=data.get("b_thumbnail"),
                template_type=data.get("template_type") or "默认",
                material_type=data.get("material_type") or "内层",
                board_thickness=float(data.get("board_thickness") or 1.0),
                etch_value=int(data.get("etch_value") or 20),
                board_polarity=data.get("board_polarity") or "正极",
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error("解析料号记录失败: %s", e)
            return None
