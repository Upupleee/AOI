"""SQLite 数据库管理。

负责数据库连接管理和表结构初始化,提供统一的连接获取入口。
"""
import logging
import sqlite3

from common.config import config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite 数据库管理器。

    负责初始化数据库连接、创建表结构,提供统一的连接获取入口。
    数据库路径优先从 common.config.config 读取,若无配置则使用默认 "aoi_system.db"。
    """

    def __init__(self, db_path: str = None):
        # 优先使用显式传入的路径,其次从 config 读取,最后使用默认值
        if db_path is None:
            db_path = getattr(config, "database_path", None) or "aoi_system.db"
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库,创建表结构。"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            # 创建料号表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS materials (
                    material_id     TEXT PRIMARY KEY,
                    a_image_path    TEXT,
                    b_image_path    TEXT,
                    a_slices        TEXT,
                    b_slices        TEXT,
                    a_thumbnail     TEXT,
                    b_thumbnail     TEXT,
                    template_type   TEXT,
                    material_type   TEXT,
                    board_thickness REAL,
                    etch_value      INTEGER,
                    board_polarity  TEXT
                )
            """)
            # 创建扫描记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_records (
                    scan_id         TEXT PRIMARY KEY,
                    material_id     TEXT,
                    mode            TEXT,
                    ab_cycle        INTEGER,
                    status          TEXT,
                    start_time      TEXT,
                    end_time        TEXT,
                    coco_json_path  TEXT,
                    vrs_path        TEXT,
                    result_dir      TEXT,
                    defect_count    INTEGER,
                    notes           TEXT,
                    FOREIGN KEY (material_id) REFERENCES materials(material_id)
                )
            """)
            conn.commit()
            logger.info("数据库初始化完成: %s", self.db_path)
        except sqlite3.Error as e:
            logger.error("初始化数据库失败: %s", e)
            raise
        finally:
            if conn:
                conn.close()

    def get_connection(self) -> sqlite3.Connection:
        """获取数据库连接。

        调用方负责关闭连接(建议使用 try/finally 或 with 语句确保关闭)。
        """
        return sqlite3.connect(self.db_path)
