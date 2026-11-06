"""数据持久化层。

提供 SQLite 数据库管理、Repository 模式的具体实现以及 JSON 到 SQLite 的数据迁移工具。
"""
from .database import DatabaseManager
from .repositories.material_repository import SqliteMaterialRepository
from .repositories.scan_record_repository import SqliteScanRecordRepository
from .json_migration import JsonToSqliteMigrator

__all__ = [
    "DatabaseManager",
    "SqliteMaterialRepository",
    "SqliteScanRecordRepository",
    "JsonToSqliteMigrator",
]
