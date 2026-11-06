"""仓储实现层。

提供基于 SQLite 的 IMaterialRepository 和 IScanRecordRepository 具体实现。
"""
from .material_repository import SqliteMaterialRepository
from .scan_record_repository import SqliteScanRecordRepository

__all__ = [
    "SqliteMaterialRepository",
    "SqliteScanRecordRepository",
]
