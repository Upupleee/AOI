"""数据持久化抽象接口(Repository 模式)。"""
from abc import ABC, abstractmethod
from typing import List, Optional

from core.models.material import Material
from core.models.scan_record import ScanRecord


class IMaterialRepository(ABC):
    """料号数据仓储接口。"""

    @abstractmethod
    def get_all(self) -> dict:
        """获取所有料号数据。"""

    @abstractmethod
    def get_by_id(self, material_id: str) -> Optional[Material]:
        """按 ID 获取料号。"""

    @abstractmethod
    def save(self, material: Material):
        """保存料号。"""

    @abstractmethod
    def delete(self, material_id: str):
        """删除料号。"""


class IScanRecordRepository(ABC):
    """扫描记录仓储接口。"""

    @abstractmethod
    def get_by_material(self, material_id: str) -> List[ScanRecord]:
        """获取指定料号的扫描记录。"""

    @abstractmethod
    def save(self, record: ScanRecord):
        """保存扫描记录。"""

    @abstractmethod
    def delete(self, scan_id: str):
        """删除扫描记录。"""
