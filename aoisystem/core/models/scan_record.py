"""扫描记录数据模型。"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ScanStatus(Enum):
    """扫描状态。"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ScanRecord:
    """单次扫描记录。

    对应一次完整的检测流程(A/B/AB 面)。
    """
    scan_id: str                           # 扫描 ID(料号_时间戳)
    material_id: str                       # 关联料号
    mode: str = "A"                        # 检测模式: A / B / AB
    ab_cycle: int = 0                      # AB 面循环次数
    status: ScanStatus = ScanStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    coco_json_path: str = ""               # 检测结果 COCO JSON 路径
    vrs_path: str = ""                     # VRS 文件路径
    result_dir: str = ""                   # 结果目录
    defect_count: int = 0                  # 缺陷数量
    notes: str = ""                        # 备注

    def to_dict(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "material_id": self.material_id,
            "mode": self.mode,
            "ab_cycle": self.ab_cycle,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "coco_json_path": self.coco_json_path,
            "vrs_path": self.vrs_path,
            "result_dir": self.result_dir,
            "defect_count": self.defect_count,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScanRecord":
        return cls(
            scan_id=data["scan_id"],
            material_id=data["material_id"],
            mode=data.get("mode", "A"),
            ab_cycle=data.get("ab_cycle", 0),
            status=ScanStatus(data.get("status", "pending")),
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            coco_json_path=data.get("coco_json_path", ""),
            vrs_path=data.get("vrs_path", ""),
            result_dir=data.get("result_dir", ""),
            defect_count=data.get("defect_count", 0),
            notes=data.get("notes", ""),
        )
