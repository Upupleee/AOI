"""料号数据模型。"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Material:
    """料号信息。

    对应 material_data.json 中的一条记录。
    """
    material_id: str                        # 料号名称(不含 material 根目录)
    a_image_path: str = ""                  # A 面原图路径
    b_image_path: str = ""                  # B 面原图路径
    a_slices: list = field(default_factory=list)  # A 面切分后图片路径列表
    b_slices: list = field(default_factory=list)  # B 面切分后图片路径列表
    a_thumbnail: Optional[str] = None       # A 面缩略图
    b_thumbnail: Optional[str] = None       # B 面缩略图
    template_type: str = "默认"             # 模板类型
    material_type: str = "内层"             # 料号类型
    board_thickness: float = 1.0            # 板厚
    etch_value: int = 20                    # 蚀刻值
    board_polarity: str = "正极"            # 板极性

    def to_dict(self) -> dict:
        """序列化为字典(兼容旧 JSON 格式)。"""
        return {
            "A面": self.a_image_path,
            "B面": self.b_image_path,
            "A面切分后": self.a_slices,
            "B面切分后": self.b_slices,
            "A面缩略图": self.a_thumbnail,
            "B面缩略图": self.b_thumbnail,
            "templatevalue": self.template_type,
            "materialvalue": self.material_type,
            "boardthicknessvalue": self.board_thickness,
            "etchvalue": self.etch_value,
            "boardpolarityvalue": self.board_polarity,
        }

    @classmethod
    def from_dict(cls, material_id: str, data: dict) -> "Material":
        """从字典反序列化(兼容旧 JSON 格式)。"""
        return cls(
            material_id=material_id,
            a_image_path=data.get("A面", ""),
            b_image_path=data.get("B面", ""),
            a_slices=data.get("A面切分后", []),
            b_slices=data.get("B面切分后", []),
            a_thumbnail=data.get("A面缩略图"),
            b_thumbnail=data.get("B面缩略图"),
            template_type=data.get("templatevalue", "默认"),
            material_type=data.get("materialvalue", "内层"),
            board_thickness=float(data.get("boardthicknessvalue", 1.0)),
            etch_value=int(data.get("etchvalue", 20)),
            board_polarity=data.get("boardpolarityvalue", "正极"),
        )
