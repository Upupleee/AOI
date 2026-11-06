"""模板匹配抽象接口。"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class MatchResult:
    """模板匹配结果。"""

    def __init__(self, overlay_image=None, file_name: str = "",
                 img_name: str = ""):
        self.overlay_image = overlay_image
        self.file_name = file_name
        self.img_name = img_name


class ITemplateMatcher(ABC):
    """模板匹配接口,封装 LoFTR + LightGlue 对齐流程。"""

    @abstractmethod
    def match(self, sample_image_path: str,
              template_image_paths: list,
              params: dict) -> Optional[MatchResult]:
        """执行模板匹配,返回匹配结果。"""
