"""检测参数数据模型。"""
from dataclasses import dataclass


@dataclass
class DetectionParams:
    """检测流程参数(从 motion_settings.ini 加载)。"""
    rows: int = 6
    cols: int = 6
    auto_split: bool = True
    overlap_temp: int = 0
    overlap_samp: int = 800
    save_image: bool = True
    flag: int = 6                         # 模型选择标志
    slice_win: int = 650                  # 切片窗口大小
    conf_thres: float = 0.2               # 置信度阈值
    mode: str = "A"                       # 当前检测面
    part_infer: bool = False
    grid_size: int = 7
    max_sampling_per_grid: int = 100
    enhance_algorithms: str = "clahe"
    enhance_param: list = None
    extractor_name: str = "aliked"

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "auto_split": self.auto_split,
            "overlap_temp": self.overlap_temp,
            "overlap_samp": self.overlap_samp,
            "save_image": self.save_image,
            "flag": self.flag,
            "slice_win": self.slice_win,
            "conf_thres": self.conf_thres,
            "mode": self.mode,
            "part_infer": self.part_infer,
            "grid_size": self.grid_size,
            "max_sampling_per_grid": self.max_sampling_per_grid,
            "enhance_algorithms": self.enhance_algorithms,
            "enhance_param": self.enhance_param or [50, 100, "None"],
            "extractor_name": self.extractor_name,
        }

    @classmethod
    def from_ini(cls, parser, section: str = "General") -> "DetectionParams":
        """从 ConfigParser 加载参数。"""
        def get(key, default, cast=str):
            if parser.has_option(section, key):
                val = parser.get(section, key)
                try:
                    return cast(val)
                except (ValueError, TypeError):
                    return default
            return default

        return cls(
            rows=get("row", 6, int),
            cols=get("col", 6, int),
            auto_split=parser.getboolean(section, "Auto_split", fallback=True) if parser.has_option(section, "Auto_split") else True,
            overlap_temp=get("overlap_temp", 0, int),
            overlap_samp=get("overlap_samp", 800, int),
            save_image=parser.getboolean(section, "SaveImage", fallback=True) if parser.has_option(section, "SaveImage") else True,
            flag=get("flag", 6, int),
            slice_win=get("slice", 650, int),
            conf_thres=get("thres", 0.2, float),
            part_infer=parser.getboolean(section, "PartInfer", fallback=False) if parser.has_option(section, "PartInfer") else False,
        )
