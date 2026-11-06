"""AOI 系统共享模块。

提供配置加载、常量定义和工具函数，供各业务模块复用，消除重复代码。
"""
from .config import config, AppConfig
from .utils import (
    extract_number_from_path,
    generate_thumbnail,
    load_json_file,
    save_json_file,
    ensure_dir,
)

__all__ = [
    "config",
    "AppConfig",
    "extract_number_from_path",
    "generate_thumbnail",
    "load_json_file",
    "save_json_file",
    "ensure_dir",
]
