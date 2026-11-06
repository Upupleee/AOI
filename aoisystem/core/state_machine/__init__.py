"""扫描流程状态机模块。

提供扫描检测流程的状态枚举与状态机,纯 Python 实现,不依赖 PyQt5。
"""
from .scan_state_machine import ScanState, ScanStateMachine

__all__ = ["ScanState", "ScanStateMachine"]
