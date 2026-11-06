"""视图模型层。

将业务逻辑从视图中抽出,视图只负责 UI 显示和事件转发,
所有业务操作委托给 ViewModel,ViewModel 再委托给各 Service。
"""
from .main_viewmodel import MainViewModel

__all__ = ["MainViewModel"]
