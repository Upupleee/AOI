"""视图层。

只负责 UI 显示和事件转发,不包含业务逻辑。
视图加载 .ui 文件,将 UI 事件转发到 ViewModel 的 slot,
并将 ViewModel 的信号连接到 UI 更新方法。
"""
