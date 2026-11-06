"""UI 层。

包含视图(View)和视图模型(ViewModel):
    - views:        只负责 UI 显示和事件转发,不包含业务逻辑
    - viewmodels:   业务逻辑代理,委托给各 Service 完成具体操作

视图通过 MainViewModel 间接访问 core 层服务,不直接依赖容器与服务。
"""
