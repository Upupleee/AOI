"""相机抽象接口。"""
from abc import ABC, abstractmethod


class ICamera(ABC):
    """相机控制接口,封装外部触发采图进程。"""

    @abstractmethod
    def start(self) -> bool:
        """启动相机触发进程,成功返回 True。"""

    @abstractmethod
    def send_trigger_command(self):
        """发送触发采图命令。"""

    @abstractmethod
    def send_exit_command(self):
        """发送退出命令。"""

    @abstractmethod
    def is_running(self) -> bool:
        """相机进程是否在运行。"""
