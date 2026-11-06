"""ACS 运动控制器抽象接口。"""
from abc import ABC, abstractmethod


class IAcsComm(ABC):
    """ACS 运动控制器通信接口。"""

    @abstractmethod
    def connect(self, ip: str, port: int) -> int:
        """连接控制器,返回连接句柄(-1 表示失败)。"""

    @abstractmethod
    def disconnect(self):
        """断开连接。"""

    @abstractmethod
    def is_connected(self) -> bool:
        """是否已连接。"""

    @abstractmethod
    def run_buffer(self, buffer_id: int, buffer_file_path: str):
        """加载并运行指定 buffer 文件。"""
