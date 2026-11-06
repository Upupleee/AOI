"""PLC 客户端抽象接口。"""
from abc import ABC, abstractmethod


class IPlcClient(ABC):
    """PLC 通信接口,抽象 Modbus 读写操作。"""

    @abstractmethod
    def connect(self) -> bool:
        """连接 PLC,成功返回 True。"""

    @abstractmethod
    def disconnect(self):
        """断开连接。"""

    @abstractmethod
    def reconnect(self):
        """重连。"""

    @abstractmethod
    def read_bits(self, address: int, count: int) -> list:
        """读取线圈状态。"""

    @abstractmethod
    def write_bit(self, address: int, value: int):
        """写单个线圈。"""

    @abstractmethod
    def read_registers(self, address: int, count: int) -> list:
        """读取保持寄存器。"""

    @abstractmethod
    def write_register(self, address: int, value: int):
        """写单个保持寄存器。"""

    @abstractmethod
    def read_float(self, address: int) -> float:
        """读取浮点数(占 2 个寄存器)。"""

    @abstractmethod
    def write_float(self, address: int, value: float):
        """写浮点数。"""
