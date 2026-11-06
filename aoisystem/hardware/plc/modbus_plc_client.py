"""基于 pymodbus 的 PLC 客户端实现。"""
import logging
import time

from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
from pymodbus.constants import Endian

from core.interfaces.i_plc_client import IPlcClient

logger = logging.getLogger(__name__)


class ModbusPlcClient(IPlcClient):
    """封装 pymodbus.client.ModbusTcpClient 的 PLC 通信实现。

    构造函数不建立连接,延迟到 connect() 调用时再连接,
    连接失败会重试 3 次(每次间隔 5 秒)。读写操作带错误处理和自动重连。
    """

    # 连接失败最大重试次数
    MAX_CONNECT_RETRIES = 3
    # 每次重试间隔(秒)
    RETRY_INTERVAL_SECONDS = 5

    def __init__(self, ip: str, port: int = 502):
        """初始化 PLC 客户端,仅保存地址信息,不建立连接。

        Args:
            ip: PLC 的 IP 地址。
            port: PLC 的 Modbus TCP 端口,默认 502。
        """
        self._ip = ip
        self._port = port
        # 延迟到 connect() 再创建并连接实际客户端
        self._client = ModbusTcpClient(self._ip, port=self._port)

    def connect(self) -> bool:
        """连接 PLC,失败重试 3 次(每次间隔 5 秒)。

        Returns:
            连接成功返回 True,否则返回 False。
        """
        for attempt in range(1, self.MAX_CONNECT_RETRIES + 1):
            try:
                if self._client.connect():
                    logger.info("成功连接到 PLC %s:%s", self._ip, self._port)
                    return True
                logger.warning("连接 PLC 失败(第 %d/%d 次): %s:%s",
                               attempt, self.MAX_CONNECT_RETRIES, self._ip, self._port)
            except Exception as e:
                logger.warning("连接 PLC 异常(第 %d/%d 次): %s",
                               attempt, self.MAX_CONNECT_RETRIES, e)
            if attempt < self.MAX_CONNECT_RETRIES:
                time.sleep(self.RETRY_INTERVAL_SECONDS)
        logger.error("连接 PLC 失败,已达到最大重试次数 %d", self.MAX_CONNECT_RETRIES)
        return False

    def disconnect(self):
        """断开与 PLC 的连接。"""
        try:
            self._client.close()
            logger.info("已断开 PLC 连接 %s:%s", self._ip, self._port)
        except Exception as e:
            logger.error("断开 PLC 连接异常: %s", e)

    def reconnect(self):
        """重连 PLC:先断开再重新连接。"""
        logger.info("开始重连 PLC %s:%s", self._ip, self._port)
        self.disconnect()
        self.connect()

    def read_bits(self, address: int, count: int) -> list:
        """读取线圈状态,失败时自动重连并重试一次。"""
        try:
            result = self._client.read_coils(address, count)
            if result.isError():
                logger.error("读取线圈失败: %s,地址=%s,数量=%s", result, address, count)
                self.reconnect()
                return self.read_bits(address, count)
            return result.bits
        except Exception as e:
            logger.error("读取线圈异常: %s,地址=%s,数量=%s,尝试重连", e, address, count)
            self.reconnect()
            return self.read_bits(address, count)

    def write_bit(self, address: int, value: int):
        """写单个线圈,失败时自动重连并重试一次。"""
        try:
            result = self._client.write_coil(address, value)
            if result.isError():
                logger.error("写线圈失败: %s,地址=%s,值=%s", result, address, value)
                self.reconnect()
                self.write_bit(address, value)
        except Exception as e:
            logger.error("写线圈异常: %s,地址=%s,值=%s,尝试重连", e, address, value)
            self.reconnect()
            self.write_bit(address, value)

    def read_registers(self, address: int, count: int) -> list:
        """读取保持寄存器,失败时自动重连并重试一次。"""
        try:
            result = self._client.read_holding_registers(address, count, slave=1)
            if result.isError():
                logger.error("读取寄存器失败: %s,地址=%s,数量=%s", result, address, count)
                self.reconnect()
                return self.read_registers(address, count)
            return result.registers
        except Exception as e:
            logger.error("读取寄存器异常: %s,地址=%s,数量=%s,尝试重连", e, address, count)
            self.reconnect()
            return self.read_registers(address, count)

    def write_register(self, address: int, value: int):
        """写单个保持寄存器,失败时自动重连并重试一次。"""
        try:
            result = self._client.write_register(address, value)
            if result.isError():
                logger.error("写寄存器失败: %s,地址=%s,值=%s", result, address, value)
                self.reconnect()
                self.write_register(address, value)
        except Exception as e:
            logger.error("写寄存器异常: %s,地址=%s,值=%s,尝试重连", e, address, value)
            self.reconnect()
            self.write_register(address, value)

    def read_float(self, address: int) -> float:
        """读取浮点数(占 2 个寄存器),使用大端字节序、小端字序。"""
        registers = self.read_registers(address, 2)
        decoder = BinaryPayloadDecoder.fromRegisters(
            registers, byteorder=Endian.BIG, wordorder=Endian.LITTLE
        )
        return decoder.decode_32bit_float()

    def write_float(self, address: int, value: float):
        """写浮点数,使用大端字节序、小端字序,失败时自动重连并重试一次。"""
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.LITTLE)
        builder.add_32bit_float(value)
        registers = builder.to_registers()
        try:
            result = self._client.write_registers(address, registers)
            if result.isError():
                logger.error("写浮点数失败: %s,地址=%s,值=%s", result, address, value)
                self.reconnect()
                self.write_float(address, value)
        except Exception as e:
            logger.error("写浮点数异常: %s,地址=%s,值=%s,尝试重连", e, address, value)
            self.reconnect()
            self.write_float(address, value)
