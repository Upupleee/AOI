"""基于 SPiiPlusPython 的 ACS 运动控制器通信实现。"""
import logging

import SPiiPlusPython as sp

from Acs.acs_test.acs_load_buffer_file import run_buffer_from_file
from core.interfaces.i_acs_comm import IAcsComm

logger = logging.getLogger(__name__)


class SpiiPlusAcsComm(IAcsComm):
    """封装 SPiiPlusPython 的 ACS 运动控制器通信实现。

    通过 OpenCommEthernetTCP 建立以太网 TCP 连接,
    支持加载并运行 buffer 文件。
    """

    def __init__(self):
        """初始化 ACS 通信,未连接状态。"""
        # 通信句柄,-1 表示未连接
        self._conn = -1
        self._connected = False

    def connect(self, ip: str, port: int) -> int:
        """连接 ACS 控制器,返回连接句柄。

        Args:
            ip: 控制器 IP 地址。
            port: 控制器 TCP 端口。

        Returns:
            连接句柄,失败返回 -1。
        """
        try:
            self._conn = sp.OpenCommEthernetTCP(ip, port)
            if self._conn == -1:
                logger.error("无法连接到 ACS 控制器 %s:%s", ip, port)
                self._connected = False
                return -1
            self._connected = True
            logger.info("成功连接 ACS 控制器 %s:%s,句柄=%s", ip, port, self._conn)
            return self._conn
        except Exception as e:
            logger.error("连接 ACS 控制器异常: %s", e)
            self._connected = False
            return -1

    def disconnect(self):
        """断开与 ACS 控制器的连接。"""
        if self._conn != -1:
            try:
                sp.CloseComm(self._conn)
                logger.info("已断开 ACS 连接,句柄=%s", self._conn)
            except Exception as e:
                logger.error("断开 ACS 连接异常: %s", e)
            finally:
                self._conn = -1
                self._connected = False

    def is_connected(self) -> bool:
        """返回当前是否已连接 ACS 控制器。"""
        return self._connected and self._conn != -1

    def run_buffer(self, buffer_id: int, buffer_file_path: str):
        """加载并运行指定 buffer 文件。

        Args:
            buffer_id: 目标 buffer 编号。
            buffer_file_path: buffer 文件路径。
        """
        if not self.is_connected():
            logger.error("ACS 未连接,无法运行 buffer %s", buffer_id)
            raise RuntimeError("ACS 未连接,请先调用 connect()")
        logger.info("开始运行 ACS buffer %s,文件=%s", buffer_id, buffer_file_path)
        run_buffer_from_file(self._conn, buffer_id, buffer_file_path)
        logger.info("ACS buffer %s 运行完成", buffer_id)
