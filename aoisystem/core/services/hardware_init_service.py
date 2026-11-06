"""硬件初始化业务服务。

将 LoginWindow.py 中 Initialization() 方法及 MyWindow 中 ACS_disconnect 的
硬件连接/断开逻辑抽出为纯 Python 服务,不依赖 PyQt5。
服务通过 IPlcClient 与 IAcsComm 接口访问硬件,通过 config 单例访问配置。
"""
import logging

from core.interfaces.i_acs_comm import IAcsComm
from core.interfaces.i_plc_client import IPlcClient

logger = logging.getLogger(__name__)


class HardwareInitializationService:
    """硬件初始化服务,负责 PLC 与 ACS 的连接和断开。"""

    def __init__(self, plc_client: IPlcClient,
                 acs_comm: IAcsComm, config):
        """构造函数注入 PLC 客户端、ACS 通信与配置。

        Args:
            plc_client: PLC 客户端接口。
            acs_comm: ACS 运动控制器通信接口。
            config: 配置单例,提供连接参数与 buffer 路径。
        """
        self.plc_client = plc_client
        self.acs_comm = acs_comm
        self.config = config

    def initialize(self) -> bool:
        """初始化硬件连接。

        流程:
            1. 连接 PLC,读取 ready_status 线圈
            2. 如果 PLC 就绪,连接 ACS
            3. 返回是否全部成功

        参考 LoginWindow.py 中 Initialization() 方法的逻辑。

        Returns:
            全部连接成功返回 True,否则返回 False。
        """
        # 1. 连接 PLC
        try:
            if not self.plc_client.connect():
                logger.error("PLC 连接失败")
                return False
        except Exception as e:
            logger.error("PLC 连接异常: %s", e)
            return False

        # 2. 读取 ready_status 线圈,确认 PLC 就绪
        try:
            ready_status = self.plc_client.read_bits(
                self.config.coil('ready_status'), 1)[0]
            if not ready_status:
                logger.error("PLC 未就绪 (ready_status=0)")
                return False
        except Exception as e:
            logger.error("读取 PLC ready_status 失败: %s", e)
            return False

        # 3. PLC 就绪后连接 ACS
        try:
            conn = self.acs_comm.connect(self.config.acs_ip, self.config.acs_port)
            if conn == -1:
                logger.error("ACS 连接失败")
                return False
        except Exception as e:
            logger.error("ACS 连接异常: %s", e)
            return False

        logger.info("硬件初始化成功")
        return True

    def shutdown(self):
        """断开所有硬件连接。

        流程:
            1. 写 PLC software_ready=0
            2. ACS 回零(依次运行 x_home / y_home / z_home / k2_home / init buffer)
            3. 断开 ACS 与 PLC 连接

        参考 main_window.py 中 reset 与 ACS_disconnect 方法的逻辑。
        """
        # 1. PLC 软件就绪位清零
        try:
            self.plc_client.write_bit(self.config.coil('software_ready'), 0)
            logger.info("已清除 software_ready")
        except Exception as e:
            logger.warning("清除 software_ready 失败: %s", e)

        # 2. ACS 回零(依次运行各轴回零 buffer,最后运行 init buffer)
        try:
            if self.acs_comm.is_connected():
                # buffer 编号与 AcsThread.py 中 homing 操作保持一致
                self.acs_comm.run_buffer(1, self.config.acs_buffer_path('x_home'))
                self.acs_comm.run_buffer(2, self.config.acs_buffer_path('y_home'))
                self.acs_comm.run_buffer(3, self.config.acs_buffer_path('z_home'))
                self.acs_comm.run_buffer(5, self.config.acs_buffer_path('k2_home'))
                self.acs_comm.run_buffer(4, self.config.acs_buffer_path('init'))
                logger.info("ACS 已回零")
        except Exception as e:
            logger.warning("ACS 回零失败: %s", e)

        # 3. 断开连接
        try:
            self.acs_comm.disconnect()
        except Exception as e:
            logger.warning("断开 ACS 连接失败: %s", e)
        try:
            self.plc_client.disconnect()
        except Exception as e:
            logger.warning("断开 PLC 连接失败: %s", e)
        logger.info("硬件已关闭")
