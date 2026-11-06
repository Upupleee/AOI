"""真空与风机控制业务服务。

将 MyWindow 中真空/风机/报警相关的业务逻辑抽出为纯 Python 服务。
服务通过 IPlcClient 接口访问 PLC,通过 config 单例访问配置。
仅使用标准库 threading.Timer 实现风机延迟关闭,不依赖 PyQt5。
"""
import time
import logging
import threading

from core.interfaces.i_plc_client import IPlcClient

logger = logging.getLogger(__name__)


class VacuumService:
    """真空与风机控制服务,封装吸真空/破真空/风机的协同控制。"""

    def __init__(self, plc_client: IPlcClient, config):
        """构造函数注入 PLC 客户端与配置。

        Args:
            plc_client: PLC 客户端接口。
            config: 配置单例,提供线圈地址与风机延迟参数。
        """
        self.plc_client = plc_client
        self.config = config
        # 运行时状态
        self.vacuum_on = False       # 当前吸真空是否开启
        self.blower_on = False       # 当前风机是否开启
        self.last_click_time = 0.0   # 上次切换真空的时间戳
        # 风机延迟关闭定时器
        self.blower_timer = None

    def toggle_vacuum(self) -> bool:
        """切换真空状态,返回新的真空状态。

        参考main_window.py toggle_vacuum 的逻辑:
            - 如果当前未开真空,调用 start_vacuum
            - 如果当前已开真空,调用 stop_vacuum
        """
        current_time = time.time()
        time_interval = current_time - self.last_click_time
        self.last_click_time = current_time

        try:
            if not self.vacuum_on:
                self.start_vacuum(time_interval)
            else:
                self.stop_vacuum(time_interval)
            self.vacuum_on = not self.vacuum_on
        except Exception as e:
            logger.error("切换真空状态出错: %s", e)

        return self.vacuum_on

    def start_vacuum(self, time_interval: float):
        """开启吸真空。

        - 如果风机未开或时间间隔超过阈值,调用 turn_on_blower
        - 写 PLC: vacuum_release=0, vacuum=1

        Args:
            time_interval: 距上次切换的时间间隔(秒)。
        """
        threshold = self.config.vacuum_time_interval_threshold
        try:
            if not self.blower_on or time_interval > threshold:
                self.turn_on_blower()
            self.plc_client.write_bit(self.config.coil('vacuum_release'), 0)
            self.plc_client.write_bit(self.config.coil('vacuum'), 1)
            logger.info("吸真空已开启")
        except Exception as e:
            logger.error("吸真空操作出错: %s", e)
            self.vacuum_on = False

    def stop_vacuum(self, time_interval: float):
        """关闭吸真空。

        - 写 PLC: vacuum_release=1, vacuum=0
        - 如果时间间隔超过阈值,启动定时器延迟关风机

        Args:
            time_interval: 距上次切换的时间间隔(秒)。
        """
        threshold = self.config.vacuum_time_interval_threshold
        try:
            self.plc_client.write_bit(self.config.coil('vacuum_release'), 1)
            self.plc_client.write_bit(self.config.coil('vacuum'), 0)
            logger.info("吸真空已关闭")

            # 超过阈值才延迟关风机,避免频繁启停
            if time_interval > threshold:
                if self.blower_timer:
                    self.blower_timer.cancel()
                delay = self.config.blower_delay_seconds
                self.blower_timer = threading.Timer(delay, self.turn_off_blower)
                self.blower_timer.start()
        except Exception as e:
            logger.error("破真空操作出错: %s", e)
            self.vacuum_on = True

    def turn_on_blower(self):
        """开启风机,写 PLC blower=1。"""
        self.plc_client.write_bit(self.config.coil('blower'), 1)
        self.blower_on = True
        logger.info("风机已开启")

    def turn_off_blower(self):
        """关闭风机,写 PLC blower=0。"""
        self.plc_client.write_bit(self.config.coil('blower'), 0)
        self.blower_on = False
        logger.info("风机已关闭")

    def reset_alarm(self):
        """报警复位。

        写 PLC: software_ready=0, alarm_reset=1, light_curtain=0
        清除 alarm_1~4。
        参考 main_window.py 中 ResetAlarm 和 reset 方法。
        """
        try:
            self.plc_client.write_bit(self.config.coil('software_ready'), 0)
            self.plc_client.write_bit(self.config.coil('alarm_reset'), 1)
            self.plc_client.write_bit(self.config.coil('light_curtain'), 0)
            # 清除所有报警位(仅当对应位为 1 时才写 0)
            for name in ('alarm_1', 'alarm_2', 'alarm_3', 'alarm_4'):
                if self.plc_client.read_bits(self.config.coil(name), 1)[0]:
                    self.plc_client.write_bit(self.config.coil(name), 0)
            logger.info("报警已复位")
        except Exception as e:
            logger.error("报警复位失败: %s", e)

    def read_sensors(self) -> dict:
        """读取温度、湿度、负压。

        参考 main_window.py 中 update_readings 的传感器读取逻辑。

        Returns:
            {'temperature': float, 'wetness': float, 'pressure': float}
            读取失败的字段返回 0.0。
        """
        result = {"temperature": 0.0, "wetness": 0.0, "pressure": 0.0}
        try:
            result["wetness"] = self.plc_client.read_float(
                self.config.register('wetness'))
        except Exception as e:
            logger.warning("读取湿度失败: %s", e)
        try:
            result["temperature"] = self.plc_client.read_float(
                self.config.register('temperature'))
        except Exception as e:
            logger.warning("读取温度失败: %s", e)
        try:
            result["pressure"] = self.plc_client.read_float(
                self.config.register('pressure'))
        except Exception as e:
            logger.warning("读取负压失败: %s", e)
        return result

    def check_safety(self) -> bool:
        """检查光幕状态,如果触发光幕则触发报警。

        参考 main_window.py 中 update_readings 的光幕检查逻辑。

        Returns:
            True 表示安全(光幕未触发), False 表示不安全(光幕已触发并报警)。
        """
        try:
            light_curtain_status = self.plc_client.read_bits(
                self.config.coil('light_curtain'), 1)[0]
            if light_curtain_status:
                self.plc_client.write_bit(self.config.coil('alarm_3'), 1)
                self.plc_client.write_bit(self.config.coil('alarm_4'), 1)
                logger.warning("光幕触发,已触发报警")
                return False
        except Exception as e:
            logger.error("检查安全状态失败: %s", e)
            return False
        return True
