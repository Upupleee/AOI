"""基于 QProcess 的相机触发进程实现。"""
import logging
from abc import ABCMeta

from PyQt5.QtCore import QObject, QProcess, QTimer, pyqtSignal

from common.config import config
from core.interfaces.i_camera import ICamera

logger = logging.getLogger(__name__)


class QABCMeta(type(QObject), ABCMeta):
    """组合 QObject 元类与 ABCMeta,解决多重继承元类冲突。"""


class ProcessCamera(ICamera, QObject, metaclass=QABCMeta):
    """封装 QProcess 调用 external_trigger 的相机触发实现。

    通过启动外部进程并写入命令实现相机触发采图,
    提供进程生命周期信号便于上层订阅。
    """

    # 进程启动信号
    started_process = pyqtSignal()
    # 进程退出信号,携带退出码
    exited = pyqtSignal(int)
    # 进程错误信号,携带错误描述
    error_occurred = pyqtSignal(str)
    # 进程输出信号,携带标准输出内容
    output_received = pyqtSignal(str)

    # 等待进程启动的超时时间(毫秒)
    START_TIMEOUT_MS = 5000
    # 发送 exit 后等待进程退出的超时时间(毫秒)
    FINALIZE_TIMEOUT_MS = 10000
    # 命令字节数据写入等待超时(毫秒)
    BYTES_WRITTEN_TIMEOUT_MS = 1000

    def __init__(self):
        """初始化相机进程控制器,创建 QProcess 并连接信号。"""
        QObject.__init__(self)
        # 创建外部进程
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(config.camera_working_dir)
        # 合并标准输出和标准错误通道
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        # 连接进程信号到内部处理槽
        self._proc.readyRead.connect(self._handle_output)
        self._proc.finished.connect(self._on_process_finished)
        self._proc.errorOccurred.connect(self._on_process_error)
        logger.info("相机进程控制器初始化完成,工作目录=%s", config.camera_working_dir)

    def is_running(self) -> bool:
        """返回相机进程是否正在运行。"""
        return self._proc.state() == QProcess.Running

    def start(self) -> bool:
        """启动相机触发进程,等待 5 秒确认启动。

        Returns:
            启动成功返回 True,否则返回 False。
        """
        if self.is_running():
            msg = "相机进程已在运行"
            logger.warning(msg)
            self.error_occurred.emit(msg)
            return False

        logger.info("启动相机进程: %s", config.camera_executable)
        self._proc.start(config.camera_executable)
        if not self._proc.waitForStarted(self.START_TIMEOUT_MS):
            msg = "相机进程启动失败"
            logger.error(msg)
            self.error_occurred.emit(msg)
            self._proc.kill()
            return False

        logger.info("相机进程启动成功")
        self.started_process.emit()
        return True

    def send_trigger_command(self):
        """发送触发采图命令(写入 b"start\\n")。"""
        if self.is_running():
            try:
                self._proc.write(b"start\n")
                self._proc.waitForBytesWritten(self.BYTES_WRITTEN_TIMEOUT_MS)
                logger.info("已发送相机触发命令 'start'")
            except Exception as e:
                msg = f"发送 'start' 命令失败: {e}"
                logger.error(msg)
                self.error_occurred.emit(msg)
        else:
            msg = "相机进程未就绪,无法发送命令"
            logger.warning(msg)
            self.error_occurred.emit(msg)

    def send_exit_command(self):
        """发送退出命令:先写 b"stop\\n",延迟 2 秒写 b"exit\\n",延迟 4 秒后强制结束。"""
        if self.is_running():
            try:
                self._proc.write(b"stop\n")
                self._proc.waitForBytesWritten(800)
                logger.info("已发送 'stop' 命令")
                # 延迟 2 秒发送 exit
                QTimer.singleShot(2000, lambda: self._proc.write(b"exit\n"))
                # 延迟 4 秒后执行最终清理
                QTimer.singleShot(4000, self._finalize_process)
            except Exception as e:
                msg = f"发送退出命令异常: {e}"
                logger.error(msg)
                self.error_occurred.emit(msg)
        else:
            msg = "相机进程未运行,无法发送退出命令"
            logger.warning(msg)
            self.error_occurred.emit(msg)

    def _finalize_process(self):
        """最终清理进程:等待退出,超时则强制 kill。"""
        if self._proc and self._proc.state() != QProcess.NotRunning:
            if not self._proc.waitForFinished(self.FINALIZE_TIMEOUT_MS):
                msg = "相机进程未正常退出,强制结束"
                logger.warning(msg)
                self.error_occurred.emit(msg)
                self._proc.kill()
                self._proc.waitForFinished(2000)
        self.exited.emit(self._proc.exitCode())

    def _handle_output(self):
        """处理进程输出,转发到 output_received 信号。"""
        if self._proc:
            data = bytes(self._proc.readAll()).decode().strip()
            if data:
                logger.debug("相机进程输出: %s", data)
                self.output_received.emit(data)

    def _on_process_error(self, error):
        """处理进程错误,转发到 error_occurred 信号。"""
        msg = f"相机进程错误: {self._proc.errorString()} (code={self._proc.exitCode()})"
        logger.error(msg)
        self.error_occurred.emit(msg)

    def _on_process_finished(self, exit_code, exit_status):
        """处理进程结束,转发到 exited 信号。"""
        logger.info("相机进程结束,退出码=%s", exit_code)
        self.exited.emit(exit_code)
