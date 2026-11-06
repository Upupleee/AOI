from PyQt5.QtCore import QObject, pyqtSlot, QProcess, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
import logging

from common.config import config

logger = logging.getLogger(__name__)


class ProcessController(QObject):
    started_process = pyqtSignal()
    exited = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    output_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(config.camera_working_dir)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyRead.connect(self.handle_output)
        self.proc.finished.connect(self.on_process_finished)
        self.proc.errorOccurred.connect(self.on_process_error)

    def is_running(self):
        return self.proc.state() == QProcess.Running

    @pyqtSlot()
    def start_process(self):
        if self.is_running():
            self.error_occurred.emit("Process is already running")
            return

        self.proc.start(config.camera_executable)
        if not self.proc.waitForStarted(5000):
            self.error_occurred.emit("Failed to start process")
            self.proc.kill()
        else:
            self.started_process.emit()

    @pyqtSlot()
    def send_start_command(self):
        if self.is_running():
            try:
                self.proc.write(b"start\n")
                self.proc.waitForBytesWritten(1000)
            except Exception as e:
                self.error_occurred.emit(f"Failed to send 'start': {str(e)}")
        else:
            self.error_occurred.emit("Process not ready for commands")

    @pyqtSlot()
    def send_exit_command(self):
        if self.is_running():
            try:
                self.proc.write(b"stop\n")
                self.proc.waitForBytesWritten(800)
                QTimer.singleShot(2000, lambda: self.proc.write(b"exit\n"))
                QTimer.singleShot(4000, self._finalize_process)
            except Exception as e:
                self.error_occurred.emit(f"Exception during exit: {str(e)}")
        else:
            self.error_occurred.emit("Process is not running")

    def _finalize_process(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            if not self.proc.waitForFinished(10000):
                self.error_occurred.emit("Process did not exit cleanly, forcing kill")
                self.proc.kill()
                self.proc.waitForFinished(2000)
        self.exited.emit(self.proc.exitCode())

    def handle_output(self):
        if self.proc:
            data = bytes(self.proc.readAll()).decode().strip()
            if data:
                self.output_received.emit(data)

    def on_process_error(self, error):
        msg = f"Process error: {self.proc.errorString()} (code={self.proc.exitCode()})"
        self.error_occurred.emit(msg)

    def on_process_finished(self, exit_code, exit_status):
        self.exited.emit(exit_code)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = ProcessController()

        self.btn_start = QPushButton("Start", self)
        self.btn_send_exit = QPushButton("Send Exit", self)

        self.btn_start.move(20, 20)
        self.btn_send_exit.move(20, 60)

        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_send_exit.clicked.connect(self.controller.send_exit_command)

        self.controller.output_received.connect(self.on_output)
        self.controller.error_occurred.connect(self.on_error)
        self.controller.exited.connect(self.on_exit)

    def on_start_clicked(self):
        if self.controller.is_running():
            self.controller.send_start_command()
        else:
            # 防止多次连接重复发送
            try:
                self.controller.started_process.disconnect()
            except TypeError:
                pass
            self.controller.started_process.connect(self.controller.send_start_command)
            self.controller.start_process()

    def on_output(self, text):
        logger.info("[OUTPUT] %s", text)

    def on_error(self, text):
        logger.error("[ERROR] %s", text)

    def on_exit(self, code):
        logger.info("[EXITED] with code %s", code)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Camera Trigger Controller")
    window.resize(300, 120)
    window.show()
    sys.exit(app.exec_())