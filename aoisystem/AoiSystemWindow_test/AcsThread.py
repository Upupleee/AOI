from PyQt5.QtCore import QThread, pyqtSignal
import os
import logging
from Acs.acs_test.acs_load_buffer_file import run_buffer_from_file
from common.config import config

logger = logging.getLogger(__name__)


class AcsThread(QThread):
    signal_received_reset = pyqtSignal(str)
    signal_received_finish = pyqtSignal(str)
    signal_received_path = pyqtSignal(str)

    def __init__(self, acs_comm):
        super().__init__()
        self.conn = acs_comm
        self.loop_iterations = None
        # mode 为 'homing' 表示回零操作，'moving' 表示运动操作
        self.operations = []
        self._is_running = True
        self.file_queue = None

    def add_iteration(self, value):
        self.loop_iterations = value

    def add_queue(self, queue):
        self.file_queue = queue

    def add_operation(self, mode):
        self.operations.append(mode)

    def run(self):
        if self.conn == -1:
            logger.error("无法连接到控制器")
            return
        try:
            for mode in self.operations:
                if self._is_running:
                    if mode == 'homing':
                        run_buffer_from_file(self.conn, 1, config.acs_buffer_path('x_home'))
                        run_buffer_from_file(self.conn, 2, config.acs_buffer_path('y_home'))
                        run_buffer_from_file(self.conn, 3, config.acs_buffer_path('z_home'))
                        run_buffer_from_file(self.conn, 5, config.acs_buffer_path('k2_home'))
                        run_buffer_from_file(self.conn, 4, config.acs_buffer_path('init'))
                        self.signal_received_reset.emit('RESET')
                    elif mode == 'moving':
                        for i in range(self.loop_iterations):
                            logger.info("循环第 %d 次", i+1)
                            run_buffer_from_file(self.conn, 0, config.acs_buffer_path('forward'))
                            run_buffer_from_file(self.conn, 7, config.acs_buffer_path('backward'))
                            file = os.path.join(config.temp_images_dir, f"test_imgs{i+1}.tif")
                            self.file_queue.put(file)
                            self.signal_received_path.emit(file)
                        run_buffer_from_file(self.conn, 9, config.acs_buffer_path('reset_k2'))
                        self.signal_received_finish.emit('FINISH')

        except Exception as e:
            logger.error("线程执行出错: %s", e)
        finally:
            self.operations.clear()

    def stop(self):
        self._is_running = False
        self.wait()