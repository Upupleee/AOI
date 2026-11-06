import time
import logging
from PyQt5.QtCore import QThread, pyqtSignal
from Acs.acs_test.acs_load_buffer_file import run_buffer_from_file

logger = logging.getLogger(__name__)


class PlcThread(QThread):
    signal_received_updated = pyqtSignal(dict)
    signal_received_reset = pyqtSignal(str)
    signal_received_finish = pyqtSignal(str)

    def __init__(self, plc_comm):
        super().__init__()
        self.conn = plc_comm
        self._is_running = True

    def run(self):
        if self.conn == -1:
            logger.error("无法连接到控制器")
            return
        platform_status_address = 1
        light_status_address = 2
        

        while self._is_running:      
            try:
                status_bits = self.conn.read_bits(0,5)
                self.signal_received_reset.emit('RESET')

                self.signal_received_finish.emit('FINISH')
                time.sleep(0.01)
            except Exception as e:
                logger.error("PLC通讯出错: %s", e)


    def stop(self):
        self._is_running = False
        self.wait()
    