from PyQt5 import uic
from PyQt5.QtGui import QIcon, QMovie
from PyQt5.QtWidgets import QDialog, QMessageBox
from AoiSystemWindow_test.AoiSystemMain import MyWindow
from AoiSystemWindow_test.AcsThread import AcsThread
from PLC.PlcThread import PlcThread

class InitializeWindow(QDialog):
    def __init__(self, app, plc_client, acs_connection):
        super().__init__()
        self.app = app
        self.plc_client = plc_client
        self.acs_comm = acs_connection
        self.ui = uic.loadUiType("./InitializeWindow/InitializeWindow.ui")[0]()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon('./icons/xjz.png'))
        self.movie = QMovie('./InitializeWindow/loading.gif')
        self.ui.label_gif.setMovie(self.movie)
        self.movie.start()


    def homing_thread(self):
        self.homing_thread_obj = AcsThread(self.acs_comm)
        self.homing_thread_obj.add_operation('homing')
        self.homing_thread_obj.signal_received_reset.connect(self.close_and_show_main)
        self.homing_thread_obj.start()

    # def initialize_thread(self):
    #     self.plc_initialize_thread = PlcThread(self.plc_client)

    def close_and_show_main(self):
        self.close()
        self.main_window = MyWindow(self.app, self.plc_client, self.acs_comm)
        self.main_window.show()


    
    def closeEvent(self, event):
        if self.homing_thread_obj and self.homing_thread_obj.isRunning():
            self.homing_thread_obj.stop()
            self.homing_thread_obj.wait()
        
        super().closeEvent(event)

