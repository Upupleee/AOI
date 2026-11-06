# This Python file uses the following encoding: utf-8
import sys
import os
from PyQt5 import uic
from PyQt5.QtCore import QProcess, pyqtSignal
# from AOI.AoiSystmWindow.AoiSystemMain import MyWindow
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QRadioButton, QDialogButtonBox

cur_dir = os.path.dirname(sys.argv[0])
if cur_dir == "":
    cur_dir = '.'

class LogoutDialog(QDialog):
    closeMainWindowSignal = pyqtSignal()
    def __init__(self):
        super(LogoutDialog, self).__init__()
        self.ui = uic.loadUiType("./LogoutWindow/LogoutWindow.ui")[0]()
        self.ui.setupUi(self)


        self.ui.buttonBox.accepted.connect(self.execute_function)
        self.ui.buttonBox.rejected.connect(self.close)



    def execute_function(self):
        if self.ui.radioButton_logout_app.isChecked():
            self.closeMainWindowSignal.emit()
            self.close()
            QApplication.quit()
        elif self.ui.radioButton_restart_app.isChecked():
            self.closeMainWindowSignal.emit()
            self.close()
            QApplication.quit()
            QProcess.startDetached(QApplication.applicationFilePath(), [os.path.join(cur_dir, 'LoginWindow.py')])
        elif self.ui.radioButton_restart_system.isChecked():
            self.closeMainWindowSignal.emit()
            self.close()
            QApplication.quit()
            process = QProcess()
            process.startDetached("shutdown -r now")
        elif self.ui.radioButton_shutdown_system.isChecked():
            self.closeMainWindowSignal.emit()
            self.close()
            QApplication.quit()
            process = QProcess()
            process.startDetached("shutdown -h now")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LogoutDialog()
    window.show()
    sys.exit(app.exec_())
