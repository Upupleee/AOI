# This Python file uses the following encoding: utf-8
import os
import sys
import json
import logging
import SPiiPlusPython as sp
from PyQt5 import uic
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PLC.PLCmodbus import PLCConnection
from AoiSystemWindow_test.AoiSystemMain import MyWindow
from common.config import config
from common.utils import load_json_file, save_json_file


class LoginWidget(QWidget):
    def __init__(self):
        super(LoginWidget, self).__init__()
        self.plc_client = None
        self.acs_connection = None
        self.ui = uic.loadUiType("LoginWindow.ui")[0]()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon('./icons/xjz.png'))
        self.ui.pushButton_login.clicked.connect(self.login)
        self.load_remembered_credentials()


    def Initialization(self):
        try:
            self.plc_client = PLCConnection(config.plc_ip, port=config.plc_port)
            ready_status = self.plc_client.read_bits(config.coil('ready_status'), 1)[0]

            if ready_status:
                self.acs_connection = sp.OpenCommEthernetTCP(config.acs_ip, config.acs_port)
                if self.acs_connection == -1:
                    return False
                else:
                    return True
            else:
                return False
        except Exception as e:
            logging.error("初始化失败: %s", e)
            return False

    def login(self):
        username = self.ui.username_input.text()
        password = self.ui.password_input.text()
        initialization_flag = self.Initialization()
        if initialization_flag:
            self.close()
            self.main_window = MyWindow(app, self.plc_client, self.acs_connection)
            self.main_window.show()
        else:
            QMessageBox.information(self, '警告', '请检查PLC或ACS通讯', QMessageBox.Yes)

    def load_remembered_credentials(self):
        data = load_json_file(config.remembered_credentials_json)
        if data.get('remember'):
            self.ui.username_input.setText(data.get('username'))
            self.ui.password_input.setText(data.get('password'))
            self.ui.checkBox_remember_password.setChecked(True)

    def save_remembered_credentials(self, username, password):
        data = {
            'username': username,
            'password': password,
            'remember': self.ui.checkBox_remember_password.isChecked()
        }
        save_json_file(config.remembered_credentials_json, data)


if __name__ == "__main__":
    app = QApplication([])
    widget = LoginWidget()
    widget.show()
    sys.exit(app.exec_())
