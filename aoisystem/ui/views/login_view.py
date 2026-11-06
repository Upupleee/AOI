"""登录视图。

新的登录视图,使用 container.initialize_hardware() 完成硬件初始化,
登录成功后实例化 MainViewModel 与 MainView 并显示主窗口。
"""
import logging

from PyQt5 import uic
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QMessageBox

from common.config import config
from common.utils import load_json_file, save_json_file
from core.container import container
from ui.viewmodels.main_viewmodel import MainViewModel
from ui.views.main_view import MainView

logger = logging.getLogger(__name__)


class LoginView(QWidget):
    """登录窗口视图。"""

    def __init__(self):
        super().__init__()
        # 加载 UI 文件
        self.ui = uic.loadUiType("LoginWindow.ui")[0]()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon("./icons/xjz.png"))

        # 主窗口引用(登录成功后创建)
        self.main_view = None

        # 连接登录按钮
        self.ui.pushButton_login.clicked.connect(self.login)

        # 加载记住的凭据
        self._load_remembered_credentials()

    def login(self):
        """登录按钮点击:验证凭据 -> 初始化硬件 -> 显示主窗口。"""
        username = self.ui.username_input.text()
        password = self.ui.password_input.text()

        if not self._authenticate(username, password):
            QMessageBox.information(self, '警告', '用户名或密码错误')
            return

        # 通过容器初始化硬件(PLC + ACS)
        if container.initialize_hardware():
            # 保存凭据(若勾选记住密码)
            self._save_remembered_credentials(username, password)
            # 关闭登录窗口
            self.close()
            # 创建并显示主窗口
            self.main_view = MainView(MainViewModel())
            self.main_view.show()
        else:
            QMessageBox.information(self, '警告', 'PLC或ACS通讯失败')

    def _authenticate(self, username: str, password: str) -> bool:
        """简单验证。

        暂时返回 True,后续可改为从配置或数据库读取。
        """
        return True

    def _load_remembered_credentials(self):
        """从 JSON 文件加载记住的凭据。"""
        data = load_json_file(config.remembered_credentials_json)
        if data.get('remember'):
            self.ui.username_input.setText(data.get('username', ''))
            self.ui.password_input.setText(data.get('password', ''))
            self.ui.checkBox_remember_password.setChecked(True)

    def _save_remembered_credentials(self, username: str, password: str):
        """保存凭据到 JSON 文件。"""
        data = {
            'username': username,
            'password': password,
            'remember': self.ui.checkBox_remember_password.isChecked()
        }
        save_json_file(config.remembered_credentials_json, data)
