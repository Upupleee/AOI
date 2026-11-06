"""AOI 系统入口点(重构后)。

组装依赖并启动应用。
"""
import sys
import logging
import os

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

from core.container import container
from ui.views.login_view import LoginView


def setup_logging():
    """配置全局日志。"""
    # 确保日志目录存在
    log_dir = "./Logger"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("./Logger/aoi_system.log", encoding="utf-8"),
            logging.StreamHandler(),
        ]
    )


if __name__ == "__main__":
    setup_logging()
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("./icons/xjz.png"))
    login = LoginView()
    login.show()
    sys.exit(app.exec_())
