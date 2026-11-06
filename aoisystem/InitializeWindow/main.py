# import sys
# import mmap
# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
# from PyQt5.QtCore import QThread, pyqtSignal, QSharedMemory

# class LoginWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.initUI()

#     def initUI(self):
#         layout = QVBoxLayout()
#         login_button = QPushButton('登录')
#         login_button.clicked.connect(self.login)
#         layout.addWidget(login_button)
#         self.setLayout(layout)
#         self.setWindowTitle('登录界面')
#         self.resize(300,200)

#     def login(self):
#         self.close()
#         self.loading_window = LoadingWindow()
#         self.loading_window.show()

# class LoadingWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.initUI()
        
#         self.button.clicked.connect(self.monitor_thread)


#     def initUI(self):
#         layout = QVBoxLayout()
#         self.label = QLabel('正在加载...')
#         self.button = QPushButton('初始化')
#         layout.addWidget(self.label)
#         layout.addWidget(self.button)
#         self.setLayout(layout)
#         self.setWindowTitle('加载界面')
#         self.resize(300,300)

#     def monitor_thread(self):
#         monitor_thread = MonitorThread()
#         monitor_thread.signal_received.connect(self.close_and_show_main)
#         monitor_thread.start()   

#     def close_and_show_main(self):
#         # self.hide()
#         main_window = MainWindow()
#         main_window.show()
#         self.close()

# class MonitorThread(QThread):
#     signal_received = pyqtSignal()

#     def __init__(self):
#         super().__init__()
#         self.size = 1024
#         self.shm_file = '/home/kabuda/AOI_Ui/PyUi/AOI/my_shm'
#         # self.shared_memory = QSharedMemory("my_shm")
#         # if not self.shared_memory.create(1024):
#         #     print("无法创建共享内存。")


#     def read_from_shared_memory(self):
#         with open(self.shm_file, 'r+b') as f:
#             shm = mmap.mmap(f.fileno(), self.size)
#             shm.seek(0)
#             data = shm.read(self.size).decode('utf-8').rstrip('\x00')  # 去除末尾的空字符
#         return data
    
#     def write_to_shared_memory(self, size, shm_file, message):
#         encoded_message = message.encode('utf-8')
#         #size = len(encoded_message) #动态设置共享内存大小，需要修改分配的共享内文文件大小

#         if len(encoded_message) > size:
#             raise ValueError(f"Message is too long. Maximum size is {size} bytes.")
        
#         with open(shm_file, 'r+b') as f:
#             shm = mmap.mmap(f.fileno(), size)
#             shm.seek(0)
#             shm.write(encoded_message)
#             shm.flush()


#     def run(self):
#         while True:
#             data = self.read_from_shared_memory()
#             if data == "RESET":
#                 self.write_to_shared_memory(self.size, self.shm_file, '\x00' * len(data)) 
#                 self.signal_received.emit()
#                 break

#             self.msleep(100)

# class MainWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.initUI()

#     def initUI(self):
#         layout = QVBoxLayout()
#         self.label = QLabel('主界面')
#         layout.addWidget(self.label)
#         self.setLayout(layout)
#         self.setWindowTitle('主界面')
#         self.resize(200,200)

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     login_window = LoginWindow()
#     login_window.show()
#     sys.exit(app.exec_())


import sys
import mmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        login_button = QPushButton('登录')
        login_button.clicked.connect(self.login)
        layout.addWidget(login_button)
        self.setLayout(layout)
        self.setWindowTitle('登录界面')
        self.resize(300, 200)

    def login(self):
        self.close()
        self.loading_window = LoadingWindow()
        self.loading_window.show()

class LoadingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.monitor_thread = MonitorThread(self)
        self.monitor_thread.signal_received.connect(self.close_and_show_main)
        self.monitor_thread.start()

    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel('正在加载...')
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setWindowTitle('加载界面')
        self.resize(300, 300)

    def close_and_show_main(self):
        self.close()
        self.main_window = MainWindow()
        self.main_window.show()
        # self.close()

class MonitorThread(QThread):
    signal_received = pyqtSignal()

    def __init__(self, loading_window):
        super().__init__()
        self.loading_window = loading_window
        self.size = 1024
        self.shm_file = '/home/kabuda/AOI_Ui/PyUi/AOI/my_shm'

    def read_from_shared_memory(self):
        with open(self.shm_file, 'r+b') as f:
            shm = mmap.mmap(f.fileno(), self.size)
            shm.seek(0)
            data = shm.read(self.size).decode('utf-8').rstrip('\x00')  # 去除末尾的空字符
        return data

    def write_to_shared_memory(self, size, shm_file, message):
        encoded_message = message.encode('utf-8')
        if len(encoded_message) > size:
            raise ValueError(f"Message is too long. Maximum size is {size} bytes.")
        with open(shm_file, 'r+b') as f:
            shm = mmap.mmap(f.fileno(), size)
            shm.seek(0)
            shm.write(encoded_message)
            shm.flush()

    def run(self):
        while True:
            data = self.read_from_shared_memory()
            if data == "RESET":
                self.write_to_shared_memory(self.size, self.shm_file, '\x00' * len(data))
                self.signal_received.emit()
                break
            self.msleep(100)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel('主界面')
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setWindowTitle('主界面')
        self.resize(200, 200)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec_())