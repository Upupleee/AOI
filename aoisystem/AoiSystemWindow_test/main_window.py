"""主窗口模块。

从 AoiSystemMain.py 拆分而来，包含主窗口 MyWindow 类。
"""
import time
import re
import os
import gc
import logging
import math
import yaml
import threading
import configparser
import shutil
import numpy as np
import json
from pathlib import Path
from queue import Queue

import SPiiPlusPython as sp
from PIL import Image
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import (QTimer, Qt, QDateTime, QThread, pyqtSignal, QObject,
                          QSettings, QPoint)
from PyQt5.QtGui import (QPixmap, QIcon, QPainter, QPen, QColor, QPixmapCache,
                         QBrush, QGuiApplication, QCursor, QFont)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                            QHBoxLayout, QLabel, QPushButton, QTableWidget,
                            QTableWidgetItem, QHeaderView, QProgressBar,
                            QComboBox, QMenu, QInputDialog, QMessageBox,
                            QFileDialog, QDialog, QAction, QLineEdit,
                            QWidget, QScrollArea)

from common.config import config
from common.utils import generate_thumbnail, extract_number_from_path, load_json_file, save_json_file
from .ImageView import Viewer
from .ViewerWindow import ViewerDialog
from .AcsThread import AcsThread
from .CameraControl import ProcessController
from .WorkerThread import DetectionController
from .material_import_dialog import MaterialImportDialog
from Logger.LoggerDialog import LoggerDialog
from ConfigWindow.ConfigDialog import ConfigDialog
from LogoutWindow.LogoutDialog import LogoutDialog
from SystemSettingWindow.SystemSettingWindow import SystemSettingDialog

Image.MAX_IMAGE_PIXELS = None  # 禁用图片像素限制
os.makedirs(config.material_root, exist_ok=True)  # 确保料号根目录存在

logger = logging.getLogger(__name__)


class MyWindow(QMainWindow):
    def __init__(self, app, plc_client, acs_comm):
        super(MyWindow, self).__init__()
        self.app = app
        self.plc_client = plc_client
        self.acs_comm = acs_comm

        # 初始化UI和基本设置
        self._init_ui()
        self._init_variables()
        self._init_logger()
        self._init_timers()
        self._init_acs()

        # 设置UI组件
        self._setup_toolbars()
        self._setup_graphics_views()
        self._setup_tables()

        # 连接信号和槽
        self._connect_menu_signals()
        self._connect_button_signals()
        self._connect_checkbox_signals()
        self._connect_combo_signals()

        # 加载保存的设置和数据
        self.load_settings()
        self.load_material_data()


    def _init_ui(self):
        """初始化UI界面"""
        self.ui = uic.loadUiType("./AoiSystemWindow_test/AoiSystemtest.ui")[0]()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon("./icons/xjz.png"))


    def _init_variables(self):
        """初始化成员变量"""
        # 文件路径相关
        self.folder_path = ''
        self.inipath = config.motion_settings_ini
        self.temp_folder_path = config.temp_images_dir
        self.acs_config_path = config.acs_config_path
        self.ccf_path = config.ccf_path
        self.settings_path = config.settings_ini_path

        self.current_board_width = None
        self.current_board_height = None
        self.current_board_thickness = None

        # 图像和检测相关
        self.image_paths = []
        self.slice_paths = []
        self.temp_slice_widths = None
        self.temp_height = None
        self.targets = None
        self.selected_image = None
        self.selected_temp_image = None
        self.selector_image = None
        self.gerber_image = False
        self.gerber_image_path = None
        self.current_index = 0
        self.sample_data_display = False
        self.defect_data_display = False
        self.sample_data_path = None
        self.template_fileName = None
        self.a_path = None
        self.b_path = None

        # 线程和控制器
        self.quick_start_running = False
        self.combine_th = None
        self.matching_thread = None
        self.detection_project = None
        self.controller = ProcessController()
        self.detection_controller = DetectionController(self)
        self.detection_controller.detection_round_finished.connect(self.on_detection_round_complete)
        self.start_sample_detection_running = False


        # 数据存储
        self.material_data = {}  # 料号数据（键为不含material根目录的料号名称）
        self.current_material = None  # 当前选中的料号（不含material根目录）
        self.detection_mode = "A"  # 默认检测模式
        self.inference_mode = "整图批量推理"
        self.ab_cycle_count = 0  # AB面模式计数器
        self.scan_records = {}  # 扫描记录（键为不含material根目录的料号名称）
        self.current_scan_id = None  # 当前扫描的ID
        self.current_ab_cycle = 0  # 当前AB面循环次数
        self.current_ab_folder = None  # 当前AB面结果文件夹（包含material根目录）
        self.auto_confirm = False  # 自动确认标志

        # 真空和风机控制
        self.vacuum_on = False
        self.blower_on = False
        self.blower_timer = None
        self.last_click_time = 0
        self.time_interval_threshold = config.vacuum_time_interval_threshold

        # 设置
        self.settings = QSettings(self.settings_path, QSettings.IniFormat)


    def _init_logger(self):
        """初始化日志记录器"""
        logging.basicConfig(
            filename=config.log_file,
            level=config.log_level,
            format=config.log_format,
            filemode='a'
        )


    def _init_timers(self):
        """初始化定时器"""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_readings)
        self.timer.start(100)


    def _init_acs(self):
        """初始化ACS通信线程"""
        self.acs_thread_obj = AcsThread(self.acs_comm)


    def _setup_toolbars(self):
        """设置工具栏"""
        # 样本图片工具栏
        self._setup_raw_toolbar()

        # 模板图片工具栏
        self._setup_template_toolbar()


    def _setup_raw_toolbar(self):
        """设置样本图片工具栏"""
        toolbar = QtWidgets.QToolBar(self.ui.Atoolbar_widget)
        self.ui.Atoolbar_verticalLayout.addWidget(toolbar)

        actions = [
            QAction('放大', self.ui.Atoolbar_widget),
            QAction('缩小', self.ui.Atoolbar_widget),
            QAction('重置大小', self.ui.Atoolbar_widget),
            QAction('适应窗口', self.ui.Atoolbar_widget)
        ]

        for action in actions:
            toolbar.addAction(action)

        actions[0].triggered.connect(self.sample_zoom_in)
        actions[1].triggered.connect(self.sample_zoom_out)
        actions[2].triggered.connect(self.sample_original_size)
        actions[3].triggered.connect(self.sample_fit_to_window)


    def _setup_template_toolbar(self):
        """设置模板图片工具栏"""
        toolbar = QtWidgets.QToolBar(self.ui.Btoolbar_widget)
        self.ui.Btoolbar_verticalLayout.addWidget(toolbar)

        actions = [
            QAction('放大', self.ui.Btoolbar_widget),
            QAction('缩小', self.ui.Btoolbar_widget),
            QAction('重置大小', self.ui.Btoolbar_widget),
            QAction('适应窗口', self.ui.Btoolbar_widget)
        ]

        for action in actions:
            toolbar.addAction(action)

        actions[0].triggered.connect(self.template_zoom_in)
        actions[1].triggered.connect(self.template_zoom_out)
        actions[2].triggered.connect(self.template_original_size)
        actions[3].triggered.connect(self.template_fit_to_window)


    def _setup_graphics_views(self):
        """设置图形视图"""
        # A面视图
        self._setup_graphics_view(
            parent_layout=self.ui.A_verticalLayout,
            old_widget=self.ui.graphicsView_left,
            new_widget_name='graphicsView_A',
            frame_title='A窗口'
        )

        # B面视图
        self._setup_graphics_view(
            parent_layout=self.ui.B_verticalLayout,
            old_widget=self.ui.graphicsView_right,
            new_widget_name='graphicsView_B',
            frame_title='B窗口'
        )


    def _setup_graphics_view(self, parent_layout, old_widget, new_widget_name, frame_title):
        """通用的图形视图设置方法"""
        # 创建新的Viewer
        new_viewer = Viewer()
        new_viewer.setFrameShape(QtWidgets.QFrame.Box)
        new_viewer.setFrameShadow(QtWidgets.QFrame.Sunken)
        new_viewer.setLineWidth(2)
        new_viewer.clear()

        # 替换旧的widget
        parent_layout.replaceWidget(old_widget, new_viewer)
        old_widget.deleteLater()
        # 设置新widget的属性
        setattr(self.ui, new_widget_name, new_viewer)


    def _setup_tables(self):
        """设置表格属性"""
        # 资料料号表格
        self.ui.materialTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.materialTable.setSelectionBehavior(QTableWidget.SelectRows)
        self.ui.materialTable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.materialTable.customContextMenuRequested.connect(self.show_context_menu)
        self.ui.materialTable.cellClicked.connect(self.table_cell_clicked)

        # 扫描记录表格
        self.ui.boardTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.boardTable.setSelectionBehavior(QTableWidget.SelectRows)
        self.ui.boardTable.cellClicked.connect(self.board_table_clicked)


    def _connect_menu_signals(self):
        """连接菜单信号"""
        # viewer_menu
        self.ui.Viewer_action.triggered.connect(self.onViewer)

        # setting_menu
        self.ui.ConfigSet_action.triggered.connect(self.onConfigSet)
        self.ui.System_action.triggered.connect(self.onSystemSet)

        # help_menu
        self.ui.About_action.triggered.connect(self.onHelpAbout)
        self.ui.Log_action.triggered.connect(self.onHelpLog)

        # exit_menu
        self.ui.Close_action.triggered.connect(self.onExit)


    def _connect_button_signals(self):
        """连接按钮信号"""
        # 真空控制按钮
        self._setup_button(
            button=self.ui.SorptionButton,
            color="gray",
            slot=self.toggle_vacuum
        )

        # 开始样本检测按钮
        self._setup_button(
            button=self.ui.StartSampleButton,
            color="gray",
            slot=self.start_sample_detection
        )

        # 开始缺陷检测按钮
        # self._setup_button(
        #     button=self.ui.StartDetectionButton,
        #     color="gray",
        #     slot=self.start_match_detection
        # )

        # 重置按钮
        self._setup_button(
            button=self.ui.ResetButton,
            color="gray",
            slot=self.reset
        )

        # 设置光源按钮
        self._setup_button(
            button=self.ui.pushButton,
            color="gray",
            slot=self.setLight
        )

        # 报警复位按钮
        self._setup_button(
            button=self.ui.AlarmRestButton,
            color="gray",
            slot=self.ResetAlarm
        )


    def _setup_button(self, button, color, slot):
        """通用的按钮设置方法"""
        button.setStyleSheet(f"background-color: {color}")
        button.clicked.connect(slot)


    def _connect_checkbox_signals(self):
        """连接复选框信号"""
        self.ui.lock_ab_checkbox.toggled.connect(self.toggle_lock_ab_mode)
        self.ui.auto_confirm_checkbox.toggled.connect(self.toggle_auto_confirm)


    def _connect_combo_signals(self):
        """连接下拉框信号"""
        self.ui.mode_combo.currentTextChanged.connect(self.change_detection_mode)
        self.ui.inference_comboBox.currentTextChanged.connect(self.change_inference_mode)


    def update_readings(self):
        light_curtain_status = self.plc_client.read_bits(config.coil('light_curtain'), 1)[0]
        if light_curtain_status:
            self.plc_client.write_bit(config.coil('alarm_3'), 1)
            self.plc_client.write_bit(config.coil('alarm_4'), 1)
        start_status = self.plc_client.read_bits(config.coil('start_signal'), 1)[0]
        if start_status:
            if not hasattr(self, 'vacuum_status') or self.vacuum_status is False:
                self.toggle_vacuum()
            self.start_sample_detection()

        wetness = self.plc_client.read_float(config.register('wetness'))
        temperature = self.plc_client.read_float(config.register('temperature'))
        pressure = self.plc_client.read_float(config.register('pressure'))
        self.ui.label_temperature.setText(f"温度: {temperature:.2f} °C")
        self.ui.label_pressure.setText(f"负压: {pressure:.2f} Pa")
        self.ui.label_wetness.setText(f"湿度: {wetness:.2f} %")


    def setLight(self):
        self.ui.pushButton.setStyleSheet("background-color: green")
        value = self.ui.spinBox.value()
        self.plc_client.write_register(config.register('light_source'), value)
        self.ui.pushButton.setStyleSheet("background-color: gray")


    def reset(self):
        self.ui.ResetButton.setStyleSheet("background-color: green")
        self.plc_client.write_bit(config.coil('software_ready'), 0)
        self.plc_client.write_bit(config.coil('alarm_reset'), 1)
        self.plc_client.write_bit(config.coil('light_curtain'), 0)
        self.ACS_disconnect()


    def toggle_vacuum(self):
        current_time = time.time()
        time_interval = current_time - self.last_click_time
        self.last_click_time = current_time

        try:
            if not self.vacuum_on:
                self.start_vacuum(time_interval)
            else:
                self.stop_vacuum(time_interval)
            self.vacuum_on = not self.vacuum_on
        except Exception as e:
            logger.error("操作出错: %s", e)


    def start_vacuum(self, time_interval):
        self.ui.SorptionButton.setStyleSheet("background-color: green")
        try:
            if not self.blower_on or time_interval > self.time_interval_threshold:
                self.turn_on_blower()
            self.plc_client.write_bit(config.coil('vacuum_release'), 0)
            self.plc_client.write_bit(config.coil('vacuum'), 1)  # 吸真空打开
            self.vacuum_status = True
        except Exception as e:
            logger.error("吸真空操作出错: %s", e)
            self.ui.SorptionButton.setStyleSheet("background-color: gray")
            self.vacuum_on = False


    def stop_vacuum(self, time_interval):
        self.ui.SorptionButton.setStyleSheet("background-color: gray")
        try:
            self.plc_client.write_bit(config.coil('vacuum_release'), 1)  # 破真空打开
            self.plc_client.write_bit(config.coil('vacuum'), 0)  # 吸真空关闭
            self.vacuum_status = False
            if time_interval > self.time_interval_threshold:
                if self.blower_timer:
                    self.blower_timer.cancel()
                self.blower_timer = threading.Timer(config.blower_delay_seconds, self.turn_off_blower)
                self.blower_timer.start()
        except Exception as e:
            logger.error("破真空操作出错: %s", e)
            self.ui.SorptionButton.setStyleSheet("background-color: green")
            self.vacuum_on = True


    def turn_on_blower(self):
        self.plc_client.write_bit(config.coil('blower'), 1)
        self.blower_on = True


    def turn_off_blower(self):
        self.plc_client.write_bit(config.coil('blower'), 0)
        self.blower_on = False


    def on_settings_update(self, row, col):
        self.row = row
        self.col = col
        self.ui.tableWidget.setItem(row, col, QTableWidgetItem("123"))


    def onViewer(self, *args):
        ViewerWidget = ViewerDialog()
        ViewerWidget.exec_()


    # 帮助菜单
    def onHelpAbout(self):
        logging.info('打开帮助窗口')
        QtWidgets.QMessageBox.information(self, 'VersionAbout', '机器视觉前端演示版')


    def onConfigSet(self, *args):
        logging.info('打开参数设置窗口')
        ConfigWidget = ConfigDialog()
        ConfigWidget.exec_()


    def onSystemSet(self, *args):
        logging.info('打开系统设置窗口')
        SystemSetWidget = SystemSettingDialog()
        SystemSetWidget.exec_()


    def onHelpLog(self, *args):
        logging.info('打开帮助日志窗口')
        LoggerWidget = LoggerDialog()
        LoggerWidget.exec_()


    def onExit(self):
        logging.info('打开关闭登出窗口')
        LogoutWidget = LogoutDialog()
        LogoutWidget.exec_()


    # ACS线程
    def acs_thread(self):
        self.acs_thread_obj.add_operation('moving')
        value = math.ceil(int(self.current_board_width) / config.board_width_divisor)
        if not self.ui.only_sample_checkBox.isChecked():
            self.detection_controller.reset(value)
            self.detection_controller.prepare_detection(self.detection_mode)
        logger.debug("value: %s", value)
        if value is not None:
            self.file_queue = Queue()
            self.acs_thread_obj.add_iteration(value)
            self.acs_thread_obj.add_queue(self.file_queue)
            self.acs_thread_obj.signal_received_path.connect(self.on_new_file)
            self.acs_thread_obj.signal_received_finish.connect(self.on_new_files_found)
            self.acs_thread_obj.start()

    def on_new_file(self, _):
        logger.debug("on_new_file path: %s", _)
        if not self.ui.only_sample_checkBox.isChecked():
            self.detection_controller.enqueue_file(_)


    def on_new_files_found(self):
        if self.ui.only_sample_checkBox.isChecked():
            self.ui.StartSampleButton.setStyleSheet("background-color: gray")
            self.quick_start_running = False
            self.start_sample_detection_running = False
        QMessageBox.information(self, '提示', '采图结束', QMessageBox.Yes)
        self.plc_client.write_bit(config.coil('software_ready'), 0)
        self.controller.send_exit_command()
        if self.acs_thread_obj is not None:
            try:
                self.acs_thread_obj.signal_received_finish.disconnect()
                self.acs_thread_obj.signal_received_path.disconnect()
            except TypeError:
                pass

    def on_detection_round_complete(self):
        if self.detection_mode =="AB":
            self.ab_cycle_count +=1
        self.ui.StartSampleButton.setStyleSheet("background-color: gray")
        self.quick_start_running = False
        self.start_sample_detection_running = False


    # 缩放图片
    def sample_zoom_in(self):
        if self.sample_data_display:
            self.ui.graphicsView_A.zoomIn()
        elif self.defect_data_display:
            self.ui.graphicsView_A.zoomIn()
        else:
            QMessageBox.warning(self, "警告", "请先加载图像！")


    def sample_zoom_out(self):
        if self.sample_data_display:
            self.ui.graphicsView_A.zoomOut()
        elif self.defect_data_display:
            self.ui.graphicsView_A.zoomOut()
        else:
            QMessageBox.warning(self, "警告", "请先加载图像！")


    def sample_original_size(self):
        if self.sample_data_display:
            self.ui.graphicsView_A.resetZoom()
        elif self.defect_data_display:
            self.ui.graphicsView_A.resetZoom()
        else:
            QMessageBox.warning(self, "警告", "请先加载图像！")


    def sample_fit_to_window(self):
        if self.sample_data_display:
            self.ui.graphicsView_A.zoomFit()
        elif self.defect_data_display:
            self.ui.graphicsView_A.zoomFit()
        else:
            QMessageBox.warning(self, "警告", "请先加载图像！")


    def template_zoom_in(self):
        if self.gerber_image:
            self.ui.graphicsView_B.zoomIn()
        else:
            QMessageBox.warning(self, "警告", "请先导入template文件！")


    def template_zoom_out(self):
        if self.gerber_image:
            self.ui.graphicsView_B.zoomOut()
        else:
            QMessageBox.warning(self, "警告", "请先导入template文件！")


    def template_original_size(self):
        if self.gerber_image:
            self.ui.graphicsView_B.resetZoom()
        else:
            QMessageBox.warning(self, "警告", "请先导入template文件！")


    def template_fit_to_window(self):
        if self.gerber_image:
            self.ui.graphicsView_B.zoomFit()
        else:
            QMessageBox.warning(self, "警告", "请先导入template文件！")


    def template_crop(self):
        if self.CropActionTemplate.isChecked():
            self.ui.graphicsView_B.set_crop_flag(True)
        else:
            self.ui.graphicsView_B.set_crop_flag(False)


    def sample_crop(self):
        if self.CropActionSample.isChecked():
            self.ui.graphicsView_A.set_crop_flag(True)
        else:
            self.ui.graphicsView_A.set_crop_flag(False)


    def ACS_disconnect(self):
        self.acs_thread_obj.add_operation('homing')
        self.acs_thread_obj.signal_received_reset.connect(self.disconnect_reset)
        self.acs_thread_obj.start()


    def disconnect_reset(self):
        self.ui.ResetButton.setStyleSheet("background-color: gray")


    def ResetAlarm(self):
        self.plc_client.write_bit(config.coil('software_ready'), 0)
        self.plc_client.write_bit(config.coil('alarm_reset'), 1)
        self.ui.AlarmRestButton.setStyleSheet("background-color: green")
        if self.plc_client.read_bits(config.coil('alarm_1'), 1)[0]:
            self.plc_client.write_bit(config.coil('alarm_1'), 0)
        if self.plc_client.read_bits(config.coil('alarm_2'), 1)[0]:
            self.plc_client.write_bit(config.coil('alarm_2'), 0)
        if self.plc_client.read_bits(config.coil('alarm_3'), 1)[0]:
            self.plc_client.write_bit(config.coil('alarm_3'), 0)
        if self.plc_client.read_bits(config.coil('alarm_4'), 1)[0]:
            self.plc_client.write_bit(config.coil('alarm_4'), 0)
        self.ui.AlarmRestButton.setStyleSheet("background-color: gray")


    def start_sample_detection(self):
        if not self.current_material:
            self.show_message("警告", "请先选择一个料号!")
            return
        if self.start_sample_detection_running is True:
            QMessageBox.warning(self, "错误", "正在运行，请等待完成！")
            return
        else:
            self.start_sample_detection_running = True
        self.ui.StartSampleButton.setStyleSheet("background-color: green")
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

        QMessageBox.information(self, "提示", "开始采图!")
        time.sleep(5)
        self.plc_client.write_bit(config.coil('software_ready'), 1)
        self.acs_thread()
        return

    def read_yaml_width(self):
        try:
            with open(self.acs_config_path, 'r') as file:
                data = yaml.safe_load(file)
                board_width = data.get('Board', {}).get('width')
                return math.ceil(board_width / config.board_width_divisor)
        except FileNotFoundError:
            logger.warning("错误: 文件未找到!")
        except yaml.YAMLError as e:
            logger.error("错误: YAML 文件解析出错: %s", e)
        except Exception as e:
            logger.error("错误: 发生了一个未知错误: %s", e)
        return None


    def read_ini_param(self, file_path, param_name, expected_type):
        parser = configparser.ConfigParser()
        parser.read(file_path)

        if 'General' in parser:
            section = parser['General']
            if param_name in section:
                value = section.get(param_name)
                try:
                    if expected_type == int:
                        value = int(value)
                    elif expected_type == float:
                        value = float(value)
                    elif expected_type == bool:
                        value = parser.getboolean('General', param_name)
                except (ValueError, configparser.NoOptionError):
                    pass
                return value
            else:
                logger.warning("在 'General' 部分中未找到参数 '%s'。", param_name)
        else:
            logger.warning("在.ini文件中未找到 'General' 部分。")

        return None

    def get_slice_image_paths(self, side):
            """获取当前检测面的所有切分后图片路径（包含material根目录）"""
            if not self.current_material or self.current_material not in self.material_data:
                return []

            # 从料号数据中获取切分后的路径
            material_data = self.material_data[self.current_material]
            if f"{side}面切分后" in material_data:
                return material_data[f"{side}面切分后"]

            # 如果没有直接保存的切分路径，尝试从料号目录中查找
            full_material_id = os.path.join(config.material_root, self.current_material)
            side_folder = os.path.join(full_material_id, side)

            if not os.path.exists(side_folder):
                return []

            # 获取该目录下的所有图片文件（切分后的）
            slice_paths = []
            for file in os.listdir(side_folder):
                if file.lower().endswith(('.bmp', '.jpg', '.png', '.tif', '.tiff')):
                    slice_path = os.path.join(side_folder, file)
                    slice_paths.append(slice_path)

            # 按文件名排序
            slice_paths.sort()
            # 更新料号数据中的切分后路径
            material_data[f"{side}面切分后"] = slice_paths
            self.save_material_data()

            return slice_paths

    def toggle_lock_ab_mode(self, checked):
        """切换锁定AB模式状态"""
        if checked:
            self.ui.mode_combo.setCurrentText("AB")
            self.ui.mode_combo.setEnabled(False)
            self.detection_mode = "AB"
        else:
            self.ui.mode_combo.setEnabled(True)
        self.save_settings()


    def load_settings(self):
        """加载设置"""
        self.auto_confirm = self.settings.value("auto_confirm", False, type=bool)
        self.ui.auto_confirm_checkbox.setChecked(self.auto_confirm)

        lock_ab = self.settings.value("lock_ab_mode", False, type=bool)
        self.ui.lock_ab_checkbox.setChecked(lock_ab)

        if lock_ab:
            self.ui.mode_combo.setCurrentText("AB")
            self.ui.mode_combo.setEnabled(False)
            self.detection_mode = "AB"
        else:
            last_mode = self.settings.value("last_detection_mode", "A")
            self.ui.mode_combo.setCurrentText(last_mode)
            self.detection_mode = last_mode


    def save_settings(self):
        """保存设置"""
        self.settings.setValue("auto_confirm", self.auto_confirm)
        self.settings.setValue("lock_ab_mode", self.ui.lock_ab_checkbox.isChecked())
        if not self.ui.lock_ab_checkbox.isChecked():
            self.settings.setValue("last_detection_mode", self.detection_mode)
        self.settings.sync()


    def load_material_data(self):
        """从文件加载料号数据"""
        self.material_data = load_json_file(config.material_data_json)
        self.update_material_table()


    def load_scan_records(self):
        """从文件加载扫描记录"""
        self.scan_records = load_json_file(config.scan_records_json)
        self.update_board_table()


    def save_material_data(self):
        """保存料号数据到文件"""
        save_json_file(config.material_data_json, self.material_data)


    def save_scan_records(self):
        """保存扫描记录到文件"""
        save_json_file(config.scan_records_json, self.scan_records)


    def update_material_table(self):
        """更新料号表格"""
        self.ui.materialTable.setRowCount(0)
        for row, (material_id, data) in enumerate(self.material_data.items()):
            self.ui.materialTable.insertRow(row)

            # 表格中显示的是不含material根目录的料号名称
            id_item = QTableWidgetItem(material_id)
            self.ui.materialTable.setItem(row, 0, id_item)

            a_path = data.get("A面", "")
            a_text = os.path.basename(a_path) if a_path else "未设置"
            a_item = QTableWidgetItem(a_text)
            self.ui.materialTable.setItem(row, 1, a_item)

            b_path = data.get("B面", "")
            b_text = os.path.basename(b_path) if b_path else "未设置"
            b_item = QTableWidgetItem(b_text)
            self.ui.materialTable.setItem(row, 2, b_item)


    def update_board_table(self):
        """更新板材表格"""
        self.ui.boardTable.setRowCount(0)
        if not self.current_material or self.current_material not in self.scan_records:
            return

        records = self.scan_records[self.current_material]
        row_index = 0

        for record in records:
            scan_id = record["scan_id"]

            # 添加A面记录
            a_defects = record.get("A面", {}).get("defects", [])
            a_image = record.get("A面", {}).get("image_path", "")
            if a_defects or a_image:
                self.ui.boardTable.insertRow(row_index)
                self.ui.boardTable.setItem(row_index, 0, QTableWidgetItem(scan_id))
                self.ui.boardTable.setItem(row_index, 1, QTableWidgetItem("A"))
                self.ui.boardTable.setItem(row_index, 2, QTableWidgetItem(record["scan_time"]))

                view_btn = QPushButton("查看详情")
                view_btn.clicked.connect(lambda checked, sid=scan_id, s="A": self.view_scan_details(sid, s))
                self.ui.boardTable.setCellWidget(row_index, 3, view_btn)
                row_index += 1

            # 添加B面记录
            b_defects = record.get("B面", {}).get("defects", [])
            b_image = record.get("B面", {}).get("image_path", "")
            if b_defects or b_image:
                self.ui.boardTable.insertRow(row_index)
                self.ui.boardTable.setItem(row_index, 0, QTableWidgetItem(scan_id))
                self.ui.boardTable.setItem(row_index, 1, QTableWidgetItem("B"))
                self.ui.boardTable.setItem(row_index, 2, QTableWidgetItem(record["scan_time"]))

                view_btn = QPushButton("查看详情")
                view_btn.clicked.connect(lambda checked, sid=scan_id, s="B": self.view_scan_details(sid, s))
                self.ui.boardTable.setCellWidget(row_index, 3, view_btn)
                row_index += 1


    def show_context_menu(self, position):
        """显示右键菜单"""
        menu = QMenu()
        new_action = menu.addAction("新建料号")
        load_action = menu.addAction("载入当前料号")
        delete_action = menu.addAction("删除料号")

        action = menu.exec_(self.ui.materialTable.mapToGlobal(position))
        if action == new_action:
            self.open_material_import_dialog()
        elif action == load_action:
            self.load_current_material()
        elif action == delete_action:
            self.delete_material()


    def open_material_import_dialog(self):
        """打开料号导入对话框"""
        dialog = MaterialImportDialog(self)
        dialog.auto_confirm = self.auto_confirm
        dialog.material_imported.connect(self.import_material)
        dialog.exec_()


    def import_material(self, material_id, data):
        """导入料号数据（material_id不含material根目录）"""
        if material_id in self.material_data and not self.auto_confirm:
            reply = QMessageBox.question(self, "确认覆盖", f"料号 '{material_id}' 已存在，是否覆盖?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        self.material_data[material_id] = data
        self.save_material_data()
        self.update_material_table()

        if not self.auto_confirm:
            # 显示的路径包含material根目录
            full_material_path = os.path.join(config.material_root, material_id)
            QMessageBox.information(self, "成功", f"料号 '{material_id}' 导入成功，保存路径: {full_material_path}")


    def load_current_material(self):
        """载入当前选中的料号"""
        selected_rows = set(index.row() for index in self.ui.materialTable.selectedIndexes())
        if not selected_rows:
            self.show_message("警告", "请先选择一个料号!")
            return

        row = selected_rows.pop()
        material_id = self.ui.materialTable.item(row, 0).text()  # 不含material根目录
        self.current_material = material_id
        self.load_material_images()
        self.update_board_table()

         # 加载并显示板材参数
        material_data = self.material_data.get(self.current_material, {})
        self.current_board_width = material_data.get("板宽(mm)")
        self.current_board_height = material_data.get("板高(mm)")
        self.current_board_thickness = material_data.get("板厚(mm)")
        self.temp_slice_widths = material_data.get("条带宽")
        self.temp_height = material_data.get("资料高")
        self.targets = material_data.get("靶点数据")
        self.save_ccf_settings("crop_height")


        self.ab_cycle_count = 0
        self.current_ab_cycle = self.get_next_ab_cycle()

        if not self.auto_confirm:
            # 显示的路径包含material根目录
            full_material_path = os.path.join(config.material_root, material_id)
            self.show_message("成功", f"已加载料号: {material_id}，路径: {full_material_path}")

    def save_ccf_settings(self, key):
        if key == "crop_height":
            new_value = int(self.current_board_height)*125+9070
            # new_value = (int(self.current_board_height)+45+12.883)*129.706
            logger.debug("current_board_height: %s", self.current_board_height)
        lines = []
        with open(self.ccf_path, 'r') as f:
            lines = f.readlines()
        with open(self.ccf_path, "w") as f:
            for line in lines:
                line = line.strip()  # 去除行首尾的空白字符
                if line.startswith('Crop Height' + '='):
                    f.write(f"Crop Height={new_value}\n")
                else:
                    f.write(line + '\n')

    def get_next_ab_cycle(self):
        """获取下一个AB循环编号"""
        if not self.current_material:
            return 0

        max_cycle = 0
        for record in self.scan_records.get(self.current_material, []):
            cycle = int(record["scan_id"].split("_")[-1]) if "_" in record["scan_id"] else 0
            if cycle > max_cycle:
                max_cycle = cycle
        return max_cycle + 1


    def delete_material(self):
        """删除选中的料号（同时删除实际文件夹）"""
        selected_rows = set(index.row() for index in self.ui.materialTable.selectedIndexes())
        if not selected_rows:
            self.show_message("警告", "请先选择一个料号!")
            return

        row = selected_rows.pop()
        material_id = self.ui.materialTable.item(row, 0).text()  # 不含material根目录

        # 包含material根目录的完整路径
        full_material_path = os.path.join(config.material_root, material_id)

        if not self.auto_confirm:
            reply = QMessageBox.question(self, "确认删除",
                                        f"确定要删除料号 '{material_id}' 及其所有文件吗?\n路径: {full_material_path}\n此操作不可撤销!",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        # 从数据中删除
        if material_id in self.material_data:
            del self.material_data[material_id]

        if self.current_material == material_id:
            self.current_material = None
            self.clear_material_images()

        if material_id in self.scan_records:
            del self.scan_records[material_id]

        # 删除实际文件夹
        try:
            if os.path.exists(full_material_path):
                shutil.rmtree(full_material_path)
                logger.info("已删除文件夹: %s", full_material_path)
        except Exception as e:
            logger.error("删除文件夹失败: %s", e)
            self.show_message("警告", f"料号数据已删除，但文件夹删除失败: {e}")

        self.save_material_data()
        self.save_scan_records()
        self.update_material_table()
        self.update_board_table()


    def load_material_images(self):
        """加载当前料号的图片（优先显示缩略图）"""
        if not self.current_material or self.current_material not in self.material_data:
            return

        material_data = self.material_data[self.current_material]

        # 加载A面图片（优先缩略图，路径已包含material根目录）
        a_image_path = material_data.get("A面", "")
        aparts = a_image_path.split('/')
        self.a_path = f"{aparts[-2]}/{aparts[-1].split('.')[0]}"
        a_thumbnail = material_data.get("A面缩略图", "")

        if a_thumbnail and os.path.exists(a_thumbnail):
            self.ui.graphicsView_A.set_image(a_thumbnail)
            self.ui.A_label.setText(f"A面图片: {os.path.basename(a_image_path)}")
        elif a_image_path and os.path.exists(a_image_path):
            self.ui.graphicsView_A.set_image(a_image_path)
            self.ui.A_label.setText(f"A面图片: {os.path.basename(a_image_path)}")
        else:
            self.ui.A_label.setText("A面检测结果: 未检测")

        # 加载B面图片（优先缩略图，路径已包含material根目录）
        b_image_path = material_data.get("B面", "")
        bparts = b_image_path.split('/')
        self.b_path = f"{bparts[-2]}/{bparts[-1].split('.')[0]}"
        b_thumbnail = material_data.get("B面缩略图", "")

        if b_thumbnail and os.path.exists(b_thumbnail):
            self.ui.graphicsView_B.set_image(b_thumbnail)
            self.ui.B_label.setText(f"B面图片: {os.path.basename(b_image_path)}")
        elif b_image_path and os.path.exists(b_image_path):
            self.ui.graphicsView_B.set_image(b_image_path)
            self.ui.B_label.setText(f"B面图片: {os.path.basename(b_image_path)}")
        else:
            self.ui.B_label.setText("B面检测结果: 未检测")


    def clear_material_images(self):
        """清除当前料号的图片显示"""
        self.ui.A_label.setText("A面检测结果: 未检测")
        self.ui.B_label.setText("B面检测结果: 未检测")
        self.ui.AprogressBar.setValue(0)
        self.ui.BprogressBar.setValue(0)


    def change_inference_mode(self, mode):
        self.inference_mode = mode
        self.save_settings()

    def change_detection_mode(self, mode):
        """更改检测模式"""
        self.detection_mode = mode
        self.save_settings()


    def toggle_auto_confirm(self, checked):
        """切换自动确认模式"""
        self.auto_confirm = checked
        self.save_settings()


    def table_cell_clicked(self, row, col):
        """表格单元格点击事件"""
        if col in [1, 2] and row >= 0:
            material_id = self.ui.materialTable.item(row, 0).text()  # 不含material根目录
            side = "A" if col == 1 else "B"

            file_path, _ = QFileDialog.getOpenFileName(
                self, f"选择{side}面图片", "", "图片文件 (*.bmp *.jpg *.png *.tif)"
            )

            if file_path:
                if material_id not in self.material_data:
                    self.material_data[material_id] = {}

                # 保存原始路径，后续处理会自动保存到material目录下
                self.material_data[material_id][f"{side}面"] = file_path
                self.material_data[material_id][f"{side}面全部"] = [file_path]

                # 生成新的缩略图（路径包含material根目录）
                full_material_id = os.path.join(config.material_root, material_id)
                thumbnails_dir = os.path.join(full_material_id, "thumbnails")
                os.makedirs(thumbnails_dir, exist_ok=True)

                try:
                    with Image.open(file_path) as img:
                        thumbnail_path = generate_thumbnail(img, os.path.basename(file_path), thumbnails_dir)
                        if thumbnail_path:
                            self.material_data[material_id][f"{side}面缩略图"] = thumbnail_path
                except Exception as e:
                    logger.error("生成缩略图失败: %s", e)

                self.update_material_table()
                self.save_material_data()

                if self.current_material == material_id:
                    self.load_material_images()


    def get_current_image_path(self, side):
        """获取当前检测面的图片路径（包含material根目录）"""
        if not self.current_material or self.current_material not in self.material_data:
            return None
        return self.material_data[self.current_material].get(f"{side}面", "")


    def view_scan_details(self, scan_id, side):
        """查看扫描详情"""
        if self.current_material not in self.scan_records:
            self.show_message("错误", "找不到此料号的扫描记录")
            return

        record = next((r for r in self.scan_records[self.current_material] if r["scan_id"] == scan_id), None)

        if not record or side not in record:
            self.show_message("错误", "找不到此扫描记录")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"扫描详情 - {scan_id} - {side}面")
        dialog.setMinimumSize(800, 600)

        layout = QVBoxLayout(dialog)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(600, 400)
        image_label.setStyleSheet("border: 1px solid #ccc;")
        layout.addWidget(image_label)

        defects_label = QLabel("缺陷信息")
        layout.addWidget(defects_label)

        defects_table = QTableWidget()
        defects_table.setColumnCount(3)
        defects_table.setHorizontalHeaderLabels(["缺陷类型", "位置", "大小"])
        defects_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(defects_table)

        image_path = record[side]["image_path"]  # 包含material根目录
        defects = record[side]["defects"]

        self.display_image_with_defects(image_label, image_path, defects)

        defects_table.setRowCount(len(defects))
        for row, defect in enumerate(defects):
            defects_table.setItem(row, 0, QTableWidgetItem(defect["type"]))
            defects_table.setItem(row, 1, QTableWidgetItem(f"({defect['position'][0]}, {defect['position'][1]})"))
            defects_table.setItem(row, 2, QTableWidgetItem(str(defect['size'])))

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec_()


    def board_table_clicked(self, row, col):
        """板材表格点击事件"""
        if col == 3:
            return


    def show_message(self, title, message):
        """显示消息框"""
        if self.auto_confirm:
            return
        QMessageBox.information(self, title, message)


    def closeEvent(self, event):
        # self.disconnect()
        self.plc_client.write_bit(config.coil('blower'), 0)  # CLOSE
        self.plc_client.write_bit(config.coil('vacuum'), 0)
        self.plc_client.write_bit(config.coil('vacuum_release'), 0)
        self.plc_client.write_register(config.register('light_source'), 0)
        sp.DisableAll(self.acs_comm)
        self.plc_client.disconnect()

        event.accept()
        super().closeEvent(event)
