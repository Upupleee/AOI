"""主窗口视图。

新的主窗口视图,使用 MainViewModel 完成业务操作。
本类只负责 UI 显示和事件转发,不包含业务逻辑:
    - 加载 AoiSystemtest.ui
    - 连接 UI 事件到 ViewModel 的 slot
    - 连接 ViewModel 的信号到 UI 更新方法
"""
import logging

from PyQt5 import uic
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QHeaderView, QTableWidgetItem

from ui.viewmodels.main_viewmodel import MainViewModel

logger = logging.getLogger(__name__)


class MainView(QMainWindow):
    """主窗口视图,委托业务操作给 MainViewModel。"""

    def __init__(self, view_model: MainViewModel):
        super().__init__()
        self._vm = view_model

        # 加载 UI 文件
        self.ui = uic.loadUiType("./AoiSystemWindow_test/AoiSystemtest.ui")[0]()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon("./icons/xjz.png"))

        # 连接信号
        self._connect_ui_signals()
        self._connect_vm_signals()

        # 初始化表格属性
        self._setup_tables()

        # 刷新料号表格
        self._refresh_material_table()

    # ---- 信号连接 ----
    def _connect_ui_signals(self):
        """将 UI 控件信号连接到 ViewModel 的 slot。"""
        # 真空控制按钮
        self.ui.SorptionButton.clicked.connect(self._vm.toggle_vacuum)
        # 开始样本检测按钮
        self.ui.StartSampleButton.clicked.connect(self._on_start_clicked)
        # 重置按钮
        self.ui.ResetButton.clicked.connect(self._vm.reset_system)
        # 报警复位按钮
        self.ui.AlarmRestButton.clicked.connect(self._vm.reset_alarm)
        # 设置光源按钮
        self.ui.pushButton.clicked.connect(self._on_set_light)

    def _connect_vm_signals(self):
        """将 ViewModel 的信号连接到 UI 更新方法。"""
        self._vm.vacuum_state_changed.connect(self._update_vacuum_button)
        self._vm.sensor_data_updated.connect(self._update_sensor_labels)
        self._vm.status_message.connect(self._show_status)
        self._vm.error_occurred.connect(self._show_error)
        self._vm.material_data_changed.connect(self._refresh_material_table)
        self._vm.detection_round_finished.connect(self._on_round_finished)
        self._vm.state_changed.connect(self._update_state_display)

    def _setup_tables(self):
        """设置表格属性。"""
        self.ui.materialTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.boardTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    # ---- UI 更新方法 ----
    def _update_vacuum_button(self, on: bool):
        """根据真空状态更新按钮颜色。"""
        if on:
            self.ui.SorptionButton.setStyleSheet("background-color: green")
        else:
            self.ui.SorptionButton.setStyleSheet("background-color: gray")

    def _update_sensor_labels(self, data: dict):
        """更新温度/湿度/负压显示。"""
        temperature = data.get("temperature", 0.0)
        wetness = data.get("wetness", 0.0)
        pressure = data.get("pressure", 0.0)
        self.ui.label_temperature.setText(f"温度: {temperature:.2f} °C")
        self.ui.label_wetness.setText(f"湿度: {wetness:.2f} %")
        self.ui.label_pressure.setText(f"负压: {pressure:.2f} Pa")

    def _show_status(self, msg: str):
        """在状态栏显示消息。"""
        self.statusBar().showMessage(msg)

    def _show_error(self, msg: str):
        """用消息框显示错误。"""
        QMessageBox.warning(self, "错误", msg)

    def _refresh_material_table(self):
        """刷新料号表格。"""
        materials = self._vm.get_all_materials()
        self.ui.materialTable.setRowCount(0)
        for row, (material_id, data) in enumerate(materials.items()):
            self.ui.materialTable.insertRow(row)
            self.ui.materialTable.setItem(row, 0, QTableWidgetItem(material_id))
            a_path = data.get("A面", "") if isinstance(data, dict) else getattr(data, "a_image_path", "")
            b_path = data.get("B面", "") if isinstance(data, dict) else getattr(data, "b_image_path", "")
            a_text = a_path.split("/")[-1] if a_path else "未设置"
            b_text = b_path.split("/")[-1] if b_path else "未设置"
            self.ui.materialTable.setItem(row, 1, QTableWidgetItem(a_text))
            self.ui.materialTable.setItem(row, 2, QTableWidgetItem(b_text))

    def _on_round_finished(self):
        """一轮检测完成处理。"""
        self.ui.StartSampleButton.setStyleSheet("background-color: gray")
        self.statusBar().showMessage("一轮检测完成")

    def _update_state_display(self, state_name: str):
        """显示当前状态机状态。"""
        self.statusBar().showMessage(f"状态: {state_name}")
        logger.info("UI 更新状态显示: %s", state_name)

    # ---- UI 事件转发 ----
    def _on_start_clicked(self):
        """开始按钮点击,委托给 ViewModel。"""
        self.ui.StartSampleButton.setStyleSheet("background-color: green")
        self._vm.start_sample_detection()

    def _on_set_light(self):
        """设置光源按钮点击,读取 spinBox 值并委托给 ViewModel。"""
        self.ui.pushButton.setStyleSheet("background-color: green")
        value = self.ui.spinBox.value()
        self._vm.set_light_source(value)
        self.ui.pushButton.setStyleSheet("background-color: gray")
