"""料号导入对话框。

从 AoiSystemMain.py 拆分而来，负责料号导入的 UI 交互、图片批量解析、靶点选择等。
"""
import os
import gc

from PIL import Image
from PyQt5.QtCore import Qt, QDateTime, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPixmapCache
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                            QLineEdit, QTableWidget, QTableWidgetItem, QComboBox,
                            QProgressBar, QFileDialog, QInputDialog, QMessageBox,
                            QHeaderView, QApplication)

from common.config import config
from common.utils import generate_thumbnail, ensure_dir
from .widgets import ThumbnailDisplayWidget, ImagePreviewWindow
from .workers import ImageProcessingWorker


class MaterialImportDialog(QDialog):
    """料号导入对话框（优化线程管理）。"""
    material_imported = pyqtSignal(str, dict)
    processing_started = pyqtSignal()
    processing_updated = pyqtSignal(int, str)
    processing_completed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("料号导入")
        self.setMinimumSize(1000, 600)

        self.image_data = []  # 存储 {path, name, side, thumbnail}
        self.material_id = None  # 仅存储料号名称（不含 material 根目录）
        self.processing_thread = QThread()   # 处理线程
        self.processing_worker = None  # 处理工作对象
        self.auto_confirm = False

        self.init_ui()

    def init_ui(self):
        """初始化界面。"""
        main_layout = QVBoxLayout()
        # 料号和板材参数布局
        params_layout = QVBoxLayout()

        # 料号输入
        id_layout = QHBoxLayout()
        id_label = QLabel("料号:")
        self.id_input = QLabel("未设置")
        id_btn = QPushButton("设置料号")
        id_btn.clicked.connect(self.set_material_id)

        id_layout.addWidget(id_label)
        id_layout.addWidget(self.id_input)
        id_layout.addWidget(id_btn)
        id_layout.addStretch()

        # 板材参数输入
        board_params_layout = QHBoxLayout()

        width_layout = QVBoxLayout()
        width_label = QLabel("板宽 (mm):")
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("请输入板宽")
        width_layout.addWidget(width_label)
        width_layout.addWidget(self.width_input)

        height_layout = QVBoxLayout()
        height_label = QLabel("板高 (mm):")
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("请输入板高")
        height_layout.addWidget(height_label)
        height_layout.addWidget(self.height_input)

        thickness_layout = QVBoxLayout()
        thickness_label = QLabel("板厚 (mm):")
        self.thickness_input = QLineEdit()
        self.thickness_input.setPlaceholderText("请输入板厚")
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thickness_input)

        board_params_layout.addLayout(width_layout)
        board_params_layout.addLayout(height_layout)
        board_params_layout.addLayout(thickness_layout)
        board_params_layout.addStretch()

        params_layout.addLayout(id_layout)
        params_layout.addLayout(board_params_layout)

        main_layout.addLayout(params_layout)

        # 文件夹选择区域
        select_layout = QHBoxLayout()
        self.folder_label = QLabel("未选择文件夹")
        self.select_folder_btn = QPushButton("选择图片文件夹")
        self.select_folder_btn.clicked.connect(self.select_folder)

        select_layout.addWidget(self.folder_label)
        select_layout.addWidget(self.select_folder_btn)

        main_layout.addLayout(select_layout)

        # 表格区域
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["图片名称", "A/B面设置", "缩略图", "靶点坐标", "预览"])
        # 第0列（图片名称）：固定宽度200px
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 200)
        # 第1列（A/B面设置）：固定宽度200px
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 200)
        # 第2列（缩略图）：固定宽度200px
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.setColumnWidth(2, 200)
        # 第3列（靶点坐标）：固定宽度200px
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.table.setColumnWidth(3, 200)
        # 第4列（预览）：固定宽度200px
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Fixed)
        self.table.setColumnWidth(4, 200)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        # 连接表格尺寸变化信号，用于调整缩略图
        self.table.horizontalHeader().sectionResized.connect(self.on_table_resized)
        self.table.verticalHeader().sectionResized.connect(self.on_table_resized)

        main_layout.addWidget(self.table)

        # 进度显示
        self.processing_progress = QProgressBar()
        self.processing_progress.setTextVisible(True)
        self.processing_progress.setValue(0)
        self.processing_status = QLabel("就绪")
        main_layout.addWidget(self.processing_status)
        main_layout.addWidget(self.processing_progress)

        # 按钮区域
        btn_layout = QHBoxLayout()
        self.process_all_btn = QPushButton("全部解析")
        self.process_all_btn.clicked.connect(self.process_all_images)
        self.process_all_btn.setEnabled(False)

        self.confirm_btn = QPushButton("确定")
        self.confirm_btn.clicked.connect(self.confirm_import)
        self.confirm_btn.setEnabled(False)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(self.process_all_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.confirm_btn)
        btn_layout.addWidget(cancel_btn)

        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

        # 连接进度信号
        self.processing_updated.connect(self.update_processing_progress)
        self.processing_completed.connect(self.on_processing_completed)

    def on_table_resized(self, *args):
        """表格尺寸变化时重新调整所有缩略图。"""
        for row in range(self.table.rowCount()):
            # 获取缩略图单元格部件
            thumbnail_widget = self.table.cellWidget(row, 2)
            if isinstance(thumbnail_widget, ThumbnailDisplayWidget):
                # 重新调整缩略图大小
                thumbnail_widget.adjust_thumbnail_size()

    def set_material_id(self):
        """设置料号。"""
        material_id, ok = QInputDialog.getText(self, "设置料号", "请输入料号:", text=self.material_id)
        if ok and material_id:
            self.material_id = material_id
            self.id_input.setText(material_id)

    def select_folder(self):
        """选择图片文件夹。"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder_path:
            self.folder_label.setText(f"当前文件夹: {folder_path}")

            # 自动根据文件夹名称和时间设置料号
            folder_name = os.path.basename(folder_path)
            current_time = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            self.material_id = f"{folder_name}_{current_time}"
            self.id_input.setText(self.material_id)

            self.load_images(folder_path)

    def load_images(self, folder_path):
        """加载图片到表格。"""
        # 清空表格和数据
        self.table.setRowCount(0)
        self.image_data = []

        # 获取所有图片文件
        img_files = [f for f in os.listdir(folder_path)
                    if f.lower().endswith(('.bmp', '.jpg', '.png', '.tif'))]

        if not img_files:
            self.show_message("警告", "所选文件夹中没有图片文件")
            return

        # 创建缩略图目录（路径包含 material 根目录）
        full_material_id = os.path.join(config.material_root, self.material_id)
        thumbnails_dir = os.path.join(full_material_id, "thumbnails")
        ensure_dir(thumbnails_dir)

        # 填充表格
        self.table.setRowCount(len(img_files))
        for row in range(len(img_files)):
            self.table.setRowHeight(row, 50)
        self.table.updateGeometry()  # 重新计算表格整体布局
        QApplication.processEvents()

        for row, filename in enumerate(img_files):
            file_path = os.path.join(folder_path, filename)

            # 图片名称
            name_item = QTableWidgetItem(filename)
            self.table.setItem(row, 0, name_item)

            # A/B面设置下拉框
            combo_box = QComboBox()
            combo_box.addItems(["未设置", "A", "B"])
            combo_box.currentIndexChanged.connect(lambda idx, r=row: self.on_side_changed(r, idx))
            self.table.setCellWidget(row, 1, combo_box)

            # 缩略图预览（仅展示，不支持点击选择靶点）
            thumbnail_container = ThumbnailDisplayWidget(row, self)
            thumbnail_container.setStyleSheet("border: 1px solid #ccc;")
            self.table.setCellWidget(row, 2, thumbnail_container)

            # 靶点坐标显示
            targets_label = QLabel("未选择靶点")
            targets_label.setAlignment(Qt.AlignCenter)
            self.table.setCellWidget(row, 3, targets_label)

            # 预览按钮
            preview_btn = QPushButton("预览")
            preview_btn.clicked.connect(lambda checked, r=row: self.open_preview_window(r))
            self.table.setCellWidget(row, 4, preview_btn)

            # 保存图片数据
            self.image_data.append({
                "path": file_path,
                "name": filename,
                "side": None,
                "thumbnail": None,
                "targets": [],  # 存储靶点坐标 (x, y)
                "original_size": None,  # 原始图片尺寸 (width, height)
                "thumbnail_size": None,  # 缩略图尺寸 (width, height)
                "slice_widths": None,
                "temp_height": None
            })

            # 异步生成缩略图
            self.generate_thumbnail_async(file_path, row, thumbnail_container)

        self.process_all_btn.setEnabled(True)
        self.confirm_btn.setEnabled(True)

    def on_side_changed(self, row, index):
        """处理A/B面选择变化，确保只有一个A面，B面时清除并禁用靶点。"""
        selected_side = ["未设置", "A", "B"][index]

        # 如果选择了A面，检查是否已有其他A面
        if selected_side == "A":
            for r in range(self.table.rowCount()):
                if r != row:
                    combo = self.table.cellWidget(r, 1)
                    if combo.currentText() == "A":
                        combo.setCurrentIndex(0)  # 重置为未设置
                        self.image_data[r]["side"] = None

        # 如果选择了B面，清除并禁用靶点功能
        elif selected_side == "B":
            # 清除该图片的所有靶点
            if self.image_data[row]["targets"]:
                self.image_data[row]["targets"] = []
                self.update_targets_display(row)

        # 更新当前行的side值
        self.image_data[row]["side"] = selected_side if selected_side != "未设置" else None

        # 更新缩略图和按钮状态
        thumbnail_container = self.table.cellWidget(row, 2)
        if isinstance(thumbnail_container, ThumbnailDisplayWidget):
            # 对于B面，添加提示文字
            if selected_side == "B":
                thumbnail_container.show_b_side_indicator()
            else:
                thumbnail_container.hide_b_side_indicator()

            # 调整缩略图大小以适应单元格
            thumbnail_container.adjust_thumbnail_size()

    def open_preview_window(self, row):
        """打开缩略图预览窗口（模态窗口）。"""
        if row >= len(self.image_data):
            return

        image_info = self.image_data[row]
        if not image_info["thumbnail"] or not os.path.exists(image_info["thumbnail"]):
            self.show_message("提示", "没有可用的缩略图进行预览")
            return

        # 检查是否为B面，如果是则提示不需要选择靶点
        if image_info["side"] == "B":
            self.show_message("提示", "B面图片不需要选择靶点")
            return

        # 保存当前靶点数据，供预览窗口使用
        current_targets = image_info["targets"].copy()

        # 创建新的预览窗口（模态对话框）
        self.preview_window = ImagePreviewWindow(
            row,
            self,
            image_info["thumbnail"],
            image_info["original_size"],
            image_info["thumbnail_size"],
            current_targets
        )
        self.preview_window.setWindowTitle(f"预览: {image_info['name']}")
        self.preview_window.resize(800, 600)

        # 以模态方式显示窗口，用户必须关闭它才能返回主界面
        result = self.preview_window.exec_()

        # 窗口关闭后，获取更新后的靶点数据
        if result == QDialog.Accepted:
            updated_targets = self.preview_window.get_targets()
            if updated_targets is not None:
                # 更新主窗口的靶点数据
                self.image_data[row]["targets"] = updated_targets
                self.update_targets_display(row)

        # 清理预览窗口引用
        self.preview_window = None

    def generate_thumbnail_async(self, image_path, row, thumbnail_container):
        """异步生成缩略图并获取原始图片尺寸。"""
        try:
            full_material_id = os.path.join(config.material_root, self.material_id)
            thumbnails_dir = os.path.join(full_material_id, "thumbnails")
            ensure_dir(thumbnails_dir)

            with Image.open(image_path) as img:
                # 保存原始图片尺寸
                original_width, original_height = img.size
                self.image_data[row]["original_size"] = (original_width, original_height)

                base_name = os.path.basename(image_path)
                thumbnail_path = generate_thumbnail(img, base_name, thumbnails_dir)

                if thumbnail_path and os.path.exists(thumbnail_path):
                    self.image_data[row]["thumbnail"] = thumbnail_path

                    # 加载缩略图并获取尺寸
                    pixmap = QPixmap(thumbnail_path)
                    if not pixmap.isNull():
                        # 保存缩略图尺寸
                        self.image_data[row]["thumbnail_size"] = (pixmap.width(), pixmap.height())
                        # 设置缩略图到容器
                        thumbnail_container.set_thumbnail(pixmap)
                        # 调整缩略图大小以适应单元格
                        thumbnail_container.adjust_thumbnail_size()
                    else:
                        thumbnail_container.set_text("无法加载")
                else:
                    thumbnail_container.set_text("生成失败")
        except Exception as e:
            print(f"生成缩略图失败: {str(e)}")
            thumbnail_container.set_text("生成失败")
        finally:
            # 清理资源
            QPixmapCache.clear()
            QApplication.processEvents()  # 强制刷新UI

    def add_target(self, row, thumbnail_pos):
        """添加靶点坐标（适配缩略图比例转换）。"""
        # 检查是否有必要的尺寸信息
        if not self.image_data[row].get("original_size") or not self.image_data[row].get("thumbnail_size"):
            self.show_message("错误", "无法获取图片尺寸信息，无法添加靶点")
            return

        # 获取尺寸信息
        orig_width, orig_height = self.image_data[row]["original_size"]
        thumb_width, thumb_height = self.image_data[row]["thumbnail_size"]

        # 计算缩放比例（原图/缩略图）
        scale_x = orig_width / thumb_width
        scale_y = orig_height / thumb_height

        # 转换坐标到原始图片
        orig_x = int(thumbnail_pos.x() * scale_x)
        orig_y = int(thumbnail_pos.y() * scale_y)

        # 添加到靶点列表
        self.image_data[row]["targets"].append((orig_x, orig_y))

        # 更新显示
        self.update_targets_display(row)

    def update_targets_display(self, row):
        """更新靶点显示。"""
        targets = self.image_data[row]["targets"]
        targets_label = self.table.cellWidget(row, 3)

        if not targets:
            targets_label.setText("未选择靶点")
            return

        # 显示靶点坐标，最多显示3个
        display_text = "\n".join([f"靶点 {i+1}: ({x}, {y})" for i, (x, y) in enumerate(targets) if i < 3])
        if len(targets) > 3:
            display_text += f"\n...共{len(targets)}个"

        targets_label.setText(display_text)

    def validate_board_parameters(self):
        """验证板宽、板高、板厚参数是否已填写。"""
        self.board_width = self.width_input.text().strip()
        self.board_height = self.height_input.text().strip()
        self.board_thickness = self.thickness_input.text().strip()

        if not self.board_width or not self.board_height or not self.board_thickness:
            return False

        # 检查是否为数字
        try:
            float(self.board_width)
            float(self.board_height)
            float(self.board_thickness)
            return True
        except ValueError:
            return False

    def validate_targets(self, images_to_process):
        """验证是否已选择靶点（只验证A面）。"""
        for path, side in images_to_process:
            # 只有A面需要验证靶点
            if side != "A":
                continue

            # 查找对应的图片数据
            found = False
            for info in self.image_data:
                if info["path"] == path:
                    found = True
                    if not info["targets"]:  # 没有靶点
                        return False, os.path.basename(path)
                    break
            if not found:
                return False, os.path.basename(path)
        return True, None

    def show_loading_status(self, message):
        """显示图片加载状态。"""
        self.processing_status.setText(message)

    def process_all_images(self):
        """处理所有图片（修复线程销毁问题）。"""
        if not self.validate_board_parameters():
            self.show_message("警告", "请先正确填写板宽、板高、板厚参数（必须为数字）")
            return
        images_to_process = []
        for row, info in enumerate(self.image_data):
            combo_box = self.table.cellWidget(row, 1)
            side_text = combo_box.currentText()
            if side_text != "未设置":
                info["side"] = side_text
                images_to_process.append((info["path"], side_text))

        if not images_to_process:
            self.show_message("警告", "没有设置A面或B面的图片")
            return

         # 验证靶点是否已选择（只验证A面）
        has_valid_targets, missing_file = self.validate_targets(images_to_process)
        if not has_valid_targets:
            self.show_message("警告", f"A面图片 {missing_file} 未选择靶点，请先选择靶点")
            return

        # 停止现有线程
        self._stop_processing_thread()

        # 创建工作对象和线程
        self.processing_worker = ImageProcessingWorker(
            images_and_sides=images_to_process,
            material_id=self.material_id,
            image_data=self.image_data
        )

        self.processing_worker.moveToThread(self.processing_thread)

        # 连接信号槽（核心：明确任务流程，避免中途销毁）
        self.processing_thread.started.connect(self.processing_worker.start_processing)
        self.processing_worker.progress_updated.connect(self.processing_updated)
        self.processing_worker.processing_completed.connect(self.processing_completed)
        self.processing_worker.processing_failed.connect(self.on_processing_failed)
        self.processing_worker.image_loading.connect(self.show_loading_status)

        # 修复：线程清理函数（确保线程停止后再销毁）
        def cleanup():
            # 关键：先确保线程已停止
            if self.processing_thread and self.processing_thread.isRunning():
                self.processing_thread.quit()  # 强制停止
                self.processing_thread.wait()  # 等待1秒确保停止

            # 安全断开信号（避免重复触发）
            try:
                if self.processing_worker:
                    self.processing_worker.progress_updated.disconnect()
                    self.processing_worker.processing_completed.disconnect()
                    self.processing_worker.processing_failed.disconnect()
                    self.processing_worker.image_loading.disconnect()
                if self.processing_thread:
                    self.processing_thread.started.disconnect()
            except TypeError:
                pass  # 忽略已断开的连接

            # 销毁对象并清除引用
            if self.processing_worker:
                self.processing_worker.deleteLater()

            self.processing_worker = None

        # 修复：仅在所有任务完全结束后才触发清理（避免中途调用）
        self.processing_worker.processing_completed.connect(cleanup)  # 所有任务成功完成
        self.processing_worker.processing_failed.connect(cleanup)     # 任务出错终止
        self.processing_thread.finished.connect(cleanup)              # 线程自然结束

        # 更新UI状态
        self.processing_status.setText("正在批量解析图片...")
        self.processing_progress.setValue(0)
        self.process_all_btn.setEnabled(False)
        self.confirm_btn.setEnabled(False)
        self.processing_started.emit()

        # 启动线程
        self.processing_thread.start()

    def _stop_processing_thread(self):
        """安全停止处理线程（核心修复）。"""
        # 使用 try-except 捕获已销毁对象的异常
        try:
            if self.processing_thread and self.processing_thread.isRunning():
                if self.processing_worker:
                    self.processing_worker.stop_processing()
                    self.processing_worker = None
                self.processing_thread.quit()
                self.processing_thread.wait()
        except RuntimeError as e:
            print(f"停止线程时出错: {e}")
        finally:
            self.processing_worker = None

    def _cleanup_resources(self):
        """清理图片资源和表格部件（核心优化）。"""
        # 1. 清空表格，释放所有单元格部件（尤其是 QPixmap）
        self.table.clearContents()  # 清除单元格内容
        self.table.setRowCount(0)   # 重置行数，销毁行对象

        # 2. 清除图片数据引用
        self.image_data = []

        # 3. 强制释放 QPixmap 缓存
        QPixmapCache.clear()

        # 4. 强制 Python 垃圾回收
        gc.collect(generation=0)

    def reject(self):
        """对话框关闭时确保线程资源释放。"""
        self._stop_processing_thread()
        self._cleanup_resources()
        self.image_data.clear()
        gc.collect(generation=0)
        super().reject()

    def update_processing_progress(self, progress, filename):
        """更新处理进度。"""
        self.processing_progress.setValue(progress)
        self.processing_status.setText(f"正在处理: {filename} ({progress}%)")

    def on_processing_completed(self, results):
        """批量处理完成回调。"""
        success_count = len(results["success"])
        failed_count = len(results["failed"])

        # 显示的路径包含 material 根目录
        full_material_path = os.path.join(config.material_root, self.material_id)
        message = f"图片解析与解析完成，已保存到 {full_material_path} 文件夹\n"
        message += f"成功: {success_count} 张\n"

        # 更新缩略图信息
        for success_info in results["success"]:
            original_path = success_info["original"]
            slice_widths = success_info["slice_widths"]
            temp_height = success_info["temp_height"]
            targets = success_info["targets"]

            # 查找对应的图片数据并更新缩略图路径
            for info in self.image_data:
                if info["path"] == original_path:
                    info["slice_widths"] = slice_widths
                    info["temp_height"] = temp_height
                    info["targets"] = targets
                    break

        if failed_count > 0:
            message += f"失败: {failed_count} 张"
            if not self.auto_confirm:
                error_list = "\n".join([f"{os.path.basename(f['path'])} ({f['side']}面): {f['error']}" for f in results["failed"]])
                message += f"\n\n失败详情:\n{error_list}"

        self.show_message("成功", message)
        self.process_all_btn.setEnabled(True)
        self.confirm_btn.setEnabled(True)
        self.processing_status.setText("就绪")
        self.processing_progress.setValue(0)

    def on_processing_failed(self, error, filename):
        """处理失败回调。"""
        self.show_message("错误", f"处理{'批量' if filename == '批量处理' else filename}时出错: {error}")
        self.process_all_btn.setEnabled(True)
        self.confirm_btn.setEnabled(True)
        self.processing_status.setText("就绪")
        self.processing_progress.setValue(0)

    def confirm_import(self):
        """确认导入。"""
        # 验证板材参数
        if not self.validate_board_parameters():
            self.show_message("警告", "请先正确填写板宽、板高、板厚参数（必须为数字）")
            return

        if not self.material_id:
            self.show_message("警告", "请先设置料号!")
            return

        # 收集A面和B面图片（路径已包含 material 根目录）
        a_images = []
        b_images = []
        a_thumbnail = None
        b_thumbnail = None
        slice_widths = None
        temp_height = None
        targets = None

        for info in self.image_data:
            slice_widths = info["slice_widths"]
            temp_height = info["temp_height"]
            if info["side"] == "A":
                targets = info["targets"]
                a_images.append(info["path"])
                if not a_thumbnail and info.get("thumbnail"):
                    a_thumbnail = info["thumbnail"]
            elif info["side"] == "B":
                b_images.append(info["path"])
                if not b_thumbnail and info.get("thumbnail"):
                    b_thumbnail = info["thumbnail"]

        if not a_images and not b_images:
            self.show_message("警告", "至少需要选择一张A面或B面图片!")
            return

        # 准备导入数据
        import_data = {
            "A面": a_images[0] if a_images else "",
            "B面": b_images[0] if b_images else "",
            "A面全部": a_images,
            "B面全部": b_images,
            "A面缩略图": a_thumbnail,
            "B面缩略图": b_thumbnail,
            "板宽(mm)": self.board_width,
            "板高(mm)": self.board_height,
            "板厚(mm)": self.board_thickness,
            "条带宽": slice_widths,
            "资料高": temp_height,
            "靶点数据": targets  # 只包含A面靶点
        }

        # 发送导入信号（传递不含 material 根目录的料号名称）
        self.material_imported.emit(self.material_id, import_data)
        self._cleanup_resources()  # 清理资源
        self.accept()

    def show_message(self, title, message):
        """显示消息框。"""
        if hasattr(self, 'auto_confirm') and self.auto_confirm:
            return
        QMessageBox.information(self, title, message)
