import sys
import os
import logging
from pathlib import Path
from AoiSystemWindow_new.ImageView import Viewer
from PIL import Image
from PyQt5.QtWidgets import QDialog, QApplication,  QFileDialog, QMessageBox, \
                            QSizePolicy, QLabel, QTreeWidgetItem, \
                            QListWidgetItem, QTableWidgetItem, QAction
from PyQt5.QtGui import QDoubleValidator, QColor
from PyQt5 import uic, QtWidgets, QtGui, QtCore
Image.MAX_IMAGE_PIXELS = None

class ViewerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 加载UI文件
        self.ui = uic.loadUiType("./AoiSystemWindow_new/viewer.ui")[0]()
        self.ui.setupUi(self)
        self.setWindowTitle("图片查看器")
        
        # 初始化变量
        self.folder_path = ""
        self.image_paths = []
        self.current_index = 0
        self.selected_image = None
        self.template_image_paths = []
        self.template_list = []
        self.selected_temp_image = None
        self.sample_data_display = False
        self.sample_data_path = ""
        self.gerber_image = False
        self.gerber_image_path = ""
        
        # 创建工具栏和按钮
        self.setup_toolbars()
        
        # 设置图形视图（关键修复部分）
        self.setup_graphics_views()
        
        # 连接信号和槽
        self.setup_connections()
    
    def setup_toolbars(self):
        """设置工具栏和按钮"""
        # 样本图片工具栏
        raw_toolbar = QtWidgets.QToolBar(self.ui.rawtoolbar_widget)
        self.ui.rawtoolbar_verticalLayout.addWidget(raw_toolbar)
        ZoomInActionSample = QAction('放大', self.ui.rawtoolbar_widget)
        ZoomOutActionSample = QAction('缩小', self.ui.rawtoolbar_widget)
        ResetZoomActionSample = QAction('重置大小', self.ui.rawtoolbar_widget)
        FitToWinActionSample = QAction('适应窗口', self.ui.rawtoolbar_widget)

        raw_toolbar.addAction(ZoomInActionSample)
        raw_toolbar.addAction(ZoomOutActionSample)
        raw_toolbar.addAction(ResetZoomActionSample)
        raw_toolbar.addAction(FitToWinActionSample)

        ZoomInActionSample.triggered.connect(self.sample_zoom_in)
        ZoomOutActionSample.triggered.connect(self.sample_zoom_out)
        ResetZoomActionSample.triggered.connect(self.sample_original_size)
        FitToWinActionSample.triggered.connect(self.sample_fit_to_window)

        # 模板图片工具栏
        template_toolbar = QtWidgets.QToolBar(self.ui.pcbtoolbar_widget)
        self.ui.pcbtoolbar_verticalLayout.addWidget(template_toolbar)
        ZoomInActionTemplate = QAction('放大', self.ui.pcbtoolbar_widget)
        ZoomOutActionTemplate = QAction('缩小', self.ui.pcbtoolbar_widget)
        ResetZoomActionTemplate = QAction('重置大小', self.ui.pcbtoolbar_widget)
        FitToWinActionTemplate = QAction('适应窗口', self.ui.pcbtoolbar_widget)

        template_toolbar.addAction(ZoomInActionTemplate)
        template_toolbar.addAction(ZoomOutActionTemplate)
        template_toolbar.addAction(ResetZoomActionTemplate)
        template_toolbar.addAction(FitToWinActionTemplate)

        ZoomInActionTemplate.triggered.connect(self.template_zoom_in)
        ZoomOutActionTemplate.triggered.connect(self.template_zoom_out)
        ResetZoomActionTemplate.triggered.connect(self.template_original_size)
        FitToWinActionTemplate.triggered.connect(self.template_fit_to_window)
    
    def setup_graphics_views(self):
        """修复主界面调用时替换控件多一个QWidget的问题"""
        # 替换graphicsView_pcb为Viewer
        if hasattr(self.ui, 'graphicsView_pcb') and self.ui.graphicsView_pcb:
            old_view = self.ui.graphicsView_pcb
            layout = self.ui.pcb_verticalLayout
            index = layout.indexOf(old_view)
            
            # 1. 保存旧控件属性（尺寸、策略等）
            size_policy = old_view.sizePolicy()
            min_size = old_view.minimumSize()
            max_size = old_view.maximumSize()
            
            # 2. 强制从布局和父对象中移除旧控件（同步操作，不依赖deleteLater）
            layout.removeWidget(old_view)
            old_view.setParent(None)  # 解除父对象关联
            old_view.destroy()  # 立即销毁（替代deleteLater，避免延迟）
            
            # 3. 创建新控件，明确父对象为UI的容器部件（避免成为主界面子控件）
            new_view = Viewer(self.ui.pcbtoolbar_widget.parentWidget())  # 父对象设为对话框内的容器
            new_view.setFrameShape(QtWidgets.QFrame.Box)
            new_view.setFrameShadow(QtWidgets.QFrame.Sunken)
            new_view.setLineWidth(2)
            new_view.clear()
            # 继承旧控件属性
            new_view.setSizePolicy(size_policy)
            new_view.setMinimumSize(min_size)
            new_view.setMaximumSize(max_size)
            
            # 4. 添加新控件到布局，并强制刷新布局
            layout.insertWidget(index, new_view)
            layout.update()  # 强制布局重新计算
            self.ui.graphicsView_pcb_gerber = new_view  # 更新引用

        # 替换graphicsView_result为Viewer（同上述逻辑）
        if hasattr(self.ui, 'graphicsView_result') and self.ui.graphicsView_result:
            old_view = self.ui.graphicsView_result
            layout = self.ui.raw_verticalLayout
            index = layout.indexOf(old_view)
            
            size_policy = old_view.sizePolicy()
            min_size = old_view.minimumSize()
            max_size = old_view.maximumSize()
            
            layout.removeWidget(old_view)
            old_view.setParent(None)
            old_view.destroy()
            
            new_view = Viewer(self.ui.rawtoolbar_widget.parentWidget())
            new_view.setFrameShape(QtWidgets.QFrame.Box)
            new_view.setFrameShadow(QtWidgets.QFrame.Sunken)
            new_view.setLineWidth(2)
            new_view.clear()
            new_view.setSizePolicy(size_policy)
            new_view.setMinimumSize(min_size)
            new_view.setMaximumSize(max_size)
            
            layout.insertWidget(index, new_view)
            layout.update()
            self.ui.graphicsView_result_defect = new_view
    
    def setup_connections(self):
        """连接信号和槽"""
        self.ui.pushButton_image.clicked.connect(self.import_images)
        self.ui.pushButton_template.clicked.connect(self.import_gerber_list)
        self.ui.templatelistWidget.itemClicked.connect(self.template_name_clicked)
        self.ui.imagelistWidget.itemClicked.connect(self.image_name_clicked)
    
    def import_images(self):
        """导入样本图片"""
        if self.selected_image:
            self.selected_image.setForeground(QColor('black'))  # 恢复文本颜色
            self.selected_image = None
            self.current_index = 0
            self.image_paths = []
            self.ui.imagelistWidget.clear()

        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.folder_path = folder
            self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if
                                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
            if self.image_paths:
                self.current_index = 0
                self.display_image_names(self.image_paths)
                # 自动显示第一张图片
                self.show_image(self.current_index)
            else:
                QMessageBox.warning(self, "警告", "所选文件夹中没有支持的图片格式")
        else:
            return
    
    def import_gerber_list(self):
        """导入模板图片"""
        if self.selected_temp_image:
            self.selected_temp_image.setForeground(QColor('black'))  # 恢复文本颜色
            self.selected_temp_image = None
        self.ui.templatelistWidget.clear()

        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.template_image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if
                                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            self.template_list = sorted(self.template_image_paths, key=lambda x: os.path.basename(x).lower())
            if self.template_list:
                for image_path in self.template_list:
                    image_name = os.path.basename(image_path)
                    self.ui.templatelistWidget.addItem(image_name)
                # 自动显示第一张模板图片
                self.show_template_image(0)
            else:
                QMessageBox.warning(self, "警告", "所选文件夹中没有支持的图片格式")
        else:
            return
    
    def display_image_names(self, image_paths):
        """在列表中显示图片名称"""
        self.ui.imagelistWidget.clear()
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            self.ui.imagelistWidget.addItem(image_name)
    
    def image_name_clicked(self, item):
        """处理样本图片列表项点击事件"""
        index = self.ui.imagelistWidget.row(item)
        self.show_image(index)
    
    def show_image(self, index):
        """显示样本图片"""
        if 0 <= index < len(self.image_paths):
            if self.selected_image:
                self.selected_image.setForeground(QColor('black'))  # 恢复之前选中项的文本颜色
            self.selected_image = self.ui.imagelistWidget.item(index)
            self.selected_image.setForeground(QColor('red'))
            self.current_index = index
            if self.image_paths:
                self.ui.graphicsView_result_defect.clear()
                self.ui.graphicsView_result_defect.set_image(self.image_paths[self.current_index])
                fileStr = Path(self.image_paths[self.current_index])
                fileName = fileStr.stem
                self.ui.out_label.setText(f"B窗口：{fileName}")
                self.sample_data_display = True
                self.sample_data_path = self.image_paths[self.current_index]
    
    def template_name_clicked(self, item):
        """处理模板图片列表项点击事件"""
        index = self.ui.templatelistWidget.row(item)
        self.show_template_image(index)
    
    def show_template_image(self, index):
        """显示模板图片"""
        if 0 <= index < len(self.template_list):
            if self.selected_temp_image:
                self.selected_temp_image.setForeground(QColor('black'))  # 恢复之前选中项的文本颜色
            self.selected_temp_image = self.ui.templatelistWidget.item(index)
            self.selected_temp_image.setForeground(QColor('red'))
            self.ui.graphicsView_pcb_gerber.clear()
            self.ui.graphicsView_pcb_gerber.set_image(self.template_list[index])
            fileStr = Path(self.template_list[index])
            fileName = fileStr.stem
            self.ui.pcb_label.setText(f"A窗口：{fileName}")
            self.gerber_image = True
            self.gerber_image_path = self.template_list[index]
    
    # 缩放图片
    def sample_zoom_in(self):
        if self.sample_data_display:
            self.ui.graphicsView_result_defect.zoomIn()
        else:
            QMessageBox.warning(self, "警告", "请先加载图像！")

    def sample_zoom_out(self):
        if self.sample_data_display:
            self.ui.graphicsView_result_defect.zoomOut()
        else:
            QMessageBox.warning(self, "警告", "请先加载图像！")

    def sample_original_size(self):
        if self.sample_data_display:
            self.ui.graphicsView_result_defect.resetZoom()
        else:
            QMessageBox.warning(self, "警告", "请先加载图像！")

    def sample_fit_to_window(self):
        if self.sample_data_display:
            self.ui.graphicsView_result_defect.zoomFit()
        else:
            QMessageBox.warning(self, "警告", "请先加载图像！")

    def template_zoom_in(self):
        if self.gerber_image:
            self.ui.graphicsView_pcb_gerber.zoomIn()
        else:
            QMessageBox.warning(self, "警告", "请先导入template文件！")

    def template_zoom_out(self):
        if self.gerber_image:
            self.ui.graphicsView_pcb_gerber.zoomOut()
        else:
            QMessageBox.warning(self, "警告", "请先导入template文件！")

    def template_original_size(self):
        if self.gerber_image:
            self.ui.graphicsView_pcb_gerber.resetZoom()
        else:
            QMessageBox.warning(self, "警告", "请先导入template文件！")

    def template_fit_to_window(self):
        if self.gerber_image:
            self.ui.graphicsView_pcb_gerber.zoomFit()
        else:
            QMessageBox.warning(self, "警告", "请先导入template文件！")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ViewerDialog()
    window.show()
    sys.exit(app.exec_())
