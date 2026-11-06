"""缩略图与图片预览相关自定义控件。

从 AoiSystemMain.py 拆分而来。
"""
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QColor, QPixmapCache,
                         QBrush, QCursor, QFont, QGuiApplication)
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout,
                            QHBoxLayout, QScrollArea, QDialog)


class ThumbnailDisplayWidget(QWidget):
    """仅用于展示缩略图的部件，支持自适应单元格大小。"""
    def __init__(self, row, parent_dialog):
        super().__init__()
        self.row = row
        self.parent_dialog = parent_dialog
        self.thumbnail = None  # 原始缩略图
        self.displayed_pixmap = None  # 显示的缩放后的缩略图
        self.b_side_overlay = None  # B面提示覆盖层
        self.setMinimumSize(200, 50)
        self.setMaximumSize(200, 50)

        self.init_ui()

        # 监听自身大小变化事件
        self.resizeEvent = self.on_resize

    def init_ui(self):
        self.label = QLabel("等待缩略图...", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setGeometry(0, 0, self.width(), self.height())

    def on_resize(self, event):
        """当部件大小变化时调整缩略图。"""
        self.label.setGeometry(0, 0, self.width(), self.height())
        self.adjust_thumbnail_size()
        # 调整B面提示位置
        if self.b_side_overlay:
            self.b_side_overlay.setGeometry(0, self.height()//2 - 15, self.width(), 30)
        super().resizeEvent(event)

    def show_b_side_indicator(self):
        """显示B面提示文字覆盖层。"""
        # 如果已存在则移除
        if self.b_side_overlay:
            self.b_side_overlay.deleteLater()

        # 创建覆盖层
        self.b_side_overlay = QLabel("B面无需选择靶点", self)
        self.b_side_overlay.setStyleSheet("color: #888; font-weight: bold; background-color: rgba(255, 255, 255, 0.7);")
        self.b_side_overlay.setAlignment(Qt.AlignCenter)
        self.b_side_overlay.setGeometry(0, self.height()//2 - 15, self.width(), 30)
        self.b_side_overlay.show()

    def hide_b_side_indicator(self):
        """隐藏B面提示文字覆盖层。"""
        if self.b_side_overlay:
            self.b_side_overlay.deleteLater()
            self.b_side_overlay = None

    def set_thumbnail(self, pixmap):
        """设置原始缩略图。"""
        self.thumbnail = pixmap
        self.displayed_pixmap = pixmap
        self.adjust_thumbnail_size()  # 立即调整大小

    def adjust_thumbnail_size(self):
        """调整缩略图大小以适应当前部件大小。"""
        if self.thumbnail and not self.thumbnail.isNull():
            # 计算可用空间（减去边距）
            available_width = self.width() - 10  # 左右边距各5px
            available_height = self.height() - 10  # 上下边距各5px

            # 按比例缩放图片以适应可用空间
            scaled_pixmap = self.thumbnail.scaled(
                available_width,
                available_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation  # 使用平滑缩放算法
            )

            self.label.setPixmap(scaled_pixmap)
            self.displayed_pixmap = scaled_pixmap

    def set_text(self, text):
        """显示文本替代缩略图。"""
        self.label.setText(text)
        self.label.setPixmap(QPixmap())  # 清除图片
        self.thumbnail = None
        self.displayed_pixmap = None


class ImagePreviewWindow(QDialog):
    """图片预览窗口（模态对话框），支持缩放、拖拽和右键选点，绘制红色靶点。"""
    def __init__(self, row, parent_dialog, image_path, original_size, thumbnail_size, initial_targets, parent=None):
        super().__init__(parent)
        self.row = row
        self.parent_dialog = parent_dialog
        self.image_path = image_path
        self.original_size = original_size  # (width, height)
        self.thumbnail_size = thumbnail_size  # (width, height)
        self.targets = initial_targets.copy()  # 存储原始图片坐标的靶点
        self.scale = 1.0  # 缩放比例
        self.offset = QPoint(0, 0)  # 偏移量（用于拖拽）
        self.dragging = False  # 是否正在拖拽
        self.last_pos = QPoint()  # 上一次鼠标位置

        # 设置为模态对话框
        self.setModal(True)

        # 窗口标题和布局
        self.setWindowTitle("图片预览")
        self.init_ui()

        # 加载带靶点的图片
        self.load_image_with_targets()

        # 设置字体用于靶点编号显示
        self.font = QFont()
        self.font.setPointSize(10)
        self.font.setBold(True)

    def init_ui(self):
        """初始化界面。"""
        layout = QVBoxLayout()

        # 添加操作按钮
        button_layout = QHBoxLayout()

        self.clear_last_btn = QPushButton("清除上一个靶点")
        self.clear_last_btn.clicked.connect(self.clear_last_target)
        self.clear_last_btn.setEnabled(len(self.targets) > 0)

        self.clear_all_btn = QPushButton("清除全部靶点")
        self.clear_all_btn.clicked.connect(self.clear_all_targets)
        self.clear_all_btn.setEnabled(len(self.targets) > 0)

        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.clear_last_btn)
        button_layout.addWidget(self.clear_all_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        # 添加提示标签
        self.info_label = QLabel("操作提示: 按住CTRL+鼠标滚轮缩放，鼠标左键拖拽，鼠标右键添加靶点")
        layout.addWidget(self.info_label)

        # 创建滚动区域用于显示大图
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        # 创建用于显示图片的部件
        self.image_widget = QWidget()
        self.image_layout = QVBoxLayout(self.image_widget)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_layout.addWidget(self.image_label)

        self.scroll_area.setWidget(self.image_widget)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

        # 设置鼠标跟踪
        self.setMouseTracking(True)
        self.scroll_area.setMouseTracking(True)
        self.image_widget.setMouseTracking(True)
        self.image_label.setMouseTracking(True)

    def load_image_with_targets(self):
        """加载图片并绘制已有的红色靶点。"""
        # 加载原始图片
        self.base_pixmap = QPixmap(self.image_path)
        if self.base_pixmap.isNull():
            self.image_label.setText("无法加载图片")
            return

        # 绘制靶点到图片上
        self.update_image_with_targets()

    def update_image_with_targets(self):
        """更新图片，绘制所有靶点为红色点。"""
        if self.base_pixmap.isNull():
            return

        # 创建原始图片的副本用于绘制靶点
        marked_pixmap = QPixmap(self.base_pixmap)
        painter = QPainter(marked_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # 计算原始图片到缩略图的缩放比例
        if self.original_size[0] > 0 and self.original_size[1] > 0 and self.thumbnail_size[0] > 0:
            scale_x = self.thumbnail_size[0] / self.original_size[0]
            scale_y = self.thumbnail_size[1] / self.original_size[1]

            # 绘制每个靶点为红色点
            for i, (orig_x, orig_y) in enumerate(self.targets):
                # 转换原始坐标到缩略图坐标
                thumb_x = int(orig_x * scale_x)
                thumb_y = int(orig_y * scale_y)

                # 绘制红色实心圆作为靶点
                painter.setPen(QPen(QColor(255, 0, 0), 2))  # 红色边框
                painter.setBrush(QBrush(QColor(255, 0, 0)))  # 红色填充
                painter.drawEllipse(thumb_x - 5, thumb_y - 5, 10, 10)  # 10x10的红色圆点

                # 绘制靶点编号
                painter.setPen(QPen(QColor(255, 255, 255), 1))  # 白色文字边框
                painter.drawText(thumb_x + 6, thumb_y + 6, f"{i+1}")

        painter.end()

        # 应用当前缩放
        scaled_pixmap = marked_pixmap.scaled(
            int(marked_pixmap.width() * self.scale),
            int(marked_pixmap.height() * self.scale),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

    def get_targets(self):
        """获取当前靶点数据。"""
        return self.targets

    def mousePressEvent(self, event):
        """鼠标按下事件。"""
        if event.button() == Qt.RightButton:
            # 右键点击添加靶点
            self.add_target(event.pos())
        elif event.button() == Qt.LeftButton:
            # 左键点击开始拖拽
            self.dragging = True
            self.last_pos = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))

    def mouseReleaseEvent(self, event):
        """鼠标释放事件。"""
        if event.button() == Qt.LeftButton and self.dragging:
            # 结束拖拽
            self.dragging = False
            self.setCursor(QCursor(Qt.ArrowCursor))

    def mouseMoveEvent(self, event):
        """鼠标移动事件。"""
        if self.dragging:
            # 计算拖拽偏移
            delta = event.pos() - self.last_pos
            self.offset += delta
            self.last_pos = event.pos()

            # 调整滚动区域
            scroll_bar_h = self.scroll_area.horizontalScrollBar()
            scroll_bar_v = self.scroll_area.verticalScrollBar()
            scroll_bar_h.setValue(scroll_bar_h.value() - delta.x())
            scroll_bar_v.setValue(scroll_bar_v.value() - delta.y())

    def wheelEvent(self, event):
        """鼠标滚轮事件 - 缩放图片。"""
        # 只有按住Ctrl键时才缩放
        if QGuiApplication.keyboardModifiers() & Qt.ControlModifier:
            # 获取当前鼠标位置作为缩放中心
            mouse_pos = event.pos()

            # 计算缩放因子
            if event.angleDelta().y() > 0:
                # 放大
                self.scale *= 1.1
            else:
                # 缩小
                self.scale /= 1.1
                # 限制最小缩放
                if self.scale < 0.1:
                    self.scale = 0.1

            # 更新带靶点的图片
            self.update_image_with_targets()
        else:
            # 没有按Ctrl键，交给父类处理（滚动）
            super().wheelEvent(event)

    def add_target(self, pos):
        """添加靶点（适配缩略图比例转换）。"""
        if self.base_pixmap.isNull() or not self.thumbnail_size[0]:
            return

        # 计算点击位置在原始缩略图上的坐标
        label_pos = self.image_label.mapTo(self, QPoint(0, 0))
        label_size = self.image_label.size()
        pixmap_size = self.base_pixmap.size()

        # 计算缩放比例（图片实际显示大小 / 原始缩略图大小）
        scale_x = (pixmap_size.width() * self.scale) / self.thumbnail_size[0]
        scale_y = (pixmap_size.height() * self.scale) / self.thumbnail_size[1]

        # 转换点击位置到原始缩略图坐标
        if scale_x > 0 and scale_y > 0:
            thumb_x = (pos.x() - label_pos.x()) / scale_x
            thumb_y = (pos.y() - label_pos.y()) / scale_y

            # 确保坐标在缩略图范围内
            if 0 <= thumb_x < self.thumbnail_size[0] and 0 <= thumb_y < self.thumbnail_size[1]:
                # 转换为原始图片坐标
                orig_width, orig_height = self.original_size
                thumb_width, thumb_height = self.thumbnail_size

                scale_x_orig = orig_width / thumb_width
                scale_y_orig = orig_height / thumb_height

                orig_x = int(thumb_x * scale_x_orig)
                orig_y = int(thumb_y * scale_y_orig)

                # 添加到靶点列表
                self.targets.append((orig_x, orig_y))
                self.update_image_with_targets()

                # 更新按钮状态
                self.clear_last_btn.setEnabled(True)
                self.clear_all_btn.setEnabled(True)

    def clear_last_target(self):
        """清除上一个靶点。"""
        if self.targets:
            self.targets.pop()
            self.update_image_with_targets()

            # 更新按钮状态
            has_targets = len(self.targets) > 0
            self.clear_last_btn.setEnabled(has_targets)
            self.clear_all_btn.setEnabled(has_targets)

    def clear_all_targets(self):
        """清除全部靶点。"""
        if self.targets:
            self.targets = []
            self.update_image_with_targets()

            # 更新按钮状态
            self.clear_last_btn.setEnabled(False)
            self.clear_all_btn.setEnabled(False)
