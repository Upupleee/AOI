import sys
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QGraphicsPixmapItem, QGraphicsScene
from PyQt5.QtGui import QPixmap, QWheelEvent, QImage
from AoiSystemWindow_new.ImageEditor import ImageEditor

class Viewer(QtWidgets.QGraphicsView):
    """
    图像查看器组件，支持缩放、拖拽和裁剪功能
    """
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QGraphicsScene(), parent)
        
        # 初始化场景和图像项
        self.pixmap_item = self.scene().addPixmap(QtGui.QPixmap())
        self.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.setBackgroundRole(QtGui.QPalette.Dark)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.rubberBandChanged.connect(self.onRubberBandChanged)
        
        # 状态变量
        self.last_rect = QtCore.QPointF()
        self.crop_bool = False
        self.save_path = None
        self.image_path = None
        self.crop_flag = False
        self.start_drag_pos = None
        self.zoomInTimes = 0
        self.maxZoomInTimes = 100
        
        # 窗口尺寸配置（用于自适应缩放）
        self.window_width = 464
        self.window_height = 460

    def clear(self):
        """清除当前显示的图片"""
        if self.pixmap_item is not None:
            self.scene().removeItem(self.pixmap_item)
            self.pixmap_item = self.scene().addPixmap(QtGui.QPixmap())
        self.image_path = None
        self.crop_bool = False
        self.save_path = None

    def setImage(self, image):
        """通过numpy数组设置显示的图片"""
        self.clear()
        
        try:
            height, width = image.shape
            bytes_per_line = width
            
            # 创建QImage对象
            qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            
            self._setup_pixmap(pixmap)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image: {str(e)}")

    def set_image(self, img_path):
        """通过文件路径设置显示的图片"""
        self.clear()
        self.image_path = img_path
        
        if not img_path or not Path(img_path).exists():
            QMessageBox.warning(self, "Error", f"File not found: {img_path}")
            return
            
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", f"Failed to load image from {self.image_path}")
            return
            
        self._setup_pixmap(pixmap)

    def _setup_pixmap(self, pixmap):
        """内部方法：设置并显示pixmap，包含自适应逻辑"""
        if pixmap.isNull():
            return
            
        original_width = pixmap.width()
        original_height = pixmap.height()
        
        if original_width <= 0:
            return
            
        # 计算自适应缩放比例
        new_width = self.window_width / original_width
        new_height = self.window_height / original_height
        
        if new_width > new_height:
            self.scale_factor = new_height
        else:
            self.scale_factor = new_width
            
        self.pixmap_item.setPixmap(pixmap)
        self.resetZoom()
        self.zoomFit()

    def setPixmap(self, pixmap):
        """直接设置pixmap显示"""
        if pixmap.isNull():
            return
            
        original_width = pixmap.width()
        original_height = pixmap.height()
        
        if original_width <= 0:
            return
            
        # 计算自适应缩放比例
        new_width = self.window_width / original_width
        new_height = self.window_height / original_height
        
        if new_width > new_height:
            self.scale_factor = new_height
        else:
            self.scale_factor = new_width
            
        self.pixmap_item.setPixmap(pixmap)
        self.zoomFit()

    def wheelEvent(self, e: QWheelEvent):
        """鼠标滚轮事件处理"""
        if e.modifiers() & Qt.ControlModifier:
            if e.angleDelta().y() > 0 and self.zoomInTimes < self.maxZoomInTimes:
                self.zoomIn()
            else:
                self.zoomOut()
        else:
            super().wheelEvent(e)

    def zoomFit(self):
        """根据窗口大小自适应缩放"""
        self.scale(self.scale_factor, self.scale_factor)

    def zoomIn(self):
        """放大图像"""
        self.scale(1.25, 1.25)
        self.zoomInTimes += 1

    def zoomOut(self):
        """缩小图像"""
        self.scale(0.8, 0.8)
        if self.zoomInTimes > 0:
            self.zoomInTimes -= 1

    def resetZoom(self):
        """重置缩放比例"""
        self.resetTransform()
        self.zoomInTimes = 0

    def fitToWindow(self):
        """使图像完全适应窗口"""
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        """鼠标按下事件处理"""
        if not self.crop_flag and event.button() == QtCore.Qt.LeftButton:
            self.start_drag_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标移动事件处理"""
        if not self.crop_flag and self.start_drag_pos is not None:
            offset = event.pos() - self.start_drag_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - offset.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - offset.y())
            self.start_drag_pos = event.pos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """鼠标释放事件处理"""
        if not self.crop_flag and event.button() == QtCore.Qt.LeftButton:
            self.start_drag_pos = None
        super().mouseReleaseEvent(event)

    @QtCore.pyqtSlot(QtCore.QRect, QtCore.QPointF, QtCore.QPointF)
    def onRubberBandChanged(self, rubberBandRect, fromScenePoint, toScenePoint):
        """橡皮筋选择区域变化事件处理"""
        if self.crop_flag:
            if rubberBandRect.isNull():
                # 选择完成，执行裁剪
                pixmap = self.pixmap_item.pixmap()
                rect = self.pixmap_item.mapFromScene(self.last_rect).boundingRect().toRect()
                
                if not rect.intersected(pixmap.rect()).isNull():
                    # 执行裁剪
                    crop_pixmap = pixmap.copy(rect)
                    crop_image = crop_pixmap.toImage()
                    
                    # 确保为灰度图
                    gray_crop_image = crop_image.convertToFormat(QtGui.QImage.Format_Grayscale8)
                    
                    # 保存临时裁剪文件
                    temp_save_path = '/home/aoi/pretest/cropped_image.tif'
                    gray_crop_image.save(temp_save_path)
                    
                    # 打开图像编辑器
                    gray_crop_pixmap = QPixmap.fromImage(gray_crop_image)
                    self.crop_bool = True
                    
                    view = ImageEditor()
                    view.open_image(gray_crop_pixmap)
                    self.save_path = view.save_image_path(gray_crop_pixmap, self.image_path)
                    
                    # 显示编辑对话框
                    dialog = QtWidgets.QDialog(self)
                    dialog.resize(600, 600)
                    lay = QtWidgets.QVBoxLayout(dialog)
                    lay.addWidget(view)
                    dialog.exec_()
                    
                    self.save_flag = view.get_save_flag()
                
                self.last_rect = QtCore.QRectF()
            else:
                # 正在选择，记录选择区域
                self.last_rect = QtCore.QRectF(fromScenePoint, toScenePoint)

    def get_image_path(self):
        """获取当前显示的图像路径"""
        return self.image_path if self.image_path is not None else None

    def get_cropped_path(self):
        """获取裁剪后的图像路径"""
        if self.crop_bool:
            self.crop_bool = False
            return self.save_path
        else:
            return None

    def set_crop_flag(self, flag):
        """设置是否启用裁剪模式"""
        self.crop_flag = flag

    def get_crop_flag(self):
        """获取当前是否处于裁剪模式"""
        return self.crop_flag
