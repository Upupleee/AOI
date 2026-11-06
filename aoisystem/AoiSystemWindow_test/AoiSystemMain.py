"""AoiSystemMain 模块（向后兼容）。

原有的大型文件已拆分为以下独立模块：
- main_window.py: MyWindow 主窗口类
- material_import_dialog.py: MaterialImportDialog 料号导入对话框
- widgets.py: ThumbnailDisplayWidget / ImagePreviewWindow 控件
- workers.py: ImageProcessingWorker 图片处理工作线程

本文件仅做重新导出，保持 `from AoiSystemWindow_test.AoiSystemMain import MyWindow`
等原有导入语句继续可用。新代码请直接从对应模块导入。
"""
from .main_window import MyWindow
from .material_import_dialog import MaterialImportDialog
from .widgets import ThumbnailDisplayWidget, ImagePreviewWindow
from .workers import ImageProcessingWorker

__all__ = [
    "MyWindow",
    "MaterialImportDialog",
    "ThumbnailDisplayWidget",
    "ImagePreviewWindow",
    "ImageProcessingWorker",
]
