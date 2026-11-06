from PIL import Image, ImageFile
import cv2
import numpy as np

# 解除 Pillow 的像素限制
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image_gray(image_path):
    """
    使用 Pillow 加载图像，并转换为 NumPy 数组
    """
    img = Image.open(image_path).convert('L')
    return np.array(img)



def load_color_image(image_path):
    """
    使用Pillow加载图像并转换为RGB（再转为BGR以兼容OpenCV），然后转换为numpy array
    """
    with Image.open(image_path) as img:
        rgb_img = img.convert('RGB')
        rgb_array = np.array(rgb_img)
        # OpenCV 默认使用BGR，所以这里转换一下
        bgr_img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        print(rgb_img.size)
        return bgr_img




def display_interactive(image, var_name=""):
    """
    改进后的交互式图像可视化：
      - 固定显示窗口尺寸（1024×768）。
      - 维护一个原图中的 ROI，初始为全图。
      - 支持两种模式：
         1. 区域选取模式（默认）：左键拖拽选取区域，放大后更新 ROI。
         2. 拖拽平移模式：按下 "p" 键切换到平移模式，左键拖拽可平移当前 ROI。
      - 鼠标滚轮缩放时以鼠标所在位置为锚点进行缩放。
      - 右键重置 ROI 为全图。
    """
    # 固定显示窗口尺寸
    win_w, win_h = 1920, 1080
    orig_h, orig_w = image.shape[:2]
    # 初始 ROI 为全图，[x, y, width, height]
    current_roi = [0, 0, orig_w, orig_h]
    window_name = f"Interactive Viewer - {var_name}" if var_name else "Interactive Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 当前模式：'select' 或 'pan'
    current_mode = "select"
    
    # 状态变量：用于左键拖拽
    dragging = False
    drag_start = (0, 0)         # 用于平移模式，记录窗口坐标起点
    pan_origin = current_roi[:]  # 平移模式下，记录拖拽开始时的 ROI
    select_start = (0, 0)       # 区域选取模式下，记录起点（窗口坐标）
    select_rect = (0, 0, 0, 0)    # 当前选取矩形（窗口坐标）

    def clamp_roi(roi):
        """
        将 ROI 限制在原图范围内，并保证宽度和高度至少为 1 像素
        roi: [x, y, width, height]
        """
        x, y, w, h = roi
        x = max(0, x)
        y = max(0, y)
        if x >= orig_w:
            x = orig_w - 1
        if y >= orig_h:
            y = orig_h - 1
        if x + w > orig_w:
            w = orig_w - x
        if y + h > orig_h:
            h = orig_h - y
        w = max(1, int(w))
        h = max(1, int(h))
        return [int(x), int(y), w, h]
    
    def get_display_image():
        """
        根据当前 ROI 生成显示图像：
          1. 裁剪出当前 ROI 区域。
          2. 根据窗口尺寸计算缩放因子（保持 ROI 高宽比）。
          3. 将缩放后的图像居中绘制到黑色背景画布上，并在左上角显示当前模式。
        """
        roi_x, roi_y, roi_w, roi_h = current_roi
        crop = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        # 如果裁剪区域为空，则返回一个白色画布
        if crop.size == 0:
            if len(image.shape) == 3:
                return np.full((win_h, win_w, 3), 255, dtype=np.uint8)
            else:
                return np.full((win_h, win_w), 255, dtype=np.uint8)
        
        scale = min(win_w / roi_w, win_h / roi_h)
        new_w = int(roi_w * scale)
        new_h = int(roi_h * scale)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # 创建白色画布
        if len(image.shape) == 3:
            canvas = np.full((win_h, win_w, 3), 255, dtype=np.uint8)
        else:
            canvas = np.full((win_h, win_w), 255, dtype=np.uint8)
        offset_x = (win_w - new_w) // 2
        offset_y = (win_h - new_h) // 2
        canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
        # 在左上角显示当前模式（绿色字体）
        cv2.putText(canvas, f"Mode: {current_mode}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
        return canvas

    def window_to_image(x, y):
        """
        将显示窗口中的坐标 (x, y) 映射回原图坐标，
        依据当前 ROI 与显示窗口尺寸的缩放和居中偏移计算。
        """
        roi_x, roi_y, roi_w, roi_h = current_roi
        scale = min(win_w / roi_w, win_h / roi_h)
        new_w = int(roi_w * scale)
        new_h = int(roi_h * scale)
        offset_x = (win_w - new_w) // 2
        offset_y = (win_h - new_h) // 2
        # 限制 x, y 在图像显示区域内
        x_clamped = min(max(x, offset_x), offset_x + new_w)
        y_clamped = min(max(y, offset_y), offset_y + new_h)
        image_x = int(roi_x + (x_clamped - offset_x) / scale)
        image_y = int(roi_y + (y_clamped - offset_y) / scale)
        return image_x, image_y

    def update_display():
        disp_img = get_display_image()
        # 如果处于区域选取状态，绘制选取矩形（窗口坐标）
        if current_mode == "select" and dragging:
            x1, y1, x2, y2 = select_rect
            cv2.rectangle(disp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow(window_name, disp_img)

    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging, drag_start, pan_origin, select_start, select_rect, current_roi, current_mode
        # 根据当前模式分别处理
        if current_mode == "pan":
            # 拖拽平移模式：左键拖拽改变 ROI 的位置
            if event == cv2.EVENT_LBUTTONDOWN:
                dragging = True
                drag_start = (x, y)
                pan_origin = current_roi[:]
            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                dx = x - drag_start[0]
                dy = y - drag_start[1]
                # 始终使用 pan_origin 的尺寸，保持放大倍率不变
                roi_x, roi_y, roi_w, roi_h = pan_origin  
                scale = min(win_w / roi_w, win_h / roi_h)
                new_roi = [pan_origin[0] - int(dx / scale),
                        pan_origin[1] - int(dy / scale),
                        roi_w, roi_h]
                current_roi = clamp_roi(new_roi)
                update_display()
            elif event == cv2.EVENT_LBUTTONUP and dragging:
                dragging = False
                update_display()
        else:
            # 区域选取模式：左键拖拽选择放大区域
            if event == cv2.EVENT_LBUTTONDOWN:
                dragging = True
                select_start = (x, y)
                select_rect = (x, y, x, y)
                update_display()
            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                select_rect = (select_start[0], select_start[1], x, y)
                update_display()
            elif event == cv2.EVENT_LBUTTONUP and dragging:
                dragging = False
                # 计算选取矩形，并映射回原图坐标更新 ROI
                x1, y1, x2, y2 = select_rect
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                img_x1, img_y1 = window_to_image(x1, y1)
                img_x2, img_y2 = window_to_image(x2, y2)
                if abs(img_x2 - img_x1) > 10 and abs(img_y2 - img_y1) > 10:
                    current_roi = clamp_roi([min(img_x1, img_x2), min(img_y1, img_y2),
                                               abs(img_x2 - img_x1), abs(img_y2 - img_y1)])
                update_display()
        # 鼠标滚轮缩放：以鼠标所在位置（映射到原图）为锚点缩放
        if event == cv2.EVENT_MOUSEWHEEL:
            # OpenCV中 flags > 0 表示向上滚（缩小 ROI，即放大图像），flags < 0 表示向下滚（放大 ROI，即缩小图像）
            factor = 0.9 if flags > 0 else 1.1
            # 以鼠标在窗口中的位置为锚点
            anchor = window_to_image(x, y)
            roi_x, roi_y, roi_w, roi_h = current_roi
            # 计算锚点在当前 ROI 内的相对位置
            rel_x = (anchor[0] - roi_x) / roi_w
            rel_y = (anchor[1] - roi_y) / roi_h
            new_w = roi_w * factor
            new_h = roi_h * factor
            new_x = anchor[0] - rel_x * new_w
            new_y = anchor[1] - rel_y * new_h
            current_roi = clamp_roi([new_x, new_y, new_w, new_h])
            update_display()
        # 右键重置 ROI 为全图
        if event == cv2.EVENT_RBUTTONDOWN:
            current_roi = [0, 0, orig_w, orig_h]
            update_display()

    cv2.setMouseCallback(window_name, mouse_callback)
    
    update_display()
    
    # 主循环：响应键盘事件切换模式
    while True:
        key = cv2.waitKey(20) & 0xFF
        # 按 'p' 切换平移/区域选取模式
        if key == ord('p'):
            current_mode = "pan" if current_mode == "select" else "select"
            update_display()
        # 按 'q' 或 ESC 退出
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()


# 示例用法
if __name__ == "__main__":
    # 分别加载左右图像（请替换为实际图像路径）
    image_one = load_color_image("/home/aoi/image_stiching/splits_overlay/slice_001.bmp")
    # image_right = load_image("/home/aoi/AOI/TemplateMatching/EfficientLoFTR/object_region_full_res.jpg")
    # image_left = load_color_image("/home/aoi/AOI/TemplateMatching/EfficientLoFTR/corresponding_sample_region_full_res.tiff")
    image_two = load_color_image("/home/aoi/image_stiching/sample_new_2/test_imgs1.tif")

    image_three = load_color_image("/home/aoi/image_stiching/splits_overlay/slice_002.bmp")
    
    image_four = load_color_image("/home/aoi/image_stiching/sample_new_2/test_imgs2.tif")

    image_five = load_color_image("/home/aoi/image_stiching/splits_overlay/slice_005.bmp")

    image_six = load_color_image("/home/aoi/image_stiching/sample_new_2/cropped_sample_image_5.tiff")

    display_interactive(image_one, "image_1")

    display_interactive(image_two, "image_2")

                                                                                                        
    display_interactive(image_three,"image_3")

    display_interactive(image_four,"image_4")

    display_interactive(image_five,"image_5")    

    display_interactive(image_six,"image_6")       


    cv2.waitKey(0)
    cv2.destroyAllWindows()