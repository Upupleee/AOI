# export_yolov12_trt.py
from ultralytics import YOLO

# 1. 加载你已经训练好的 YOLOv12 模型
model = YOLO("/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/best_2025_5_26.pt")

# 2. （可选）先导出 ONNX，便于后续调试
# model.export(format="onnx")  # 生成 yolov12.onnx :contentReference[oaicite:0]{index=0}

# 3. 直接导出为 TensorRT engine
#    format="engine" 会生成 yolov12.engine 文件
#    默认使用 FP16；如需 INT8 量化可添加 int8=True 并提供校准数据集
model.export(
    format="engine",    # 指定导出格式为 TensorRT engine
    device="0",         # 指定使用的 GPU 号
    dynamic=True,       # 是否开启动态批次(可根据需要关/开)
    fp16=True,          # 开启半精度加速（RTX 4090 强烈推荐）
    # int8=False,       # 如需 INT8，设为 True 并添加 data="calib_data.yaml"
    # batch=8,          # 如果你想一次导出支持更大 batch
    # workspace=4       # TensorRT 最大 workspace (GB)，可按需调整
)
# 运行后会在当前目录生成 yolov12.engine 文件 :contentReference[oaicite:1]{index=1}
