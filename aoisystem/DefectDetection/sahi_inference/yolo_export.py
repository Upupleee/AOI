import os
from ultralytics import YOLO

# 设置要检索的文件夹路径
folder_path = '/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1207_pm'

# 获取所有.pt文件
pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

# 遍历所有权重文件并转换为TensorRT engine
for pt_file in pt_files:
    pt_file_path = os.path.join(folder_path, pt_file)
    print(f'Converting {pt_file_path} to TensorRT...')
    
    # 加载模型
    model = YOLO(pt_file_path)

    # 转换到TRT模型
    model.export(format='engine', batch=1, imgsz=640)

    print(f'Converted {pt_file_path} to TensorRT successfully.')

print("Converted all weights successfully!")
