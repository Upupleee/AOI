
import cv2
from sahi.predict import predict, get_sliced_prediction
from sahi import AutoDetectionModel
import sys

# detector = AutoDetectionModel.from_pretrained(model_type="yolov8", model_path="C:\\Users\\zhang\\Desktop\\Workspace\\runs\\detect\\train2\\weights\\best.pt", device="cuda:0")

def main():

    if len(sys.argv) < 2:
        print("请提供图像路径作为参数。")
        return
    
    image_path = sys.argv[1]
    print(f"处理的图像路径: {image_path}")

    detector = AutoDetectionModel.from_pretrained(model_type="yolov8engine", 
                                                model_path="/home/aoi/AOI/sahi_inference/models/weights1031/best-1031.engine",
                                                device="cuda:0", 
                                                config_path = "/home/aoi/AOI/sahi_inference/tensorrt_cfg/cfg_engine_bicolor_1031.json"
                                                )
    '''
    # Must be passed a config_path to a json with the below fields. 
    # Fields must be obtained from base PyTorch model before quantization.
    # {
    #     "task": "segment", # or 'detect'
    #     "names": [
    #         "dog",
    #         "cat",
    #     ],
    #     "imgsz": [
    #         640,
    #         640
    #     ],
    #     "half": true
    # }
    '''
    # inference on a full image with sahi
    result = predict(
        source = image_path,
        detection_model = detector,
        slice_height = 640,
        slice_width = 640,
        overlap_height_ratio = 0.1,
        overlap_width_ratio = 0.1,
        verbose = 2,
        visual_bbox_thickness = 2,
        visual_hide_labels = False,
        visual_hide_conf= True,
        model_confidence_threshold=0.75,
        visual_text_size=0.5,
        visual_text_thickness=1,
        no_standard_prediction=True,
        project="./prediction_visual",
        name="exp"
    )

if __name__ == '__main__':
    main()
