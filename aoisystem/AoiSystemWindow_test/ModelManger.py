###############################################################################
#GUIDING INDEX
#RYAN   #SEND_TO_VRS_DEBUG_HERE
###############################################################################
import logging
import torch
from copy import deepcopy
from TemplateMatching.EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from TemplateMatching.EfficientLoFTR.lightglue import LightGlue, SuperPoint,ALIKED,DISK
from DefectDetection.sahi.sahi.auto_model import AutoDetectionModel
from common.config import config

logger = logging.getLogger(__name__)


class ModelManager:
    _instance = None
    _model_cache = {}  # 缓存不同类型和参数的模型
    _device = None     # 统一计算设备
    
    # 模型类型常量
    MODEL_TYPE_LOFTR = "loftr"
    MODEL_TYPE_DETECTION = "detection"
    MODEL_TYPE_ALIKED = "aliked"
    MODEL_TYPE_LIGHTGLUE = "lightglue"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 初始化设备（优先使用CUDA）
            cls._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info("ModelManager 初始化设备: %s", cls._device)
        return cls._instance

    def get_model(self, model_type, **kwargs):
        """
        获取指定类型的模型实例
        参数:
            model_type: 模型类型（使用类常量）
            **kwargs: 模型特定参数（如flag、conf_threshold、max_num_keypoints等）
        返回:
            预加载的模型实例
        """
        # 构建唯一缓存键
        cache_key = self._build_cache_key(model_type,** kwargs)
        
        # 缓存未命中时加载模型
        if cache_key not in self._model_cache:
            logger.info("首次加载模型: %s (参数: %s)", model_type, kwargs)
            if model_type == self.MODEL_TYPE_LOFTR:
                self._model_cache[cache_key] = self._load_loftr_model()
            elif model_type == self.MODEL_TYPE_DETECTION:
                self._model_cache[cache_key] = self._load_detection_model(
                    flag=kwargs.get('flag'),
                    conf_threshold=kwargs.get('conf_threshold')
                )
            elif model_type == self.MODEL_TYPE_ALIKED:
                self._model_cache[cache_key] = self._load_aliked_model(
                    max_num_keypoints=kwargs.get('max_num_keypoints', 2048*3)
                )
            elif model_type == self.MODEL_TYPE_LIGHTGLUE:
                self._model_cache[cache_key] = self._load_lightglue_model(
                    features=kwargs.get('features', 'aliked'),
                    depth_confidence=kwargs.get('depth_confidence', -1),
                    width_confidence=kwargs.get('width_confidence', -1)
                )
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
        
        return self._model_cache[cache_key]

    def _build_cache_key(self, model_type, **kwargs):
        """构建唯一缓存键，区分不同参数的模型实例"""
        key_parts = [model_type]
        # 根据模型类型添加关键参数
        if model_type == self.MODEL_TYPE_DETECTION:
            key_parts.extend([str(kwargs.get('flag')), str(kwargs.get('conf_threshold'))])
        elif model_type == self.MODEL_TYPE_ALIKED:
            key_parts.append(str(kwargs.get('max_num_keypoints', 2048*3)))
        elif model_type == self.MODEL_TYPE_LIGHTGLUE:
            key_parts.extend([
                kwargs.get('features', 'aliked'),
                str(kwargs.get('depth_confidence', -1)),
                str(kwargs.get('width_confidence', -1))
            ])
        return "_".join(key_parts)

    def _load_loftr_model(self):

        cfg = deepcopy(full_default_cfg)
        cfg["match_coarse"]["thr"] = 0.015  # 配置粗匹配阈值

        matcher = LoFTR(config=cfg)
        matcher.load_state_dict(
            torch.load(
                config.loftr_weights_path,
                weights_only=False
            )['state_dict']
        )
        matcher = reparameter(matcher)
        matcher = matcher.eval().to(self._device)
        
        return matcher, cfg, self._device

    def _load_detection_model(self, flag, conf_threshold):

        # 检测模型路径映射
        model_paths = {
            0: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/weights1031/best-1031.engine",
            1: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1207/best-30-95um-1207.engine",
            2: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_lowpass/best-lowpass-1206.engine",
            3: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_1/best-008f-1203.engine",
            4: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_2/best-008f-bg1.engine",
            5: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30um-ryan_1/best-30um-1.engine",
            #6: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1207_pm/best-30-95um-1207-2.engine",
            6: "/home/aoi/pretest/AOI_V1/DefectDetection/models/drew_models/aug12_finetunning.pt",

            7: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/35um-08f_3/last_008f_1205.engine",
            8: "/home/aoi/pretest/AOI_V1/DefectDetection/sahi_inference/models/30_95um_1206_lowpass/best-1206-30um.engine",
            9: "/home/aoi/pretest/AOI_V1/DefectDetection/models/drew_models/yolo11s-p2_aug9.pt",
            10: "/home/aoi/Downloads/best-v12n-0408.pt/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0812_0856.pt",
            #11: "/home/aoi/Downloads/best-v12n-0409.pt",
            #11: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0810_1537_aggressive.pt",
            #11: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0819_1010_640by640_aggressive.pt",
            #11: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0820_0922_640by640_aggressive.pt",
            #11: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0820_1439_640by640_12n_cleandata.pt",
            #11: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0821_1253_verify_annotationmethod.pt",
            11: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0827_v12n_784.pt",
            #11: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0811_0935_aggresive.pt",
            #11: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0812_0856.pt",
            #11: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0815_1447_1024by1024_aggressive.pt",   #work
            #11: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_08160300_1024by1024_aggressive.pt",
            #11: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0818_1900_1024by1024_aggressive.pt",
            #12: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0807_1527.pt"
            #12: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0809_1107.pt"
            12: "/home/aoi/pretest/AOI_V1/DefectDetection/models/aoi_0828_11sp2.pt"
        }
        
        model_path = model_paths.get(flag, model_paths[0])
        detector = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=conf_threshold,
            device=str(self._device)  # 使用统一设备
        )
        detector.load_model()
        return detector

    def _load_aliked_model(self, max_num_keypoints):
        extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(self._device)
        return extractor, self._device  # 返回提取器和设备信息

    def _load_lightglue_model(self, features, depth_confidence, width_confidence): 
        matcher = LightGlue(
            features=features,
            depth_confidence=depth_confidence,
            width_confidence=width_confidence
        ).eval().to(self._device)
        return matcher, self._device  # 返回匹配器和设备信息
