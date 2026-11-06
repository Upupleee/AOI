
# from DefectDetection.batchinference.vrs_generation import convert_coco_to_vrs_with_varing_area,convert_coco_to_vrs_huashengxin
from pathlib import Path
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

####Founction Definition
def convert_coco_to_vrs_with_varing_area(defects, output_path, image_height,scan_area_input):
    """
    将COCO格式检测结果直接转换为VRS格式
    
    Args:
        defects: COCO格式的检测结果列表
        output_path: 输出文件路径，如果为None则返回VRS内容字符串
        image_height: 图像高度，用于坐标系转换
    
    Returns:
        如果output_path为None，返回VRS内容字符串；否则返回None
    """
    # === 用户配置区域 ===
    scale = 0.333  # AOI像素 → 实际坐标
    offset_x = 0  
    offset_y = 0
    type_mapping = {"green": 0, "blue": 1}
    
    # 原始输入区域 
    # scan_area_input         = [0, 0, 67360, 50703]
    #UPPER_LEFT_POINT
    target1_input           = [0, 0]
    #BOTTOM_DOWN_POINT JUST CHANGE THE TWO POINTS TO SEND TO VRS
    target2_input           = [scan_area_input[2], scan_area_input[3]]
    align_input             = [0, 0]
    margin_input            = [0, 0]
    calib_coordinates_input = [0, 0]
    shift_input             = [0, 0]
    target1_online_input    = [0, 0]
    target2_online_input    = [0, 0]
    align_shift_input       = [0, 0]
    
    # === 坐标变换函数（左下角原点 + 缩放） ===
    def transform_xy(x, y):
        y_flipped = image_height - y
        return int(x * scale), int(y_flipped * scale)
    
    def transform_list(coords):
        return transform_xy(coords[0], coords[1])
    
    # === 计算缺陷中心位置 ===
    scaled_defects = []
    for d in defects:
        x, y, w, h = d["bbox"]
        cx = x + w / 2 + shift_input[0]
        cy = y + h / 2 + shift_input[1]
        x_out, y_out = transform_xy(cx, cy)
        
        # 获取类别名称，如果不存在就使用category_id转换
        if "category_name" in d:
            category_name = d["category_name"]
        elif "category_id" in d:
            # 这里需要根据您的类别ID映射关系进行调整
            category_name = "green" if d["category_id"] == 0 else "blue"
        else:
            category_name = "green"  # 默认类别
            
        defect_type = type_mapping.get(category_name, -1)
        scaled_defects.append((x_out + offset_x, y_out + offset_y, defect_type))
    
    # === 排序逻辑：先按类别（0,1），再按y坐标升序排序 ===
    scaled_defects.sort(key=lambda item: (item[2], item[1]))
    
    # === 区域参数缩放 ===
    scan_area      = [int(v * scale) for v in scan_area_input]
    target1        = transform_list(target1_input)
    target2        = transform_list(target2_input)
    align          = [int(v * scale) for v in align_input]
    margin         = [int(v * scale) for v in margin_input]
    calib_coords   = [int(v * scale) for v in calib_coordinates_input]
    shift          = [int(v * scale) for v in shift_input]
    target1_online = transform_list(target1_online_input)
    target2_online = transform_list(target2_online_input)
    align_shift    = [int(v * scale) for v in align_shift_input]
    
    # === 紧凑头部 ===
    header = "\n".join([
        str(len(scaled_defects)),
        "0.0",
        "1 0 0 0 0",
        "0",
        f"{scan_area[0]} {scan_area[1]} {scan_area[2]} {scan_area[3]}",
        f"{target1[0]} {target1[1]}",
        f"{target2[0]} {target2[1]}",
        "0 0",
        "1 0 0 1 0 0"
    ] + [
        f"{x} {y} {t}" for x, y, t in scaled_defects
    ])
    
    # === general 区块 ===
    general = f"""[general]
    version=0.0
    status=0
    created_by=DSV
    number_of_defects={len(scaled_defects)}
    defects_to_pcbs = 0
    transformation=1 0 0 1 0 0
    scan_area={scan_area[0]} {scan_area[1]} {scan_area[2]} {scan_area[3]}
    align={align[0]} {align[1]}
    margin={margin[0]} {margin[1]}
    calib_coordinates={calib_coords[0]} {calib_coords[1]}
    shift={shift[0]} {shift[1]}
    target1={target1[0]} {target1[1]}
    target1_online={target1_online[0]} {target1_online[1]}
    target2={target2[0]} {target2[1]}
    target2_online={target2_online[0]} {target2_online[1]}
    align_shift={align_shift[0]} {align_shift[1]}
    """
    
    # === defects 区块 ===
    defects_block = "[defects]\n" + "\n".join(
        f"{x} {y} 0 {t}" for x, y, t in scaled_defects
    )
    
    # === 输出组合 ===
    vrs_content = header + "\n\n" + general + "\n" + defects_block
    
    # 如果指定了输出路径，则保存到文件
    if output_path:
        with open(output_path, "w") as f:
            f.write(vrs_content)
        return None
    else:
        return vrs_content



def convert_coco_to_vrs_huashengxin(defects, output_path, image_height,scan_area_input):
    """
    将COCO格式检测结果直接转换为VRS格式

    Args:
        defects: COCO格式的检测结果列表
        output_path: 输出文件路径，如果为None则返回VRS内容字符串
        image_height: 图像高度，用于坐标系转换

    Returns:
        如果output_path为None，返回VRS内容字符串；否则返回None
    """
    # === 用户配置区域 ===
    scale = 0.333  # AOI像素 → 实际坐标
    offset_x = 0  
    offset_y = 0
    type_mapping = {"green": 0, "blue": 1}

    # 原始输入区域 
    # scan_area_input         = [0, 0, 67360, 50703]
    #UPPER_LEFT_POINT
    target1_input           = [0, 0]
    #BOTTOM_DOWN_POINT JUST CHANGE THE TWO POINTS TO SEND TO VRS
    target2_input           = [scan_area_input[2], scan_area_input[3]]
    align_input             = [0, 0]
    margin_input            = [0, 0]
    calib_coordinates_input = [0, 0]
    shift_input             = [0, 0]
    target1_online_input    = [0, 0]
    target2_online_input    = [0, 0]
    align_shift_input       = [0, 0]

    # === 坐标变换函数（左下角原点 + 缩放） ===
    def transform_xy(x, y):
        y_flipped = image_height - y
        return int(x * scale), int(y_flipped * scale)

    def transform_list(coords):
        return transform_xy(coords[0], coords[1])

    # === 计算缺陷中心位置 ===
    scaled_defects = []
    for d in defects:
        x, y, w, h = d["bbox"]
        cx = x + w / 2 + shift_input[0]
        cy = y + h / 2 + shift_input[1]
        x_out, y_out = transform_xy(cx, cy)

        # 获取类别名称，如果不存在就使用category_id转换
        if "category_name" in d:
            category_name = d["category_name"]
        elif "category_id" in d:
            # 这里需要根据您的类别ID映射关系进行调整
            category_name = "green" if d["category_id"] == 0 else "blue"
        else:
            category_name = "green"  # 默认类别

        defect_type = type_mapping.get(category_name, -1)
        scaled_defects.append((x_out + offset_x, y_out + offset_y, defect_type))

    # === 排序逻辑：先按类别（0,1），再按y坐标升序排序 ===
    scaled_defects.sort(key=lambda item: (item[2], item[1]))

    # === 区域参数缩放 ===
    scan_area      = [int(v * scale) for v in scan_area_input]
    target1        = transform_list(target1_input)
    target2        = transform_list(target2_input)
    align          = [int(v * scale) for v in align_input]
    margin         = [int(v * scale) for v in margin_input]
    calib_coords   = [int(v * scale) for v in calib_coordinates_input]
    shift          = [int(v * scale) for v in shift_input]
    target1_online = transform_list(target1_online_input)
    target2_online = transform_list(target2_online_input)
    align_shift    = [int(v * scale) for v in align_shift_input]

    # === 紧凑头部 ===
    # header = "\n".join([
    #     str(len(scaled_defects)),
    #     "0.0",
    #     "1 0 0 0 0",
    #     "0",
    #     f"{scan_area[0]} {scan_area[1]} {scan_area[2]} {scan_area[3]}",
    #     f"{target1[0]} {target1[1]}",
    #     f"{target2[0]} {target2[1]}",
    #     "0 0",
    #     "1 0 0 1 0 0"
    # ] + [
    #     f"{x} {y} {t}" for x, y, t in scaled_defects
    # ])

    header = f"""{str(len(scaled_defects))}0.01 0 0 0 00{scan_area[0]} {scan_area[1]} {scan_area[2]} {scan_area[3]}{target1[0]} {target1[1]}{target2[0]} {target2[1]}0 01 0 0 1 0 0"""+"".join(f"{x} {y} {t}" for x, y, t in scaled_defects)

    # === general 区块 ===
    general = f"""[general]version=0.0status=0created_by=DSVnumber_of_defects={len(scaled_defects)}defects_to_pcbs = 0transformation=1 0 0 1 0 0scan_area={scan_area[0]} {scan_area[1]} {scan_area[2]} {scan_area[3]}align={align[0]} {align[1]}margin={margin[0]} {margin[1]}calib_coordinates={calib_coords[0]} {calib_coords[1]}shift={shift[0]} {shift[1]}target1={target1[0]} {target1[1]}target1_online={target1_online[0]} {target1_online[1]}target2={target2[0]} {target2[1]}target2_online={target2_online[0]} {target2_online[1]}align_shift={align_shift[0]} {align_shift[1]}"""

    # === defects 区块 ===
    defects_block = "[defects]" + "".join(
        f"{x} {y} 0 {t}" for x, y, t in scaled_defects
    )

    # === 输出组合 ===
    vrs_content = header + "" + general + "" + defects_block

    # 如果指定了输出路径，则保存到文件
    if output_path:
        with open(output_path, "w") as f:
            f.write(vrs_content)
        return None
    else:
        return vrs_content





# ========= 配置路径（请修改） =========
IMAGE_PATH = "/home/aoi/pretest/AOI_V1/hsx_experiment/Ver_gerber/p2830p4h018545a0/gtl.bmp"          # 单张大图（bmp/jpg/png/tif 等 PIL 可读格式）
COCO_JSON  = "/home/aoi/pretest/AOI_V1/prediction_visual/gtl/detection_20250813_193404.json"          # 只针对这张图的检测结果
OUT_DIR    = "/home/aoi/pretest/AOI_V1/results/vrs_compare"              # 输出目录
# ===================================


def load_image_size(image_path: str):
    with Image.open(image_path) as im:
        w, h = im.size
    return int(w), int(h)

def read_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_defects(coco_obj):
    """
    将 COCO 内容统一成列表[{"bbox":[x,y,w,h], "category_id":..., "category_name":...}, ...]
    由于只有一张图：若是标准 COCO(dict)，直接取其全部 annotations；
    若是检测结果列表(list)，直接全部使用。
    """
    if isinstance(coco_obj, dict):
        anns = coco_obj.get("annotations", [])
        cats = coco_obj.get("categories", [])
        id2name = {c.get("id"): c.get("name") for c in cats if "id" in c and "name" in c}

        defects = []
        for a in anns:
            bbox = a.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            cid = a.get("category_id")
            cname = id2name.get(cid)
            defects.append({
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "category_id": cid,
                "category_name": cname
            })
        return defects

    if isinstance(coco_obj, list):
        defects = []
        for d in coco_obj:
            bbox = d.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            cid = d.get("category_id")
            cname = d.get("category_name")  # 若上游没带，这里可能是 None
            defects.append({
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "category_id": cid,
                "category_name": cname
            })
        return defects

    raise ValueError("Unsupported COCO format: expected dict or list.")

def main():
    # 1) 读取图像尺寸
    img_w, img_h = load_image_size(IMAGE_PATH)
    image_height = img_h
    scan_area_input = [0, 0, img_w, img_h]

    # 2) 读取 COCO（仅一张图，直接全取）
    coco_obj = read_json(COCO_JSON)
    defects = normalize_defects(coco_obj)
    if not defects:
        print("⚠️ COCO 中未找到有效缺陷（bbox）。")
        return

    # 3) 输出路径
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(IMAGE_PATH).stem
    vrs_var_path = out_dir / f"{stem}_varing_area.vrs"
    vrs_hsx_path = out_dir / f"{stem}_huashengxin.vrs"

    # 4) 生成两个 VRS
    try:
        convert_coco_to_vrs_with_varing_area(
            defects=defects,
            output_path=str(vrs_var_path),
            image_height=image_height,
            scan_area_input=scan_area_input
        )
        print(f"✅ 生成（varing_area）：{vrs_var_path}")
    except Exception as e:
        print(f"❌ convert_coco_to_vrs_with_varing_area 失败：{e}")

    try:
        convert_coco_to_vrs_huashengxin(
            defects=defects,
            output_path=str(vrs_hsx_path),
            image_height=image_height,
            scan_area_input=scan_area_input
        )
        print(f"✅ 生成（huashengxin）：{vrs_hsx_path}")
    except Exception as e:
        print(f"❌ convert_coco_to_vrs_huashengxin 失败：{e}")

    # 5) 基本信息
    print("\n—— 基本信息 ——")
    print(f"图片尺寸: {img_w} x {img_h}")
    print(f"缺陷数量: {len(defects)}")
    by_cat = {}
    for d in defects:
        key = d.get("category_name") or d.get("category_id")
        by_cat[key] = by_cat.get(key, 0) + 1
    print("类别统计：", by_cat)

if __name__ == "__main__":
    main()
