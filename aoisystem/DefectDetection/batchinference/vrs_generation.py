#CONVERT_COCO_TO_VRS_FOR_MERGE_RESULT

def convert_coco_to_vrs_for_merged_result(defects, output_path=None, image_height=40985):
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
    scan_area_input         = [0, 0, 67360, 50703]
    #UPPER_LEFT_POINT
    target1_input           = [0, 0]
    #BOTTOM_DOWN_POINT JUST CHANGE THE TWO POINTS TO SEND TO VRS
    target2_input           = [67360, 50703]
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

        #x, y, w, h = d["bbox"]
        #CONVERT_COCO_TO_VRS_FOR_MERGE_RESULT
        str_x, str_y, str_w, str_h = d[0]["bbox"]
        x, y, w, h = int(str_x), int(str_y), int(str_w), int(str_h)

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

def convert_coco_to_vrs(defects, output_path=None, image_height=40985):
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
    scan_area_input         = [0, 0, 67360, 50703]
    #UPPER_LEFT_POINT
    target1_input           = [0, 0]
    #BOTTOM_DOWN_POINT JUST CHANGE THE TWO POINTS TO SEND TO VRS
    target2_input           = [67360, 50703]
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
    target1_input           = list(scan_area_input[0])
    #BOTTOM_DOWN_POINT JUST CHANGE THE TWO POINTS TO SEND TO VRS
    target2_input           = list(scan_area_input[1])
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

# coco=[
#   {
#     "image_id": 0,
#     "category_id": 1,
#     "bbox": [100.5, 50.0, 80.0, 200.0],
#     "score": 0.987
#   },
# {     "image_id": 1,     "category_id": 3,     "bbox": [300.0, 120.0, 150.0, 60.0],     "score": 0.852   }
# ]
# image_height = 50703
# scan_area_input = [0, 0, 67360, 50703]

# vrs_text = convert_coco_to_vrs(coco, output_path='vrs.txt',
#                                image_height=image_height,
#                                scan_area_input=scan_area_input)
