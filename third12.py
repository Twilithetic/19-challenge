import cv2
import numpy as np

# ---------------------------- 常量定义 ----------------------------
# 这些常量用于控制检测的灵敏度和精度，可根据实际场景调整

# Canny边缘检测阈值
CANNY_THRESH_LOW = 50  # 低阈值：低于此值的边缘不保留
CANNY_THRESH_HIGH = 150  # 高阈值：高于此值的边缘保留

# 轮廓近似精度（用于多边形拟合）
CONTOUR_EPSILON_RATIO = 0.02  # 以轮廓周长的2%作为近似精度

# 最小轮廓面积（过滤噪点）
MIN_CONTOUR_AREA = 500  # 面积小于此值的轮廓视为噪点

# 圆形检测的圆形度阈值
CIRCULARITY_THRESH = 0.6  # 圆形度公式：4π×面积/(周长²)，越接近1越接近正圆

# 等边三角形边长差异容忍度
EQUILATERAL_TOLERANCE = 0.05  # 边长差异在5%以内视为等边三角形

# A4纸实际尺寸（厘米）
A4_WIDTH_CM = 17.0  # 宽度
A4_HEIGHT_CM = 25.7  # 高度
A4_AREA_CM2 = A4_WIDTH_CM * A4_HEIGHT_CM  # 面积

# 距离计算比例因子（需根据实际设备校准）
DISTANCE_SCALE_FACTOR = 1110

# 带黑边A4纸的检测参数
INNER_RECT_AREA_RATIO = 0.6  # 内矩形面积至少为外矩形的60%
MAX_EDGE_DISTANCE_RATIO = 0.05  # 内矩形距离图像边缘至少5%
BLACK_BORDER_PX_MIN = 10  # 黑边最小像素宽度（约对应2cm）


# ---------------------------- 核心函数定义 ----------------------------

def detect_outer_rectangle(gray_img):
    """检测带黑边A4纸的内矩形（内容区域）"""
    # 高斯模糊去除噪声
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # Canny边缘检测提取边缘
    edges = cv2.Canny(blurred, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)
    # 查找轮廓（RETR_TREE保留层级关系，便于区分内外轮廓）
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:  # 无轮廓时返回空
        return None, None, None, None, None
    
    h, w = gray_img.shape[:2]  # 获取图像尺寸
    quad_contours = []  # 存储符合条件的四边形轮廓
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 2000:  # 过滤过小的轮廓（非A4纸区域）
            continue
        
        # 轮廓周长计算
        perimeter = cv2.arcLength(contour, True)
        # 多边形拟合（近似为直线段）
        epsilon = CONTOUR_EPSILON_RATIO * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) != 4:  # 只保留四边形（A4纸为矩形）
            continue
        
        # 获取边界矩形并计算宽高比
        x, y, rect_w, rect_h = cv2.boundingRect(approx)
        aspect_ratio = float(rect_w) / rect_h
        # A4纸宽高比约0.707，过滤比例不符的四边形
        if not (0.6 < aspect_ratio < 0.8):
            continue
        
        # 计算矩形到图像边缘的距离，过滤边缘过近的矩形
        edge_distances = [x, y, w - (x + rect_w), h - (y + rect_h)]
        if min(edge_distances) < MAX_EDGE_DISTANCE_RATIO * min(w, h):
            continue
        
        # 判断是否为子轮廓（用于区分内外矩形）
        is_child = hierarchy[0][i][3] != -1
        quad_contours.append({
            "contour": approx, "area": area, "is_child": is_child,
            "x": x, "y": y, "w": rect_w, "h": rect_h, "parent_idx": hierarchy[0][i][3]
        })
    
    if not quad_contours:  # 无符合条件的四边形时返回空
        return None, None, None, None, None
    
    # 筛选子轮廓（内矩形）
    child_contours = [c for c in quad_contours if c["is_child"]]
    if child_contours:
        valid_inner = []
        for child in child_contours:
            parent_idx = child["parent_idx"]
            if parent_idx < 0 or parent_idx >= len(quad_contours):
                continue
            parent = quad_contours[parent_idx]
            # 计算黑边宽度（内矩形与外矩形的间距）
            border_width = min(
                child["x"] - parent["x"], child["y"] - parent["y"],
                parent["x"] + parent["w"] - (child["x"] + child["w"]),
                parent["y"] + parent["h"] - (child["y"] + child["h"])
            )
            if border_width >= BLACK_BORDER_PX_MIN:  # 黑边宽度达标
                valid_inner.append(child)
        
        # 选择面积最大的有效内矩形
        candidate = max(valid_inner or child_contours, key=lambda x: x["area"])
    else:
        # 无子轮廓时选择最大面积的四边形
        candidate = max(quad_contours, key=lambda x: x["area"])
    
    # 过滤面积过小的内矩形
    if candidate["area"] < INNER_RECT_AREA_RATIO * max([c["area"] for c in quad_contours]):
        return None, None, None, None, None
    
    return candidate["contour"], candidate["x"], candidate["y"], candidate["w"], candidate["h"]


def extract_rectangle_roi(image):
    """提取A4纸的ROI区域并计算像素到厘米的转换比例"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 检测A4纸内矩形
    best_rect, x, y, w, h = detect_outer_rectangle(gray)
    
    if best_rect is None:  # 未检测到A4纸
        return None, None, None, 0.0
    
    # 提取ROI区域
    roi = image[y:y+h, x:x+w]
    # 提取矩形四个顶点
    pts = best_rect.reshape(4, 2)
    # 计算四条边的长度
    edge_lengths = [np.linalg.norm(pts[i] - pts[(i+1) % 4]) for i in range(4)]
    
    # 计算对边平均长度
    avg_side1 = (edge_lengths[0] + edge_lengths[2]) / 2.0
    avg_side2 = (edge_lengths[1] + edge_lengths[3]) / 2.0
    a4_ratio = A4_WIDTH_CM / A4_HEIGHT_CM  # A4纸实际宽高比
    
    # 匹配A4纸比例，确定像素尺寸对应的实际尺寸
    side_ratio = min(avg_side1, avg_side2) / max(avg_side1, avg_side2)
    if abs(side_ratio - a4_ratio) < 0.15:  # 比例匹配时区分宽和高
        width_pixel, height_pixel = (avg_side1, avg_side2) if avg_side1 < avg_side2 else (avg_side2, avg_side1)
    else:  # 比例不匹配时直接使用平均长度
        width_pixel, height_pixel = avg_side1, avg_side2
    
    # 计算像素到厘米的转换比例（平均宽和高的转换比例）
    if width_pixel > 0 and height_pixel > 0:
        pixel_to_cm_ratio = (A4_WIDTH_CM / width_pixel + A4_HEIGHT_CM / height_pixel) / 2.0
    else:
        pixel_to_cm_ratio = 0.0
    
    return roi, (x, y, w, h), best_rect, pixel_to_cm_ratio


def calculate_distance_cm(pixel_area, real_area_cm2):
    """根据A4纸的像素面积和实际面积计算距离"""
    if pixel_area <= 0:
        return 0.0
    # 距离公式：sqrt(实际面积/像素面积) × 比例因子
    return np.sqrt(real_area_cm2 / pixel_area) * DISTANCE_SCALE_FACTOR


def is_equilateral_triangle(approx):
    """判断轮廓是否为等边三角形"""
    if len(approx) != 3:  # 非三角形直接返回
        return False, 0
    
    # 提取三个顶点坐标
    p1, p2, p3 = approx[0][0], approx[1][0], approx[2][0]
    # 计算三条边的长度
    d1, d2, d3 = np.linalg.norm(p1 - p2), np.linalg.norm(p2 - p3), np.linalg.norm(p3 - p1)
    
    if d1 <= 0 or d2 <= 0 or d3 <= 0:  # 边长为0时无效
        return False, 0
    
    # 计算平均边长和最大偏差
    avg_length = (d1 + d2 + d3) / 3
    max_diff = max(abs(d1 - avg_length), abs(d2 - avg_length), abs(d3 - avg_length))
    # 判断偏差是否在容忍范围内
    is_equilateral = max_diff / avg_length < EQUILATERAL_TOLERANCE
    return is_equilateral, avg_length


def detect_shapes_in_roi(roi, pixel_to_cm_ratio):
    """在ROI区域内检测形状（包括圆形、等边三角形、普通正方形）"""
    if roi is None or roi.size == 0 or pixel_to_cm_ratio <= 0:
        return []

    detected = []  # 存储检测到的形状

    # 转换为灰度图
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 自适应阈值处理（将灰度图转换为二值图，突出形状边缘）
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:  # 过滤小轮廓
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:  # 周长为0时跳过
            continue
        
        # 多边形拟合
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        # 计算圆形度
        circularity = 4 * np.pi * area / (perimeter **2)
        
        # 圆形：圆形度高且边数多（>5）
        if circularity > CIRCULARITY_THRESH and len(approx) > 5:
            (_, _), radius = cv2.minEnclosingCircle(cnt)  # 最小外接圆
            detected.append(('circle', cnt, 2 * radius * pixel_to_cm_ratio))  # 直径（厘米）
        # 等边三角形：3条边且符合边长差异条件
        elif len(approx) == 3:
            is_equilateral, avg_side_pixel = is_equilateral_triangle(approx)
            if is_equilateral:
                detected.append(('equilateral_triangle', cnt, avg_side_pixel * pixel_to_cm_ratio))  # 边长（厘米）
        # 正方形：4条边且边长差异小
        elif len(approx) == 4:
            pts = approx.reshape(-1, 2)
            # 计算四条边的长度
            dists = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
            # 所有边长为正且最大最小边长差异小于平均边长的20%
            if all(d > 0 for d in dists) and max(dists) - min(dists) < 0.2 * np.mean(dists):
                detected.append(('square', cnt, np.mean(dists) * pixel_to_cm_ratio))  # 边长（厘米）
    
    return detected


def draw_detected_shapes(image, offset, shapes):
    """在图像上绘制检测到的形状及标签（英文）"""
    x_off, y_off = offset  # ROI区域在原图中的偏移量
    for shape, cnt, size in shapes:
        if size <= 0:  # 尺寸无效时跳过
            continue
        
        # 调整轮廓坐标到原图坐标系
        cnt += np.array([x_off, y_off])
        
        # 根据形状类型设置颜色和标签
        if shape == 'circle':
            color, label = (255, 255, 0), f'Circle: {size:.1f}cm'
            (x, y), r = cv2.minEnclosingCircle(cnt)  # 绘制外接圆
            cv2.circle(image, (int(x), int(y)), int(r), color, 2)
            text_pos = (int(x) - 50, int(y) - 10)  # 标签位置
        elif shape == 'equilateral_triangle':
            color, label = (0, 255, 0), f'Equi Triangle: {size:.1f}cm'
            cv2.drawContours(image, [cnt], -1, color, 2)  # 绘制三角形轮廓
            center = np.mean(cnt.reshape(-1, 2), axis=0).astype(int)  # 计算中心
            text_pos = (center[0] - 70, center[1] - 10)
        elif shape == 'square':
            color, label = (255, 0, 255), f'Square: {size:.1f}cm'
            cv2.drawContours(image, [cnt], -1, color, 2)  # 绘制正方形轮廓
            center = np.mean(cnt.reshape(-1, 2), axis=0).astype(int)
            text_pos = (center[0] - 50, center[1] - 10)
        
        # 绘制标签文字（英文）
        cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def print_shape_info(shapes, distance):
    """打印检测到的形状信息（控制台输出）"""
    if not shapes:
        print("No shapes detected")
        return
    
    print("\nDetected Shape Information:")
    print(f"Distance: {distance:.1f}cm")
    for i, (shape, _, size) in enumerate(shapes):
        if shape == 'circle':
            print(f"Circle {i+1}: Diameter = {size:.1f}cm")
        elif shape == 'equilateral_triangle':
            print(f"Equilateral Triangle {i+1}: Side Length = {size:.1f}cm")
        elif shape == 'square':
            print(f"Square {i+1}: Side Length = {size:.1f}cm")
    print("")


def crop_and_detect(roi, pixel_to_cm_ratio, crop_percent=0.1, max_crops=5):
    """裁剪ROI边缘并重新检测形状（解决边缘干扰问题）"""
    crop_count = 0  # 裁剪次数
    detected_shapes = []  # 检测到的形状
    
    while crop_count < max_crops:
        h, w = roi.shape[:2]
        # 计算裁剪像素（每次裁剪10%）
        crop_x, crop_y = int(w * crop_percent), int(h * crop_percent)
        # 裁剪ROI（去除边缘）
        roi = roi[crop_y:h-crop_y, crop_x:w-crop_x]
        
        # 裁剪后区域过小则停止
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            break
        
        crop_count += 1
        # 重新检测形状
        detected_shapes = detect_shapes_in_roi(roi, pixel_to_cm_ratio)
        if detected_shapes:  # 检测到形状则停止裁剪
            break
    
    return roi, crop_count, detected_shapes


# ---------------------------- 主程序 ----------------------------
if __name__ == "__main__":
    # 打开摄像头（索引0为默认摄像头，1为外接摄像头）
    cap = cv2.VideoCapture(1)
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 状态变量
    last_shapes, last_distance, last_crop_count = [], 0.0, 0
    contour_history, STABLE_FRAMES = [], 3  # 轮廓历史（用于稳定检测）
    detect_shapes_flag = False  # 形状检测开关

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # 读取失败则退出
            break

        # 提取A4纸ROI及相关参数
        roi, rect, approx, pixel_ratio = extract_rectangle_roi(frame)
        shapes, distance, crop_count = [], 0.0, 0
        
        # 轮廓稳定性判断（连续3帧检测到相同轮廓视为稳定）
        if approx is not None:
            contour_history.append(approx)
            if len(contour_history) > STABLE_FRAMES:
                contour_history.pop(0)
            # 未达到稳定帧数则视为无效
            if len(contour_history) != STABLE_FRAMES:
                approx = None
        
        # 当检测到稳定的A4纸区域时
        if roi is not None and pixel_ratio > 0 and approx is not None:
            x, y, w, h = rect
            # 计算A4纸的像素面积
            pixel_area = cv2.contourArea(approx)
            # 计算距离
            distance = calculate_distance_cm(pixel_area, A4_AREA_CM2)
            
            # 在原图上绘制距离和面积信息（英文）
            cv2.putText(frame, f"Distance: {distance:.1f}cm", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Area: {pixel_area:.0f}px", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # 绘制ROI边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "ROI", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 显示ROI区域
            cv2.imshow("ROI Region", roi)
            
            # 当开启形状检测时
            if detect_shapes_flag:
                shapes = detect_shapes_in_roi(roi, pixel_ratio)
                if shapes:  # 检测到形状则绘制
                    draw_detected_shapes(frame, (x, y), shapes)
                    last_shapes = shapes
                # 更新状态变量
                last_distance = distance
                last_crop_count = crop_count
                
                # 显示裁剪次数（如果有）
                if last_crop_count > 0:
                    cv2.putText(frame, f"Cropped: {last_crop_count} times", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示主窗口
        cv2.imshow("Distance and Shape Detection", frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):  # 按q退出
            break
        elif key == ord('2'):  # 按a开启形状检测
            detect_shapes_flag = True
            if roi is None or pixel_ratio <= 0:
                print("No valid A4 paper region detected")
                continue
                
            # 检测形状
            shapes = detect_shapes_in_roi(roi, pixel_ratio)
            if shapes:
                print_shape_info(shapes, distance)
            else:
                print("No shapes detected, attempting to crop ROI...")
                # 裁剪ROI后重新检测
                cropped_roi, crop_count, shapes = crop_and_detect(roi, pixel_ratio)
                
                if shapes:
                    # 计算裁剪偏移量并绘制形状
                    crop_x, crop_y = int(w * 0.1 * crop_count), int(h * 0.1 * crop_count)
                    draw_detected_shapes(frame, (x + crop_x, y + crop_y), shapes)
                    last_shapes, last_crop_count = shapes, crop_count
                    print(f"Detected shapes after cropping {crop_count} times")
                    print_shape_info(shapes, distance)
                    cv2.putText(frame, f"Cropped: {crop_count} times", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Distance and Shape Detection", frame)
                    cv2.waitKey(500)  # 短暂停留显示结果
                else:
                    print(f"No shapes detected after cropping {crop_count} times")
        elif key == ord('3'):  # 按d关闭形状检测
            detect_shapes_flag = False
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()