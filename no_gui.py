import cv2
import numpy as np
from single_char_input import get_single_char

# 定义常量
CANNY_THRESH_LOW = 50
CANNY_THRESH_HIGH = 150
CONTOUR_EPSILON_RATIO = 0.02
MIN_CONTOUR_AREA = 500
CIRCULARITY_THRESH = 0.6
EQUILATERAL_TOLERANCE = 0.05
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
A4_AREA_CM2 = A4_WIDTH_CM * A4_HEIGHT_CM
DISTANCE_SCALE_FACTOR = 1150

# 针对2cm黑边的内矩形筛选参数（核心调整）
INNER_RECT_AREA_RATIO = 0.6  # 内矩形面积至少为外矩形的60%（2cm黑边约占20%面积）
MAX_EDGE_DISTANCE_RATIO = 0.05  # 内矩形距离图像边缘至少5%（避免外轮廓）
BLACK_BORDER_PX_MIN = 10  # 黑边最小像素宽度（根据分辨率换算2cm，如1080p下约20px）

def detect_outer_rectangle(gray_img):
    """优先识别有黑边A4纸的内矩形"""
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)
    
    # 获取轮廓及层次结构（关键：RETR_TREE保留父子关系）
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None, None
    
    quad_contours = []
    h, w = gray_img.shape[:2]
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 2000:  # 过滤过小轮廓（小于A4纸1%面积）
            continue
        
        # 近似为多边形，筛选四边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) != 4:
            continue
        
        # 筛选宽高比符合A4纸的矩形
        x, y, rect_w, rect_h = cv2.boundingRect(approx)
        aspect_ratio = float(rect_w) / rect_h
        if not (0.6 < aspect_ratio < 0.8):  # 收紧A4比例范围（21/29.7≈0.707）
            continue
        
        # 计算轮廓到图像边缘的距离（内矩形通常远离边缘）
        edge_distances = [x, y, w - (x + rect_w), h - (y + rect_h)]
        min_edge_distance = min(edge_distances)
        if min_edge_distance < MAX_EDGE_DISTANCE_RATIO * min(w, h):
            continue
        
        # 判断是否为子轮廓（内轮廓一定是外轮廓的子轮廓）
        is_child = hierarchy[0][i][3] != -1  # 有父轮廓的是子轮廓
        quad_contours.append({
            "contour": approx,
            "area": area,
            "is_child": is_child,
            "x": x, "y": y, "w": rect_w, "h": rect_h,
            "parent_idx": hierarchy[0][i][3]  # 记录父轮廓索引
        })
    
    if not quad_contours:
        return None, None, None, None, None
    
    # 步骤1：筛选所有子轮廓（内轮廓候选）
    child_contours = [c for c in quad_contours if c["is_child"]]
    if child_contours:
        # 步骤2：验证子轮廓与父轮廓的间距是否符合2cm黑边
        valid_inner = []
        for child in child_contours:
            # 找到对应的父轮廓（外轮廓）
            parent_idx = child["parent_idx"]
            if parent_idx < 0 or parent_idx >= len(quad_contours):
                continue
            parent = quad_contours[parent_idx]
            
            # 计算父子轮廓的边缘间距（黑边宽度）
            border_width = min(
                child["x"] - parent["x"],  # 左黑边
                child["y"] - parent["y"],  # 上黑边
                parent["x"] + parent["w"] - (child["x"] + child["w"]),  # 右黑边
                parent["y"] + parent["h"] - (child["y"] + child["h"])   # 下黑边
            )
            
            # 黑边宽度需大于最小阈值（2cm对应的像素）
            if border_width >= BLACK_BORDER_PX_MIN:
                valid_inner.append(child)
        
        # 步骤3：从有效内轮廓中选面积最大的
        if valid_inner:
            candidate = max(valid_inner, key=lambda x: x["area"])
        else:
            # 无有效内轮廓时，从子轮廓中选面积最大的（放宽黑边限制）
            candidate = max(child_contours, key=lambda x: x["area"])
    else:
        # 无任何子轮廓时，退而求其次选外轮廓（不推荐，仅备用）
        candidate = max(quad_contours, key=lambda x: x["area"])
    
    # 最终过滤：确保内轮廓面积符合比例
    all_areas = [c["area"] for c in quad_contours]
    max_area = max(all_areas) if all_areas else 0
    if candidate["area"] < INNER_RECT_AREA_RATIO * max_area:
        return None, None, None, None, None
    
    return (candidate["contour"], candidate["x"], candidate["y"], 
            candidate["w"], candidate["h"])

def extract_rectangle_roi(image):
    """提取矩形ROI并计算像素到厘米的转换比例"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 接收5个返回值（修复解包错误）
    best_rect, x, y, w, h = detect_outer_rectangle(gray)
    
    if best_rect is None:
        return None, None, None, 0.0
    
    roi = image[y:y+h, x:x+w]
    
    # 计算A4纸边长及比例
    pts = best_rect.reshape(4, 2)
    edge_lengths = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i+1) % 4]
        edge_lengths.append(np.linalg.norm(p1 - p2))
    
    avg_side1 = (edge_lengths[0] + edge_lengths[2]) / 2.0
    avg_side2 = (edge_lengths[1] + edge_lengths[3]) / 2.0
    
    a4_ratio = A4_WIDTH_CM / A4_HEIGHT_CM
    side_ratio = min(avg_side1, avg_side2) / max(avg_side1, avg_side2)
    
    if abs(side_ratio - a4_ratio) < 0.15:
        if avg_side1 < avg_side2:
            width_pixel, height_pixel = avg_side1, avg_side2
        else:
            width_pixel, height_pixel = avg_side2, avg_side1
    else:
        width_pixel, height_pixel = avg_side1, avg_side2
    
    if width_pixel > 0 and height_pixel > 0:
        ratio_width = A4_WIDTH_CM / width_pixel
        ratio_height = A4_HEIGHT_CM / height_pixel
        pixel_to_cm_ratio = (ratio_width + ratio_height) / 2.0
    else:
        pixel_to_cm_ratio = 0.0
    
    return roi, (x, y, w, h), best_rect, pixel_to_cm_ratio

def calculate_distance_cm(pixel_area, real_area_cm2):
    if pixel_area <= 0:
        return 0.0
    return np.sqrt(real_area_cm2 / pixel_area) * DISTANCE_SCALE_FACTOR

def is_equilateral_triangle(approx):
    if len(approx) != 3:
        return False, 0
    
    p1, p2, p3 = approx[0][0], approx[1][0], approx[2][0]
    d1 = np.linalg.norm(p1 - p2)
    d2 = np.linalg.norm(p2 - p3)
    d3 = np.linalg.norm(p3 - p1)
    
    if d1 <= 0 or d2 <= 0 or d3 <= 0:
        return False, 0
    
    avg_length = (d1 + d2 + d3) / 3
    if avg_length <= 0:
        return False, 0
    
    max_diff = max(abs(d1 - avg_length), abs(d2 - avg_length), abs(d3 - avg_length))
    if max_diff / avg_length < EQUILATERAL_TOLERANCE:
        return True, avg_length
    return False, 0

def detect_shapes_in_roi(roi, pixel_to_cm_ratio):
    if roi is None or roi.size == 0 or pixel_to_cm_ratio <= 0:
        return []
        
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
            
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity > CIRCULARITY_THRESH and len(approx) > 5:
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            diameter_cm = 2 * radius * pixel_to_cm_ratio
            detected.append(('circle', cnt, diameter_cm))
        
        elif len(approx) == 3:
            is_equilateral, avg_side_pixel = is_equilateral_triangle(approx)
            if is_equilateral and avg_side_pixel > 0:
                avg_side_cm = avg_side_pixel * pixel_to_cm_ratio
                detected.append(('equilateral_triangle', cnt, avg_side_cm))
        
        elif len(approx) == 4:
            pts = approx.reshape(-1, 2)
            dists = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
            
            if any(d <= 0 for d in dists):
                continue
                
            if max(dists) - min(dists) < 0.2 * np.mean(dists):
                side_length_cm = np.mean(dists) * pixel_to_cm_ratio
                detected.append(('square', cnt, side_length_cm))
    
    return detected

def draw_detected_shapes(image, offset, shapes):
    x_off, y_off = offset
    for shape, cnt, size in shapes:
        if size <= 0:
            continue
            
        cnt += np.array([x_off, y_off])
        
        if shape == 'circle':
            color = (255, 255, 0)
            label = f'Circle: {size:.1f}cm'
            (x, y), r = cv2.minEnclosingCircle(cnt)
            cv2.circle(image, (int(x), int(y)), int(r), color, 2)
            text_pos = (int(x) - 50, int(y) - 10)
            
        elif shape == 'equilateral_triangle':
            color = (0, 255, 0)
            label = f'Equi Triangle: {size:.1f}cm'
            cv2.drawContours(image, [cnt], -1, color, 2)
            center = np.mean(cnt.reshape(-1, 2), axis=0).astype(int)
            text_pos = (center[0] - 70, center[1] - 10)
            
        elif shape == 'square':
            color = (255, 0, 255)
            label = f'Square: {size:.1f}cm'
            cv2.drawContours(image, [cnt], -1, color, 2)
            center = np.mean(cnt.reshape(-1, 2), axis=0).astype(int)
            text_pos = (center[0] - 50, center[1] - 10)
        
        cv2.putText(image, label, text_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def print_shape_info(shapes, distance):
    if not shapes:
        print("未检测到任何形状")
        return
    
    print("\n检测到的形状信息:")
    print(f"距离: {distance:.1f}cm")
    for i, (shape, _, size) in enumerate(shapes):
        if shape == 'circle':
            print(f"圆形 {i+1}: 直径 = {size:.1f}cm")
        elif shape == 'equilateral_triangle':
            print(f"等边三角形 {i+1}: 边长 = {size:.1f}cm")
        elif shape == 'square':
            print(f"正方形 {i+1}: 边长 = {size:.1f}cm")
    print("")

def crop_and_detect(roi, pixel_to_cm_ratio, crop_percent=0.1, max_crops=5):
    original_roi = roi.copy()
    crop_count = 0
    detected_shapes = []
    
    while crop_count < max_crops:
        h, w = roi.shape[:2]
        crop_x = int(w * crop_percent)
        crop_y = int(h * crop_percent)
        roi = roi[crop_y:h-crop_y, crop_x:w-crop_x]
        
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            break
        
        crop_count += 1
        detected_shapes = detect_shapes_in_roi(roi, pixel_to_cm_ratio)
        if detected_shapes:
            break
    
    return roi, crop_count, detected_shapes

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    last_shapes = []
    last_distance = 0.0
    last_crop_count = 0
    contour_history = []
    STABLE_FRAMES = 3
    # 新增标志位，控制形状检测
    detect_shapes_flag = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        roi, rect, approx, pixel_ratio = extract_rectangle_roi(frame)
        shapes = []
        distance = 0.0
        crop_count = 0
        
        if approx is not None:
            contour_history.append(approx)
            if len(contour_history) > STABLE_FRAMES:
                contour_history.pop(0)
            if len(contour_history) != STABLE_FRAMES:
                approx = None
        
        if roi is not None and pixel_ratio > 0 and approx is not None:
            x, y, w, h = rect
            pixel_area = cv2.contourArea(approx)
            distance = calculate_distance_cm(pixel_area, A4_AREA_CM2)
            
            # 只显示距离和面积信息，不检测形状
            cv2.putText(frame, f"Distance: {distance:.1f}cm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Area: {pixel_area:.0f}px", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 只有当按下'a'键时才检测形状
            if detect_shapes_flag:
                shapes = detect_shapes_in_roi(roi, pixel_ratio)
                if shapes:
                    draw_detected_shapes(frame, (x, y), shapes)
                    last_shapes = shapes
                last_distance = distance
                last_crop_count = crop_count
                
                # 显示裁剪信息（如果有）
                if last_crop_count > 0:
                    cv2.putText(frame, f"Cropped: {last_crop_count} times", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Distance and Shape Detection", frame)
        
        key = ord(get_single_char())
        if key == ord('q'):
            break
        elif key == ord('a'):
            detect_shapes_flag = True  # 设置标志位，开始检测形状
            if roi is None or pixel_ratio <= 0:
                print("未检测到有效A4纸区域")
                continue
                
            shapes = detect_shapes_in_roi(roi, pixel_ratio)
            if shapes:
                print_shape_info(shapes, distance)
            else:
                print("未检测到形状，尝试裁剪ROI...")
                cropped_roi, crop_count, shapes = crop_and_detect(roi, pixel_ratio)
                
                if shapes:
                    crop_x = int(w * 0.1 * crop_count)
                    crop_y = int(h * 0.1 * crop_count)
                    new_x = x + crop_x
                    new_y = y + crop_y
                    
                    draw_detected_shapes(frame, (new_x, new_y), shapes)
                    last_shapes = shapes
                    last_crop_count = crop_count
                    
                    print(f"裁剪{crop_count}次后检测到形状")
                    print_shape_info(shapes, distance)
                    
                    cv2.putText(frame, f"Cropped: {crop_count} times", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Distance and Shape Detection", frame)
                    cv2.waitKey(500)
                else:
                    print(f"裁剪{crop_count}次后仍未检测到形状")
        elif key == ord('d'):
            detect_shapes_flag = False  # 关闭形状检测
    
    cap.release()
    cv2.destroyAllWindows()