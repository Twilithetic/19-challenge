import cv2
import numpy as np

# 定义常量（新增内矩形筛选参数）
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
# 新增：内矩形筛选参数（根据黑边厚度调整）
INNER_RECT_AREA_RATIO = 0.5  # 内矩形面积至少为外矩形的50%（过滤过小轮廓）
MAX_EDGE_DISTANCE_RATIO = 0.1  # 内矩形距离图像边缘至少为图像尺寸的10%（避免贴边外轮廓）

def detect_outer_rectangle(gray_img):
    """识别图像中的内矩形（优先选择内部轮廓）"""
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)
    
    # 关键：使用RETR_TREE获取轮廓层次，区分父子轮廓（外轮廓包含内轮廓）
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None
    
    # 筛选所有四边形轮廓（可能是外轮廓或内轮廓）
    quad_contours = []
    h, w = gray_img.shape[:2]  # 图像尺寸（用于判断边缘距离）
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 2000:
            continue
        
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) != 4:
            continue
        
        # 计算轮廓边界框
        x, y, rect_w, rect_h = cv2.boundingRect(approx)
        aspect_ratio = float(rect_w) / rect_h
        if not (0.5 < aspect_ratio < 2.0):
            continue
        
        # 计算轮廓到图像边缘的距离（内矩形通常远离边缘）
        edge_distances = [x, y, w - (x + rect_w), h - (y + rect_h)]
        min_edge_distance = min(edge_distances)
        if min_edge_distance < MAX_EDGE_DISTANCE_RATIO * min(w, h):
            continue  # 过滤过于靠近边缘的轮廓（可能是外轮廓）
        
        # 记录：轮廓、面积、层次（是否为子轮廓，即内轮廓）
        is_child = hierarchy[0][i][3] != -1  # 有父轮廓的是子轮廓（内轮廓）
        quad_contours.append({
            "contour": approx,
            "area": area,
            "is_child": is_child,
            "x": x, "y": y, "w": rect_w, "h": rect_h
        })
    
    if not quad_contours:
        return None, None, None, None
    
    # 优先选择内轮廓（子轮廓）
    child_contours = [c for c in quad_contours if c["is_child"]]
    if child_contours:
        # 从内轮廓中选面积最大且符合A4比例的
        candidate = max(child_contours, key=lambda x: x["area"])
    else:
        # 无内轮廓时，选最可能的外轮廓（备用）
        candidate = max(quad_contours, key=lambda x: x["area"])
    
    # 再次过滤：确保内轮廓面积不过小（避免黑边内部的噪声轮廓）
    all_areas = [c["area"] for c in quad_contours]
    max_area = max(all_areas) if all_areas else 0
    if candidate["area"] < INNER_RECT_AREA_RATIO * max_area:
        return None, None, None, None
    
    return (candidate["contour"], candidate["x"], candidate["y"], 
            candidate["w"], candidate["h"])

# 以下函数与原代码一致，省略重复部分（extract_rectangle_roi、calculate_distance_cm等）
# 【注意：需保留原代码中这些函数的实现】

def extract_rectangle_roi(image):
    """提取矩形ROI并计算像素到厘米的转换比例（不受旋转影响）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    best_rect, x, y, w, h = detect_outer_rectangle(gray)  # 调用优化后的检测函数
    
    if best_rect is None:
        return None, None, None, 0.0
    
    roi = image[y:y+h, x:x+w]
    
    # 计算A4纸四边形的实际边长（像素）
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

# 【此处省略原代码中其他函数（与问题无关，保持不变）】

def calculate_distance_cm(pixel_area, real_area_cm2):
    """计算距离（单位：cm）"""
    if pixel_area <= 0:
        return 0.0
    return np.sqrt(real_area_cm2 / pixel_area) * DISTANCE_SCALE_FACTOR

def is_equilateral_triangle(approx):
    """判断三角形是否为等边三角形"""
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
    """在ROI中检测形状（仅保留圆形、正方形、等边三角形）"""
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

        # 圆形检测
        if circularity > CIRCULARITY_THRESH and len(approx) > 5:
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            diameter_cm = 2 * radius * pixel_to_cm_ratio  # 用准确比例计算
            detected.append(('circle', cnt, diameter_cm))
        
        # 仅保留等边三角形检测（去掉普通三角形）
        elif len(approx) == 3:
            is_equilateral, avg_side_pixel = is_equilateral_triangle(approx)
            if is_equilateral and avg_side_pixel > 0:  # 只添加等边三角形
                avg_side_cm = avg_side_pixel * pixel_to_cm_ratio
                detected.append(('equilateral_triangle', cnt, avg_side_cm))
        
        # 正方形检测
        elif len(approx) == 4:
            pts = approx.reshape(-1, 2)
            dists = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
            
            if any(d <= 0 for d in dists):
                continue
                
            if max(dists) - min(dists) < 0.2 * np.mean(dists):
                side_length_cm = np.mean(dists) * pixel_to_cm_ratio  # 用准确比例计算
                detected.append(('square', cnt, side_length_cm))
    
    return detected

def draw_detected_shapes(image, offset, shapes):
    """绘制检测到的形状（仅处理圆形、正方形、等边三角形）"""
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
    """打印形状信息（仅包含圆形、正方形、等边三角形）"""
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
    """逐步裁剪ROI区域并尝试检测形状"""
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
    # 新增：轮廓跟踪（连续3帧相同轮廓才确认，减少跳变）
    contour_history = []
    STABLE_FRAMES = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        roi, rect, approx, pixel_ratio = extract_rectangle_roi(frame)
        shapes = []
        distance = 0.0
        crop_count = 0
        
        # 轮廓稳定性过滤
        if approx is not None:
            contour_history.append(approx)
            if len(contour_history) > STABLE_FRAMES:
                contour_history.pop(0)
            # 连续STABLE_FRAMES帧都检测到轮廓才确认
            if len(contour_history) == STABLE_FRAMES:
                pass  # 已稳定，使用当前轮廓
            else:
                approx = None  # 未稳定，不更新
        
        if roi is not None and pixel_ratio > 0 and approx is not None:
            x, y, w, h = rect
            pixel_area = cv2.contourArea(approx)
            distance = calculate_distance_cm(pixel_area, A4_AREA_CM2)
            shapes = detect_shapes_in_roi(roi, pixel_ratio)
            draw_detected_shapes(frame, (x, y), shapes)
            
            last_shapes = shapes
            last_distance = distance
            last_crop_count = crop_count
            
            cv2.putText(frame, f"Distance: {distance:.1f}cm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Area: {pixel_area:.0f}px", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if last_crop_count > 0:
                cv2.putText(frame, f"Cropped: {last_crop_count} times", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Distance and Shape Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            if roi is None or pixel_ratio <= 0:
                print("未检测到有效A4纸区域")
                continue
                
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
    
    cap.release()
    cv2.destroyAllWindows()