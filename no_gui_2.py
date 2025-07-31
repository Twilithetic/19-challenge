import cv2
import numpy as np

# -------------------------- 常量定义 --------------------------
# 图像处理参数
CANNY_THRESH_LOW = 50
CANNY_THRESH_HIGH = 150
MIN_CONTOUR_AREA = 500  # 最小轮廓面积（过滤小噪声）
CIRCULARITY_THRESH = 0.6  # 圆形度阈值（越接近1越圆）
EQUILATERAL_TOLERANCE = 0.05  # 等边三角形边长误差容忍度

# A4纸物理参数（厘米）
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
A4_AREA_CM2 = A4_WIDTH_CM * A4_HEIGHT_CM
DISTANCE_SCALE_FACTOR = 1150  # 距离计算缩放因子

# 内矩形（带黑边A4纸）检测参数
INNER_RECT_AREA_RATIO = 0.6  # 内矩形面积至少为外矩形的60%
MAX_EDGE_DISTANCE_RATIO = 0.05  # 内矩形距离图像边缘至少5%
BLACK_BORDER_PX_MIN = 10  # 黑边最小像素宽度（约2cm）


# -------------------------- A4纸检测与距离计算模块 --------------------------
def detect_a4_paper(gray_img):
    """检测A4纸轮廓，返回最佳矩形及位置信息"""
    # 预处理：模糊降噪 + 边缘检测
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)
    
    # 提取轮廓及层次（区分父子轮廓）
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None, None  # 未检测到轮廓
    
    quad_contours = []  # 存储符合条件的四边形
    h, w = gray_img.shape[:2]
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 2000:  # 过滤过小轮廓（小于A4纸1%面积）
            continue
        
        # 多边形近似（筛选四边形）
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) != 4:
            continue
        
        # 筛选宽高比符合A4纸的矩形（21/29.7≈0.707）
        x, y, rect_w, rect_h = cv2.boundingRect(approx)
        aspect_ratio = float(rect_w) / rect_h
        if not (0.6 < aspect_ratio < 0.8):
            continue
        
        # 过滤靠近图像边缘的轮廓（内矩形通常在中间）
        edge_distances = [x, y, w - (x + rect_w), h - (y + rect_h)]
        if min(edge_distances) < MAX_EDGE_DISTANCE_RATIO * min(w, h):
            continue
        
        # 记录轮廓信息（是否为子轮廓、位置等）
        is_child = hierarchy[0][i][3] != -1  # 有父轮廓的是内轮廓
        quad_contours.append({
            "contour": approx,
            "area": area,
            "is_child": is_child,
            "x": x, "y": y, "w": rect_w, "h": rect_h,
            "parent_idx": hierarchy[0][i][3]
        })
    
    if not quad_contours:
        return None, None, None, None, None  # 无符合条件的四边形
    
    # 优先选择内轮廓（带黑边的A4纸内框）
    child_contours = [c for c in quad_contours if c["is_child"]]
    if child_contours:
        # 验证内轮廓与外轮廓的黑边宽度
        valid_inner = []
        for child in child_contours:
            parent_idx = child["parent_idx"]
            if 0 <= parent_idx < len(quad_contours):
                parent = quad_contours[parent_idx]
                # 计算黑边宽度（内框与外框的间距）
                border_width = min(
                    child["x"] - parent["x"],
                    child["y"] - parent["y"],
                    parent["x"] + parent["w"] - (child["x"] + child["w"]),
                    parent["y"] + parent["h"] - (child["y"] + child["h"])
                )
                if border_width >= BLACK_BORDER_PX_MIN:
                    valid_inner.append(child)
        
        # 选择有效内轮廓中面积最大的
        candidate = max(valid_inner, key=lambda x: x["area"]) if valid_inner else max(child_contours, key=lambda x: x["area"])
    else:
        # 无内轮廓时选择最大外轮廓（备用）
        candidate = max(quad_contours, key=lambda x: x["area"])
    
    # 最终过滤：面积需符合比例
    max_area = max([c["area"] for c in quad_contours]) if quad_contours else 0
    if candidate["area"] < INNER_RECT_AREA_RATIO * max_area:
        return None, None, None, None, None
    
    return (candidate["contour"], candidate["x"], candidate["y"], candidate["w"], candidate["h"])


def extract_a4_roi(image):
    """提取A4纸ROI区域，并计算像素到厘米的转换比例"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    best_rect, x, y, w, h = detect_a4_paper(gray)
    
    if best_rect is None:
        return None, None, None, 0.0  # 未检测到A4纸
    
    # 裁剪A4纸区域作为ROI
    roi = image[y:y+h, x:x+w]
    
    # 计算A4纸边长（像素）及像素到厘米的转换比例
    pts = best_rect.reshape(4, 2)
    edge_lengths = [np.linalg.norm(pts[i] - pts[(i+1) % 4]) for i in range(4)]
    avg_side1 = (edge_lengths[0] + edge_lengths[2]) / 2.0  # 对边平均
    avg_side2 = (edge_lengths[1] + edge_lengths[3]) / 2.0
    
    # 匹配A4纸宽高比（确保比例正确）
    a4_ratio = A4_WIDTH_CM / A4_HEIGHT_CM
    side_ratio = min(avg_side1, avg_side2) / max(avg_side1, avg_side2)
    if abs(side_ratio - a4_ratio) < 0.15:
        width_pixel, height_pixel = (avg_side1, avg_side2) if avg_side1 < avg_side2 else (avg_side2, avg_side1)
    else:
        width_pixel, height_pixel = avg_side1, avg_side2
    
    # 计算像素到厘米的转换比例（平均宽高比例）
    if width_pixel > 0 and height_pixel > 0:
        ratio_width = A4_WIDTH_CM / width_pixel
        ratio_height = A4_HEIGHT_CM / height_pixel
        pixel_to_cm_ratio = (ratio_width + ratio_height) / 2.0
    else:
        pixel_to_cm_ratio = 0.0
    
    return roi, (x, y, w, h), best_rect, pixel_to_cm_ratio


def calculate_a4_distance(best_rect):
    """计算镜头到A4纸的距离（厘米）"""
    if best_rect is None:
        return 0.0
    pixel_area = cv2.contourArea(best_rect)
    if pixel_area <= 0:
        return 0.0
    # 距离公式：基于面积比例计算
    return np.sqrt(A4_AREA_CM2 / pixel_area) * DISTANCE_SCALE_FACTOR


# -------------------------- 形状检测模块 --------------------------
def is_equilateral_triangle(approx):
    """判断是否为等边三角形，返回是否等边及平均边长（像素）"""
    if len(approx) != 3:
        return False, 0
    
    p1, p2, p3 = approx[0][0], approx[1][0], approx[2][0]
    d1, d2, d3 = np.linalg.norm(p1-p2), np.linalg.norm(p2-p3), np.linalg.norm(p3-p1)
    if d1 <= 0 or d2 <= 0 or d3 <= 0:
        return False, 0
    
    avg_length = (d1 + d2 + d3) / 3
    max_diff = max(abs(d - avg_length) for d in [d1, d2, d3])
    return (max_diff / avg_length < EQUILATERAL_TOLERANCE), avg_length


def detect_shapes_in_a4(roi, pixel_to_cm_ratio):
    """检测A4纸ROI内的形状（圆、正方形、等边三角形）"""
    if roi is None or roi.size == 0 or pixel_to_cm_ratio <= 0:
        return []
    
    # 预处理：灰度化 + 自适应阈值（突出形状边缘）
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 提取轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_shapes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue  # 过滤小噪声
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        
        # 多边形近似与形状判断
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        circularity = 4 * np.pi * area / (perimeter **2)  # 圆形度（圆=1，方<1）
        
        # 1. 检测圆形（圆形度高且边数多）
        if circularity > CIRCULARITY_THRESH and len(approx) > 5:
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            diameter_cm = 2 * radius * pixel_to_cm_ratio
            detected_shapes.append(('circle', cnt, diameter_cm))
        
        # 2. 检测等边三角形（3条边）
        elif len(approx) == 3:
            is_equi, avg_side_pixel = is_equilateral_triangle(approx)
            if is_equi and avg_side_pixel > 0:
                avg_side_cm = avg_side_pixel * pixel_to_cm_ratio
                detected_shapes.append(('equilateral_triangle', cnt, avg_side_cm))
        
        # 3. 检测正方形（4条边且边长接近）
        elif len(approx) == 4:
            pts = approx.reshape(-1, 2)
            dists = [np.linalg.norm(pts[i] - pts[(i+1) % 4]) for i in range(4)]
            if all(d > 0 for d in dists) and (max(dists) - min(dists) < 0.2 * np.mean(dists)):
                side_cm = np.mean(dists) * pixel_to_cm_ratio
                detected_shapes.append(('square', cnt, side_cm))
    
    return detected_shapes


# -------------------------- 辅助功能模块 --------------------------
def crop_roi_until_shapes(roi, pixel_ratio, max_crops=5):
    """逐步裁剪ROI直到检测到形状，返回裁剪后的ROI、次数和形状"""
    current_roi = roi.copy()
    crop_count = 0
    detected = []
    while crop_count < max_crops:
        h, w = current_roi.shape[:2]
        # 每次裁剪10%边缘
        crop_x, crop_y = int(w * 0.1), int(h * 0.1)
        current_roi = current_roi[crop_y:h-crop_y, crop_x:w-crop_x]
        
        if current_roi.size == 0:  # 裁剪到过小则停止
            break
        
        crop_count += 1
        detected = detect_shapes_in_a4(current_roi, pixel_ratio)
        if detected:  # 检测到形状则停止
            break
    return current_roi, crop_count, detected


def draw_detection_results(image, offset, shapes, distance, crop_count=0):
    """在图像上绘制检测结果（形状、距离、裁剪次数）"""
    x_off, y_off = offset
    
    # 绘制距离和面积
    cv2.putText(image, f"Distance: {distance:.1f}cm", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 绘制形状
    for shape, cnt, size in shapes:
        if size <= 0:
            continue
        cnt += np.array([x_off, y_off])  # 调整形状坐标到原始图像
        
        if shape == 'circle':
            color = (255, 255, 0)
            label = f'Circle: {size:.1f}cm'
            (x, y), r = cv2.minEnclosingCircle(cnt)
            cv2.circle(image, (int(x), int(y)), int(r), color, 2)
            text_pos = (int(x)-50, int(y)-10)
        
        elif shape == 'equilateral_triangle':
            color = (0, 255, 0)
            label = f'Triangle: {size:.1f}cm'
            cv2.drawContours(image, [cnt], -1, color, 2)
            center = np.mean(cnt.reshape(-1, 2), axis=0).astype(int)
            text_pos = (center[0]-70, center[1]-10)
        
        elif shape == 'square':
            color = (255, 0, 255)
            label = f'Square: {size:.1f}cm'
            cv2.drawContours(image, [cnt], -1, color, 2)
            center = np.mean(cnt.reshape(-1, 2), axis=0).astype(int)
            text_pos = (center[0]-50, center[1]-10)
        
        cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 绘制裁剪次数
    if crop_count > 0:
        cv2.putText(image, f"Cropped: {crop_count} times", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def print_shape_details(shapes, distance):
    """在终端打印形状检测结果"""
    if not shapes:
        print("未检测到任何形状")
        return
    print("\n检测到的形状信息:")
    print(f"距离A4纸: {distance:.1f}cm")
    for i, (shape, _, size) in enumerate(shapes):
        if shape == 'circle':
            print(f"圆形 {i+1}: 直径 = {size:.1f}cm")
        elif shape == 'equilateral_triangle':
            print(f"等边三角形 {i+1}: 边长 = {size:.1f}cm")
        elif shape == 'square':
            print(f"正方形 {i+1}: 边长 = {size:.1f}cm")


# -------------------------- 主函数（简洁版） --------------------------
def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 状态变量
    contour_history = []  # 轮廓历史（用于稳定性检测）
    STABLE_FRAMES = 3     # 连续3帧稳定才算有效
    detect_shapes_flag = False  # 形状检测开关
    last_shapes = []
    last_crop_count = 0

    print("操作说明：")
    print("  a键：开启形状检测")
    print("  d键：关闭形状检测")
    print("  q键：退出程序")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 读取失败则退出
        
        # 1. 提取A4纸ROI及相关信息
        roi, rect, best_rect, pixel_ratio = extract_a4_roi(frame)
        # 修复：明确检查best_rect是否为None（而非直接用if best_rect）
        distance = calculate_a4_distance(best_rect) if best_rect is not None else 0.0
        
        # 2. 轮廓稳定性过滤（连续3帧相同轮廓才确认）
        if best_rect is not None:
            contour_history.append(best_rect)
            if len(contour_history) > STABLE_FRAMES:
                contour_history.pop(0)
            if len(contour_history) != STABLE_FRAMES:
                best_rect = None  # 未稳定则不处理
        
        # 3. 处理形状检测（根据开关状态）
        shapes = []
        if roi is not None and pixel_ratio > 0 and best_rect is not None:
            x, y, w, h = rect
            if detect_shapes_flag:
                shapes = detect_shapes_in_a4(roi, pixel_ratio)
                last_shapes = shapes
        
        # 4. 绘制结果
        if rect and best_rect is not None:  # 修复：明确检查best_rect是否为None
            draw_detection_results(frame, (x, y), last_shapes, distance, last_crop_count)
        
        # 显示图像
        cv2.imshow("A4 Detection", frame)
        
        # 5. 处理键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break  # 退出
        elif key == ord('a'):
            # 开启形状检测并强制检测一次
            detect_shapes_flag = True
            if roi is None or pixel_ratio <= 0:
                print("未检测到有效A4纸区域")
                continue
            
            shapes = detect_shapes_in_a4(roi, pixel_ratio)
            if shapes:
                print_shape_details(shapes, distance)
            else:
                print("未检测到形状，尝试裁剪ROI...")
                _, crop_count, shapes = crop_roi_until_shapes(roi, pixel_ratio)
                if shapes:
                    last_crop_count = crop_count
                    print(f"裁剪{crop_count}次后检测到形状：")
                    print_shape_details(shapes, distance)
                else:
                    print(f"裁剪{crop_count}次后仍未检测到形状")
        elif key == ord('d'):
            # 关闭形状检测
            detect_shapes_flag = False
            last_shapes = []
            last_crop_count = 0
            print("已关闭形状检测")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
