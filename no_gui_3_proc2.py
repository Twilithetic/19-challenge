import cv2
import numpy as np
import math  # 导入数学模块（用于开平方和π值）
import no_gui_3_proc  # 导入整个模块，而非单独变量
DEBUG6 = 1
DEBUG7 = 1
def get_X(A4_frame, area_cm2):

    if A4_frame is None:
        return None
    
    # 1. 转为灰度图
    gray = cv2.cvtColor(A4_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # 只保留接近纯白的区域
    #blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    #edges = cv2.Canny(thresh, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)
    # 3. 寻找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
    # 3. 筛选最大的轮廓（假设目标是最大的那个）
    max_contour = max(contours, key=cv2.contourArea)
    
    # 4. 多边形近似：提取轮廓的顶点（保留4个顶点，适合矩形）
    epsilon = 0.02 * cv2.arcLength(max_contour, True)  # 控制近似精度
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    approx_vertices = approx.reshape(-1, 2)  # 转换为顶点坐标列表 (n, 2)
    
    # 确保是四边形（4个顶点）
    if len(approx_vertices) != 4:
        print("未检测到四边形轮廓，无法生成内接矩形")
        return A4_frame  #  fallback到原图
    
    # 5. 对4个顶点排序（确保顺序是顺时针或逆时针，方便透视变换）
    # 计算中心点
    center = np.mean(approx_vertices, axis=0)
    # 计算每个点相对于中心点的角度，用于排序
    angles = np.arctan2(approx_vertices[:, 1] - center[1], approx_vertices[:, 0] - center[0])
    # 按角度排序，得到顺时针的4个顶点
    sorted_vertices = approx_vertices[np.argsort(angles)]
    
    # 6. 透视变换：将四边形映射为正矩形（内接区域）
    # 目标矩形的宽高（根据原四边形的宽高比例计算）
    width = int(np.linalg.norm(sorted_vertices[0] - sorted_vertices[1]))  # 第0到1点的距离作为宽
    height = int(np.linalg.norm(sorted_vertices[1] - sorted_vertices[2]))  # 第1到2点的距离作为高
    
    # 目标矩形的4个顶点（正矩形）
    dst_vertices = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # 计算透视变换矩阵
    src_vertices = sorted_vertices.astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
    
    # 应用透视变换，得到内接矩形区域
    inner_rect = cv2.warpPerspective(A4_frame, matrix, (width, height))
    
    # 调试显示结果
    if DEBUG6:
        cv2.imshow("A4 Detection", inner_rect)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    x, type_name = distinguish_contours(inner_rect, area_cm2)
    return x, type_name
    
    
    
    
    
    
    
    
    
    
def distinguish_contours(inner_rect, area_cm2):
    CANNY_THRESH_LOW = 50
    CANNY_THRESH_HIGH = 100
    MIN_CONTOUR_AREA = 500  # 最小轮廓面积（过滤小噪声）
    CIRCULARITY_THRESH = 0.6  # 圆形度阈值（越接近1越圆）
    gray = cv2.cvtColor(inner_rect, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # 保留像素值 > 230 的“接近纯白”区域，设为 255（纯白），其他变黑
    ret, thresh = cv2.threshold(blurred, thresh=150, maxval=255, type=cv2.THRESH_BINARY)  # 只保留接近纯白的区域
    edges = cv2.Canny(thresh, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)  # 边缘检测
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtter_conrours = filter_siimilar(contours)
    contour_cnt = len(filtter_conrours)

    if DEBUG7:# 3. 显示这一帧(全部的轮廓)
        for contour in filtter_conrours:
            print(f"轮廓数量：{contour_cnt}")
            frame7 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 灰度图→三通道BGR
            cv2.drawContours(frame7, [contour], -1, (0, 255, 0), 2)
            cv2.imshow("A4 Detection", frame7)
            # 4. 关键：用waitKey(0)阻塞程序，等待用户按任意键再继续（不刷新画面）
            # 0表示无限等待，直到有按键输入
            cv2.waitKey(0)
            cv2.destroyAllWindows()  # 关闭窗口
    x = None
    type_name = ""
    if(contour_cnt == 1):
        contour = filtter_conrours[0]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        # 多边形近似与形状判断
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        circularity = 4 * np.pi * area / (perimeter **2)  # 圆形度（圆=1，方<1）
        
        # 1. 检测圆形（圆形度高且边数多）
        if circularity > CIRCULARITY_THRESH and len(approx) > 5:
            type_name = "圆"
            (_, _), radius = cv2.minEnclosingCircle(contour)
            x = 2 * math.sqrt(area * area_cm2 / math.pi)

        
        # 2. 检测等边三角形（3条边）
        elif len(approx) == 3:
            is_equi, avg_side_pixel = is_equilateral_triangle(approx)
            if is_equi and avg_side_pixel > 0:
                x = avg_side_cm = avg_side_pixel * math.sqrt(area_cm2)
                type_name = "等边三角形"
        
        # 3. 检测正方形（4条边且边长接近）
        elif len(approx) == 4:
            pts = approx.reshape(-1, 2)
            dists = [np.linalg.norm(pts[i] - pts[(i+1) % 4]) for i in range(4)]
            if all(d > 0 for d in dists) and (max(dists) - min(dists) < 0.2 * np.mean(dists)):
                x = side_cm = np.mean(dists) * math.sqrt(area_cm2)
                type_name = "正方形"
        else:
            x = -1
            type_name = "哟呀正方形"
    if(contour_cnt > 1):
        
        free_square = 0
        for contour in filtter_conrours:# 
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if(len(approx) == 4): 
                free_square += 1
        if(free_square == len(filtter_conrours)):#
            squares_x = [] 
            type_name = "多个分离的正方形(最小x)"
            for contour in filtter_conrours:
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                pts = approx.reshape(-1, 2)
                dists = [np.linalg.norm(pts[i] - pts[(i+1) % 4]) for i in range(4)]
                if all(d > 0 for d in dists) and (max(dists) - min(dists) < 0.2 * np.mean(dists)):
                    squares_x.append(np.mean(dists) * math.sqrt(area_cm2))
            x = min(squares_x)
        else: # 又有游离的，又有重叠的正方形
            pass
            

    return x, type_name
            



def filter_siimilar(contours):
    filtered_contours = []
    for i, contour in enumerate(contours):
        # 检查当前轮廓是否与已保留的轮廓相似
        similar = False
        for filtered in filtered_contours:
            if are_contours_similar(contour, filtered):
                similar = True
                break
        if not similar:
            filtered_contours.append(contour)

    return filtered_contours



def are_contours_similar(contour1, contour2, shape_threshold=0.01, area_threshold=0.01, center_threshold=0.01):

    """判断两个轮廓是否相似"""
    # 比较面积
    area1 = cv2.contourArea(contour1)
    area2 = cv2.contourArea(contour2)
    area_diff = abs(area1 - area2) / max(area1, area2)
    if area_diff > area_threshold:
        return False
    
    # 比较边界框和中心位置
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    
    # 计算中心距离
    cx1, cy1 = x1 + w1//2, y1 + h1//2
    cx2, cy2 = x2 + w2//2, y2 + h2//2
    center_dist = np.hypot(cx1 - cx2, cy1 - cy2)
    max_dim = max(w1, h1, w2, h2)
    if center_dist > max_dim * center_threshold:
        return False
    
    # 比较形状（使用OpenCV的形状匹配）
    shape_dist = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
    if shape_dist > shape_threshold:
        return False
    
    return True



EQUILATERAL_TOLERANCE = 0.05  # 等边三角形边长误差容忍度
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






