import cv2
import numpy as np
from no_gui_3_proc4 import print_image


DEBUG = 1
DEBUG2 = 1
DEBUG3 = 1
DEBUG4 = 0
DEBUG5 = 0

def get_distance(frame):

    CANNY_THRESH_LOW = 50
    CANNY_THRESH_HIGH = 100
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #转换成灰度
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)
    # 提取轮廓及层次（区分父子轮廓）
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if DEBUG:
        # 3. ???????????
        frame1 = frame.copy()
        cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
        print_image(frame1)

    in_out_rect_contours, in_out_hierarchy = filter_contours(frame, contours, hierarchy)
    inner_contour = None
    outer_contour = None

        # 检查是否有有效轮廓
    if not in_out_rect_contours:
        print("filter_contours未返回任何轮廓，无法提取内/外轮廓")
        return -1, -1, -1

    for i, contour in enumerate(in_out_rect_contours):
        # 外轮廓：没有父轮廓（hierarchy[3] == -1）
        if in_out_hierarchy[i][3] == -1:
            outer_contour = contour
        else:
            # 内轮廓：有父轮廓（hierarchy[3] != -1）
            inner_contour = contour
    
    if inner_contour is None:
        print("inner_contour为空，重试")
        return -1, -1, -1

   

    if DEBUG4:# 3. 显示这一帧(全部的轮廓)
        frame4 = frame.copy()  # 每次用原图复制，避免叠加之前的绘制
        cv2.drawContours(frame4, [inner_contour], 0, (0, 255, 0), 2)
        cv2.putText(frame4, f" should be one inner", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("A4 Detection", frame4)
        # 4. 关键：用waitKey(0)阻塞程序，等待用户按任意键再继续（不刷新画面）
        # 0表示无限等待，直到有按键输入
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # 关闭窗口
    distance = calculate_a4_distance(inner_contour)
    area_cm2 = update_pixel_area_to_cm2(outer_contour)
    A4_frame = cut_ROI_from_frame(frame, inner_contour)
    if DEBUG5:# 3. 显示这一帧(全部的轮廓)
        print(f"area_cm2:{area_cm2}")
        cv2.imshow("A4 Detection", A4_frame)
        # 4. 关键：用waitKey(0)阻塞程序，等待用户按任意键再继续（不刷新画面）
        # 0表示无限等待，直到有按键输入
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # 关闭窗口

    return distance, A4_frame, area_cm2
    print(f"距离(D): {distance}")

def filter_contours(frame, contours, hierarchy):
    MAX_EDGE_DISTANCE_RATIO = 0.5  # 内矩形距离图像边缘至少5%
    candidate_contours = []
    candidate_hierarchy = []  # 保存候选轮廓对应的层级信息
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

        h, w = frame.shape[:2]
        # 过滤靠近图像边缘的轮廓（内矩形通常在中间）
        edge_distances = [x, y, w - (x + rect_w), h - (y + rect_h)]
        if min(edge_distances) < MAX_EDGE_DISTANCE_RATIO * min(w, h):
            continue

        if DEBUG2:# 3. 显示这一帧(全部的轮廓)
                # 绘制距离和面积
            
            frame2 = frame.copy()  # 每次用原图复制，避免叠加之前的绘制
            cv2.drawContours(frame2, [contour], 0, (0, 255, 0), 2)
            cv2.putText(frame2, f"filter1 should be A4 rect only", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("A4 Detection", frame2)
            # 4. 关键：用waitKey(0)阻塞程序，等待用户按任意键再继续（不刷新画面）
            # 0表示无限等待，直到有按键输入
            cv2.waitKey(0)
            cv2.destroyAllWindows()  # 关闭窗口
        
        candidate_contours.append(contour)
        candidate_hierarchy.append(hierarchy[0][i])  # 记录当前轮廓的层级信息
    
     # 过滤相似轮廓
    filtered_contours = []
    filtered_hierarchy = []  # 同步保存过滤后轮廓的层级信息
    for i, contour in enumerate(candidate_contours):
        # 检查当前轮廓是否与已保留的轮廓相似
        similar = False
        for filtered in filtered_contours:
            if are_contours_similar(contour, filtered):
                similar = True
                break
        if not similar:
            filtered_contours.append(contour)
            filtered_hierarchy.append(candidate_hierarchy[i])  # 同步添加层级
    
    for i, contour in enumerate(filtered_contours):
        if DEBUG3:# 3. 显示这一帧(全部的轮廓)
            frame3 = frame.copy()  # 每次用原图复制，避免叠加之前的绘制
            cv2.drawContours(frame3, [contour], 0, (0, 255, 0), 2)
            cv2.putText(frame3, f"A4 rect {1 + i}/2", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("A4 Detection", frame3)
            # 4. 关键：用waitKey(0)阻塞程序，等待用户按任意键再继续（不刷新画面）
            # 0表示无限等待，直到有按键输入
            cv2.waitKey(0)
            cv2.destroyAllWindows()  # 关闭窗口
    return filtered_contours, filtered_hierarchy
    


        
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




# A4纸物理参数（厘米）
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
A4_AREA_CM2 = A4_WIDTH_CM * A4_HEIGHT_CM
DISTANCE_SCALE_FACTOR = 1385  # 距离计算缩放因子
def calculate_a4_distance(contour):
    """计算镜头到A4纸的距离（厘米）"""
    pixel_area = cv2.contourArea(contour)
    # 距离公式：基于面积比例计算
    return np.sqrt(A4_AREA_CM2 / pixel_area) * DISTANCE_SCALE_FACTOR




def cut_ROI_from_frame(frame, inner_contour):
     # 切割inner_contour包围的区域
    A4_frame = None  # 初始化A4_frame
    if inner_contour is not None:
        # 1. 获取inner_contour的边界矩形（x,y为左上角坐标，w,h为宽高）
        x, y, w, h = cv2.boundingRect(inner_contour)
        
        # 2. 确保坐标在图像范围内（避免越界）
        h_frame, w_frame = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        # 3. 切割区域（OpenCV图像是numpy数组，用切片提取）
        A4_frame = frame[y:y+h, x:x+w]  # [y开始:y结束, x开始:x结束]
    return A4_frame

def update_pixel_area_to_cm2(outer_contour):
    try:
        outer_pixel_area = cv2.contourArea(outer_contour)
        if outer_pixel_area <= 0:
            raise ValueError("外轮廓像素面积为0，无法计算比例")
        # A4纸实际面积是 21cm * 29.7cm
        area_cm2 = (21 * 29.7) / outer_pixel_area
        #print(f"更新 area_cm2: {area_cm2} cm²/像素")  # 调试用
    except Exception as e:
        print(f"更新 area_cm2 失败: {e}")
        area_cm2 = None  # 出错时显式设为 None
    return area_cm2