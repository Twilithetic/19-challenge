import cv2
import numpy as np



DEBUG = 1
DEBUG2 = 0
DEBUG3 = 1

def get_distance(frame):
    CANNY_THRESH_LOW = 50
    CANNY_THRESH_HIGH = 150
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #转换成灰度
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)
    # 提取轮廓及层次（区分父子轮廓）
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if DEBUG:# 3. 显示这一帧(全部的轮廓)
        frame1 = frame.copy()  # 用copy()避免修改原图
        cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
        cv2.imshow("A4 Detection", frame1)
        # 4. 关键：用waitKey(0)阻塞程序，等待用户按任意键再继续（不刷新画面）
        # 0表示无限等待，直到有按键输入
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # 关闭窗口
    filter_contours(frame, contours)
    

def filter_contours(frame, contours):
    MAX_EDGE_DISTANCE_RATIO = 0.05  # 内矩形距离图像边缘至少5%
    candidate_contours = None
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
            frame2 = frame.copy()  # 每次用原图复制，避免叠加之前的绘制
            cv2.drawContours(frame2, [contour], 0, (0, 255, 0), 2)
            cv2.imshow("A4 Detection", frame2)
            # 4. 关键：用waitKey(0)阻塞程序，等待用户按任意键再继续（不刷新画面）
            # 0表示无限等待，直到有按键输入
            cv2.waitKey(0)
            cv2.destroyAllWindows()  # 关闭窗口
        
        candidate_contours.append(contour)
    
     # 过滤相似轮廓
    filtered_contours = []
    for contour in candidate_contours:
        # 检查当前轮廓是否与已保留的轮廓相似
        similar = False
        for filtered in filtered_contours:
            if are_contours_similar(contour, filtered):
                similar = True
                break
        if not similar:
            filtered_contours.append(contour)
    for contour in filter_contours:
        if DEBUG3:# 3. 显示这一帧(全部的轮廓)
            frame3 = frame.copy()  # 每次用原图复制，避免叠加之前的绘制
            cv2.drawContours(frame3, [contour], 0, (0, 255, 0), 2)
            cv2.imshow("A4 Detection", frame3)
            # 4. 关键：用waitKey(0)阻塞程序，等待用户按任意键再继续（不刷新画面）
            # 0表示无限等待，直到有按键输入
            cv2.waitKey(0)
            cv2.destroyAllWindows()  # 关闭窗口
    


        
def are_contours_similar(contour1, contour2, shape_threshold=0.1, area_threshold=0.1, center_threshold=0.1):
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