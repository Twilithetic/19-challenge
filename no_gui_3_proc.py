import cv2
import numpy as np
from no_gui_3_proc4 import print_image


DEBUG = 0
DEBUG2 = 0 #第一次筛选结果
DEBUG3 = 0 #第三次筛选结果
DEBUG4 = 1 #顯示內A4
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
        print("所有輪廓結構")
        # 3. ???????????
        frame1 = frame.copy()
        cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
        print_image(frame1)

    in_out_rect_contours, in_out_hierarchy = filter_contours(frame, contours, hierarchy)
    inner_contour = None
    outer_contour = None


        # ?????????
    if not in_out_rect_contours:
        print("filter_contours没有找到A4之")
        return -1, -1, -1

    # ??????????????????
    # ???[(??, ??), ...]
    contour_with_area = [
        (cv2.contourArea(contour), contour) 
        for contour in in_out_rect_contours
    ]
    # ???????
    contour_with_area.sort(key=lambda x: x[0], reverse=True)

    # ??????????
    if len(contour_with_area) == 1:
        # ??????????????
        inner_contour = contour_with_area[0][1]
        if(DEBUG): print("仅检测到一个轮廓，默认为内轮廓")
    else:  
        # ???????????????????????
        outer_contour = contour_with_area[0][1]  # ????
        inner_contour = contour_with_area[-1][1]  # ????
        if(DEBUG): print(f"检测到{len(contour_with_area)}个轮廓，已按面积分配内外轮廓")

    # ?????????????????
    if inner_contour is None:
        print("inner_contour没被找到")
        return -1, -1, -1
    
    if DEBUG4:# 3. 显示这一帧(全部的轮廓)
        print("内輪廓結果")
        frame4 = frame.copy()  # 每次用原图复制，避免叠加之前的绘制
        cv2.drawContours(frame4, [inner_contour], 0, (0, 255, 0), 2)
        cv2.putText(frame4, f" should be one inner", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print_image(frame4)

   


    distance = calculate_a4_distance(inner_contour)
    if(outer_contour is not None):
        area_cm2 = update_pixel_area_to_cm2(outer_contour, True)
    else:
        area_cm2 = update_pixel_area_to_cm2(inner_contour, False)
    A4_frame = cut_ROI_from_frame(frame, inner_contour)
    if DEBUG5:# 3. 显示这一帧(全部的轮廓)
        print(f"area_cm2:{area_cm2}")
        print_image(A4_frame)

    return distance, A4_frame, area_cm2
    print(f"距离(D): {distance}")

def filter_contours(frame, contours, hierarchy):
    MAX_EDGE_DISTANCE_RATIO = 0.01  # 内矩形距离图像边缘至少5%
    CONTOUR_EPSILON_RATIO = 0.02  # 以轮廓周长的2%作为近似精度
    candidate_contours = []
    candidate_hierarchy = []  # 保存候选轮廓对应的层级信息
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 2000:  # 过滤过小轮廓（小于A4纸1%面积）
            continue
        
               # 轮廓周长计算
        perimeter = cv2.arcLength(contour, True)
        # 多边形拟合（近似为直线段）
        epsilon = CONTOUR_EPSILON_RATIO * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) != 4:  # 只保留四边形（A4纸为矩形）
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


        candidate_contours.append(contour)
        candidate_hierarchy.append(hierarchy[0][i])  # 记录当前轮廓的层级信息
    if DEBUG2:# 第一篩選
            # 绘制距离和面积
        print(f"第一次篩選結果{len(candidate_contours)}")
        frame2 = frame.copy()  # 每次用原图复制，避免叠加之前的绘制
        cv2.drawContours(frame2, candidate_contours, 0, (0, 255, 0), 2)
        cv2.putText(frame2, f"filter1 should be A4 rect only", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print_image(frame2)    
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
    
    # 第三次过滤：保留距离屏幕中心最近的两个轮廓
    if len(filtered_contours) > 2:
        # 获取屏幕中心坐标
        h, w = frame.shape[:2]
        screen_center = (w // 2, h // 2)
        
        # 计算每个轮廓中心到屏幕中心的距离
        contour_distances = []
        for i, contour in enumerate(filtered_contours):
            # 计算轮廓的矩（用于求中心）
            M = cv2.moments(contour)
            if M["m00"] == 0:  # 避免除以零
                continue
            # 轮廓中心坐标
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # 计算欧氏距离（平方，避免开方运算）
            distance = (cX - screen_center[0])**2 + (cY - screen_center[1])** 2
            contour_distances.append((distance, i))  # 存储距离和原始索引
        
        # 按距离排序（升序），取前两个
        contour_distances.sort()
        selected_indices = [idx for (dist, idx) in contour_distances[:2]]
        
        # 更新为筛选后的结果
        filtered_contours = [filtered_contours[i] for i in selected_indices]
        filtered_hierarchy = [filtered_hierarchy[i] for i in selected_indices]
        
        if DEBUG3:
            print(f"第三次筛选：从{len(contour_distances)}个轮廓中保留了2个最近的")
    
    for i, contour in enumerate(filtered_contours):
        if DEBUG3:# 3. 显示这一帧(全部的轮廓)
            print("第三筛选结果")
            frame3 = frame.copy()  # 每次用原图复制，避免叠加之前的绘制
            cv2.drawContours(frame3, filtered_contours, 0, (0, 255, 0), 2)
            cv2.putText(frame3, f"A4 rect {1 + i}/2", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print_image(frame3)
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

def update_pixel_area_to_cm2(outer_contour, is_outter):
    try:
        # 1. 检查轮廓是否为空或无效
        if outer_contour is None:
            raise ValueError("外轮廓为None，无效")
        if len(outer_contour) == 0:
            raise ValueError("外轮廓为空，不包含任何点")
        
        # 2. 检查轮廓格式（必须是三维数组 (N,1,2)）
        if outer_contour.ndim != 3:
            # 尝试修复格式：如果是二维数组 (N,2)，转换为 (N,1,2)
            outer_contour = outer_contour.reshape(-1, 1, 2)
            print("修复轮廓格式：转换为三维数组")
        
        # 3. 检查点数量（至少需要1个点）
        npoints = len(outer_contour)
        if npoints < 1:
            raise ValueError(f"外轮廓点数量不足（{npoints}个点）")
        
        # 4. 检查并转换数据类型（必须是CV_32F或CV_32S）
        if outer_contour.dtype not in (np.int32, np.float32):
            outer_contour = outer_contour.astype(np.int32)  # 转换为32位整数
            print("转换轮廓数据类型为np.int32")
        
        # 5. 计算轮廓面积
        outer_pixel_area = cv2.contourArea(outer_contour)
        if outer_pixel_area <= 0:
            raise ValueError(f"外轮廓像素面积无效（{outer_pixel_area}），必须大于0")
        
        # 6. 计算实际面积比例（A4纸实际尺寸）
        if is_outter:
            # 外轮廓对应A4纸总面积：21cm * 29.7cm
            area_cm2 = (21 * 29.7) / outer_pixel_area
        else:
            # 内轮廓对应A4纸有效区域（假设17cm*25.7cm）
            area_cm2 = (17 * 25.7) / outer_pixel_area
        
        # print(f"更新 area_cm2: {area_cm2} cm²/像素")  # 调试用
        return area_cm2
    
    except Exception as e:
        print(f"更新 area_cm2 失败: {e}")
        return None  # 出错时显式设为 None
