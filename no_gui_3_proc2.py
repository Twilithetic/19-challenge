import cv2
import numpy as np
DEBUG6 = 1
DEBUG7 = 1
def get_X(A4_frame):

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
    distinguish_contours(inner_rect)
    
    
    
    
    
    
    
    
    
    
def distinguish_contours(inner_rect):
    CANNY_THRESH_LOW = 50
    CANNY_THRESH_HIGH = 100
    gray = cv2.cvtColor(inner_rect, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # 保留像素值 > 230 的“接近纯白”区域，设为 255（纯白），其他变黑
    ret, thresh = cv2.threshold(blurred, thresh=150, maxval=255, type=cv2.THRESH_BINARY)  # 只保留接近纯白的区域
    edges = cv2.Canny(thresh, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)  # 边缘检测
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG7:# 3. 显示这一帧(全部的轮廓)
        frame7 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 灰度图→三通道BGR
        cv2.drawContours(frame7, contours, -1, (0, 255, 0), 2)
        cv2.imshow("A4 Detection", frame7)
        # 4. 关键：用waitKey(0)阻塞程序，等待用户按任意键再继续（不刷新画面）
        # 0表示无限等待，直到有按键输入
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # 关闭窗口


