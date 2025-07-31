import cv2
import numpy as np


def find_largest_rectangle(image):
    """在图像中寻找最大矩形轮廓"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 寻找最大矩形
    max_area = 0
    largest_rect = None
    
    for cnt in contours:
        # 近似多边形
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 如果是四边形
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                largest_rect = approx
    
    return largest_rect, max_area

def calculate_distance_cm(pixel_area, real_area_cm2, pixel_to_cm=0.5):
    """计算摄像头到目标的距离（单位：cm）"""
    # 计算像素面积对应的实际面积（cm²）
    pixel_area_cm2 = pixel_area * (pixel_to_cm **2)
    
    # 面积比例 = (实际面积)/(成像面积) = (距离/焦距)^2
    # 使用简化计算，假设物体距离与面积平方根成反比
    distance_cm = np.sqrt(real_area_cm2 / pixel_area_cm2) * np.sqrt(real_area_cm2)
    
    return distance_cm
