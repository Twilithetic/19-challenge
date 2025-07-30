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
    """
    计算摄像头到目标的距离（单位：cm）
    
    参数:
    - pixel_area: 矩形的像素面积
    - real_area_cm2: 矩形的实际面积（cm²）
    - pixel_to_cm: 1像素对应的厘米数（默认0.5cm/像素，即5mm/像素）
    
    返回:
    - 距离（cm）
    """
    # 计算像素面积对应的实际面积（cm²）
    pixel_area_cm2 = pixel_area * (pixel_to_cm ** 2)
    
    # 面积比例 = (实际面积)/(成像面积) = (距离/焦距)^2
    # 使用简化计算，假设物体距离与面积平方根成反比
    distance_cm = np.sqrt(real_area_cm2 / pixel_area_cm2) * np.sqrt(real_area_cm2)
    
    return distance_cm

# 示例使用
if __name__ == "__main__":
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 设置摄像头分辨率（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 已知实际参数（示例：一个边长为15cm的正方形，面积225cm²）
    REAL_AREA_CM2 =470   # 15cm × 15cm = 225cm²
    
    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 找到最大矩形
        rect, pixel_area = find_largest_rectangle(frame)
        
        if rect is not None:
            # 计算距离
            distance = calculate_distance_cm(pixel_area, REAL_AREA_CM2)*10
            
            # 在图像上绘制结果
            cv2.drawContours(frame, [rect], -1, (0, 255, 0), 3)
            
            # 计算矩形的中心点
            M = cv2.moments(rect)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # 在中心点显示距离
                cv2.putText(frame, f"{distance:.1f}cm", (cX - 50, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 在左上角显示信息
            cv2.putText(frame, f"Distance: {distance:.1f}cm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Area: {pixel_area}px", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 显示结果
        cv2.imshow("Distance Measurement", frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()