import cv2
import numpy as np
import math

COLORS_LIST = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

DEBUG1 = 0
DEBUG2 = 0

def proc_overlay_square(A4_frame, contour, area_cm2):
    EPSILON_FACTOR = 0.01           # 减小该值可保留更多顶点，有助于检测细微凹点
    # 遍历每个要显示的轮廓
    dist_sqs = []
    for contour_idx, contour in enumerate([contour]):
        # 多边形近似（保留更多细节，有助于检测细微凹点）
        perimeter = cv2.arcLength(contour, True)
        epsilon = EPSILON_FACTOR * perimeter  # 更小的epsilon值，保留更多顶点
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx_points = [p[0] for p in approx]  # 所有顶点序列
        num_points = len(approx_points)

        # if DEBUG1:# 3. 显示这一帧(全部的轮廓)

        #     print(f"\n[轮廓 #{contour_idx+1}] 顶点总数: {num_points}（含内凹点和直角点）")
        #     color = COLORS_LIST[contour_idx % len(COLORS_LIST)]
        #     frame2 = A4_frame.copy()  # 用copy()避免修改原图
        #     cv2.drawContours(frame2, [contour], -1, color, 2)
        #     cv2.imshow("A4 Detection", frame2)
        #     # 4. 关键：用waitKey(0)阻塞程序，等待用户按任意键再继续（不刷新画面）
        #     # 0表示无限等待，直到有按键输入
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()  # 关闭窗口

        
        # 1. 标记所有内凹点并记录索引（使用增强版检测函数）
        concave_indices = get_concave_indices(A4_frame, contour, approx_points, contour_idx)


        
  
        frame2 = A4_frame.copy()  # 用copy()避免修改原图


                        # 2. 检测直角点
        right_angle_info = []
        for i in range(num_points):
            p1 = np.array(approx_points[(i-1) % num_points])
            p2 = np.array(approx_points[i])
            p3 = np.array(approx_points[(i+1) % num_points])
            angle, concavity = angle_between_points(p1, p2, p3)
            point_id = f"{contour_idx+1}-{i}"
            
            # 标记直角点
            if 80 <= angle <= 120 and concavity == "外凸":
                right_angle_info.append( (p2, i, point_id) )
                cv2.circle(frame2, tuple(p2), 10, (0, 0, 255), 2)
                cv2.putText(frame2, point_id, (p2[0]-15, p2[1]-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                #print(f"[直角点] 编号 {point_id}，角度: {angle:.1f}°，坐标: {p2}")
            elif i not in concave_indices:
                pass#print(f"[普通顶点] 编号 {point_id}，角度: {angle:.1f}°（{concavity}），坐标: {p2}")

        # 3. 判断直角点是否相邻
        num_right_angles = len(right_angle_info)
        if num_right_angles >= 2:
            print(f"\n[相邻直角边判断] 共{num_right_angles}个直角点：")
            for i in range(num_right_angles):
                p_current, idx_current, id_current = right_angle_info[i]
                p_next, idx_next, id_next = right_angle_info[(i+1) % num_right_angles]
                
                # 确定区间
                if idx_current < idx_next:
                    start, end = idx_current + 1, idx_next
                else:
                    start, end = idx_current + 1, num_points + idx_next
                
                # 检查区间内是否有内凹点
                has_concave_between = False
                for idx_p in range(start, end):
                    if idx_p % num_points in concave_indices:
                        has_concave_between = True
                        break
                
                if not has_concave_between:
                    dist_sq = distance_squared(p_current, p_next)
                    dist_sqs.append(dist_sq)
                    print(f"[相邻对 {i+1}] {id_current} 与 {id_next}，距离平方: {dist_sq}")
                    mid_point = ((p_current[0]+p_next[0])//2, (p_current[1]+p_next[1])//2)
                    cv2.line(frame2, tuple(p_current), tuple(p_next), (255, 255, 0), 2)
                    cv2.putText(frame2, f"sq={dist_sq}", mid_point,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                else:
                    print(f"[相邻对 {i+1}] {id_current} 与 {id_next}，中间有内凹点")

                # 标注轮廓序号
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cX, cY = int(moments["m10"]/moments["m00"]), int(moments["m01"]/moments["m00"])
            cv2.putText(frame2, f"轮廓{contour_idx+1}", (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if DEBUG2:# 3. 显示这一帧(全部的轮廓)
            
            print(f"\n[轮廓 #{contour_idx+1}] 顶点总数: {num_points}（含内凹点和直角点）")
            for idx_p in concave_indices:
                p = approx_points[idx_p]
                point_id = f"{contour_idx+1}-{idx_p}"  # 内凹点编号
                cv2.circle(frame2, tuple(p), 8, (0, 255, 255), -1)  # 黄色实心圆标记内凹点
                cv2.putText(frame2, point_id, (p[0]+10, p[1]+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                print(f"[内凹点] 编号 {point_id}，坐标: {p}")
            cv2.imshow("A4 Detection", frame2)
            # 4. 关键：用waitKey(0)阻塞程序，等待用户按任意键再继续（不刷新画面）
            # 0表示无限等待，直到有按键输入
            cv2.waitKey(0)
            cv2.destroyAllWindows()  # 关闭窗口
        
    sq = min(dist_sqs)
    min_x = math.sqrt(sq * area_cm2)

    return f"{num_points / 4}个重叠正方形", min_x

def get_concave_indices(A4_frame, contour, approx_points, contour_idx):
    CONCAVE_DEFECT_THRESHOLD = 1000  # 降低内凹程度阈值，更容易检测到浅凹点
    POINT_MATCH_THRESHOLD = 100       # 增大点匹配距离阈值
    """获取内凹点在approx_points中的索引，增强版"""
    # 绘制凸包用于调试
    hull = cv2.convexHull(contour)
    cv2.drawContours(A4_frame, [hull], -1, (0, 0, 255), 2)  # 红色绘制凸包
    
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull_indices)
    concave_indices = set()  # 存储内凹点的索引
    
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            # 降低阈值，更容易检测到凹点
            if d > CONCAVE_DEFECT_THRESHOLD:
                # 绘制缺陷点用于调试
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                cv2.line(A4_frame, start, end, (255, 0, 0), 2)  # 蓝色线连接凸包点
                cv2.circle(A4_frame, far, 5, (0, 255, 0), -1)   # 绿色标记缺陷点
                
                # 更宽松的点匹配逻辑
                far_point = np.array(far)
                min_dist = float('inf')
                best_match_idx = -1
                
                for idx, p in enumerate(approx_points):
                    dist = distance(far_point, np.array(p))
                    if dist < min_dist:
                        min_dist = dist
                        best_match_idx = idx
                
                # 如果找到足够近的点，或者虽然稍远但角度明显内凹，都视为内凹点
                if best_match_idx != -1:
                    p1 = np.array(approx_points[(best_match_idx-1) % len(approx_points)])
                    p2 = np.array(approx_points[best_match_idx])
                    p3 = np.array(approx_points[(best_match_idx+1) % len(approx_points)])
                    angle, concavity = angle_between_points(p1, p2, p3)
                    
                    if min_dist < POINT_MATCH_THRESHOLD or (concavity == "内凹" and angle < 150):
                        concave_indices.add(best_match_idx)
                        print(f"[调试] 轮廓 #{contour_idx+1} 发现内凹点: 索引 {best_match_idx}, 距离 {min_dist:.1f}, 角度 {angle:.1f}°")
    
    # 额外检查：基于角度的内凹点检测（作为凸包缺陷检测的补充）
    num_points = len(approx_points)
    for i in range(num_points):
        p1 = np.array(approx_points[(i-1) % num_points])
        p2 = np.array(approx_points[i])
        p3 = np.array(approx_points[(i+1) % num_points])
        angle, concavity = angle_between_points(p1, p2, p3)
        
        # 角度明显小于180度且为内凹，即使凸包检测没发现，也视为内凹点
        if concavity == "内凹" and angle < 150 and i not in concave_indices:
            concave_indices.add(i)
            print(f"[补充检测] 轮廓 #{contour_idx+1} 发现内凹点: 索引 {i}, 角度 {angle:.1f}°")
    
    return concave_indices


def distance(p1, p2):
    """计算两点间的实际距离"""
    return np.sqrt(distance_squared(p1, p2))

def distance_squared(p1, p2):
    """计算两点距离的平方"""
    return (p2[0] - p1[0])**2 + (p2[1] - p1[1])** 2


def angle_between_points(p1, p2, p3):
    """计算三点形成的角度（p2为顶点），判断凹凸性"""
    v1 = p1 - p2
    v2 = p3 - p2
    cross_product = np.cross(v1, v2)  # 叉积判断凹凸（<0为内凹）
    concavity = "内凹" if cross_product < 0 else "外凸"
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0, concavity
    angle_rad = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
    return np.degrees(angle_rad), concavity