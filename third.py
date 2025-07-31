import cv2
import numpy as np

# ????
CANNY_THRESH_LOW = 80
CANNY_THRESH_HIGH = 200
CONTOUR_EPSILON_RATIO = 0.02
MIN_CONTOUR_AREA = 500
CIRCULARITY_THRESH = 0.6
EQUILATERAL_TOLERANCE = 0.05
A4_WIDTH_CM =17
A4_HEIGHT_CM = 25.5
A4_AREA_CM2 = A4_WIDTH_CM * A4_HEIGHT_CM  # ?623.7 cm�
DISTANCE_SCALE_FACTOR = 700  # ?????????

def extract_rectangle_roi(image):
    """??A4????ROI??"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    max_contour = max(contours, key=cv2.contourArea)
    epsilon = CONTOUR_EPSILON_RATIO * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    if len(approx) != 4:
        return None, None, None

    rect = cv2.boundingRect(approx)
    x, y, w, h = rect
    roi = image[y:y+h, x:x+w]
    return roi, (x, y, w, h), approx

def calculate_distance_cm(pixel_area, real_area_cm2):
    """????????cm?"""
    # ?????
    if pixel_area <= 0:
        return 0.0
        
    # ????????????
    # ?????????? (???? ? ?(????/????))
    return np.sqrt(real_area_cm2 / pixel_area) * DISTANCE_SCALE_FACTOR

def is_equilateral_triangle(approx):
    """?????????????"""
    if len(approx) != 3:
        return False, 0
    
    # ??????
    p1, p2, p3 = approx[0][0], approx[1][0], approx[2][0]
    
    # ????????
    d1 = np.linalg.norm(p1 - p2)
    d2 = np.linalg.norm(p2 - p3)
    d3 = np.linalg.norm(p3 - p1)
    
    # ?????????
    if d1 <= 0 or d2 <= 0 or d3 <= 0:
        return False, 0
    
    # ??????
    avg_length = (d1 + d2 + d3) / 3
    
    # ?????
    if avg_length <= 0:
        return False, 0
    
    # ???????????
    max_diff = max(abs(d1 - avg_length), abs(d2 - avg_length), abs(d3 - avg_length))
    if max_diff / avg_length < EQUILATERAL_TOLERANCE:
        return True, avg_length
    return False, 0

def detect_shapes_in_roi(roi):
    """?ROI???????????????????????"""
    # ??ROI????
    if roi is None or roi.size == 0:
        return []
        
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = []

    # ??ROI???
    roi_height, roi_width = roi.shape[:2]
    
    # ??ROI????
    if roi_width <= 0:
        return detected
        
    # ????????????
    pixel_to_cm_ratio = A4_WIDTH_CM / roi_width

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:  # ???????
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
            
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # ????
        if circularity > CIRCULARITY_THRESH and len(approx) > 5:
            # ???????????
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            diameter_cm = 2 * radius * pixel_to_cm_ratio
            detected.append(('circle', cnt, diameter_cm))
        
        # ???????????????????
        elif len(approx) == 3:
            is_equilateral, avg_side_pixel = is_equilateral_triangle(approx)
            if avg_side_pixel <= 0:
                continue
                
            avg_side_cm = avg_side_pixel * pixel_to_cm_ratio
            
            if is_equilateral:
                detected.append(('equilateral_triangle', cnt, avg_side_cm))
            else:
                # ?????????????????????
                (_, _), radius = cv2.minEnclosingCircle(cnt)
                size_cm = 2 * radius * pixel_to_cm_ratio
                detected.append(('triangle', cnt, size_cm))
        
        # ?????
        elif len(approx) == 4:
            pts = approx.reshape(-1, 2)
            dists = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
            
            if any(d <= 0 for d in dists):
                continue
                
            if max(dists) - min(dists) < 0.2 * np.mean(dists):
                # ??????????
                side_length_cm = np.mean(dists) * pixel_to_cm_ratio
                detected.append(('square', cnt, side_length_cm))
    
    return detected

def draw_detected_shapes(image, offset, shapes):
    """?????????????????????????"""
    x_off, y_off = offset
    for shape, cnt, size in shapes:
        if size <= 0:
            continue
            
        cnt += np.array([x_off, y_off])
        
        # ????????????????
        if shape == 'circle':
            color = (255, 255, 0)  # ??
            label = f'Circle: {size:.1f}cm'
            (x, y), r = cv2.minEnclosingCircle(cnt)
            cv2.circle(image, (int(x), int(y)), int(r), color, 2)
            text_pos = (int(x) - 50, int(y) - 10)
            
        elif shape == 'equilateral_triangle':
            color = (0, 25, 0)  # ??
            label = f'Equi Triangle: {size:.1f}cm'
            cv2.drawContours(image, [cnt], -1, color, 2)
            center = np.mean(cnt.reshape(-1, 2), axis=0).astype(int)
            text_pos = (center[0] - 70, center[1] - 10)
            
        elif shape == 'triangle':
            color = (0, 255, 255)  # ??
            label = f'Triangle: {size:.1f}cm'
            cv2.drawContours(image, [cnt], -1, color, 2)
            center = np.mean(cnt.reshape(-1, 2), axis=0).astype(int)
            text_pos = (center[0] - 60, center[1] - 10)
            
        elif shape == 'square':
            color = (255, 0, 255)  # ??
            label = f'Square: {size:.1f}cm'
            cv2.drawContours(image, [cnt], -1, color, 2)
            center = np.mean(cnt.reshape(-1, 2), axis=0).astype(int)
            text_pos = (center[0] - 50, center[1] - 10)
        
        # ????
        cv2.putText(image, label, text_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == "__main__":
    # ?????
    cap = cv2.VideoCapture(0)
    
    # ????????
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ????ROI
        roi, rect, approx = extract_rectangle_roi(frame)
        
        if roi is not None:
            x, y, w, h = rect
            
            # ????
            pixel_area = cv2.contourArea(approx)
            distance = calculate_distance_cm(pixel_area, A4_AREA_CM2)
            
            # ?ROI?????
            shapes = detect_shapes_in_roi(roi)
            draw_detected_shapes(frame, (x, y), shapes)
            
            # ??ROI??
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            
            # ??????
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # cv2.putText(frame, f"{distance:.1f}cm", (cX - 50, cY),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # ??????
            cv2.putText(frame, f"Distance: {distance:.1f}cm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Area: {pixel_area:.0f}px", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # ????
        cv2.imshow("Distance and Shape Detection", frame)
        
        # ?'q'??
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # ????
    cap.release()
    cv2.destroyAllWindows()