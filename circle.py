import cv2
import numpy as np

def main():
    # ??????
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("???????")
        return
    
    # ????????
    cv2.namedWindow('????')
    cv2.createTrackbar('????', '????', 5, 15, lambda x: None)
    cv2.createTrackbar('???', '????', 30, 100, lambda x: None)
    cv2.createTrackbar('????', '????', 10, 200, lambda x: None)
    cv2.createTrackbar('????', '????', 100, 500, lambda x: None)
    cv2.createTrackbar('????', '????', 50, 200, lambda x: None)
    
    # ???
    while True:
        # ??????
        ret, frame = cap.read()
        if not ret:
            print("?????")
            break
        
        # ???????
        blur_val = max(1, cv2.getTrackbarPos('????', '????') * 2 + 1)  # ?????
        param2 = cv2.getTrackbarPos('???', '????')
        min_radius = cv2.getTrackbarPos('????', '????')
        max_radius = cv2.getTrackbarPos('????', '????')
        min_dist = cv2.getTrackbarPos('????', '????')
        
        # ???
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, blur_val)
        
        # ?????
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=50,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        # ???????
        result_frame = frame.copy()
        
        # ?????
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # ?????????
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                
                # ??????
                cv2.circle(result_frame, center, radius, (0, 255, 0), 3)
                # ????
                cv2.circle(result_frame, center, 3, (0, 0, 255), -1)
                # ??????
                cv2.putText(result_frame, f"R:{radius}", 
                           (center[0]+radius+10, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # ?????????
            cv2.putText(result_frame, f"??? {len(circles[0])} ???", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # ????????
        param_display = np.zeros((150, 400, 3), dtype=np.uint8)
        cv2.putText(param_display, f"????: {blur_val}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        cv2.putText(param_display, f"???: {param2}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        cv2.putText(param_display, f"????: {min_radius}-{max_radius}", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        cv2.putText(param_display, f"??????: {min_dist}", (10, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        cv2.putText(param_display, "? 'Q' ??", (10, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        
        # ????
        cv2.imshow('????????', result_frame)
        cv2.imshow('????', param_display)
        
        # ????
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # ????
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()