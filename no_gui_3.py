import cv2
import numpy as np
from single_char_input import get_single_char
from no_gui_3_proc import get_distance

frame = None

def main():
    global frame
    # 初始化摄像头
    cap = cv2.VideoCapture(0)

    #key = ord(get_single_char())

    # 开始第一部分
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 读取失败则退出
        # if(key == ord('1')):
        #     print("处理1")

        get_distance(frame)
        # 显示图像
        cv2.imshow("A4 Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户按 'q' 退出")
            break  # 退出循环

    # 5. 释放资源（必须做，否则摄像头可能被占用）
    cap.release()  # 关闭摄像头
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

if __name__ == "__main__":
    main()
