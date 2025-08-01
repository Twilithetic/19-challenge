import cv2
import numpy as np
from single_char_input import get_single_char
from no_gui_3_proc import get_distance
from no_gui_3_proc2 import get_X

frame = None
DEBUG = 1

def main():
    global frame
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # 关键：设置摄像头缓冲区大小为1（只保留最新帧）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    
    is_runing = False
    is_done = False
    distance_list = []
    x_list = []
    type_name = ""
    # 开始第一部分
    print("启动中")
    while cap.isOpened():
        ret, frame = cap.read()
        if (not ret): break  # 读取失败则退出
        
        if(is_runing == False):
            print("等待一键启动")
            key = ord(get_single_char())
            if(key == ord('1')): 
                print("处理中...")
                distance_list = []
                x_list = []
                is_runing = True
        if(is_runing):
            distance, A4_frame, area_cm2 = get_distance(frame)
            if(distance >= 90 and area_cm2 is not None):
                distance_list.append(distance)
                x, type_name = get_X(A4_frame, area_cm2)
                x_list.append(x)
                type_name = type_name
                if(len(distance_list) >= 20):
                    is_done = True
            else:
                continue
        if(is_done):
            print(f"距离(D): {sum(distance_list) / len(distance_list)}cm")
            print(f"{type_name}(x): {sum(x_list) / len(x_list)}cm")
            is_runing = False

        # # 显示图像
        # cv2.imshow("A4 Detection", frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print("用户按 'q' 退出")
        #     break  # 退出循环

    # 5. 释放资源（必须做，否则摄像头可能被占用）
    cap.release()  # 关闭摄像头
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

if __name__ == "__main__":
    main()
