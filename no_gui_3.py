import cv2
import numpy as np
from single_char_input import get_single_char
from no_gui_3_proc import get_distance
from no_gui_3_proc2 import get_X
import threading
import time

class CameraReader:
    def __init__(self, cam_id=0):
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头")
        
        # 配置摄像头参数（减少延迟）
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 缓冲区大小设为1
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.running = True  # 线程运行标志
        self.latest_frame = None  # 存储最新帧
        self.lock = threading.Lock()  # 线程锁
        
        # 启动缓冲区清理线程
        self.clean_thread = threading.Thread(target=self._clean_buffer, daemon=True)
        self.clean_thread.start()

    def _clean_buffer(self):
        """专门清理缓冲区的线程：持续grab()，只抓取不解码"""
        while self.running:
            # 用grab()快速抓取帧，清理缓冲区（比read()快，因为不解码）
            ret = self.cap.grab()
            if not ret:
                print("缓冲区清理失败，重试...")
                time.sleep(0.01)
            # 控制清理频率（避免占用过多CPU）
            time.sleep(0.001)

    def get_latest_frame(self):
        """主线程调用：获取最新帧（增加有效性检查）"""
        # 从缓冲区中取回最新帧（解码）
        ret, frame = self.cap.retrieve()
        if ret and frame is not None:  # 确保帧有效
            with self.lock:
                self.latest_frame = frame.copy()  # 只在帧有效时copy
            return frame
        # 帧无效时返回None，并打印提示
        #print("获取帧失败，返回None")
        return None

    def release(self):
        """释放资源"""
        self.running = False
        self.clean_thread.join()  # 等待清理线程结束
        self.cap.release()

frame = None

DEBUG = 1

def main():
    camera = CameraReader()
    try:
        
        global frame
        # 初始化摄像头
        # cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # # 关键：设置摄像头缓冲区大小为1（只保留最新帧）
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        
        is_runing = False
        is_done = False
        double_fix = True
        distance_list = []
        x_list = []
        type_name = ""
        # 开始第一部分
        print("启动中")
        while True:
            # 获取最新帧（供距离检测使用）
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            if(is_runing == False):
                print("等待一键启动")
                key = ord(get_single_char())
                if(key == ord('1')): 
                    print("处理中...")
                    distance_list = []
                    x_list = []
                    is_runing = True
                if key == ord('2'):
                    print("退出中...")
                    break  # ?????
            if(is_runing):
                distance, A4_frame, area_cm2 = get_distance(frame)
                if(distance >= 90 and area_cm2 is not None):
                    distance_list.append(distance)
                    x, type_name = get_X(A4_frame, area_cm2)
                    x_list.append(x)
                    type_name = type_name
                    if(len(distance_list) >= 3):
                        is_done = True
                else:
                    continue
            if(is_done):
                if (double_fix) :
                    double_fix = False
                    distance_list = []
                    is_done = False
                    continue
                print(f"距离(D): {sum(distance_list) / len(distance_list)}cm")
                print(f"{type_name}(x): {sum(x_list) / len(x_list)}cm")
                is_runing = False
                distance_list = []
                double_fix = True

            # # 显示图像
            # cv2.imshow("A4 Detection", frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("用户按 'q' 退出")
            #     break  # 退出循环

    finally:
        camera.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
