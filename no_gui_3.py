import cv2
import numpy as np
from single_char_input import get_single_char



def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(1)

    key = ord(get_single_char())
    print(key)
    print("11")


if __name__ == "__main__":
    main()
