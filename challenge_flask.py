#! .conda/bin/python
from flask import Flask, Response, render_template_string, request, jsonify,render_template
import cv2
import threading
from challenge_flask_proc import find_largest_rectangle, calculate_distance_cm

app = Flask(__name__, template_folder='static')

# 全局变量控制是否进行图像处理
process_image = False
# 存储距离数据的数组
distance_data = []
filter_len = 10

# 保护共享变量的锁
lock = threading.Lock()



def generate_frames():
    # 声明使用全局变量（关键修复！）
    global process_image, latest_results

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # # 设置摄像头分辨率
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    # 已知实际参数（示例：一个面积为470cm²的矩形）
    REAL_AREA_CM2 = 470
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 检查是否需要处理图像
        with lock:
            current_process = process_image
        
        if current_process: # 按了测试才会运行下面的
            # 找到最大矩形
            rect, pixel_area = find_largest_rectangle(frame)
            
            if rect is not None:
                # 计算距离
                distance = calculate_distance_cm(pixel_area, REAL_AREA_CM2) * 10
                # 将距离添加到数据数组
                with lock:
                    distance_data.append(distance)
                    
                    # 检查数据量是否超过1000，如果是则停止处理
                    if len(distance_data) >= filter_len:
                        # 计算平均值
                        latest_results["distance"] = sum(distance_data) / len(distance_data)
                        # 停止处理
                        process_image = False
                
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
                
                with lock:
                    data_count = len(distance_data)
                # 在左上角显示信息
                cv2.putText(frame, f"Distance: {distance:.1f}cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Area: {pixel_area}px", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Data: {data_count}/1000", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # 编码为JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # 生成MJPEG流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_process', methods=['POST'])
def start_process():
    global process_image, distance_data
    with lock:
        # 重置数据
        distance_data = []
        # 开始处理
        process_image = True
    return jsonify({"status": "started"})

@app.route('/stop_process', methods=['POST'])
def stop_process():
    global process_image
    with lock:
        process_image = False
    return jsonify({"status": "stopped"})

# @app.route('/stop_process', methods=['POST'])
# def stop_process():
#     global process_image
#     with lock:
#         process_image = False
#     return jsonify({"status": "stopped"})

# 存储最新的处理结果
latest_results = {
    "distance": None,
    "area": None
}

@app.route('/get_results')
def get_results():
    return jsonify(latest_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
