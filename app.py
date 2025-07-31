from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import numpy as np
import threading

app = Flask(__name__)

# 全局变量控制是否进行图像处理
process_image = False
# 保护共享变量的锁
lock = threading.Lock()

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

def generate_frames():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 已知实际参数（示例：一个面积为470cm²的矩形）
    REAL_AREA_CM2 = 470
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 检查是否需要处理图像
        with lock:
            current_process = process_image
        
        if current_process:
            # 找到最大矩形
            rect, pixel_area = find_largest_rectangle(frame)
            
            if rect is not None:
                # 计算距离
                distance = calculate_distance_cm(pixel_area, REAL_AREA_CM2) * 10
                
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
        
        # 编码为JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # 生成MJPEG流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OpenCV矩形检测与距离计算</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdn.jsdelivr.net/npm/font-awesome@4.7.0/css/font-awesome.min.css" rel="stylesheet">
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            primary: '#3B82F6',
                            secondary: '#10B981',
                            danger: '#EF4444',
                            dark: '#1E293B',
                        },
                        fontFamily: {
                            sans: ['Inter', 'system-ui', 'sans-serif'],
                        },
                    }
                }
            }
        </script>
        <style type="text/tailwindcss">
            @layer utilities {
                .content-auto {
                    content-visibility: auto;
                }
                .shadow-soft {
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                }
                .transition-custom {
                    transition: all 0.3s ease;
                }
            }
        </style>
    </head>
    <body class="bg-gray-50 font-sans text-dark">
        <div class="container mx-auto px-4 py-8 max-w-6xl">
            <header class="mb-8 text-center">
                <h1 class="text-[clamp(1.8rem,4vw,2.5rem)] font-bold text-dark mb-2">
                    <i class="fa fa-camera mr-3 text-primary"></i>矩形检测与距离计算
                </h1>
                <p class="text-gray-600 text-lg">点击"测试"按钮开始图像处理，识别最大矩形并计算距离</p>
            </header>
            
            <main class="bg-white rounded-xl shadow-soft p-6 md:p-8 mb-8">
                <div class="flex flex-col md:flex-row gap-6">
                    <div class="w-full md:w-3/4">
                        <div class="relative rounded-lg overflow-hidden bg-gray-100 border-2 border-gray-200">
                            <img id="videoFeed" src="/video_feed" alt="摄像头实时画面" 
                                 class="w-full h-auto object-contain">
                            <div id="statusOverlay" class="absolute top-4 left-4 bg-danger/80 text-white px-3 py-1 rounded-full text-sm font-medium hidden">
                                <i class="fa fa-spinner fa-spin mr-2"></i>处理中...
                            </div>
                        </div>
                    </div>
                    
                    <div class="w-full md:w-1/4 flex flex-col justify-center gap-4">
                        <button id="testBtn" class="bg-primary hover:bg-primary/90 text-white font-semibold py-3 px-6 rounded-lg transition-custom transform hover:scale-105 flex items-center justify-center">
                            <i class="fa fa-play-circle mr-2"></i>开始测试
                        </button>
                        
                        <button id="stopBtn" class="bg-danger hover:bg-danger/90 text-white font-semibold py-3 px-6 rounded-lg transition-custom transform hover:scale-105 flex items-center justify-center hidden">
                            <i class="fa fa-stop-circle mr-2"></i>停止测试
                        </button>
                        
                        <div class="bg-gray-50 p-4 rounded-lg border border-gray-200">
                            <h3 class="font-semibold text-gray-700 mb-3 flex items-center">
                                <i class="fa fa-info-circle text-primary mr-2"></i>检测信息
                            </h3>
                            <div id="distanceInfo" class="text-gray-600 mb-2">
                                <span class="font-medium">距离:</span> <span id="distanceValue">-- cm</span>
                            </div>
                            <div id="areaInfo" class="text-gray-600">
                                <span class="font-medium">面积:</span> <span id="areaValue">-- px</span>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
            
            <footer class="text-center text-gray-500 text-sm">
                <p>使用OpenCV和Flask构建 | 实时图像处理演示</p>
            </footer>
        </div>
        
        <script>
            // 获取DOM元素
            const testBtn = document.getElementById('testBtn');
            const stopBtn = document.getElementById('stopBtn');
            const statusOverlay = document.getElementById('statusOverlay');
            const distanceValue = document.getElementById('distanceValue');
            const areaValue = document.getElementById('areaValue');
            
            // 开始测试
            testBtn.addEventListener('click', async () => {
                try {
                    const response = await fetch('/start_process', { method: 'POST' });
                    if (response.ok) {
                        testBtn.classList.add('hidden');
                        stopBtn.classList.remove('hidden');
                        statusOverlay.classList.remove('hidden');
                    }
                } catch (error) {
                    console.error('启动测试失败:', error);
                }
            });
            
            // 停止测试
            stopBtn.addEventListener('click', async () => {
                try {
                    const response = await fetch('/stop_process', { method: 'POST' });
                    if (response.ok) {
                        stopBtn.classList.add('hidden');
                        testBtn.classList.remove('hidden');
                        statusOverlay.classList.add('hidden');
                        distanceValue.textContent = '-- cm';
                        areaValue.textContent = '-- px';
                    }
                } catch (error) {
                    console.error('停止测试失败:', error);
                }
            });
            
            // 定期获取处理结果更新UI
            setInterval(async () => {
                if (stopBtn.classList.contains('hidden')) return;
                
                try {
                    const response = await fetch('/get_results');
                    if (response.ok) {
                        const data = await response.json();
                        if (data.distance !== null) {
                            distanceValue.textContent = `${data.distance.toFixed(1)} cm`;
                        }
                        if (data.area !== null) {
                            areaValue.textContent = `${data.area} px`;
                        }
                    }
                } catch (error) {
                    console.error('获取结果失败:', error);
                }
            }, 100);
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_process', methods=['POST'])
def start_process():
    global process_image
    with lock:
        process_image = True
    return jsonify({"status": "started"})

@app.route('/stop_process', methods=['POST'])
def stop_process():
    global process_image
    with lock:
        process_image = False
    return jsonify({"status": "stopped"})

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
