from flask import Flask, render_template, request, jsonify
from flask_sock import Sock
import datetime
import time
from flask import Flask, render_template, request, jsonify
import datetime
import threading
from collections import deque
import json

app = Flask(__name__, template_folder='static')
sock = Sock(app)

# 存储最新的1000条数据
data_queue = deque(maxlen=1000)
# 记录最后处理的索引，避免重复发送
last_sent_index = -1
# 线程锁，确保线程安全
data_lock = threading.Lock()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/time', methods=['GET'])
def get_time():
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return jsonify({'time': current_time})

@sock.route('/ws')
def websocket(ws):
    global last_sent_index
    try:
        while True:
            # 检查队列是否有新数据
            if data_queue and len(data_queue) > last_sent_index + 1:
                with data_lock:  # 如果有多线程访问，仍需要锁
                    # 获取所有新数据（从上次发送的位置之后）
                    new_data = list(data_queue)[last_sent_index+1:]
                
                # 发送新数据给客户端
                for data in new_data:
                    try:
                        # 将Python字典转换为JSON字符串
                        ws.send(json.dumps(data))
                        last_sent_index += 1
                    except Exception:
                        # 客户端断开，退出循环
                        print("客户端断开")
                        break
            
            # 短暂休眠，避免CPU占用过高
            time.sleep(0.1)
            
    except Exception as e:
        print(f"WebSocket错误: {e}")
    finally:
        print("客户端连接关闭")
        
@app.route('/api/data', methods=['POST'])
def receive_data():
    """接收来自串口工具的数据"""
 
    try:
        data = request.json
        # 存储数据
        with data_lock:
            data_queue.append(data)
            
        return jsonify({'status': 'success', 'message': '数据已接收'}), 200
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)    