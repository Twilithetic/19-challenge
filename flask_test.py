from flask import Flask, Response
import cv2

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # 编码为 JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>摄像头测试</title>
    </head>
    <body>
        <h1>摄像头实时画面</h1>
        <img src="/video_feed" width="800" height="600">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)