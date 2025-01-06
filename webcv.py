from flask import Flask, Response
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            print('Failed to read frame')
            # exit(0)
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    

@app.route('/video_feed')
def video_feed():
    # This route serves the live video feed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>Live Video Feed</title>
        </head>
        <body>
        <center>
            <h1>Live Video Feed</h1>
            <img src="/video_feed" width="640" height="480">
        </center>
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
