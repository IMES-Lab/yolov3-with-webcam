from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera



def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') # concat frame one by one and show result

def gen_frames_data():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        print('a')
        success, frame = camera.read()  # read the camera frame
        if not success:
            print('sjang fail')
            break
        else:
            print('sjang')
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            print(len('--frame\r\n Content-Type: image/jpeg\r\n\r\n' + str(frame) + '\r\n'))
            # yield (str(len('--frame\r\n Content-Type: image/jpeg\r\n\r\n' + str(frame) + '\r\n')))
            yield (frame) # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    data = gen_frames()

    return Response(data,
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_data')
def video_feed_data():
    """Video streaming route. Put this in the src attribute of an img tag."""

    data = gen_frames_data().__next__()

    print(data)
    return Response(data)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
