#!/usr/bin/env python3
"""
Raspberry Pi Global Shutter Camera Web Streaming Server
Streams video from the camera to a web page accessible on local network
"""

from flask import Flask, render_template_string, Response
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import threading
import io
import time
import socket

app = Flask(__name__)

# HTML template for the video stream page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Raspberry Pi Camera Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
            text-align: center;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        #videoStream {
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 8px;
        }
        .info {
            margin-top: 20px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Raspberry Pi Camera Live Stream</h1>
        <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Camera Stream">
        <div class="info">
            <p>Streaming from Raspberry Pi Global Shutter Camera</p>
            <p>Server IP: {{ server_ip }}</p>
        </div>
    </div>
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

def get_local_ip():
    """Get the local IP address of the Raspberry Pi"""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

# Initialize camera and streaming
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1640, 1232)})
picam2.configure(config)

output = StreamingOutput()
encoder = JpegEncoder(q=70)  # Quality setting (70 is good balance of quality/bandwidth)

@app.route('/')
def index():
    """Main page with video stream"""
    server_ip = get_local_ip()
    return render_template_string(HTML_TEMPLATE, server_ip=server_ip)

def generate_frames():
    """Generator function for video frames"""
    while True:
        with output.condition:
            output.condition.wait()
            frame = output.frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Simple status endpoint"""
    return {
        "status": "running",
        "camera": "Raspberry Pi Global Shutter Camera",
        "resolution": "1640x1232",
        "server_ip": get_local_ip()
    }

def start_camera():
    """Start the camera in a separate thread"""
    picam2.start_recording(encoder, FileOutput(output))

if __name__ == '__main__':
    try:
        print("Starting Raspberry Pi Camera Web Stream...")
        print("Initializing camera...")
        
        # Start camera in a separate thread
        camera_thread = threading.Thread(target=start_camera)
        camera_thread.daemon = True
        camera_thread.start()
        
        time.sleep(2)  # Give camera time to initialize
        
        server_ip = get_local_ip()
        print(f"Camera initialized successfully!")
        print(f"Starting web server on {server_ip}:5000")
        print(f"Access the stream at: http://{server_ip}:5000")
        print("Press Ctrl+C to stop the server")
        
        # Start Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            picam2.stop_recording()
            picam2.close()
        except:
            pass
