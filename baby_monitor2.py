#!/usr/bin/env python3
"""
Baby Monitor - Working version using system + virtual environment packages
Uses system packages for camera/OpenCV, virtual env for MediaPipe
Runs on port 5001 to avoid conflicts
Now includes toggle for Computer Vision capabilities and proper camera cleanup
"""

import sys
import os
import signal
import atexit
import subprocess

# Add system packages path to access picamera2, cv2, flask from system
sys.path.insert(0, '/usr/lib/python3/dist-packages')

# Check if we're in virtual environment and can import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe available - using advanced AI detection")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available - falling back to OpenCV detection")

from flask import Flask, render_template_string, Response, jsonify, request
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import time
import socket
from datetime import datetime
from collections import deque

app = Flask(__name__)

# Global variables
picam2_instance = None
baby_monitor = None
output_frame = None
frame_lock = threading.Lock()

# Camera cleanup functions
def cleanup_camera_resources():
    """Clean up camera resources and kill any lingering processes"""
    global picam2_instance
    
    print("🧹 Cleaning up camera resources...")
    
    # Stop our camera instance if it exists
    if picam2_instance is not None:
        try:
            picam2_instance.stop()
            picam2_instance.close()
            print("✅ Camera instance stopped and closed")
        except Exception as e:
            print(f"⚠️ Error stopping camera: {e}")
        finally:
            picam2_instance = None
    
    # Kill any lingering camera processes
    try:
        subprocess.run(['sudo', 'pkill', '-f', 'libcamera'], capture_output=True, timeout=5)
        subprocess.run(['sudo', 'pkill', '-f', 'rpicam'], capture_output=True, timeout=5)
        subprocess.run(['sudo', 'fuser', '-k', '/dev/video0'], capture_output=True, timeout=5)
        subprocess.run(['sudo', 'fuser', '-k', '/dev/media0'], capture_output=True, timeout=5)
        subprocess.run(['sudo', 'fuser', '-k', '/dev/media1'], capture_output=True, timeout=5)
        print("✅ Killed lingering camera processes")
    except Exception as e:
        print(f"⚠️ Error killing camera processes: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    cleanup_camera_resources()
    exit(0)

def startup_camera_cleanup():
    """Clean up any existing camera processes at startup"""
    print("🧹 Startup: Cleaning existing camera processes...")
    try:
        # Kill any existing camera processes
        subprocess.run(['sudo', 'pkill', '-f', 'libcamera-hello'], capture_output=True, timeout=5)
        subprocess.run(['sudo', 'pkill', '-f', 'rpicam-hello'], capture_output=True, timeout=5)
        subprocess.run(['sudo', 'systemctl', 'stop', 'camera-stream.service'], capture_output=True, timeout=5)
        
        # Release any locked camera devices
        subprocess.run(['sudo', 'fuser', '-k', '/dev/video0'], capture_output=True, timeout=5)
        subprocess.run(['sudo', 'fuser', '-k', '/dev/media0'], capture_output=True, timeout=5)
        subprocess.run(['sudo', 'fuser', '-k', '/dev/media1'], capture_output=True, timeout=5)
        
        print("✅ Startup cleanup completed")
        time.sleep(2)  # Give time for cleanup to complete
    except Exception as e:
        print(f"⚠️ Startup cleanup error: {e}")

# Register cleanup functions
atexit.register(cleanup_camera_resources)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# HTML template with toggle button
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Home Video System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f2f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        .video-section {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .status-section {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-height: 80vh;
            overflow-y: auto;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
            text-align: center;
        }
        #videoStream {
            width: 100%;
            max-width: 800px;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 8px;
            display: block;
            margin: 0 auto;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .cv-toggle-btn {
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 200px;
        }
        .cv-toggle-btn.disabled {
            background-color: #6c757d;
            color: white;
        }
        .cv-toggle-btn.enabled {
            background-color: #28a745;
            color: white;
        }
        .cv-toggle-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .current-state {
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            border-radius: 8px;
            font-size: 24px;
            font-weight: bold;
        }
        .state-awake {
            background-color: #e8f5e8;
            color: #2d5a2d;
            border: 2px solid #4caf50;
        }
        .state-asleep {
            background-color: #e3f2fd;
            color: #1565c0;
            border: 2px solid #2196f3;
        }
        .state-not-present {
            background-color: #fff3e0;
            color: #ef6c00;
            border: 2px solid #ff9800;
        }
        .state-unknown {
            background-color: #f5f5f5;
            color: #666;
            border: 2px solid #9e9e9e;
        }
        .state-disabled {
            background-color: #f8f9fa;
            color: #6c757d;
            border: 2px solid #dee2e6;
        }
        .detection-info {
            margin-top: 10px;
            font-size: 14px;
            color: #666;
            text-align: center;
        }
        .state-log {
            margin-top: 20px;
        }
        .log-entry {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid #ddd;
            background-color: #f9f9f9;
        }
        .log-awake {
            border-left-color: #4caf50;
            background-color: #f1f8e9;
        }
        .log-asleep {
            border-left-color: #2196f3;
            background-color: #e3f2fd;
        }
        .log-not-present {
            border-left-color: #ff9800;
            background-color: #fff8e1;
        }
        .timestamp {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        .state-text {
            font-weight: bold;
        }
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="video-section">
            <h1>🏠 Home Video System</h1>
            <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Baby Monitor Stream">
            
            <div class="controls">
                <button id="cvToggleBtn" class="cv-toggle-btn disabled" onclick="toggleCV()">
                    👶 Enable Baby Monitoring
                </button>
            </div>
            
            <div class="detection-info">
                <p id="aiTypeInfo">📹 Raw Camera Feed</p>
                <p id="frameRateInfo">🎬 Frame Rate: -- FPS</p>
                <p>Server: {{ server_ip }}:5001</p>
            </div>
        </div>
        
        <div class="status-section">
            <h2>Current Status</h2>
            <div id="currentState" class="current-state state-disabled">
                📹 Baby Monitoring Disabled
            </div>
            
            <div class="state-log">
                <h3>State Changes</h3>
                <div id="stateLog">
                    <p style="text-align: center; color: #666;">Enable Baby Monitoring to see detection logs</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let cvEnabled = false;
        
        function toggleCV() {
            const btn = document.getElementById('cvToggleBtn');
            const aiInfo = document.getElementById('aiTypeInfo');
            
            // Toggle state
            cvEnabled = !cvEnabled;
            
            // Update button appearance and text
            if (cvEnabled) {
                btn.className = 'cv-toggle-btn enabled';
                btn.innerHTML = '📹 Disable Baby Monitoring';
                aiInfo.innerHTML = '{{ ai_type }}';
            } else {
                btn.className = 'cv-toggle-btn disabled';
                btn.innerHTML = '👶 Enable Baby Monitoring';
                aiInfo.innerHTML = '📹 Raw Camera Feed';
            }
            
            // Send toggle request to server
            fetch('/toggle_cv', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({enabled: cvEnabled})
            })
            .then(response => response.json())
            .then(data => {
                console.log('CV toggle response:', data);
            })
            .catch(error => {
                console.error('Error toggling CV:', error);
                // Revert button state on error
                cvEnabled = !cvEnabled;
                toggleCV();
            });
        }
        
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const stateDiv = document.getElementById('currentState');
                    const logDiv = document.getElementById('stateLog');
                    const frameRateInfo = document.getElementById('frameRateInfo');
                    
                    // Update frame rate display
                    if (data.frame_rate) {
                        frameRateInfo.innerHTML = `🎬 Frame Rate: ${data.frame_rate.toFixed(1)} FPS`;
                    }
                    
                    if (!data.cv_enabled) {
                        stateDiv.className = 'current-state state-disabled';
                        stateDiv.innerHTML = '📹 Baby Monitoring Disabled';
                        logDiv.innerHTML = '<p style="text-align: center; color: #666;">Enable Baby Monitoring to see detection logs</p>';
                        return;
                    }
                    
                    const state = data.current_state;
                    stateDiv.className = `current-state state-${state}`;
                    
                    let stateText = '';
                    let emoji = '';
                    
                    switch(state) {
                        case 'awake':
                            stateText = 'Baby is Awake';
                            emoji = '👁️';
                            break;
                        case 'asleep':
                            stateText = 'Baby is Sleeping';
                            emoji = '😴';
                            break;
                        case 'not_present':
                            stateText = 'No Baby Detected';
                            emoji = '🛏️';
                            break;
                        default:
                            stateText = 'Analyzing...';
                            emoji = '🔍';
                    }
                    
                    stateDiv.innerHTML = `${emoji} ${stateText}`;
                    
                    logDiv.innerHTML = '';
                    
                    data.state_log.slice(-10).reverse().forEach(entry => {
                        const logEntry = document.createElement('div');
                        logEntry.className = `log-entry log-${entry.state}`;
                        logEntry.innerHTML = `
                            <div class="timestamp">${entry.timestamp}</div>
                            <div class="state-text">${entry.message}</div>
                        `;
                        logDiv.appendChild(logEntry);
                    });
                })
                .catch(error => console.error('Error fetching status:', error));
        }
        
        setInterval(updateStatus, 2000);
        updateStatus();
    </script>
</body>
</html>
"""

class BabyMonitor:
    def __init__(self):
        self.current_state = "unknown"
        self.previous_state = "unknown"
        self.state_log = deque(maxlen=50)
        self.state_counter = 0
        self.CONSECUTIVE_FRAMES = 15
        self.lock = threading.Lock()
        self.cv_enabled = False  # CV capabilities disabled by default
        
        # Frame rate calculation
        self.frame_times = deque(maxlen=30)  # Store last 30 frame times
        self.current_fps = 0.0
        
        # Initialize detection methods based on available packages
        if MEDIAPIPE_AVAILABLE:
            self.init_mediapipe()
        else:
            self.init_opencv_only()
    
    def init_mediapipe(self):
        """Initialize MediaPipe for advanced detection"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.EAR_THRESHOLD = 0.25
        print("🤖 MediaPipe initialized for advanced eye tracking")
    
    def init_opencv_only(self):
        """Initialize OpenCV-only detection"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.motion_threshold = 1000
        self.motion_history = deque(maxlen=10)
        self.no_motion_counter = 0
        self.SLEEP_THRESHOLD = 30
        print("🔍 OpenCV initialized for motion detection")
    
    def toggle_cv(self, enabled):
        """Toggle Computer Vision capabilities"""
        with self.lock:
            self.cv_enabled = enabled
            if enabled:
                print("👶 Baby Monitoring ENABLED")
                # Reset state when enabling
                self.current_state = "unknown"
                self.state_counter = 0
            else:
                print("📹 Baby Monitoring DISABLED - showing raw feed")
                # Clear state log when disabling
                self.current_state = "disabled"
                self.state_log.clear()
    
    def update_frame_rate(self):
        """Update frame rate calculation"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        if len(self.frame_times) > 1:
            # Calculate FPS based on last 30 frames
            time_span = self.frame_times[-1] - self.frame_times[0]
            if time_span > 0:
                self.current_fps = (len(self.frame_times) - 1) / time_span
    
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio for MediaPipe"""
        try:
            eye_points = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
            A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
            B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
            C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
            ear = (A + B) / (2.0 * C)
            return ear
        except:
            return 0.3
    
    def detect_with_mediapipe(self, frame):
        """Advanced detection using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        
        baby_present = False
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            if nose.visibility > 0.5:
                baby_present = True
        
        if not baby_present:
            new_state = "not_present"
        else:
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                left_eye = [362, 385, 387, 263, 373, 380]
                right_eye = [33, 160, 158, 133, 153, 144]
                
                left_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, left_eye)
                right_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                new_state = "asleep" if avg_ear < self.EAR_THRESHOLD else "awake"
                self.mp_draw.draw_landmarks(frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)
            else:
                new_state = "awake"
        
        if pose_results.pose_landmarks:
            self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        return new_state, frame
    
    def detect_with_opencv(self, frame):
        """Simple detection using OpenCV only"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        bodies = self.body_cascade.detectMultiScale(gray, 1.1, 3, minSize=(50, 50))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        baby_present = len(faces) > 0 or len(bodies) > 0
        
        if not baby_present:
            new_state = "not_present"
            self.no_motion_counter = 0
        else:
            fg_mask = self.background_subtractor.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100)
            
            if motion_area > self.motion_threshold:
                new_state = "awake"
                self.no_motion_counter = 0
            else:
                self.no_motion_counter += 1
                new_state = "asleep" if self.no_motion_counter >= self.SLEEP_THRESHOLD else self.current_state
            
            cv2.putText(frame, f"Motion: {int(motion_area)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return new_state, frame
    
    def process_frame(self, frame):
        """Process frame - either raw or with CV depending on toggle"""
        # Update frame rate calculation
        self.update_frame_rate()
        
        # Always return a copy to avoid modifying the original
        processed_frame = frame.copy()
        
        # If CV is disabled, just add timestamp and return raw frame
        if not self.cv_enabled:
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(processed_frame, f"RAW FEED - {timestamp}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"FPS: {self.current_fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(processed_frame, "Baby Monitoring: DISABLED", (10, processed_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            return processed_frame
        
        # CV is enabled - perform detection
        try:
            if MEDIAPIPE_AVAILABLE:
                new_state, processed_frame = self.detect_with_mediapipe(processed_frame)
            else:
                new_state, processed_frame = self.detect_with_opencv(processed_frame)
            
            # State change logic
            if new_state == self.current_state:
                self.state_counter = 0
            else:
                self.state_counter += 1
                if self.state_counter >= self.CONSECUTIVE_FRAMES:
                    self.update_state(new_state)
                    self.state_counter = 0
            
            # Add overlays
            state_text = f"State: {self.current_state.replace('_', ' ').title()}"
            cv2.putText(processed_frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add FPS counter
            cv2.putText(processed_frame, f"FPS: {self.current_fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(processed_frame, timestamp, (10, processed_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            print(f"Detection error: {e}")
            # On error, return frame with error message
            cv2.putText(processed_frame, f"CV Error: {str(e)[:50]}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return processed_frame
    
    def update_state(self, new_state):
        """Update state and log changes"""
        with self.lock:
            if new_state != self.current_state:
                self.previous_state = self.current_state
                self.current_state = new_state
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if self.previous_state == "unknown":
                    message = f"Initial detection: {new_state.replace('_', ' ').title()}"
                else:
                    message = f"Changed from {self.previous_state.replace('_', ' ').title()} to {new_state.replace('_', ' ').title()}"
                
                self.state_log.append({
                    "timestamp": timestamp,
                    "state": new_state,
                    "previous_state": self.previous_state,
                    "message": message
                })
                
                print(f"[{timestamp}] State Change: {message}")
    
    def get_current_state(self):
        """Get current state and log"""
        with self.lock:
            return {
                "cv_enabled": self.cv_enabled,
                "current_state": self.current_state,
                "previous_state": self.previous_state,
                "state_log": list(self.state_log),
                "frame_rate": self.current_fps
            }

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def camera_processing_thread():
    """Camera processing thread with proper cleanup"""
    global output_frame, picam2_instance
    
    try:
        # Startup cleanup
        startup_camera_cleanup()
        
        # Initialize camera
        picam2_instance = Picamera2()
        config = picam2_instance.create_video_configuration(main={"size": (1280, 720)})
        picam2_instance.configure(config)
        picam2_instance.start()
        print("✅ Camera initialized successfully")
        
        while True:
            frame = picam2_instance.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Process frame (raw or with CV based on toggle)
            processed_frame = baby_monitor.process_frame(frame_bgr)
            
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            with frame_lock:
                output_frame = buffer.tobytes()
            
            # Small sleep to prevent excessive CPU usage, but allow higher frame rates
            time.sleep(0.033)  # ~30 FPS max
            
    except Exception as e:
        print(f"❌ Camera thread error: {e}")
    finally:
        # Ensure cleanup happens
        if picam2_instance is not None:
            try:
                picam2_instance.stop()
                picam2_instance.close()
                print("✅ Camera cleaned up in finally block")
            except:
                pass
            picam2_instance = None

def generate_frames():
    """Generator for video frames"""
    global output_frame
    
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            frame = output_frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    server_ip = get_local_ip()
    ai_type = "🤖 MediaPipe AI Detection" if MEDIAPIPE_AVAILABLE else "🔍 OpenCV Motion Detection"
    return render_template_string(HTML_TEMPLATE, server_ip=server_ip, ai_type=ai_type)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_cv', methods=['POST'])
def toggle_cv():
    """Toggle Computer Vision capabilities"""
    try:
        data = request.get_json()
        enabled = data.get('enabled', False)
        baby_monitor.toggle_cv(enabled)
        
        return jsonify({
            "status": "success",
            "cv_enabled": enabled,
            "message": f"Baby Monitoring {'enabled' if enabled else 'disabled'}"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/status')
def status():
    """Status endpoint"""
    state_data = baby_monitor.get_current_state()
    ai_model = "Google MediaPipe" if MEDIAPIPE_AVAILABLE else "OpenCV Motion Detection"
    
    return jsonify({
        "status": "running",
        "camera": "Raspberry Pi Global Shutter Camera",
        "ai_model": ai_model,
        "server_ip": get_local_ip(),
        "cv_enabled": state_data["cv_enabled"],
        "current_state": state_data["current_state"],
        "previous_state": state_data["previous_state"],
        "state_log": state_data["state_log"],
        "frame_rate": state_data["frame_rate"]
    })

if __name__ == '__main__':
    try:
        # Clean up any existing camera resources first
        startup_camera_cleanup()
        
        # Initialize baby monitor
        baby_monitor = BabyMonitor()
        
        print("Starting Home Video System...")
        if MEDIAPIPE_AVAILABLE:
            print("🤖 MediaPipe available for advanced AI detection")
        else:
            print("🔍 OpenCV available for motion-based detection")
        
        print("📹 Baby Monitoring starts DISABLED - showing raw camera feed")
        print("💡 Use the toggle button in the web interface to enable baby monitoring")
        
        camera_thread = threading.Thread(target=camera_processing_thread)
        camera_thread.daemon = True
        camera_thread.start()
        
        time.sleep(3)
        
        server_ip = get_local_ip()
        print(f"Home Video System ready!")
        print(f"Access at: http://{server_ip}:5001")
        
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
        cleanup_camera_resources()
    except Exception as e:
        print(f"❌ Error: {e}")
        cleanup_camera_resources()
    finally:
        print("🧹 Final cleanup...")
        cleanup_camera_resources()