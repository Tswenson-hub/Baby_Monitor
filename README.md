# Baby Monitor - Raspberry Pi Video System

A Flask-based baby monitoring application for Raspberry Pi 5 that uses computer vision to detect whether a baby is awake, asleep, or not present in the camera view.

## Features

### Dual AI Detection Modes
- **MediaPipe (Advanced)**: Uses Google MediaPipe for eye tracking and pose detection to accurately determine sleep state
- **OpenCV (Fallback)**: Motion-based detection using background subtraction if MediaPipe is unavailable

### Smart State Detection
- **Awake**: Baby's eyes are open or significant motion detected
- **Asleep**: Eyes closed (MediaPipe) or minimal motion for 30+ frames (OpenCV)
- **Not Present**: No baby detected in camera view

### User Interface
- Real-time video streaming at ~30 FPS
- Toggle button to enable/disable baby monitoring (starts disabled, showing raw camera feed)
- Live state display with color-coded indicators
- State change history log (last 50 events)
- FPS counter and system status

### Camera Management
- Automatic cleanup of camera resources on startup and shutdown
- Graceful handling of shutdown signals (SIGTERM, SIGINT)
- Prevents camera lock conflicts with other processes

## Technical Details

### Hardware Requirements
- Raspberry Pi 5
- Raspberry Pi Camera (Global Shutter Camera recommended)

### Software Stack
- **Python 3.11** with virtual environment support
- **Flask**: Web server and streaming
- **Picamera2**: Camera interface
- **OpenCV**: Image processing
- **MediaPipe** (optional): Advanced AI detection
- System packages for camera support

### Architecture
- Runs on port 5001 to avoid conflicts
- Multi-threaded design (separate camera processing thread)
- Frame rate: ~30 FPS maximum
- Hybrid package approach: system packages for camera/OpenCV, virtual env for MediaPipe

### State Change Logic
- Requires 15 consecutive frames of the same state before changing
- Reduces false positives from momentary changes
- All state changes logged with timestamps

## Usage

1. Access the web interface at `http://[raspberry-pi-ip]:5001`
2. Click "Enable Baby Monitoring" to start detection
3. View real-time state changes in the status panel
4. Toggle off to return to raw camera feed

## Key Routes
- `/` - Main web interface
- `/video_feed` - MJPEG video stream
- `/toggle_cv` - Enable/disable computer vision
- `/status` - JSON status endpoint# Baby_Monitor
