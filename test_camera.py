# Create a simple test
from picamera2 import Picamera2
import time
cameras = Picamera2.global_camera_info()
print(f'Found {len(cameras)} cameras')
for i, cam in enumerate(cameras):
    print(f'Camera {i}: {cam}')

