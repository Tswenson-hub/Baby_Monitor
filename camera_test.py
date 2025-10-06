#!/usr/bin/env python3
import sys

try:
    from picamera2 import Picamera2
    print("✅ Picamera2 imported successfully")
    
    # Get camera info
    picam2 = Picamera2()
    cameras = picam2.global_camera_info()
    
    print(f"📷 Found {len(cameras)} camera(s):")
    for i, cam in enumerate(cameras):
        print(f"  Camera {i}: {cam}")
    
    if len(cameras) > 0:
        print("✅ Camera detected! Trying to start...")
        config = picam2.create_preview_configuration()
        picam2.configure(config)
        picam2.start()
        print("✅ Camera started successfully!")
        picam2.stop()
    else:
        print("❌ No cameras detected")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Camera error: {e}")
    print("This suggests camera drivers or hardware connection issues")
