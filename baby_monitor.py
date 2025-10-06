#!/usr/bin/env python3
"""
Baby Monitor with Arducam IMX462 on Raspberry Pi 5
Features: Live feed, timestamp overlay, snapshot capture, motion detection
Controls: 'q' to quit, 's' to save snapshot, 'm' to toggle motion detection
"""

from picamera2 import Picamera2
import cv2
import time
import os
from datetime import datetime
import numpy as np

class BabyMonitor:
    def __init__(self):
        self.picam2 = None
        self.motion_detection = False
        self.last_frame = None
        self.motion_threshold = 1000
        self.snapshots_dir = "baby_snapshots"
        
        # Create snapshots directory
        if not os.path.exists(self.snapshots_dir):
            os.makedirs(self.snapshots_dir)
    
    def initialize_camera(self):
        """Initialize the IMX462 camera with optimal settings"""
        print("Initializing IMX462 ultra low-light camera...")
        self.picam2 = Picamera2()
        
        # Configure for baby monitoring - good quality and low light performance
        config = self.picam2.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"},
            controls={
                "FrameRate": 30,
                "ExposureTime": 33000,  # Good for low light
                "AnalogueGain": 4.0,    # Boost for dark rooms
                "Brightness": 0.1       # Slight brightness boost
            }
        )
        self.picam2.configure(config)
        self.picam2.start()
        print("Camera started successfully!")
        
        # Let camera adjust to lighting
        time.sleep(2)
    
    def detect_motion(self, current_frame):
        """Simple motion detection for baby monitoring"""
        if self.last_frame is None:
            self.last_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(self.last_frame, gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Count white pixels (motion)
        motion_pixels = cv2.countNonZero(thresh)
        
        # Update last frame
        self.last_frame = gray.copy()
        
        return motion_pixels > self.motion_threshold
    
    def add_overlay(self, frame):
        """Add timestamp and status overlay to frame"""
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # Status indicators
        status_y = 70
        if self.motion_detection:
            cv2.putText(frame, "Motion Detection: ON", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Controls help
        help_text = [
            "Controls: Q=Quit, S=Snapshot, M=Motion Detection",
        ]
        
        for i, text in enumerate(help_text):
            y_pos = frame.shape[0] - 30 - (i * 25)
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
        
        return frame
    
    def save_snapshot(self, frame):
        """Save a snapshot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.snapshots_dir, f"baby_{timestamp}.jpg")
        
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr_frame)
        print(f"Snapshot saved: {filename}")
        
        # Show visual feedback
        return True
    
    def run(self):
        """Main monitoring loop"""
        try:
            self.initialize_camera()
            
            print("Baby Monitor Active!")
            print("Press 'q' to quit, 's' for snapshot, 'm' to toggle motion detection")
            
            snapshot_feedback_timer = 0
            
            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Motion detection
                motion_detected = False
                if self.motion_detection:
                    motion_detected = self.detect_motion(frame)
                    if motion_detected:
                        # Add motion alert to frame
                        cv2.putText(frame, "MOTION DETECTED!", (400, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Add overlay information
                frame = self.add_overlay(frame)
                
                # Snapshot feedback
                if snapshot_feedback_timer > 0:
                    cv2.putText(frame, "SNAPSHOT SAVED!", (400, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    snapshot_feedback_timer -= 1
                
                # Display frame
                cv2.imshow('Baby Monitor - IMX462 Ultra Low Light', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_snapshot(frame)
                    snapshot_feedback_timer = 30  # Show feedback for 30 frames
                elif key == ord('m'):
                    self.motion_detection = not self.motion_detection
                    status = "ON" if self.motion_detection else "OFF"
                    print(f"Motion detection: {status}")
                
                # Auto-snapshot on motion (optional)
                if motion_detected and self.motion_detection:
                    self.save_snapshot(frame)
                    snapshot_feedback_timer = 30
                    
        except KeyboardInterrupt:
            print("\nStopping baby monitor...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.picam2:
            self.picam2.stop()
        cv2.destroyAllWindows()
        print("Baby monitor stopped.")

def main():
    monitor = BabyMonitor()
    monitor.run()

if __name__ == "__main__":
    main()
