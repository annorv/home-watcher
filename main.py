import cv2
import time
import os
from motion_detector import detect_motion
from utils.helpers import get_timestamp, save_frame

# Create the 'captured' folder if it doesn't exist
os.makedirs("captured", exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
time.sleep(2)

print("[INFO] Starting camera feed. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    motion = detect_motion(frame)

    if motion:
        timestamp = get_timestamp()
        save_frame(frame, timestamp)
        print(f"[ALERT] Motion detected at {timestamp}")

    cv2.imshow("HomeWatcher Feed", frame)

    if cv2.waitKey(10) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
