import cv2
import numpy as np
from utils.helpers import (
    get_timestamp,
    save_frame,
    save_video_clip,
#    play_alert_sound
)

previous_frame = None
recording = False
clip_frames = []

def detect_motion(current_frame, threshold=1000):
    global previous_frame

    current_frame_resized = cv2.resize(current_frame, (640, 480))
    gray = cv2.cvtColor(current_frame_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    if previous_frame is None:
        previous_frame = blurred
        return False, []

    frame_diff = cv2.absdiff(previous_frame, blurred)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    motion_boxes = []

    for contour in contours:
        if cv2.contourArea(contour) < threshold:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        motion_boxes.append((x, y, w, h))
        motion_detected = True

    previous_frame = blurred
    return motion_detected, motion_boxes


# --- Main app loop ---
video = cv2.VideoCapture(0)

print("[INFO] Starting webcam feed... Press ESC to quit.")
while True:
    ret, frame = video.read()
    if not ret:
        break

    motion_detected, boxes = detect_motion(frame)

    if motion_detected:
        timestamp = get_timestamp()
        print(f"[ALERT] Motion detected at {timestamp}")
        save_frame(frame, timestamp)
        #play_alert_sound()

        recording = True
        clip_frames = [frame]  # start new video

        # Draw rectangles
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if recording:
        clip_frames.append(frame)
        if len(clip_frames) >= 60:  # ~3 seconds at 20 FPS
            save_video_clip(clip_frames, timestamp)
            recording = False
            clip_frames = []

    cv2.imshow("Motion Detector", frame)

    key = cv2.waitKey(10)
    if key == 27:  # ESC
        break

try:
    while True:
        # Your motion detection logic here
        pass
except KeyboardInterrupt:
    print("[INFO] Exiting safely.")
finally:
    video.release()
    out.release()
    cv2.destroyAllWindows()
