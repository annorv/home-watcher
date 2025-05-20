import cv2
import numpy as np

previous_frame = None

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

    # Find contours in the threshold image
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
