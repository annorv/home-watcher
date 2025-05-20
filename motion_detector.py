import cv2
import numpy as np

# Store the previous frame between calls
previous_frame = None

def detect_motion(current_frame, threshold=50000):
    global previous_frame

    # Resize for speed (optional)
    current_frame_resized = cv2.resize(current_frame, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(current_frame_resized, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # If first frame, just store and return no motion
    if previous_frame is None:
        previous_frame = blurred
        return False

    # Compute difference between current and previous frame
    frame_diff = cv2.absdiff(previous_frame, blurred)

    # Threshold the difference
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # Count white pixels (changed areas)
    motion_pixels = np.sum(thresh) / 255

    # Update previous frame
    previous_frame = blurred

    # Return True if enough motion is detected
    return motion_pixels > threshold
