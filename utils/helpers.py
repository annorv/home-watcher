import cv2
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_frame(frame, timestamp):
    filename = f"captured/{timestamp}.jpg"
    cv2.imwrite(filename, frame)
