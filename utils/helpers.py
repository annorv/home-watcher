import cv2
import os
from datetime import datetime
from playsound import playsound
import simpleaudio as sa
import os

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_frame(frame, timestamp):
    filename = f"captured/images/motion_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

def save_video_clip(frames, timestamp, fps=20):
    height, width, _ = frames[0].shape
    filename = f"captured/videos/motion_{timestamp}.mp4"

    # Use H.264 codec (requires OpenCV to be built with ffmpeg support)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use 'avc1' or 'X264' if 'mp4v' doesn't work

    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()

def play_alert_sound():
    sound_path = os.path.join("assets", "alert.wav")
    try:
        wave_obj = sa.WaveObject.from_wave_file(sound_path)
        wave_obj.play()
    except Exception as e:
        print(f"[ERROR] Could not play sound: {e}")
