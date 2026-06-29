import os
import cv2
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
from config import Config

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Clear handlers to avoid duplicate logging if setup is run multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Initialize main application logger
logger = setup_logger('safety_monitor', os.path.join(Config.LOGS_FOLDER, 'app.log'))

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    box = [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    if union == 0:
        return 0
        
    return intersection / union

def get_overlap_percentage(box_child, box_parent):
    """
    Checks how much of box_child is inside box_parent.
    Useful for determining if gloves/vest/helmet are inside a person's bounding box.
    Returns value between 0.0 and 1.0.
    """
    x1 = max(box_child[0], box_parent[0])
    y1 = max(box_child[1], box_parent[1])
    x2 = min(box_child[2], box_parent[2])
    y2 = min(box_child[3], box_parent[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    child_area = (box_child[2] - box_child[0]) * (box_child[3] - box_child[1])

    if child_area == 0:
        return 0.0
    return intersection_area / child_area

def is_point_in_polygon(point, polygon):
    """
    Check if a point (x, y) is inside a polygon using OpenCV.
    polygon is a list of [x, y] coordinates: [[x1, y1], [x2, y2], ...]
    """
    if not polygon or len(polygon) < 3:
        return False
    
    # Convert polygon to np array of shape (N, 1, 2)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    # point shape should be (x, y)
    result = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
    return result >= 0

def draw_restricted_area(image, polygon, is_violation=False):
    """
    Draw safety polygon onto the frame
    """
    if not polygon or len(polygon) < 3:
        return image
        
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    color = (0, 0, 255) if is_violation else (0, 255, 255) # Red if violation, Yellow/Orange otherwise
    
    # Draw transparent overlay
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.polylines(image, [pts], True, color, 2)
    
    alpha = 0.25 # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

def ensure_alarm_audio():
    """Generates a standard safety beep (alarm.wav) if it doesn't already exist"""
    audio_dir = os.path.join(Config.BASE_DIR, 'static', 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    filepath = os.path.join(audio_dir, 'alarm.wav')
    
    if os.path.exists(filepath):
        return
        
    import wave
    import struct
    import math
    
    # 8000Hz sampling rate, 16bit, mono, 0.5s beep
    sample_rate = 8000.0
    duration = 0.5
    frequency = 900.0 # 900Hz warning beep
    
    num_samples = int(duration * sample_rate)
    
    try:
        with wave.open(filepath, 'w') as wav_file:
            # 1 channel, 2 bytes/sample, sample rate, num_samples
            wav_file.setparams((1, 2, int(sample_rate), num_samples, 'NONE', 'not compressed'))
            for i in range(num_samples):
                val = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
                data = struct.pack('<h', val)
                wav_file.writeframesraw(data)
        logger.info(f"Generated default audio alarm chime at: {filepath}")
    except Exception as e:
        logger.error(f"Failed to generate alarm.wav: {e}")

# Automatically compile chime on load
ensure_alarm_audio()

