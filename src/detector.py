import os
import cv2
import numpy as np
import random
from ultralytics import YOLO
from src.utils import logger
from config import Config

class Detector:
    def __init__(self, model_path=None):
        self.model_path = model_path or Config.MODEL_PATH
        self.model = None
        self.mode = "simulated" # "custom", "coco_fallback", "simulated"
        self.classes = ["Person", "Helmet", "Safety Vest", "Gloves", "Face Mask", "Fire", "Smoke"]
        
        self.load_model()
        
    def load_model(self):
        """Load custom YOLO model or fallback to COCO model or simulation"""
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                # Verify classes count
                names = self.model.names
                if len(names) >= 7:
                    self.mode = "custom"
                    self.classes = [names[i] for i in range(7)]
                    logger.info(f"Loaded custom YOLOv11 model from {self.model_path} in '{self.mode}' mode. Classes: {self.classes}")
                    return
                else:
                    logger.info(f"Loaded model from {self.model_path} but it is not our custom model (classes count < 7). Falling back.")
            except Exception as e:
                logger.error(f"Error loading custom model {self.model_path}: {e}")
                
        # Fallback to COCO yolo11n.pt
        try:
            logger.info("Custom model not found or invalid. Attempting to load pretrained COCO 'yolo11n.pt' model...")
            self.model = YOLO('yolo11n.pt')
            self.mode = "coco_fallback"
            logger.info("Loaded pretrained COCO yolo11n.pt. Running in coco_fallback mode.")
        except Exception as e:
            logger.warning(f"Could not load COCO yolo11n.pt model: {e}. Running in pure simulation mode.")
            self.model = None
            self.mode = "simulated"

    def detect(self, frame):
        """
        Run inference on the frame.
        Returns a list of dicts: [
            {'box': [x1, y1, x2, y2], 'class_id': int, 'label': str, 'confidence': float}
        ]
        """
        height, width = frame.shape[:2]
        
        # 1. Custom model inference
        if self.mode == "custom" and self.model:
            try:
                results = self.model(frame, verbose=False, conf=Config.CONFIDENCE_THRESHOLD, iou=Config.IOU_THRESHOLD)
                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        if class_id < len(self.classes):
                            detections.append({
                                'box': [x1, y1, x2, y2],
                                'class_id': class_id,
                                'label': self.classes[class_id],
                                'confidence': conf
                            })
                return detections
            except Exception as e:
                logger.error(f"Error running inference in custom mode: {e}. Falling back to simulation.")
                
        # 2. COCO Fallback: detects actual people and simulates PPE
        if self.mode == "coco_fallback" and self.model:
            try:
                results = self.model(frame, verbose=False, conf=Config.CONFIDENCE_THRESHOLD)
                detections = []
                coco_names = self.model.names
                
                # We extract persons (COCO class 0 is 'person')
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # COCO class 0 -> person
                        if coco_names[class_id] == 'person':
                            # Add person detection
                            detections.append({
                                'box': [x1, y1, x2, y2],
                                'class_id': 0, # Custom Person
                                'label': 'Person',
                                'confidence': conf
                            })
                            
                            # Simulate PPE mapping for testing rule engine:
                            # Let's seed based on coordinates to make it consistent for the same person
                            seed = int(x1 + y1)
                            random.seed(seed)
                            
                            # Helmet: 70% probability
                            if random.random() < 0.70:
                                h_w = (x2 - x1)
                                h_h = (y2 - y1)
                                helmet_box = [
                                    x1 + h_w * 0.1,
                                    y1 - h_h * 0.05,
                                    x2 - h_w * 0.1,
                                    y1 + h_h * 0.2
                                ]
                                detections.append({
                                    'box': helmet_box,
                                    'class_id': 1,
                                    'label': 'Helmet',
                                    'confidence': 0.88
                                })
                                
                            # Vest: 80% probability
                            if random.random() < 0.80:
                                h_w = (x2 - x1)
                                h_h = (y2 - y1)
                                vest_box = [
                                    x1 - h_w * 0.05,
                                    y1 + h_h * 0.2,
                                    x2 + h_w * 0.05,
                                    y1 + h_h * 0.65
                                ]
                                detections.append({
                                    'box': vest_box,
                                    'class_id': 2,
                                    'label': 'Safety Vest',
                                    'confidence': 0.85
                                })
                                
                            # Gloves: 60% probability
                            if random.random() < 0.60:
                                h_w = (x2 - x1)
                                h_h = (y2 - y1)
                                glove_box = [
                                    x1 - h_w * 0.1,
                                    y1 + h_h * 0.65,
                                    x1 + h_w * 0.2,
                                    y1 + h_h * 0.85
                                ]
                                detections.append({
                                    'box': glove_box,
                                    'class_id': 3,
                                    'label': 'Gloves',
                                    'confidence': 0.72
                                })
                                
                            # Face mask: 50% probability
                            if random.random() < 0.50:
                                h_w = (x2 - x1)
                                h_h = (y2 - y1)
                                mask_box = [
                                    x1 + h_w * 0.25,
                                    y1 + h_h * 0.15,
                                    x2 - h_w * 0.25,
                                    y1 + h_h * 0.3
                                ]
                                detections.append({
                                    'box': mask_box,
                                    'class_id': 4,
                                    'label': 'Face Mask',
                                    'confidence': 0.79
                                })
                                
                # Add random fire or smoke with 1% probability to test emergency rules
                # Use standard system random (re-seed with None to make it truly random)
                random.seed(None)
                if random.random() < 0.01:
                    # Draw a fire box in the center
                    detections.append({
                        'box': [width * 0.4, height * 0.4, width * 0.6, height * 0.6],
                        'class_id': 5,
                        'label': 'Fire',
                        'confidence': 0.91
                    })
                return detections
            except Exception as e:
                logger.error(f"Error in COCO fallback mode: {e}")
                
        # 3. Pure Simulation Mode
        # Generate artificial workers and items walking/situated in frame
        detections = []
        random.seed(None)
        
        # Simulate two workers
        for i in range(2):
            seed = int(100 + i * 200)
            # Make the workers move slightly over time
            import time
            t = time.time()
            dx = int(50 * np.sin(t / 10.0 + i))
            dy = int(15 * np.cos(t / 5.0 + i))
            
            x1 = 100 + i * 250 + dx
            y1 = 150 + dy
            x2 = x1 + 100
            y2 = y1 + 220
            
            # Clamp coords
            x1 = max(0, min(width - 5, x1))
            y1 = max(0, min(height - 5, y1))
            x2 = max(0, min(width - 5, x2))
            y2 = max(0, min(height - 5, y2))
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'class_id': 0,
                'label': 'Person',
                'confidence': 0.95
            })
            
            # Worker 1 wears PPE, Worker 2 lacks helmet
            if i == 0:
                # Helmet
                detections.append({
                    'box': [x1 + 10, y1 - 10, x2 - 10, y1 + 35],
                    'class_id': 1,
                    'label': 'Helmet',
                    'confidence': 0.93
                })
                # Vest
                detections.append({
                    'box': [x1 - 5, y1 + 45, x2 + 5, y1 + 140],
                    'class_id': 2,
                    'label': 'Safety Vest',
                    'confidence': 0.91
                })
            else:
                # Vest but no Helmet
                detections.append({
                    'box': [x1 - 5, y1 + 45, x2 + 5, y1 + 140],
                    'class_id': 2,
                    'label': 'Safety Vest',
                    'confidence': 0.89
                })
                
        # Random fire/smoke trigger (0.5% chance)
        if random.random() < 0.005:
            detections.append({
                'box': [width * 0.7, height * 0.2, width * 0.85, height * 0.45],
                'class_id': 5,
                'label': 'Fire',
                'confidence': 0.92
            })
            
        return detections
        
    def draw_detections(self, frame, detections):
        """Helper to render boxes on screen"""
        # color definitions (BGR)
        color_map = {
            0: (255, 0, 0),    # Person - Blue
            1: (0, 165, 255),  # Helmet - Orange
            2: (0, 255, 0),    # Safety Vest - Green
            3: (0, 255, 255),  # Gloves - Yellow
            4: (255, 255, 0),  # Face Mask - Cyan
            5: (0, 0, 255),    # Fire - Red
            6: (128, 128, 128) # Smoke - Grey
        }
        
        for det in detections:
            box = [int(coord) for coord in det['box']]
            cls_id = det['class_id']
            label = det['label']
            conf = det['confidence']
            
            color = color_map.get(cls_id, (255, 255, 255))
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Put label text
            txt = f"{label} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (box[0], box[1] - text_h - 5), (box[0] + text_w, box[1]), color, -1)
            cv2.putText(frame, txt, (box[0], box[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
        return frame
