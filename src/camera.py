import cv2
import time
import numpy as np
from src.utils import logger

class CameraStream:
    def __init__(self, stream_url):
        """
        stream_url can be:
        - integer: Webcam/USB index (e.g. 0)
        - string: path to MP4 video, RTSP url, or "simulation"
        """
        self.stream_url = stream_url
        self.is_simulated = False
        
        # Check if index is integer
        try:
            self.stream_url = int(stream_url)
        except ValueError:
            if str(stream_url).lower() == 'simulation':
                self.is_simulated = True
                
        self.cap = None
        self.running = False
        
        # Simulation parameters
        self.sim_workers = [
            {'x': 100, 'y': 200, 'dx': 2, 'dy': 1, 'id': 1, 'ppe': True, 'timer': 0},
            {'x': 400, 'y': 250, 'dx': -1, 'dy': 2, 'id': 2, 'ppe': False, 'timer': 120}
        ]
        self.sim_fire_timer = 0
        self.sim_fire_active = False

    def start(self):
        if self.running:
            return
            
        if self.is_simulated:
            self.running = True
            logger.info("Started Simulated Camera Stream.")
            return True
            
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if self.cap.isOpened():
                self.running = True
                logger.info(f"Successfully opened camera stream: {self.stream_url}")
                return True
            else:
                logger.error(f"Failed to open camera stream: {self.stream_url}. Falling back to simulation.")
                self.is_simulated = True
                self.running = True
                return True
        except Exception as e:
            logger.error(f"Error starting camera {self.stream_url}: {e}. Falling back to simulation.")
            self.is_simulated = True
            self.running = True
            return True

    def read_frame(self):
        if not self.running:
            return False, None
            
        if self.is_simulated:
            return True, self._generate_simulated_frame()
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                # If video file ended, loop it
                if isinstance(self.stream_url, str) and not self.stream_url.startswith('rtsp'):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
            return ret, frame
        except Exception as e:
            logger.error(f"Error reading frame from camera: {e}")
            return False, None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info(f"Stopped camera stream: {self.stream_url}")

    def _generate_simulated_frame(self):
        """
        Draw a simulated workshop floor with machines, pathways,
        restricted zones, and moving workers.
        """
        # Create a base gray canvas (Factory Floor)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (45, 45, 45) # Dark charcoal background
        
        # Draw floor markings (walkways)
        cv2.rectangle(frame, (50, 100), (590, 380), (70, 70, 70), -1) # work area
        cv2.line(frame, (50, 100), (590, 100), (100, 255, 255), 2)   # yellow walkway bounds
        cv2.line(frame, (50, 380), (590, 380), (100, 255, 255), 2)
        
        # Draw structural columns / machines
        cv2.circle(frame, (80, 80), 30, (120, 120, 120), -1)
        cv2.circle(frame, (80, 80), 28, (80, 80, 80), 3)
        cv2.putText(frame, "CNC-1", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.circle(frame, (560, 80), 30, (120, 120, 120), -1)
        cv2.circle(frame, (560, 80), 28, (80, 80, 80), 3)
        cv2.putText(frame, "CNC-2", (540, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw a restricted zone (e.g. Danger High Voltage box)
        # Coordinates: [[320, 150], [480, 150], [480, 280], [320, 280]]
        pts = np.array([[320, 150], [480, 150], [480, 280], [320, 280]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        # Transparent red overlay for restricted zone
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 120))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.putText(frame, "DANGER: HIGH TEMPERATURE ZONE", (325, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Simulate movement of workers
        for worker in self.sim_workers:
            worker['timer'] += 1
            # Move workers
            worker['x'] += worker['dx']
            worker['y'] += worker['dy']
            
            # Bounce off boundary
            if worker['x'] < 60 or worker['x'] > 580:
                worker['dx'] *= -1
            if worker['y'] < 110 or worker['y'] > 360:
                worker['dy'] *= -1
                
            # Draw visual worker placeholder in frame
            wx, wy = worker['x'], worker['y']
            
            # Body (Person)
            cv2.rectangle(frame, (wx - 20, wy - 50), (wx + 20, wy + 20), (180, 105, 255), -1) # Pinkish person box
            # Head
            cv2.circle(frame, (wx, wy - 60), 12, (220, 200, 180), -1)
            
            # PPE Visual representation
            if worker['ppe']:
                # Helmet (Orange semi-circle)
                cv2.ellipse(frame, (wx, wy - 62), (13, 8), 0, 180, 360, (0, 128, 255), -1)
                # Vest (Neon green shirt overlay)
                cv2.rectangle(frame, (wx - 15, wy - 40), (wx + 15, wy - 5), (0, 255, 128), -1)
            else:
                # No PPE
                pass
                
        # Fire simulation triggers
        self.sim_fire_timer += 1
        if self.sim_fire_timer % 400 == 0:
            self.sim_fire_active = not self.sim_fire_active
            
        if self.sim_fire_active:
            # Draw fire (orange-red circles flickering)
            import random
            radius = random.randint(15, 25)
            cv2.circle(frame, (250, 300), radius, (0, 0, 255), -1)
            cv2.circle(frame, (250, 300), int(radius * 0.7), (0, 165, 255), -1)
            cv2.circle(frame, (250, 300), int(radius * 0.4), (0, 255, 255), -1)
            cv2.putText(frame, "FIRE EVENT SIMULATED", (210, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Add timestamp and watermarks
        t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(frame, f"FEED: SIMULATED DIGITAL FACTORY | {t_str}", (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        return frame
