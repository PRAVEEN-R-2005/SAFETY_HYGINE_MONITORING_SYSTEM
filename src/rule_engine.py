import time
import datetime
from src.utils import get_overlap_percentage, is_point_in_polygon, logger
from config import Config

class RuleEngine:
    def __init__(self, alert_cooldown=None):
        self.alert_cooldown = alert_cooldown or Config.ALERT_COOLDOWN_SECONDS
        # Structure: { (camera_id, track_id_or_type, violation_type): timestamp }
        self.active_violations = {}

    def process_detections(self, detections, camera_id, camera_polygon=None):
        """
        Apply rules on frame detections.
        detections: List of detections from Tracker (includes track_id for persons)
        camera_id: ID of the camera source
        camera_polygon: List of list coords for restricted zone [[x1, y1], ...]
        
        Returns:
            violations_found: list of dicts describing new violations to log/alert
            tracked_workers: list of person dicts updated with compliance info
        """
        current_time = time.time()
        
        # 1. Separate detections by classes
        persons = []
        helmets = []
        vests = []
        gloves = []
        masks = []
        fires = []
        smokes = []
        
        for det in detections:
            cls_id = det['class_id']
            if cls_id == 0:
                persons.append(det)
            elif cls_id == 1:
                helmets.append(det)
            elif cls_id == 2:
                vests.append(det)
            elif cls_id == 3:
                gloves.append(det)
            elif cls_id == 4:
                masks.append(det)
            elif cls_id == 5:
                fires.append(det)
            elif cls_id == 6:
                smokes.append(det)
                
        violations_found = []
        tracked_workers = []
        
        # 2. Match PPE to Person
        for person in persons:
            p_box = person['box']
            track_id = person.get('track_id', -1)
            
            has_helmet = False
            has_vest = False
            has_gloves = False
            has_mask = False
            
            # Check helmet overlap
            for helmet in helmets:
                if get_overlap_percentage(helmet['box'], p_box) > 0.5:
                    has_helmet = True
                    break
                    
            # Check vest overlap
            for vest in vests:
                if get_overlap_percentage(vest['box'], p_box) > 0.4:
                    has_vest = True
                    break
                    
            # Check gloves overlap
            for glove in gloves:
                if get_overlap_percentage(glove['box'], p_box) > 0.4:
                    has_gloves = True
                    break

            # Check mask overlap
            for mask in masks:
                if get_overlap_percentage(mask['box'], p_box) > 0.5:
                    has_mask = True
                    break
            
            # Calculate feet coordinates (midpoint of bottom of bounding box)
            feet_x = (p_box[0] + p_box[2]) / 2.0
            feet_y = p_box[3]
            in_restricted = False
            
            if camera_polygon and len(camera_polygon) >= 3:
                if is_point_in_polygon((feet_x, feet_y), camera_polygon):
                    in_restricted = True
                    
            # Store updated compliance info
            person_info = person.copy()
            person_info.update({
                'has_helmet': has_helmet,
                'has_vest': has_vest,
                'has_gloves': has_gloves,
                'has_mask': has_mask,
                'in_restricted_area': in_restricted
            })
            tracked_workers.append(person_info)
            
            # 3. Apply safety rules and generate violation items
            rules_to_check = [
                ('No Helmet', not has_helmet),
                ('No Vest', not has_vest),
                ('No Gloves', not has_gloves),
                ('Unauthorized Access', in_restricted)
            ]
            
            for violation_type, triggered in rules_to_check:
                if triggered:
                    key = (camera_id, track_id, violation_type)
                    last_alert = self.active_violations.get(key, 0)
                    
                    if current_time - last_alert > self.alert_cooldown:
                        self.active_violations[key] = current_time
                        violations_found.append({
                            'camera_id': camera_id,
                            'violation_type': violation_type,
                            'confidence': person['confidence'],
                            'track_id': track_id,
                            'worker_box': p_box
                        })

        # 4. Check for Environmental Violations (Fire & Smoke)
        for fire in fires:
            key = (camera_id, 'environment', 'Fire')
            last_alert = self.active_violations.get(key, 0)
            if current_time - last_alert > self.alert_cooldown:
                self.active_violations[key] = current_time
                violations_found.append({
                    'camera_id': camera_id,
                    'violation_type': 'Fire Alert',
                    'confidence': fire['confidence'],
                    'track_id': -1,
                    'worker_box': fire['box']
                })
                
        for smoke in smokes:
            key = (camera_id, 'environment', 'Smoke')
            last_alert = self.active_violations.get(key, 0)
            if current_time - last_alert > self.alert_cooldown:
                self.active_violations[key] = current_time
                violations_found.append({
                    'camera_id': camera_id,
                    'violation_type': 'Smoke Alert',
                    'confidence': smoke['confidence'],
                    'track_id': -1,
                    'worker_box': smoke['box']
                })
                
        return violations_found, tracked_workers
