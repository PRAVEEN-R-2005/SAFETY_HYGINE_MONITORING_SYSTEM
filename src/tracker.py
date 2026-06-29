import numpy as np
from src.utils import calculate_iou

class Tracker:
    def __init__(self, max_lost=20, iou_threshold=0.3):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.next_id = 1
        
        # Track representation: { id: {'box': [x1,y1,x2,y2], 'lost_count': int} }
        self.tracks = {}

    def update(self, person_detections):
        """
        Update tracks with new bounding boxes of class 'Person'.
        person_detections: list of dicts with 'box', 'confidence', etc.
        Returns a list of dicts with an added 'track_id' field.
        """
        updated_detections = []
        if not person_detections:
            # Increment lost counter for all existing tracks
            lost_ids = []
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['lost_count'] += 1
                if self.tracks[track_id]['lost_count'] > self.max_lost:
                    lost_ids.append(track_id)
            for track_id in lost_ids:
                del self.tracks[track_id]
            return []

        # 1. Prepare cost matrix (IoU based)
        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(person_detections)

        matched_tracks = set()
        matched_dets = set()

        if num_tracks > 0:
            iou_matrix = np.zeros((num_tracks, num_dets))
            for i, track_id in enumerate(track_ids):
                for j, det in enumerate(person_detections):
                    iou_matrix[i, j] = calculate_iou(self.tracks[track_id]['box'], det['box'])

            # Simple greedy matching
            while True:
                # Find maximum IoU
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_val = iou_matrix[max_idx]

                if max_val < self.iou_threshold:
                    break

                i, j = max_idx
                track_id = track_ids[i]

                if track_id not in matched_tracks and j not in matched_dets:
                    # Match found
                    matched_tracks.add(track_id)
                    matched_dets.add(j)
                    
                    # Update track state
                    self.tracks[track_id]['box'] = person_detections[j]['box']
                    self.tracks[track_id]['lost_count'] = 0
                    
                    det_copy = person_detections[j].copy()
                    det_copy['track_id'] = track_id
                    updated_detections.append(det_copy)

                # Set this row/col to zero to find next best match
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        # 2. Register unmatched detections as new tracks
        for j, det in enumerate(person_detections):
            if j not in matched_dets:
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'box': det['box'],
                    'lost_count': 0
                }
                
                det_copy = det.copy()
                det_copy['track_id'] = track_id
                updated_detections.append(det_copy)

        # 3. Clean up lost tracks
        unmatched_tracks = set(track_ids) - matched_tracks
        lost_ids = []
        for track_id in unmatched_tracks:
            self.tracks[track_id]['lost_count'] += 1
            if self.tracks[track_id]['lost_count'] > self.max_lost:
                lost_ids.append(track_id)
        for track_id in lost_ids:
            del self.tracks[track_id]

        return updated_detections
