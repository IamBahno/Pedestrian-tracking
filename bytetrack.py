import numpy as np
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
import cv2
from tracker_base import TrackerUtils

class ByteTrack:    
    def __init__(self, 
                 track_thresh=0.5, 
                 high_thresh=0.6, 
                 match_thresh=0.8,
                 frame_rate=30,
                 track_buffer=30):

        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.track_buffer = track_buffer
        
        self.tracked_tracks = OrderedDict()
        self.lost_tracks = OrderedDict()
        self.removed_tracks = OrderedDict()
        self.frame_id = 0
        self.next_id = 1
        
        self.utils = TrackerUtils()
        self.kalman_filter = self.utils.create_kalman_filter(7, 4, 'bytetrack')
    
    # === KONVERZIE FORMATOV ===
    
    def _tlwh_to_xyah(self, tlwh):
        return self.utils.tlwh_to_xyah(tlwh)
    
    def _xyah_to_tlwh(self, xyah):
        return self.utils.xyah_to_tlwh(xyah)
    
    # === VYPOCTY METRIK ===
    
    def _iou(self, box1, box2):
        return self.utils.iou(box1, box2)
    
    def _linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        
        cost_matrix[cost_matrix > thresh] = thresh + 1e-4
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_a = []
        unmatched_b = []
        
        for d, cost in enumerate(cost_matrix[row_indices, col_indices]):
            if cost <= thresh:
                matches.append([row_indices[d], col_indices[d]])
            else:
                unmatched_a.append(row_indices[d])
                unmatched_b.append(col_indices[d])
        
        unmatched_a += [i for i in range(cost_matrix.shape[0]) if i not in row_indices]
        unmatched_b += [i for i in range(cost_matrix.shape[1]) if i not in col_indices]
        
        return np.array(matches), unmatched_a, unmatched_b
    
    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.5):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk)
        
        matched_indices, unmatched_detections, unmatched_trackers = self._linear_assignment(
            -iou_matrix, -iou_threshold
        )
        
        return matched_indices, unmatched_detections, unmatched_trackers
    
    # === SPRACOVANIE TRACKOV ===
    
    def _update_track(self, track, detection):
        xyah = self._tlwh_to_xyah(detection[:4])
        
        measurement = np.array([xyah[0], xyah[1], xyah[2], xyah[3]], dtype=np.float32)
        track['kalman'].correct(measurement)
        
        state = track['kalman'].statePost
        track['mean'] = state[:4]
        track['covariance'] = track['kalman'].errorCovPost[:4, :4]
        track['time_since_update'] = 0
        track['hits'] += 1
        track['hit_streak'] += 1
    
    def _init_track(self, detection):
        xyah = self._tlwh_to_xyah(detection[:4])
        
        kalman = cv2.KalmanFilter(7, 4)
        kalman.transitionMatrix = self.kalman_filter.transitionMatrix.copy()
        kalman.measurementMatrix = self.kalman_filter.measurementMatrix.copy()
        kalman.processNoiseCov = self.kalman_filter.processNoiseCov.copy()
        kalman.measurementNoiseCov = self.kalman_filter.measurementNoiseCov.copy()
        kalman.errorCovPost = self.kalman_filter.errorCovPost.copy()
        
        initial_state = np.array([xyah[0], xyah[1], xyah[2], xyah[3], 0, 0, 0], dtype=np.float32)
        kalman.statePre = initial_state.copy()
        kalman.statePost = initial_state.copy()
        
        kalman.errorCovPre = np.eye(7, dtype=np.float32) * 1000
        kalman.errorCovPost = np.eye(7, dtype=np.float32) * 1000
        
        track = {
            'id': self.next_id,
            'mean': xyah,
            'covariance': np.eye(4, dtype=np.float32),
            'time_since_update': 0,
            'hits': 1,
            'hit_streak': 1,
            'age': 1,
            'kalman': kalman
        }
        
        self.next_id += 1
        return track
    
    # === HLAVNA LOGIKA TRACKOVANIA ===
    
    def update(self, detections, frame=None):
        self.frame_id += 1
        
        if len(detections) == 0:
            detections = np.empty((0, 6))
        else:
            detections = np.array(detections)
        
        # Rozdeluje detekcie podla confidence thresholdu
        high_conf_detections = detections[detections[:, 4] >= self.high_thresh]
        low_conf_detections = detections[detections[:, 4] >= self.track_thresh]
        low_conf_detections = low_conf_detections[low_conf_detections[:, 4] < self.high_thresh]
        
        # Predikuje pozicie existujucich trackov
        for track_id, track in self.tracked_tracks.items():
            track['kalman'].predict()
            track['age'] += 1
            track['time_since_update'] += 1
        
        track_states = []
        for track_id, track in self.tracked_tracks.items():
            state = track['mean']
            tlwh = self._xyah_to_tlwh(state)
            track_states.append(tlwh)
        
        # Etapa 1: Matchuje high-confidence detekcie s existujucimi trackmi
        if len(high_conf_detections) > 0 and len(track_states) > 0:
            matches, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
                high_conf_detections[:, :4], track_states, self.match_thresh
            )
            
            tracked_track_ids = list(self.tracked_tracks.keys())
            for match in matches:
                det_idx, trk_idx = match
                if trk_idx < len(tracked_track_ids):
                    track_id = tracked_track_ids[trk_idx]
                    self._update_track(self.tracked_tracks[track_id], high_conf_detections[det_idx])
            
            high_conf_detections = high_conf_detections[unmatched_dets]
            
            for trk_idx in unmatched_trks:
                if trk_idx < len(tracked_track_ids):
                    track_id = tracked_track_ids[trk_idx]
                    self.lost_tracks[track_id] = self.tracked_tracks[track_id]
                    del self.tracked_tracks[track_id]
        
        # Etapa 2: Matchuje low-confidence detekcie s lost trackmi
        if len(low_conf_detections) > 0 and len(self.lost_tracks) > 0:
            lost_track_states = []
            lost_track_ids = list(self.lost_tracks.keys())
            for track_id, track in self.lost_tracks.items():
                state = track['mean']
                tlwh = self._xyah_to_tlwh(state)
                lost_track_states.append(tlwh)
            
            matches, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
                low_conf_detections[:, :4], lost_track_states, self.match_thresh
            )
            
            for match in matches:
                det_idx, trk_idx = match
                if trk_idx < len(lost_track_ids):
                    track_id = lost_track_ids[trk_idx]
                    self._update_track(self.lost_tracks[track_id], low_conf_detections[det_idx])
                    self.tracked_tracks[track_id] = self.lost_tracks[track_id]
                    del self.lost_tracks[track_id]
        
        # Vytvara nove tracky z unmatched high-conf detekci
        for det in high_conf_detections:
            track = self._init_track(det)
            self.tracked_tracks[track['id']] = track
        
        # Odstraňuje predlhe tracky z lost poolu
        for track_id, track in list(self.lost_tracks.items()):
            if track['time_since_update'] > self.track_buffer:
                self.removed_tracks[track_id] = track
                del self.lost_tracks[track_id]
        
        # Vracia aktivne tracky
        active_tracks = []
        for track_id, track in self.tracked_tracks.items():
            if track['time_since_update'] == 0:
                state = track['mean']
                tlwh = self._xyah_to_tlwh(state)
                active_tracks.append({
                    'id': track_id,
                    'bbox': tlwh,
                    'confidence': 1.0,
                    'age': track['age'],
                    'hits': track['hits']
                })
        
        return active_tracks
