import numpy as np
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
import cv2
from tracker_base import TrackerUtils
from reid_model import ReIDExtractor

class DeepSORT:
    def __init__(self, 
                 track_thresh=0.5, 
                 match_thresh=0.5,
                 max_age=30,
                 n_init=3,
                 max_iou_distance=0.7):
        
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        
        self.tracked_tracks = OrderedDict()
        self.frame_id = 0
        self.next_id = 1
        
        self.utils = TrackerUtils()
        self.kalman_filter = self.utils.create_kalman_filter(8, 4, 'deepsort')
        
        self.feature_dim = 128
        self.reid_extractor = ReIDExtractor(device='cpu')
        self.current_frame = None
    
    # === KONVERZIE FORMATOV ===
    
    def _tlwh_to_xyah(self, tlwh):
        return self.utils.tlwh_to_xyah(tlwh)
    
    def _xyah_to_tlwh(self, xyah):
        return self.utils.xyah_to_tlwh(xyah)
    
    # === VYPOCTY METRIK ===
    
    def _iou(self, box1, box2):
        return self.utils.iou(box1, box2)
    
    # === REID FEATURES ===
    
    def _extract_features(self, detection, frame=None):
        x, y, w, h = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
        
        if frame is not None and w > 0 and h > 0 and x >= 0 and y >= 0:
            roi = frame[y:y+h, x:x+w]
            
            if roi.size > 0 and roi.shape[0] > 10 and roi.shape[1] > 10:
                try:
                    features = self.reid_extractor.extract(roi)
                    return features
                except:
                    pass
        
        aspect_ratio = w / h if h > 0 else 1.0
        area = w * h
        
        features = np.zeros(self.feature_dim, dtype=np.float32)
        features[0] = w / 1000.0
        features[1] = h / 1000.0
        features[2] = aspect_ratio
        features[3] = area / 100000.0
        features[4:] = np.random.randn(self.feature_dim - 4).astype(np.float32) * 0.01
        
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features
    
    def _cosine_distance(self, feat1, feat2):
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        return 1.0 - (dot_product / (norm1 * norm2))
    
    # === MATCHING ===
    
    def _linear_assignment(self, cost_matrix):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_a = []
        unmatched_b = []
        
        matched_rows = set()
        matched_cols = set()
        
        for d in range(len(row_indices)):
            row, col = row_indices[d], col_indices[d]
            matched_rows.add(row)
            matched_cols.add(col)
        
        for row in range(cost_matrix.shape[0]):
            if row not in matched_rows:
                unmatched_a.append(row)
        
        for col in range(cost_matrix.shape[1]):
            if col not in matched_cols:
                unmatched_b.append(col)
        
        return np.array(list(zip(row_indices, col_indices))), unmatched_a, unmatched_b
    
    # === SPRACOVANIE TRACKOV ===
    
    def _init_track(self, detection):
        xyah = self._tlwh_to_xyah(detection[:4])
        
        kalman = cv2.KalmanFilter(8, 4)
        kalman.transitionMatrix = self.kalman_filter.transitionMatrix.copy()
        kalman.measurementMatrix = self.kalman_filter.measurementMatrix.copy()
        kalman.processNoiseCov = self.kalman_filter.processNoiseCov.copy()
        kalman.measurementNoiseCov = self.kalman_filter.measurementNoiseCov.copy()
        kalman.errorCovPost = self.kalman_filter.errorCovPost.copy()
        
        initial_state = np.array([xyah[0], xyah[1], xyah[2], xyah[3], 0, 0, 0, 0], dtype=np.float32)
        kalman.statePre = initial_state.copy()
        kalman.statePost = initial_state.copy()
        
        kalman.errorCovPre = np.eye(8, dtype=np.float32) * 1000
        kalman.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
        
        features = self._extract_features(detection, self.current_frame)
        
        track = {
            'id': self.next_id,
            'mean': xyah,
            'covariance': np.eye(4, dtype=np.float32) * 10.0,
            'time_since_update': 0,
            'hits': 1,
            'hit_streak': 1,
            'age': 1,
            'is_confirmed': False,
            'n_init': 0,
            'kalman': kalman,
            'features': features
        }
        
        self.next_id += 1
        return track
    
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
        track['n_init'] += 1
        
        # Aktualizuje ReID features EMA
        new_features = self._extract_features(detection, self.current_frame)
        alpha = 0.1
        track['features'] = (1 - alpha) * track['features'] + alpha * new_features
        norm = np.linalg.norm(track['features'])
        if norm > 0:
            track['features'] = track['features'] / norm
        
        if track['n_init'] >= self.n_init:
            track['is_confirmed'] = True
    
    def _predict(self, track):
        track['kalman'].predict()
        track['age'] += 1
        track['time_since_update'] += 1
    
    # === HLAVNA LOGIKA TRACKOVANIA ===
    
    def update(self, detections, frame=None):
        self.frame_id += 1
        self.current_frame = frame
        
        if len(detections) == 0:
            detections = np.empty((0, 6))
        else:
            detections = np.array(detections)
        
        detections = detections[detections[:, 4] >= self.track_thresh]
        
        # Predikuje existujuce tracky
        for track_id, track in list(self.tracked_tracks.items()):
            self._predict(track)
        
        # Rozdeluje tracky na confirmed/unconfirmed
        confirmed_tracks = []
        unconfirmed_tracks = []
        
        for track_id, track in self.tracked_tracks.items():
            if track['is_confirmed']:
                confirmed_tracks.append(track_id)
            else:
                unconfirmed_tracks.append(track_id)
        
        # Etapa 1: Matchuje confirmed tracky
        matches, unmatched_tracks, unmatched_dets = [], list(range(len(confirmed_tracks))), list(range(len(detections)))
        
        if len(confirmed_tracks) > 0 and len(detections) > 0:
            confirmed_track_states = []
            confirmed_track_features = []
            for track_id in confirmed_tracks:
                state = self.tracked_tracks[track_id]['mean']
                tlwh = self._xyah_to_tlwh(state)
                confirmed_track_states.append(tlwh)
                confirmed_track_features.append(self.tracked_tracks[track_id]['features'])
            
            cost_matrix = np.zeros((len(confirmed_track_states), len(detections)), dtype=np.float32)
            
            for i, (trk, trk_features) in enumerate(zip(confirmed_track_states, confirmed_track_features)):
                for j, det in enumerate(detections):
                    iou_dist = 1.0 - self._iou(trk, det[:4])
                    det_features = self._extract_features(det, self.current_frame)
                    cosine_dist = self._cosine_distance(trk_features, det_features)
                    lambda_param = 0.5
                    cost_matrix[i, j] = lambda_param * iou_dist + (1 - lambda_param) * cosine_dist
            
            matches, unmatched_tracks_tmp, unmatched_dets_tmp = self._linear_assignment(cost_matrix)
            
            if len(matches) > 0:
                matches = matches.tolist()
            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_tracks_tmp]
            unmatched_dets = [unmatched_dets[i] for i in unmatched_dets_tmp]
            
            for match in matches:
                i, j = match[0], match[1]
                track_id = confirmed_tracks[i]
                self._update_track(self.tracked_tracks[track_id], detections[j])
        
        # Etapa 2: Matchuje unconfirmed tracky
        unmatched_detections = detections[unmatched_dets] if len(unmatched_dets) > 0 else np.empty((0, 6))
        remaining_detections = unmatched_detections
        
        if len(unconfirmed_tracks) > 0 and len(unmatched_detections) > 0:
            unconfirmed_track_states = []
            unconfirmed_track_features = []
            for track_id in unconfirmed_tracks:
                state = self.tracked_tracks[track_id]['mean']
                tlwh = self._xyah_to_tlwh(state)
                unconfirmed_track_states.append(tlwh)
                unconfirmed_track_features.append(self.tracked_tracks[track_id]['features'])
            
            unmatched_tracks_indices = list(range(len(unconfirmed_tracks)))
            unmatched_dets_indices_2 = list(range(len(unmatched_detections)))
            
            cost_matrix = np.zeros((len(unconfirmed_track_states), len(unmatched_detections)), dtype=np.float32)
            
            for i, (trk, trk_features) in enumerate(zip(unconfirmed_track_states, unconfirmed_track_features)):
                for j, det in enumerate(unmatched_detections):
                    iou_dist = 1.0 - self._iou(trk, det[:4])
                    det_features = self._extract_features(det, self.current_frame)
                    cosine_dist = self._cosine_distance(trk_features, det_features)
                    lambda_param = 0.5
                    cost_matrix[i, j] = lambda_param * iou_dist + (1 - lambda_param) * cosine_dist
            
            matches_2, unmatched_tracks_tmp_2, unmatched_dets_tmp_2 = self._linear_assignment(cost_matrix)
            
            if len(matches_2) > 0:
                matches_2 = matches_2.tolist()
            unmatched_tracks_indices_final = [unmatched_tracks_indices[i] for i in unmatched_tracks_tmp_2]
            unmatched_dets_indices_final = [unmatched_dets_indices_2[i] for i in unmatched_dets_tmp_2]
            
            for match in matches_2:
                i, j = match[0], match[1]
                track_id = unconfirmed_tracks[i]
                self._update_track(self.tracked_tracks[track_id], unmatched_detections[j])
            
            remaining_detections = unmatched_detections[unmatched_dets_indices_final] if len(unmatched_dets_indices_final) > 0 else np.empty((0, 6))
        
        # Vytvara nove tracky
        for det in remaining_detections:
            if len(det) > 0:
                track = self._init_track(det)
                self.tracked_tracks[track['id']] = track
        
        # Odstranuje stare tracky
        tracks_to_remove = []
        for track_id, track in self.tracked_tracks.items():
            if track['time_since_update'] > self.max_age:
                tracks_to_remove.append(track_id)
            elif not track['is_confirmed'] and track['time_since_update'] > 3:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_tracks[track_id]
        
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
