import numpy as np
import cv2


class TrackerUtils:
    # === KONVERZIE FORMATOV BOXOV ===
    
    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2] = ret[2] / ret[3]
        ret[0] = ret[0] + ret[2] * ret[3] / 2
        ret[1] = ret[1] + ret[3] / 2
        return ret
    
    @staticmethod
    def xyah_to_tlwh(xyah):
        ret = np.asarray(xyah).copy()
        ret[2] = ret[2] * ret[3]
        ret[0] = ret[0] - ret[2] / 2
        ret[1] = ret[1] - ret[3] / 2
        return ret
    
    # === IoU VYPOCTY ===
    
    @staticmethod
    def iou(box1, box2):
        x1_1, y1_1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
        x1_2, y1_2, w2, h2 = box2[0], box2[1], box2[2], box2[3]
        
        x2_1 = x1_1 + w1
        y2_1 = y1_1 + h1
        x2_2 = x1_2 + w2
        y2_2 = y1_2 + h2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # === KALMAN FILTER ===
    
    @staticmethod
    def create_kalman_filter(state_dim=8, measurement_dim=4, kalman_type='deepsort'):
        kalman = cv2.KalmanFilter(state_dim, measurement_dim)
        
        if kalman_type == 'deepsort':
            kalman.transitionMatrix = np.array([
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]
            ], dtype=np.float32)
            
            kalman.measurementMatrix = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0]
            ], dtype=np.float32)
            
            kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.2
            kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
            kalman.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
            
        else:  # bytetrack
            kalman.transitionMatrix = np.array([
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1]
            ], dtype=np.float32)
            
            kalman.measurementMatrix = np.array([
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0]
            ], dtype=np.float32)
            
            kalman.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
            kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
            kalman.errorCovPost = np.eye(7, dtype=np.float32)
        
        return kalman
