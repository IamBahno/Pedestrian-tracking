import cv2
import numpy as np
from ultralytics import YOLO
from bytetrack import ByteTrack
from deepsort import DeepSORT
import argparse
import time
import os

class PedestrianTracker:
    def __init__(self, 
                 model_path="best.pt",
                 tracker_type="bytetrack",
                 track_thresh=0.5,
                 high_thresh=0.6,
                 match_thresh=0.8,
                 conf_thresh=0.25,
                 iou_thresh=0.45,
                 imgsz=320):
        
        print(f"Loading YOLO model from {model_path}")
        self.yolo_model = YOLO(model_path)
        self.tracker_type = tracker_type.lower()
        print(f"Using tracker: {self.tracker_type}")
        
        if self.tracker_type == "deepsort":
            self.tracker = DeepSORT(track_thresh=track_thresh, match_thresh=match_thresh)
        else:
            self.tracker = ByteTrack(track_thresh=track_thresh, high_thresh=high_thresh, match_thresh=match_thresh)
        
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        
        self.frame_count = 0
        self.total_tracks = 0
        self.active_tracks = 0
        
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        self.last_fps_update = time.time()
        
    # === DETEKCE ===
    
    def detect_pedestrians(self, frame):
        results = self.yolo_model(frame, 
                                conf=self.conf_thresh,
                                iou=self.iou_thresh,
                                imgsz=320,
                                device='cpu',
                                half=False,
                                verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    x, y, w, h = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
                    
                    if w <= 0 or h <= 0 or x < 0 or y < 0:
                        continue
                    
                    if class_id == 0:
                        detections.append([x, y, w, h, confidence, class_id])
        
        return detections
    
    # === TRACKOVANIE ===
    
    def track_pedestrians(self, detections, frame=None):
        return self.tracker.update(detections, frame)
    
    # === SPRACOVANIE FRAMU ===
    
    def process_frame(self, frame):
        current_time = time.time()
        self.frame_count += 1
        self.fps_frame_count += 1
        
        if self.fps_frame_count >= 10:
            elapsed = current_time - self.last_fps_update
            if elapsed > 0:
                self.current_fps = self.fps_frame_count / elapsed
            self.last_fps_update = current_time
            self.fps_frame_count = 0
        
        resized_frame = cv2.resize(frame, (1280, 720))
        
        detections = self.detect_pedestrians(resized_frame)
        
        if self.tracker_type == "deepsort":
            tracks = self.track_pedestrians(detections, resized_frame)
        else:
            tracks = self.track_pedestrians(detections)
        
        self.active_tracks = len(tracks)
        self.total_tracks = max(self.total_tracks, self.tracker.next_id - 1)
        
        annotated_frame = self.draw_tracks(resized_frame, tracks)
        
        return annotated_frame, tracks
    
    def draw_tracks(self, frame, tracks):
        # Nakresli obdlzniky a ID na frame
        annotated_frame = frame.copy()
        
        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            confidence = track['confidence']
            age = track['age']
            hits = track['hits']
            
            x, y, w, h = map(int, bbox)
            
            color = self.get_track_color(track_id)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"ID:{track_id} ({hits})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(annotated_frame, 
                         (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), 
                         color, -1)
            
            cv2.putText(annotated_frame, label, 
                       (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        info_text = f"Frame: {self.frame_count} | FPS: {self.current_fps:.1f} | Active: {self.active_tracks} | Track: {self.total_tracks}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        tracker_text = f"Tracker: {self.tracker_type.upper()}"
        cv2.putText(annotated_frame, tracker_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        fps_text = f"FPS: {self.current_fps:.1f}"
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        fps_x = annotated_frame.shape[1] - fps_size[0] - 10
        cv2.putText(annotated_frame, fps_text, (fps_x, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return annotated_frame
    
    def get_track_color(self, track_id):
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
            (255, 192, 203), (0, 128, 0), (128, 128, 0), (0, 0, 128)
        ]
        return colors[track_id % len(colors)]
    
    # === SPRACOVANIE VIDEA ===
    
    def process_video(self, input_path, output_path=None, display=True):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        start_time = time.time()
        frame_delay = int(1000 / fps)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video reached at frame {self.frame_count}")
                    break
                
                try:
                    annotated_frame, tracks = self.process_frame(frame)
                except Exception as e:
                    print(f"Error processing frame {self.frame_count}: {e}")
                    break
                
                if out:
                    out.write(annotated_frame)
                
                if display:
                    cv2.imshow('Pedestrian Tracking', annotated_frame)
                    key = cv2.waitKey(frame_delay) & 0xFF
                    if key == ord('q'):
                        break
                
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = self.frame_count / elapsed
                    print(f"Frame {self.frame_count}/{total_frames} | "
                          f"FPS: {fps_actual:.1f} | "
                          f"Active tracks: {self.active_tracks}")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            print(f"\nProcessing complete!")
            print(f"Total frames: {self.frame_count}")
            print(f"Total time: {elapsed:.2f}s")
            print(f"Average FPS: {self.frame_count/elapsed:.2f}")
            print(f"Total unique tracks: {self.total_tracks}")
    

def main():
    parser = argparse.ArgumentParser(description='Pedestrian Tracking with YOLO + ByteTrack/DeepSORT - Video Processing Only')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, help='Output video path (optional)')
    parser.add_argument('--model', type=str, default='best.pt', help='YOLO model path (best.pt or last.pt)')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    
    parser.add_argument('--tracker', type=str, default='bytetrack', 
                       choices=['bytetrack', 'deepsort'],
                       help='Tracking algorithm: bytetrack (default) or deepsort')
    parser.add_argument('--track-thresh', type=float, default=0.5, help='Track confidence threshold')
    parser.add_argument('--high-thresh', type=float, default=0.6, help='High confidence threshold (ByteTrack only)')
    parser.add_argument('--match-thresh', type=float, default=0.8, help='IoU match threshold')
    parser.add_argument('--conf-thresh', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--iou-thresh', type=float, default=0.45, help='YOLO IoU threshold')
    parser.add_argument('--imgsz', type=int, default=320, help='YOLO image size (smaller = faster)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input video file '{args.input}' not found!")
        return
    
    tracker = PedestrianTracker(
        model_path=args.model,
        tracker_type=args.tracker,
        track_thresh=args.track_thresh,
        high_thresh=args.high_thresh,
        match_thresh=args.match_thresh,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        imgsz=args.imgsz
    )
    
    tracker.process_video(args.input, args.output, not args.no_display)

if __name__ == "__main__":
    main()
