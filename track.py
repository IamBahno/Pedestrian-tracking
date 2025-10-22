#!/usr/bin/env python3
import cv2
import argparse
from pedestrian_tracker import PedestrianTracker
import os

def main():
    parser = argparse.ArgumentParser(description='Pedestrian tracking in video')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--model', type=str, default='best.pt', 
                       choices=['best.pt', 'last.pt'], 
                       help='Model selection: best.pt (recommended) or last.pt')
    parser.add_argument('--conf-thresh', type=float, default=0.25, 
                       help='YOLO confidence threshold')
    parser.add_argument('--imgsz', type=int, default=320, 
                       help='Image size for YOLO (smaller = faster)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} not found!")
        return
    
    print("Press 'q' to quit, 'p' to pause/resume")
    
    tracker = PedestrianTracker(
        model_path=args.model,
        conf_thresh=args.conf_thresh,
        imgsz=args.imgsz
    )
    
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.input}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    frame_delay = int(1000 / fps)
    
    paused = False
    frame_count = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video")
                    break
                
                try:
                    annotated_frame, tracks = tracker.process_frame(frame)
                    frame_count += 1
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    break
            else:
                if 'annotated_frame' in locals():
                    pass
                else:
                    continue
            
            display_frame = cv2.resize(annotated_frame, (1280, 720))
            
            cv2.imshow('Pedestrian Tracking - Press Q to quit, P to pause', display_frame)
            
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}/{total_frames} | FPS: {tracker.current_fps:.1f} | Active tracks: {tracker.active_tracks}")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
