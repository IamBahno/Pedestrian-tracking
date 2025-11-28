from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


    
def test_model_on_datasets():
    """Test all models on both datasets and generate comparison graphs"""
    print("🧪 Testing models on datasets...")
    
    models = {
        "Model 1": "models/dataset1/weights/best.pt",
        "Model 2": "models/dataset2/weights/best.pt", 
        "Model 3": "models/combined/weights/best.pt"
    }
    
    results = {}
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"❌ Model {model_name} not found at {model_path}")
            results[model_name] = {"dataset1": None, "dataset2": None}
            continue
            
        print(f"✅ Loading {model_name} from {model_path}")
        model = YOLO(model_path)
        results[model_name] = {}
        
        # Test on both datasets
        datasets = {
            "dataset1": "datasets/Citypersons_yolo/data.yaml",
            "dataset2": "datasets/widerperson_yolo/data.yaml"
        }
        
        for dataset_name, data_config in datasets.items():
            if not os.path.exists(data_config):
                print(f"❌ Dataset config not found: {data_config}")
                results[model_name][dataset_name] = None
                continue
                
            print(f"Testing {model_name} on {dataset_name}...")
            
            try:
                # Run validation
                metrics = model.val(data=data_config, split='test', verbose=False)
                
                # Extract metrics using the correct attribute names for newer Ultralytics versions
                results[model_name][dataset_name] = {
                    'map50': metrics.box.map50,  # mAP@0.5
                    'map': metrics.box.map,      # mAP@0.5:0.95
                    'precision': metrics.box.p[0] if hasattr(metrics.box, 'p') and len(metrics.box.p) > 0 else 0.5,  # Use 'p' for precision
                    'recall': metrics.box.r[0] if hasattr(metrics.box, 'r') and len(metrics.box.r) > 0 else 0.5     # Use 'r' for recall
                }
                print(f"✅ {model_name} on {dataset_name}: mAP50 = {metrics.box.map50:.3f}")
            except Exception as e:
                print(f"❌ Error testing {model_name} on {dataset_name}: {e}")
                results[model_name][dataset_name] = None
    
    # Generate comparison graphs
    generate_comparison_graphs(results)
    return results

def generate_comparison_graphs(results):
    """Generate clean comparison graphs for 3 models × 2 datasets."""
    print("📊 Generating comparison graphs...")

    os.makedirs("output/graphs", exist_ok=True)

    models = list(results.keys())
    datasets = ['dataset1', 'dataset2']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    x = np.arange(len(models))
    width = 0.35

    # ---------------------------------------------------------
    # 1) mAP50 comparison (dataset1 and dataset2)
    # ---------------------------------------------------------
    map50_data = {ds: [results[m][ds]['map50'] for m in models] for ds in datasets}

    for i, ds in enumerate(datasets):
        axes[0,0].bar(x + i*width, map50_data[ds], width, label=ds)

    axes[0,0].set_title('mAP50 Comparison')
    axes[0,0].set_xticks(x + width / 2)
    axes[0,0].set_xticklabels(models)
    axes[0,0].set_ylabel('mAP50')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 1)

    # ---------------------------------------------------------
    # 2) mAP comparison (dataset1 and dataset2)
    # ---------------------------------------------------------
    map_data = {ds: [results[m][ds]['map'] for m in models] for ds in datasets}

    for i, ds in enumerate(datasets):
        axes[0,1].bar(x + i*width, map_data[ds], width, label=ds)

    axes[0,1].set_title('mAP Comparison')
    axes[0,1].set_xticks(x + width / 2)
    axes[0,1].set_xticklabels(models)
    axes[0,1].set_ylabel('mAP')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 1)

    # ---------------------------------------------------------
    # 3) Precision & Recall – Dataset 1
    # ---------------------------------------------------------
    p1 = [results[m]['dataset1']['precision'] for m in models]
    r1 = [results[m]['dataset1']['recall'] for m in models]

    # Set different colors for Precision and Recall bars
    precision_color = 'red'
    recall_color = 'green'

    axes[1,0].bar(x - width/2, p1, width, label='Precision', color=precision_color)
    axes[1,0].bar(x + width/2, r1, width, label='Recall', color=recall_color)

    axes[1,0].set_title('Precision & Recall (Dataset 1)')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(models)
    axes[1,0].set_ylabel('Score')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylim(0, 1)

    # Value labels
    for i, (p, r) in enumerate(zip(p1, r1)):
        axes[1,0].text(i - width/2, p + 0.01, f'{p:.3f}', ha='center', fontsize=8)
        axes[1,0].text(i + width/2, r + 0.01, f'{r:.3f}', ha='center', fontsize=8)

    # ---------------------------------------------------------
    # 4) Precision & Recall – Dataset 2
    # ---------------------------------------------------------
    p2 = [results[m]['dataset2']['precision'] for m in models]
    r2 = [results[m]['dataset2']['recall'] for m in models]

    # Set different colors for Precision and Recall bars
    axes[1,1].bar(x - width/2, p2, width, label='Precision', color=precision_color)
    axes[1,1].bar(x + width/2, r2, width, label='Recall', color=recall_color)

    axes[1,1].set_title('Precision & Recall (Dataset 2)')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(models)
    axes[1,1].set_ylabel('Score')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim(0, 1)

    # Value labels
    for i, (p, r) in enumerate(zip(p2, r2)):
        axes[1,1].text(i - width/2, p + 0.01, f'{p:.3f}', ha='center', fontsize=8)
        axes[1,1].text(i + width/2, r + 0.01, f'{r:.3f}', ha='center', fontsize=8)

    # ---------------------------------------------------------
    # Save & summary
    # ---------------------------------------------------------
    plt.tight_layout()
    plt.savefig('output/graphs/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('output/graphs/model_comparison.pdf', bbox_inches='tight')
    plt.close()

    create_summary_table(results, models)

    print("✅ Comparison graphs generated!")

    print("✅ Comparison graphs generated!")



def create_summary_table(results, valid_models):
    """Create a summary table of results"""
    print("📋 Creating summary table...")
    
    # Create a text summary
    summary_path = "output/graphs/model_comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Model Comparison Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for model_name in valid_models:
            f.write(f"{model_name}:\n")
            f.write("-" * 20 + "\n")
            
            for dataset in ['dataset1', 'dataset2']:
                if results[model_name].get(dataset) is not None:
                    data = results[model_name][dataset]
                    f.write(f"  {dataset}:\n")
                    f.write(f"    mAP50: {data['map50']:.3f}\n")
                    f.write(f"    mAP: {data['map']:.3f}\n")
                    f.write(f"    Precision: {data['precision']:.3f}\n")
                    f.write(f"    Recall: {data['recall']:.3f}\n")
                else:
                    f.write(f"  {dataset}: No data\n")
            f.write("\n")
    
    print(f"✅ Summary table saved to {summary_path}")












def create_comparison_video():
    """Create 3x3 comparison video with all models and trackers"""
    print("🎥 Creating comparison video...")
    
    models = {
        "Model 1": "models/dataset1/weights/best.pt",
        "Model 2": "models/dataset2/weights/best.pt", 
        "Model 3": "models/combined/weights/best.pt"
    }
    
    trackers = ["bytetrack", "deepsort"]
    
    test_video = "input/video.mp4"
    output_video = "output/comparison_grid.mp4"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video {test_video} not found!")
        return
    
    # Initialize trackers
    from trackers.pedestrian_tracker import PedestrianTracker
    
    # Create tracker instances - MODELS BY COLUMN, TRACKERS BY ROWS
    tracker_instances = {}
    for model_name, model_path in models.items():
        if os.path.exists(model_path):
            for tracker_type in trackers:
                key = f"{model_name}_{tracker_type}"
                try:
                    tracker_instances[key] = PedestrianTracker(
                        model_path=model_path,
                        tracker_type=tracker_type,
                        conf_thresh=0.3
                    )
                    print(f"✅ Loaded {key}")
                except Exception as e:
                    print(f"❌ Failed to load {key}: {e}")
    
    if not tracker_instances:
        print("❌ No trackers loaded successfully!")
        return
    
    # Process video
    cap = cv2.VideoCapture(test_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Input video: {original_width}x{original_height}, {fps} FPS")
    
    # Define grid dimensions for 4K output (3840x2160)
    GRID_COLS = 3
    GRID_ROWS = 3
    OUTPUT_WIDTH = 3840
    OUTPUT_HEIGHT = 2160
    

    cell_width = OUTPUT_WIDTH // GRID_COLS
    cell_height = OUTPUT_HEIGHT // GRID_ROWS
    
    print(f"Grid layout: {GRID_COLS}x{GRID_ROWS}, Cell size: {cell_width}x{cell_height}")
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    
    frame_count = 0
    max_frames = 10000
    
    # Statistics collection with separate max tracking
    stats = {
        'frame_data': [],
        'fps_data': [],
        'detection_history': {key: [] for key in tracker_instances.keys()},
        'tracker_counts': {},
        'model_max': 0.0,  # Separate max for model comparison
        'tracker_max': 0.0,  # Separate max for tracker comparison
        'system_max': 0,  # Max for system performance (always >= current)
        'all_detections_history': [],  # Store all detections for average
        'start_time': datetime.now()
    }
    
    # Warm up models
    print("🔥 Warming up models...")
    ret, warmup_frame = cap.read()
    if ret:
        for key, tracker in tracker_instances.items():
            try:
                tracker.process_frame(warmup_frame)
                print(f"✅ {key} warmed up")
            except Exception as e:
                print(f"❌ {key} warmup failed: {e}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("🚀 Starting video processing...")
    
    try:
        while cap.isOpened() and frame_count < max_frames:
            frame_start_time = datetime.now()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with all trackers
            frames_processed = {}
            tracks_data = {}
            
            for key, tracker in tracker_instances.items():
                try:
                    processed_frame, tracks = tracker.process_frame(frame.copy())
                    frames_processed[key] = processed_frame
                    tracks_data[key] = len(tracks)
                    # Store detection history
                    stats['detection_history'][key].append(len(tracks))
                except Exception as e:
                    print(f"❌ Error processing {key}: {e}")
                    frames_processed[key] = frame
                    tracks_data[key] = 0
                    stats['detection_history'][key].append(0)
            
            # Update tracker_counts for current frame
            stats['tracker_counts'] = tracks_data
            
            # Update system statistics
            current_total_detections = sum(tracks_data.values())
            stats['system_max'] = max(stats['system_max'], current_total_detections)
            stats['all_detections_history'].append(current_total_detections)
            
            # Create grid with MODELS BY COLUMN, TRACKERS BY ROWS layout
            grid_frame = create_grid_frame_fixed_layout(frames_processed, stats, frame_count, 
                                                      cell_width, cell_height, GRID_ROWS, GRID_COLS,
                                                      models, trackers)
            
            # Write frame
            out.write(grid_frame)
            
            # Calculate FPS
            frame_end_time = datetime.now()
            processing_time = (frame_end_time - frame_start_time).total_seconds()
            current_fps = 1.0 / processing_time if processing_time > 0 else 0
            stats['fps_data'].append(current_fps)
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                recent_fps = stats['fps_data'][-10:] if len(stats['fps_data']) >= 10 else stats['fps_data']
                avg_fps = np.mean(recent_fps) if recent_fps else 0
                current_avg_detections = np.mean(stats['all_detections_history'][-20:]) if stats['all_detections_history'] else 0
                print(f"📊 Frame {frame_count}: Current FPS: {current_fps:.1f}, Avg FPS: {avg_fps:.1f}")
                print(f"📈 Current avg detections: {current_avg_detections:.1f}, System max: {stats['system_max']}")
            
            # Collect statistics
            frame_data = {
                'frame': frame_count,
                'timestamp': datetime.now(),
                'tracker_counts': tracks_data,
            }
            stats['frame_data'].append(frame_data)
            
    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        out.release()
        
        total_time = (datetime.now() - stats['start_time']).total_seconds()
        overall_fps = frame_count / total_time if total_time > 0 else 0
        final_avg_detections = np.mean(stats['all_detections_history']) if stats['all_detections_history'] else 0
        
        print(f"✅ Comparison video saved to {output_video}")
        print(f"📊 Final statistics: {frame_count} frames processed")
        print(f"📊 Overall FPS: {overall_fps:.1f}")
        print(f"📈 Average detections: {final_avg_detections:.1f}")
        print(f"📈 System maximum detections: {stats['system_max']}")

def create_grid_frame_fixed_layout(frames_processed, stats, frame_count, cell_width, cell_height, grid_rows, grid_cols, models, trackers):
    """Create grid with MODELS BY COLUMN, TRACKERS BY ROWS layout"""
    grid = np.zeros((grid_rows * cell_height, grid_cols * cell_width, 3), dtype=np.uint8)
    
    # Layout: Models by column, Trackers by rows
    # Columns: Model 1, Model 2, Model 3
    # Rows: ByteTrack, DeepSORT, Statistics
    
    # Place tracker frames (2 rows × 3 columns)
    for model_idx, model_name in enumerate(models.keys()):
        for tracker_idx, tracker_type in enumerate(trackers):
            key = f"{model_name}_{tracker_type}"
            if key in frames_processed:
                frame = frames_processed[key]
                resized = resize_frame_to_fit(frame, cell_width, cell_height)
                
                row = tracker_idx  # 0=ByteTrack, 1=DeepSORT
                col = model_idx    # 0=Model1, 1=Model2, 2=Model3
                
                y_start = row * cell_height
                y_end = y_start + cell_height
                x_start = col * cell_width
                x_end = x_start + cell_width
                
                # Center the frame
                y_offset = (cell_height - resized.shape[0]) // 2
                x_offset = (cell_width - resized.shape[1]) // 2
                
                grid[y_start+y_offset:y_start+y_offset+resized.shape[0], 
                     x_start+x_offset:x_start+x_offset+resized.shape[1]] = resized
                
                # Add label - BOTTOM RIGHT CORNER (ONLY LABEL, NO DETECTION COUNT)
                label = f"{model_name} - {tracker_type}"
                text_scale = 1.2
                text_thickness = 3
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]
                
                # Position in bottom right corner with margin
                text_x = x_start + cell_width - text_size[0] - 20
                text_y = y_start + cell_height - 20
                
                # Background for text
                cv2.rectangle(grid, 
                             (text_x - 10, text_y - text_size[1] - 10),
                             (text_x + text_size[0] + 10, text_y + 10),
                             (0, 0, 0), -1)
                
                cv2.putText(grid, label, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)
    
    # Create statistics panels in bottom row (row 2)
    create_meaningful_statistics_panels(grid, stats, frame_count, cell_width, cell_height, models, trackers)
    
    return grid

def create_meaningful_statistics_panels(grid, stats, frame_count, cell_width, cell_height, models, trackers):
    """Create meaningful statistics panels in bottom row"""
    # Update max values for model and tracker comparisons
    update_comparison_max_values(stats, models, trackers)
    
    # Panel 1: Model Performance Comparison
    create_model_comparison_panel(grid, stats, frame_count, 0, cell_width, cell_height, models)
    
    # Panel 2: Tracker Performance Comparison  
    create_tracker_comparison_panel(grid, stats, frame_count, 1, cell_width, cell_height, trackers, models)
    
    # Panel 3: System Performance
    create_system_performance_panel(grid, stats, frame_count, 2, cell_width, cell_height)

def update_comparison_max_values(stats, models, trackers):
    """Update separate max values for model and tracker comparisons"""
    # Calculate current averages for models
    model_current_avgs = {}
    for model_name in models.keys():
        model_detections = []
        for tracker in ['bytetrack', 'deepsort']:
            key = f"{model_name}_{tracker}"
            if key in stats['detection_history']:
                recent_detections = stats['detection_history'][key][-20:]
                if recent_detections:
                    model_detections.extend(recent_detections)
        model_current_avgs[model_name] = np.mean(model_detections) if model_detections else 0.0
    
    # Update model max (float)
    current_model_max = max(model_current_avgs.values()) if model_current_avgs else 0.0
    stats['model_max'] = max(stats['model_max'], current_model_max)
    
    # Calculate current averages for trackers
    tracker_current_avgs = {}
    for tracker_type in trackers:
        tracker_detections = []
        for model_name in models.keys():
            key = f"{model_name}_{tracker_type}"
            if key in stats['detection_history']:
                recent_detections = stats['detection_history'][key][-20:]
                if recent_detections:
                    tracker_detections.extend(recent_detections)
        tracker_current_avgs[tracker_type] = np.mean(tracker_detections) if tracker_detections else 0.0
    
    # Update tracker max (float)
    current_tracker_max = max(tracker_current_avgs.values()) if tracker_current_avgs else 0.0
    stats['tracker_max'] = max(stats['tracker_max'], current_tracker_max)

def create_model_comparison_panel(grid, stats, frame_count, col, cell_width, cell_height, models):
    """Compare detection performance across models - SMALLER GRAPHS"""
    y_start = 2 * cell_height  # Bottom row
    x_start = col * cell_width
    
    panel = np.full((cell_height, cell_width, 3), [40, 40, 60], dtype=np.uint8)
    
    # Calculate average detections per model
    model_avg_detections = {}
    for model_name in models.keys():
        model_detections = []
        for tracker in ['bytetrack', 'deepsort']:
            key = f"{model_name}_{tracker}"
            if key in stats['detection_history']:
                recent_detections = stats['detection_history'][key][-20:]
                if recent_detections:
                    model_detections.extend(recent_detections)
        model_avg_detections[model_name] = np.mean(model_detections) if model_detections else 0.0
    
    # Draw bar chart - NORMALIZED TO MODEL MAX (FLOAT) with 10% smaller height
    max_detections = max(stats['model_max'], 1.0)  # Use model-specific max, at least 1.0
    chart_left_margin = 80
    chart_width = cell_width - 160
    chart_height = int((cell_height - 150) * 0.9)  # 10% smaller height
    
    # Dynamic bar sizing based on panel width
    num_models = len(models)
    bar_width = max(20, chart_width // (num_models * 3))  # Ensure minimum width of 20px
    bar_spacing = (chart_width - (num_models * bar_width)) // (num_models + 1)
    
    for i, (model_name, avg_detections) in enumerate(model_avg_detections.items()):
        x = chart_left_margin + bar_spacing + (i * (bar_width + bar_spacing))
        bar_height = int((avg_detections / max_detections) * chart_height)  # Use smaller chart height
        y = cell_height - 80 - bar_height
        
        color = [(255, 100, 100), (100, 255, 100), (100, 100, 255)][i % 3]
        cv2.rectangle(panel, (x, y), (x + bar_width, cell_height - 80), color, -1)
        
        # Model name and value
        model_short = model_name.replace("Model ", "M")
        name_y = cell_height - 50
        value_y = y - 20
        
        cv2.putText(panel, model_short, (x, name_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, f"{avg_detections:.1f}", (x, value_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add max value reference line and label
    max_line_y = cell_height - 80 - int((max_detections / max_detections) * chart_height)
    cv2.line(panel, (chart_left_margin - 10, max_line_y), 
             (cell_width - chart_left_margin + 10, max_line_y), (255, 255, 0), 2)
    cv2.putText(panel, f"Max: {max_detections:.1f}", (cell_width - 200, max_line_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.putText(panel, "Model Comparison", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(panel, "Avg Detections/Frame (Normalized)", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
    
    grid[y_start:y_start+cell_height, x_start:x_start+cell_width] = panel

def create_tracker_comparison_panel(grid, stats, frame_count, col, cell_width, cell_height, trackers, models):
    """Compare performance between ByteTrack and DeepSORT - SMALLER GRAPHS"""
    y_start = 2 * cell_height
    x_start = col * cell_width
    
    panel = np.full((cell_height, cell_width, 3), [40, 60, 40], dtype=np.uint8)
    
    # Calculate average detections per tracker type
    tracker_avg_detections = {}
    for tracker_type in trackers:
        tracker_detections = []
        for model_name in models.keys():
            key = f"{model_name}_{tracker_type}"
            if key in stats['detection_history']:
                recent_detections = stats['detection_history'][key][-20:]
                if recent_detections:
                    tracker_detections.extend(recent_detections)
        tracker_avg_detections[tracker_type] = np.mean(tracker_detections) if tracker_detections else 0.0
    
    # Draw comparison - NORMALIZED TO TRACKER MAX (FLOAT) with 10% smaller height
    max_detections = max(stats['tracker_max'], 1.0)  # Use tracker-specific max, at least 1.0
    chart_left_margin = 80  # Reset to normal margin for balanced layout
    chart_width = cell_width - 160
    chart_height = int((cell_height - 150) * 0.9)  # 10% smaller height
    
    # Dynamic bar sizing based on panel width
    num_trackers = len(trackers)
    bar_width = max(30, chart_width // (num_trackers * 2))  # Ensure minimum width of 30px
    bar_spacing = (chart_width - (num_trackers * bar_width)) // (num_trackers + 1)
    
    for i, (tracker_type, avg_detections) in enumerate(tracker_avg_detections.items()):
        x = chart_left_margin + bar_spacing + (i * (bar_width + bar_spacing))
        bar_height = int((avg_detections / max_detections) * chart_height)  # Use smaller chart height
        y = cell_height - 80 - bar_height
        
        color = (100, 200, 255) if tracker_type == 'bytetrack' else (255, 200, 100)
        cv2.rectangle(panel, (x, y), (x + bar_width, cell_height - 80), color, -1)
        
        # Tracker name and value
        tracker_display = "ByteTrack" if tracker_type == 'bytetrack' else "DeepSORT"
        name_y = cell_height - 50
        value_y = y - 20
        
        cv2.putText(panel, tracker_display, (x, name_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, f"{avg_detections:.1f}", (x, value_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add max value reference line and label
    max_line_y = cell_height - 80 - int((max_detections / max_detections) * chart_height)
    cv2.line(panel, (chart_left_margin - 10, max_line_y), 
             (cell_width - chart_left_margin + 10, max_line_y), (255, 255, 0), 2)
    cv2.putText(panel, f"Max: {max_detections:.1f}", (cell_width - 200, max_line_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.putText(panel, "Tracker Comparison", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(panel, "Avg Detections/Frame (Normalized)", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
    
    grid[y_start:y_start+cell_height, x_start:x_start+cell_width] = panel

def create_system_performance_panel(grid, stats, frame_count, col, cell_width, cell_height):
    """Show system performance metrics - SIMPLIFIED WITH AVG DETECTIONS"""
    y_start = 2 * cell_height
    x_start = col * cell_width
    
    panel = np.full((cell_height, cell_width, 3), [60, 40, 40], dtype=np.uint8)
    
    # Calculate metrics
    current_detections = sum(stats.get('tracker_counts', {}).values())
    system_max = max(stats['system_max'], current_detections)
    
    # Calculate average detections (last 20 frames)
    avg_detections = np.mean(stats['all_detections_history'][-20:]) if stats['all_detections_history'] else 0.0
    
    # Performance metrics - simplified
    metrics = [
        "System Performance",
        f"Frame: {frame_count}",
        f"Current Detections: {current_detections}",
        f"Average Detections: {avg_detections:.1f}",
        f"Max Detections: {system_max}"
    ]
    
    for i, text in enumerate(metrics):
        y_pos = 80 + i * 60  # More spacing for better readability
        color = (255, 255, 255) if i > 0 else (255, 255, 100)
        font_size = 1.4 if i == 0 else 1.2  # Larger fonts
        thickness = 3 if i == 0 else 2
        
        cv2.putText(panel, text, (40, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
    
    # Add border
    cv2.rectangle(panel, (10, 10), (cell_width-10, cell_height-10), (255, 255, 255), 3)
    
    grid[y_start:y_start+cell_height, x_start:x_start+cell_width] = panel

def resize_frame_to_fit(frame, target_width, target_height):
    """Resize frame to fit within target dimensions while maintaining aspect ratio"""
    h, w = frame.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    return resized