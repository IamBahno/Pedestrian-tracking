#Key Features Implemented:
#    Model Management: Automatically downloads or trains all three models
#    Dataset Handling: Supports two datasets with automatic setup
#    Testing & Evaluation: Tests all models on both datasets and generates comparison graphs
#
#    3x3 Video Grid: Creates a comprehensive comparison video with:
#        6 tracking windows (3 models × 2 trackers)
#        3 statistics panels
#
#    Outputs: Saves graphs and comparison video


import subprocess
import os
import sys
import platform



def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    
    if os.path.exists("requirements.txt"):
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully!")
            return True
        else:
            print("❌ Failed to install requirements!")
            print("STDERR:", result.stderr)
            return False
    else:
        print("❌ requirements.txt not found!")
        return False
    


if __name__ == "__main__":
    """Main build and run function"""
    print("🚀 PGR tracker - Build and Run")
    print("=" * 50)
    
    # Check platform
    system = platform.system()
    print(f"💻 Platform: {system} {platform.machine()}")
    print(f"🐍 Python: {sys.version}")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("output/graphs", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("input", exist_ok=True)
    
    # Install requirements first
    if not install_requirements():
        print("⚠️  Some requirements failed to install, continuing...")
    
    # Ensure all models exist
    from model_setup import ensure_models_exist, setup_datasets_for_testing
    ensure_models_exist()
    
    # Setup datasets for testing
    if setup_datasets_for_testing():
        from testing import test_model_on_datasets, create_comparison_video
        test_results = test_model_on_datasets() # Test models and generate graphs
        
        # Only create video if test video exists
        if os.path.exists("input/video.mp4"):
            create_comparison_video()
        else:
            print("⚠️  Test video not found at input/video.mp4 - skipping video creation")
    else:
        print("❌ Could not setup datasets for testing")
    
    print("✅ All tasks completed!")
    print("📁 Outputs available in: output/")
