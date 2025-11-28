import os
import gdown
import subprocess
import yaml
import shutil
from ultralytics import YOLO
from pathlib import Path
import gc
import time

# ================================================================
#  DOWNLOAD HELPERS
# ================================================================

def download_file_by_id(file_id, output_path):
    """Download ZIP from Google Drive using gdown"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"\n📥 Downloading {output_path} ...")

    gdown.download(url, output_path, quiet=False)

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Downloaded {output_path} ({size_mb:.1f} MB)")
        return True

    print(f"❌ Failed to download {output_path}")
    return False


# ================================================================
#  DATASET SETUP PIPELINE
# ================================================================

def setup_dataset(zip_filename, google_drive_id, processor_script, processed_folder):
    """
    1. If processed_folder exists → skip everything
    2. If not, check ZIP
    3. If ZIP missing → download
    4. Run processor script to extract + convert
    """

    processed_path = os.path.join("datasets", processed_folder)

    # --------------------------------------------------------
    # 1) Already processed → SKIP EVERYTHING
    # --------------------------------------------------------
    if os.path.exists(processed_path):
        print(f"✅ Processed dataset already exists: {processed_path}")
        return True

    # --------------------------------------------------------
    # 2) ZIP handling
    # --------------------------------------------------------
    os.makedirs("datasets", exist_ok=True)
    zip_path = os.path.join("datasets", zip_filename)

    if not os.path.exists(zip_path):
        print(f"📦 {zip_filename} missing → downloading...")
        if not download_file_by_id(google_drive_id, zip_path):
            print(f"❌ Failed to download: {zip_filename}")
            return False
    else:
        print(f"📁 Found ZIP: {zip_path}")

    # --------------------------------------------------------
    # 3) Run user processor script
    # --------------------------------------------------------
    print(f"🔧 Running processor script: {processor_script}")
    result = subprocess.run(["python", processor_script], capture_output=True, text=True)

    gc.collect()
    time.sleep(2)  # Let OS clean up subprocess memory

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        print(f"❌ Processing failed: {processor_script}")
        return False

    print(f"✅ Processing complete: {processed_folder}")
    return True


# ================================================================
#  TRAINING HELPERS
# ================================================================

def train_model(dataset_type, model_save_path):
    """Train YOLO model on selected dataset"""

    print(f"\n🎯 Training model on: {dataset_type}")

    epochs = 1

    if dataset_type == "dataset1":
        data_config = "datasets/Citypersons_yolo/data.yaml"
    elif dataset_type == "dataset2":
        data_config = "datasets/widerperson_yolo/data.yaml"
    else:
        data_config = create_combined_dataset()

    model = YOLO("yolov8m.pt")

    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=640,
        batch=1,
        device=0,
        workers=1,
        save=True,
        project="models",
        name=f"{dataset_type}",
        exist_ok=True
    )

    del model
    gc.collect() 
    time.sleep(3)  # Ensure file handles are released


# ================================================================
#  COMBINED DATASET GENERATION
# ================================================================

def create_combined_dataset():
    """Merge CityPersons_yolo + widerperson_yolo into one dataset"""
    combined = "datasets/combined"

    if os.path.exists(combined):
        return f"{combined}/data.yaml"

    print("🔀 Creating combined dataset...")

    for split in ["train", "valid", "test"]:
        os.makedirs(f"{combined}/{split}/images", exist_ok=True)
        os.makedirs(f"{combined}/{split}/labels", exist_ok=True)

    # Merge: CityPersons
    for split in ["train", "valid", "test"]:
        src = f"datasets/Citypersons_yolo/{split}"
        if os.path.exists(src):
            shutil.copytree(src + "/images", f"{combined}/{split}/images", dirs_exist_ok=True)
            shutil.copytree(src + "/labels", f"{combined}/{split}/labels", dirs_exist_ok=True)

    # Merge: WiderPerson
    for split in ["train", "valid", "test"]:
        src = f"datasets/widerperson_yolo/{split}"
        if os.path.exists(src):
            shutil.copytree(src + "/images", f"{combined}/{split}/images", dirs_exist_ok=True)
            shutil.copytree(src + "/labels", f"{combined}/{split}/labels", dirs_exist_ok=True)

    from dataset_convertors.dataset_utils import create_data_yaml
    create_data_yaml(combined)


    #yaml_data = {
    #    "nc": 1,
    #    "names": ["pedestrian"],
    #    "train": "train/images",
    #    "val": "valid/images",
    #    "test": "test/images"
    #}
    #
    #with open(f"{combined}/data.yaml", "w") as f:
    #    yaml.dump(yaml_data, f)

    print("✅ Combined dataset created")
    return f"{combined}/data.yaml"

def create_data_yaml(out_dir, nc=1, names=None):
    if names is None:
        names = ["pedestrian"]

    yaml_path = Path(out_dir) / "data.yaml"
    data_yaml = {
        "path": str(Path(out_dir).resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": names,
        "nc": nc
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"YAML created: {yaml_path}")



# ================================================================
#  MODEL ORCHESTRATOR - FIXED VERSION
# ================================================================

def ensure_models_exist():
    """Ensure all three models exist, download or train if missing"""
    
    # File IDs for known resources
    file_ids = {
        "dataset1": "1mWtr01Ab7LuVWAZReEh1fXAuRBWufZCf",  # CityPersons dataset
        "dataset2": "1I7OjhaomWqd8Quf7o5suwLloRlY0THbp",  # WiderPerson dataset
        "model1": "1Ayb0msbngys1UHZWySyTYcDoRoOzQABg",    # best model1
        "model2": "TODO_MODEL2_FILE_ID",                  # best model2
        "model3": "TODO_MODEL3_FILE_ID"                   # best model3
    }
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    processor_scripts = {
        "dataset1": SCRIPT_DIR / "dataset_convertors" / "citypersons_label_filter.py",
        "dataset2": SCRIPT_DIR / "dataset_convertors" / "widerperson_to_yolo.py"
    }

    print("\n==============================")
    print("  CHECKING ALL MODELS")
    print("==============================")

    # Define models with their types and pre-trained IDs
    models = {
        "1": {"type": "dataset1", "pre_trained_id": file_ids["model1"], "path": "models/dataset1/weights/best.pt"},
        "2": {"type": "dataset2", "pre_trained_id": file_ids["model2"], "path": "models/dataset2/weights/best.pt"},
        "3": {"type": "combined", "pre_trained_id": file_ids["model3"], "path": "models/combined/weights/best.pt"}
    }
    
    for model_number, model_info in models.items():
        model_path = model_info["path"]
        
        if os.path.exists(model_path):
            print(f"✅ Model {model_number} already exists at {model_path}!")
            continue
            
        print(f"🔍 Model {model_number} not found at {model_path}!")
        
        # Try to download pre-trained model first
        if model_info["pre_trained_id"] and not model_info["pre_trained_id"].startswith("TODO"):
            print(f"🔄 Attempting to download pre-trained model {model_number}...")
            if download_file_by_id(model_info["pre_trained_id"], model_path):
                print(f"✅ Model {model_number} downloaded successfully!")
                continue
            else:
                print(f"❌ Pre-trained model download failed, training model {model_number}...")
        else:
            print(f"🔄 No pre-trained model available, training model {model_number}...")
        
        # Training logic
        if model_number == "1":
            zip_file = "Citypersons.v1i.yolov8.zip"
            processor = processor_scripts["dataset1"]
            if not setup_dataset(zip_file, file_ids["dataset1"], processor, processed_folder="Citypersons_yolo"):
                print("❌ Cannot setup dataset 1 for training")
                continue
            train_model("dataset1", model_path)
        
        elif model_number == "2":
            zip_file = "WiderPerson.zip" 
            processor = processor_scripts["dataset2"]
            if not setup_dataset(zip_file, file_ids["dataset2"], processor, processed_folder="widerperson_yolo"):
                print("❌ Cannot setup dataset 2 for training")
                continue
            train_model("dataset2", model_path)
        
        elif model_number == "3":
            # For model 3, we need both datasets
            zip_file1 = "Citypersons.v1i.yolov8.zip"
            processor1 = processor_scripts["dataset1"]
            if not setup_dataset(zip_file1, file_ids["dataset1"], processor1, processed_folder="Citypersons_yolo"):
                print("❌ Cannot setup dataset 1 for combined training")
                continue
                
            zip_file2 = "WiderPerson.zip"
            processor2 = processor_scripts["dataset2"] 
            if not setup_dataset(zip_file2, file_ids["dataset2"], processor2, processed_folder="widerperson_yolo"):
                print("❌ Cannot setup dataset 2 for combined training")
                continue
                
            train_model("combined", model_path)




def setup_datasets_for_testing():
    """Set up datasets specifically for testing purposes"""
    file_ids = {
        "dataset1": "1mWtr01Ab7LuVWAZReEh1fXAuRBWufZCf",
        "dataset2": "1I7OjhaomWqd8Quf7o5suwLloRlY0THbp",
    }
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    processor_scripts = {
        "dataset1": SCRIPT_DIR / "dataset_convertors" / "citypersons_label_filter.py",
        "dataset2": SCRIPT_DIR / "dataset_convertors" / "widerperson_to_yolo.py"
    }

    print("\n==============================")
    print("  SETTING UP DATASETS FOR TESTING")
    print("==============================")

    # Setup dataset1
    zip_file1 = "Citypersons.v1i.yolov8.zip"
    processor1 = processor_scripts["dataset1"]
    if not setup_dataset(zip_file1, file_ids["dataset1"], processor1, processed_folder="Citypersons_yolo"):
        print("❌ Cannot setup dataset 1 for testing")
        return False

    # Setup dataset2  
    zip_file2 = "WiderPerson.zip"
    processor2 = processor_scripts["dataset2"]
    if not setup_dataset(zip_file2, file_ids["dataset2"], processor2, processed_folder="widerperson_yolo"):
        print("❌ Cannot setup dataset 2 for testing")
        return False

    print("✅ All datasets ready for testing!")
    return True



# ================================================================
#  ENTRYPOINT
# ================================================================

if __name__ == "__main__":
    ensure_models_exist()