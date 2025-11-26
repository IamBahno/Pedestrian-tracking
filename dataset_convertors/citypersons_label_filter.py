import zipfile
from pathlib import Path
from dataset_utils import process_split, create_data_yaml, cleanup_source
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = SCRIPT_DIR.parent / "datasets"

ZIP_NAME = "Citypersons.v1i.yolov8.zip"
ZIP_PATH = DATASETS / ZIP_NAME
UNZIPPED_DIR = DATASETS / "Citypersons.v1i.yolov8"
OUT_DIR = DATASETS / "Citypersons_yolo"

try:
    # Unzip if needed
    if not UNZIPPED_DIR.exists():
        print(f"Unzipping {ZIP_NAME} ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(UNZIPPED_DIR)
        print("Unzip complete.")
    else:
        print("Already unzipped.")

    # Process splits
    print("Processing CityPersons splits...")
    for split in ["train", "valid", "test"]:
        stats = process_split(split, UNZIPPED_DIR, OUT_DIR, dataset_type="citypersons")
        print(f"{split} stats: {stats}")

    # Create YAML
    create_data_yaml(OUT_DIR)
    print("CityPersons conversion complete.")

finally:
    # Cleanup unzipped folder
    cleanup_source(UNZIPPED_DIR)
