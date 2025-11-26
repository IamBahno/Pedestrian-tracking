import zipfile
from pathlib import Path
from dataset_utils import process_split, make_test_split, create_data_yaml, cleanup_source
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = SCRIPT_DIR.parent / "datasets"

ZIP_NAME = "WiderPerson.zip"
ZIP_PATH = DATASETS / ZIP_NAME
UNZIPPED_DIR = DATASETS / "WiderPerson"
OUT_DIR = DATASETS / "WiderPerson_yolo"

try:
    # Unzip if needed
    if not UNZIPPED_DIR.exists():
        print(f"Unzipping {ZIP_NAME} ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(UNZIPPED_DIR)
        print("Unzip complete.")
    else:
        print("Already unzipped.")

    # Process splits (tiny box removal enabled)
    print("Processing WiderPerson splits...")
    for split in ["train", "val"]:
        stats = process_split(split, UNZIPPED_DIR, OUT_DIR, dataset_type="widerperson", remove_tiny=True)
        print(f"{split} stats: {stats}")

    # Create 10% test split from train
    make_test_split(OUT_DIR, percentage=0.1)

    # Create YAML
    create_data_yaml(OUT_DIR)
    print("WiderPerson conversion complete.")

finally:
    # Cleanup unzipped folder
    cleanup_source(UNZIPPED_DIR)
