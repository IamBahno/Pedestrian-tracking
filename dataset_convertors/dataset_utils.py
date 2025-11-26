import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml
import cv2

random.seed(42)

# ======================================================
# BOX FILTERING
# ======================================================
def keep_box(parts, dataset_type="widerperson", remove_tiny=False, is_normalized=True):
    """
    Filter boxes depending on dataset_type.
    Returns new parts (list) or None, stats dict.
    """
    stats = {"removed_tiny": 0, "removed_ignored": 0}

    cls = int(parts[0])

    if dataset_type == "widerperson":
        if cls not in [1, 3]:
            stats["removed_ignored"] = 1
            return None, stats
        parts[0] = "0"
        if remove_tiny:
            w, h = map(float, parts[3:5]) if is_normalized else map(float, parts[3:5])
            threshold = 0.01 if is_normalized else 5
            if w < threshold or h < threshold:
                stats["removed_tiny"] = 1
                return None, stats
    elif dataset_type == "citypersons":
        if cls != 2:
            stats["removed_ignored"] = 1
            return None, stats
        parts[0] = "0"

    return parts, stats


# ======================================================
# PROCESS SPLIT
# ======================================================
def process_split(split, src_dir, out_dir, dataset_type="widerperson", remove_tiny=False):
    """
    Unified split processor.
    Handles CityPersons and WiderPerson.
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)

    out_img = out_dir / split / "images"
    out_lbl = out_dir / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    stats = {
        "processed": 0,
        "kept": 0,
        "skipped": 0,
        "bbox_total": 0,
        "removed_tiny": 0,
        "removed_ignored": 0
    }

    if dataset_type == "widerperson":
        images_dir = src_dir / "Images"
        ann_dir = src_dir / "Annotations"

        split_file = src_dir / f"{split}.txt"
        if not split_file.exists():
            print(f"No {split}.txt found, skipping {split}.")
            return stats

        with open(split_file, "r") as f:
            img_list = [line.strip() + ".jpg" for line in f.readlines()]

        for img_name in tqdm(img_list, desc=f"WiderPerson {split}"):
            img_path = images_dir / img_name
            ann_path = ann_dir / (img_name + ".txt")
            stats["processed"] += 1

            if not img_path.exists():
                stats["skipped"] += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                stats["skipped"] += 1
                continue
            h_img, w_img = img.shape[:2]

            yolo_lines = []

            if ann_path.exists():
                with open(ann_path, "r") as f:
                    lines = f.readlines()

                if len(lines) > 1:
                    count = int(lines[0].strip())
                    ann_entries = lines[1:count+1]

                    for entry in ann_entries:
                        parts = entry.strip().split()
                        if len(parts) < 5:
                            continue
                        label = int(parts[0])
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        w_pixel, h_pixel = x2 - x1, y2 - y1

                        # Calculate normalized coordinates
                        x_center = (x1 + x2) / 2.0 / w_img
                        y_center = (y1 + y2) / 2.0 / h_img
                        w_norm = w_pixel / w_img
                        h_norm = h_pixel / h_img

                        # Apply filtering using normalized coordinates
                        parts_normalized = [label, x_center, y_center, w_norm, h_norm]
                        parts_new, s = keep_box(parts_normalized, 
                                              dataset_type="widerperson", 
                                              remove_tiny=remove_tiny, 
                                              is_normalized=True)  # Now normalized!
                        stats["removed_tiny"] += s["removed_tiny"]
                        stats["removed_ignored"] += s["removed_ignored"]

                        if parts_new:
                            yolo_lines.append(f"0 {parts_new[1]:.6f} {parts_new[2]:.6f} {parts_new[3]:.6f} {parts_new[4]:.6f}")
                            stats["bbox_total"] += 1

            if yolo_lines:
                shutil.copy(img_path, out_img / img_name)
                label_path = out_lbl / img_name.replace(".jpg", ".txt")
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_lines))
                stats["kept"] += 1
            else:
                stats["skipped"] += 1

    elif dataset_type == "citypersons":
        src_img = src_dir / split / "images"
        src_lbl = src_dir / split / "labels"

        if not src_lbl.exists():
            print(f"No labels for {split} — skipping.")
            return stats

        lbl_files = list(src_lbl.glob("*.txt"))

        for lbl in tqdm(lbl_files, desc=f"CityPersons {split}"):
            img_path = src_img / (lbl.stem + ".jpg")
            stats["processed"] += 1

            if not img_path.exists():
                stats["skipped"] += 1
                continue

            lines = lbl.read_text().strip().splitlines()
            filtered = []

            for l in lines:
                parts = l.split()
                if not parts:
                    continue
                parts_new, s = keep_box(parts, dataset_type="citypersons")
                stats["removed_ignored"] += s["removed_ignored"]
                if parts_new:
                    filtered.append(" ".join(map(str, parts_new)))
                    stats["bbox_total"] += 1

            if filtered:
                stats["kept"] += 1
                shutil.copy(img_path, out_img / img_path.name)
                (out_lbl / lbl.name).write_text("\n".join(filtered))
            else:
                stats["skipped"] += 1

    return stats


# ======================================================
# MAKE TEST SPLIT
# ======================================================
def make_test_split(out_dir, percentage=0.1):
    train_img_dir = Path(out_dir) / "train" / "images"
    train_lbl_dir = Path(out_dir) / "train" / "labels"
    test_img_dir = Path(out_dir) / "test" / "images"
    test_lbl_dir = Path(out_dir) / "test" / "labels"

    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    train_lbl_files = list(train_lbl_dir.glob("*.txt"))
    take = int(len(train_lbl_files) * percentage)
    if take == 0:
        print("No files to move to test split.")
        return

    selected = random.sample(train_lbl_files, take)
    moved_count = 0
    for lbl in tqdm(selected, desc="Moving to test"):
        img_name = lbl.stem + ".jpg"
        img_src = train_img_dir / img_name
        img_dst = test_img_dir / img_name
        lbl_dst = test_lbl_dir / lbl.name

        if lbl.exists():
            shutil.move(lbl, lbl_dst)
        if img_src.exists():
            shutil.move(img_src, img_dst)
            moved_count += 1
    print(f"Moved {moved_count} files to test split.")


# ======================================================
# YAML CREATION
# ======================================================
def create_data_yaml(out_dir, nc=1, names=None):
    if names is None:
        names = ["pedestrian"]

    yaml_path = Path(out_dir) / "data.yaml"
    data_yaml = {
        "path": str(Path(out_dir).resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names,
        "nc": nc
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"YAML created: {yaml_path}")


# ======================================================
# CLEANUP
# ======================================================
def cleanup_source(src_dir):
    try:
        src_dir = Path(src_dir)
        if src_dir.exists():
            shutil.rmtree(src_dir)
            print(f"Cleaned up: {src_dir}")
    except Exception as e:
        print(f"Could not delete {src_dir}: {e}")
