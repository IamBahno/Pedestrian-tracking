from pathlib import Path

# Define the dataset splits
splits = ["train", "valid", "test"]

# Go through each split
for split in splits:
    label_dir = Path(split) / "labels"
    if not label_dir.exists():
        print(f"⚠️ Skipping {label_dir}, not found.")
        continue

    count_total = 0
    count_kept = 0

    for label_file in label_dir.glob("*.txt"):
        lines = label_file.read_text().strip().splitlines()
        count_total += len(lines)

        # Keep only lines starting with '2' and change it to '0'
        filtered = []
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0] == "2":
                parts[0] = "0"
                filtered.append(" ".join(parts))

        # Write filtered lines back (overwrite)
        label_file.write_text("\n".join(filtered))

        count_kept += len(filtered)

    print(f"✅ {split}: processed {count_total} → kept {count_kept} pedestrian boxes.")
