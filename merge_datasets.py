import os
import shutil
import yaml
from pathlib import Path

# === CONFIGURATION ===
extracted_dir = "datasets/extracted"
output_dir = "datasets/trash_combined"

datasets = [
    "Garbage Detection.v21i.yolov8",
    "litter.v5i.yolov8",
    "Garbage Detection.v1i.yolov8"
]

splits = ["train", "valid", "test"]

# === CREATE OUTPUT FOLDERS ===
for split in splits:
    Path(f"{output_dir}/{split}/images").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/{split}/labels").mkdir(parents=True, exist_ok=True)

total_images = {"train": 0, "valid": 0, "test": 0}

# === MERGE EACH DATASET ===
for ds_name in datasets:
    ds_path = os.path.join(extracted_dir, ds_name)
    print(f"\nProcessing: {ds_name}")

    for split in splits:
        img_src = os.path.join(ds_path, split, "images")
        lbl_src = os.path.join(ds_path, split, "labels")

        if not os.path.exists(img_src):
            print(f"  No {split} split found, skipping.")
            continue

        images = os.listdir(img_src)
        print(f"  {split}: {len(images)} images")

        for img_file in images:
            # Create unique filename to avoid collisions
            prefix = ds_name.replace(" ", "_").replace(".", "_")
            new_name = f"{prefix}_{img_file}"

            # Copy image
            shutil.copy(
                os.path.join(img_src, img_file),
                os.path.join(output_dir, split, "images", new_name)
            )

            # Copy and remap label (all classes → class 0 = litter)
            lbl_file = os.path.splitext(img_file)[0] + ".txt"
            lbl_path = os.path.join(lbl_src, lbl_file)
            out_lbl_path = os.path.join(output_dir, split, "labels", 
                                         os.path.splitext(new_name)[0] + ".txt")

            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                with open(out_lbl_path, 'w') as f:
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            # Remap any class ID → 0 (unified "litter" class)
                            parts[0] = "0"
                            f.write(" ".join(parts) + "\n")
            else:
                # Create empty label file if no annotations
                open(out_lbl_path, 'w').close()

            total_images[split] += 1

# === WRITE data.yaml ===
data_yaml = {
    "path": os.path.abspath(output_dir),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 1,
    "names": ["litter"]
}

with open(os.path.join(output_dir, "data.yaml"), 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

# === SUMMARY ===
print("\n" + "="*50)
print("MERGE COMPLETE — DATASET SUMMARY")
print("="*50)
for split, count in total_images.items():
    print(f"  {split}: {count} images")
print(f"  Classes: 1 (unified 'litter')")
print(f"  Saved to: {output_dir}/data.yaml")



