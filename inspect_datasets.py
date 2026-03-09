import zipfile
import os
import yaml

zip_dir = "datasets/raw_zips"
extract_dir = "datasets/extracted"

for zip_name in os.listdir(zip_dir):
    if zip_name.endswith(".zip"):
        zip_path = os.path.join(zip_dir, zip_name)
        out_path = os.path.join(extract_dir, zip_name.replace(".zip", ""))
        
        print(f"\nExtracting: {zip_name}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(out_path)
        print(f"Extracted to: {out_path}")
        
        # Find and read data.yaml
        for root, dirs, files in os.walk(out_path):
            for file in files:
                if file == "data.yaml":
                    yaml_path = os.path.join(root, file)
                    with open(yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                    print(f"Classes ({data.get('nc', '?')} total): {data.get('names', [])}")
