from ultralytics import YOLO
import json, os

model = YOLO('yolov8n.pt')

print("Starting training — no mid-training validation to prevent memory kills")

results = model.train(
    data='datasets/trash_combined/data.yaml',
    epochs=20,
    imgsz=416,
    batch=4,
    name='harmony_trash_v3',
    patience=0,        # Disable early stopping — no val needed
    device='mps',
    workers=0,
    cache=False,
    fraction=0.3,      # ~5000 images — lighter
    optimizer='AdamW',
    lr0=0.001,
    val=False,         # KEY FIX — skip validation during training entirely
)

print(f"\nTraining complete. Weights at: {results.save_dir}/weights/last.pt")

os.makedirs("evaluation", exist_ok=True)
with open("evaluation/training_summary.json", "w") as f:
    json.dump({
        "model": "harmony_trash_v3",
        "weights": str(results.save_dir) + "/weights/last.pt"
    }, f, indent=2)