from ultralytics import YOLO
import json

model = YOLO('yolov8n.pt')  # Original pretrained model, NOT retrained

print("Running baseline evaluation on original yolov8n.pt...")
print("(This tests how well the default model detects our litter classes)")
print()

metrics = model.val(
    data='datasets/trash_combined/data.yaml',
    split='test',
    verbose=False
)

results = {
    "model": "yolov8n.pt (baseline - COCO pretrained)",
    "mAP50": round(metrics.box.map50, 4),
    "mAP50-95": round(metrics.box.map, 4),
    "Precision": round(metrics.box.mp, 4),
    "Recall": round(metrics.box.mr, 4),
}

print("="*50)
print("BASELINE RESULTS")
print("="*50)
for k, v in results.items():
    print(f"  {k}: {v}")

# Save to file
import os
os.makedirs("evaluation", exist_ok=True)
with open("evaluation/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to evaluation/baseline_results.json")
