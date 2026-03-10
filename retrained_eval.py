from ultralytics import YOLO
import json, os

# Point to your saved weights
model = YOLO('runs/detect/harmony_trash_v3/weights/last.pt')

print("Evaluating retrained model on test set...")

metrics = model.val(
    data='datasets/trash_combined/data.yaml',
    split='test',
    conf=0.25,        # Lower confidence threshold — catches more detections
    iou=0.5,
    max_det=100,      # Limit detections per image — prevents NMS memory kill
    verbose=False
)

results = {
    "model": "harmony_trash_v3 (fine-tuned, 20 epochs)",
    "mAP50": round(metrics.box.map50, 4),
    "mAP50-95": round(metrics.box.map, 4),
    "Precision": round(metrics.box.mp, 4),
    "Recall": round(metrics.box.mr, 4),
}

print("\n" + "="*50)
print("RETRAINED MODEL RESULTS")
print("="*50)
for k, v in results.items():
    print(f"  {k}: {v}")

# Compare against baseline
baseline = {
    "mAP50": 0.1018,
    "mAP50-95": 0.0673,
    "Precision": 0.1434,
    "Recall": 0.2333
}

print("\n" + "="*50)
print("IMPROVEMENT OVER BASELINE")
print("="*50)
for metric in ["mAP50", "mAP50-95", "Precision", "Recall"]:
    improvement = ((results[metric] - baseline[metric]) / baseline[metric]) * 100
    print(f"  {metric}: {baseline[metric]} → {results[metric]} (+{improvement:.1f}%)")

os.makedirs("evaluation", exist_ok=True)
with open("evaluation/retrained_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to evaluation/retrained_results.json")