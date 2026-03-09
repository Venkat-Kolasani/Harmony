from ultralytics import YOLO
import json, os

model = YOLO('yolov8n.pt')

print("Starting fine-tuning on trash_combined dataset...")
print("This will take a while on CPU. Do not close the terminal.\n")

results = model.train(
    data='datasets/trash_combined/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,          # 8 is safe for M3 MacBook
    name='harmony_trash_v1',
    patience=10,
    device='mps',     # Apple M3 GPU — faster than CPU
    workers=4
)

print("\nTraining complete.")
print(f"Best weights saved at: {results.save_dir}/weights/best.pt")

# Save training summary
os.makedirs("evaluation", exist_ok=True)
summary = {
    "model": "harmony_trash_v1 (fine-tuned on 18485 images)",
    "epochs_completed": results.epoch + 1 if hasattr(results, 'epoch') else 50,
    "best_weights": str(results.save_dir) + "/weights/best.pt"
}
with open("evaluation/training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
