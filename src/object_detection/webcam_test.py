from ultralytics import YOLO

# Load your trained model
model = YOLO("/home/anish1234/drone_ws/golden_probe/runs/detect/probe_detector/run2/weights/best.pt")

# Run on webcam
results = model.predict(
    source=1,           # 0 = default laptop camera
    conf=0.5,
    show=True,          # opens live window with detections
    stream=True,
)

for r in results:
    pass  # processes each frame