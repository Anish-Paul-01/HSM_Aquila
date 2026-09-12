from roboflow import Roboflow
from ultralytics import YOLO

# 1. Download dataset from Roboflow
rf = Roboflow(api_key="vcod6x8b64sZ4cmr3GEx")
project = rf.workspace("anishs-workspace-shp6g").project("probe-detection-vvsvt")
version = project.version(4)
dataset = version.download("yolov8")
                

# 2. Load YOLOv8 model (nano=fastest, small/medium = better accuracy)
model = YOLO("yolov8n.pt")  # change to yolov8s.pt or yolov8m.pt if GPU is strong

# 3. Train
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=512,          # matches your dataset image size
    batch=16,           # lower to 8 if you get OOM errors
    device=0,           # GPU 0
    project="probe_detection",
    name="yolov8_run1",
    patience=20,        # early stopping
    save=True,
)

print("Training complete!")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")


                