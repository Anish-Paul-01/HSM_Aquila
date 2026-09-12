import cv2
import depthai as dai
from ultralytics import YOLO

# point this at your actual trained weights
model = YOLO("/home/anish1234/runs/detect/probe_shape_detector/yolo11_run1/weights/best.pt")

pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
video_out = cam_rgb.requestOutput((1280, 720), fps=30)
q_rgb = video_out.createOutputQueue(maxSize=4, blocking=False)

pipeline.start()

seen_ids = set()

while pipeline.isRunning():
    in_rgb = q_rgb.get()
    frame = in_rgb.getCvFrame()

    # match training distribution: model learned on grayscale, so convert
    # each frame to grayscale then back to 3-channel before feeding it in
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # track() instead of predict() — assigns persistent IDs across frames
    results = model.track(
        source=gray_3ch,
        conf=0.5,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    r = results[0]
    if r.boxes.id is not None:
        ids = r.boxes.id.int().tolist()
        seen_ids.update(ids)

    annotated_frame = r.plot()
    cv2.putText(annotated_frame, f"Unique probes: {len(seen_ids)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("OAK-D Pro - probe tracking", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
print(f"Total unique probes seen: {len(seen_ids)}")