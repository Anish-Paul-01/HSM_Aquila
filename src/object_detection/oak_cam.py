import cv2
import depthai as dai
from ultralytics import YOLO

# Load your trained model
model = YOLO("/home/anish1234/drone_ws/src/object_detection/best.pt")

# Set up DepthAI Pipeline for OAK-D Pro
pipeline = dai.Pipeline()

# Define Camera Node (v3 API)
cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

# Request an output stream at a given size/fps
video_out = cam_rgb.requestOutput((1280, 720), fps=30)

# Create an output queue directly from the output (no XLinkOut needed)
q_rgb = video_out.createOutputQueue(maxSize=4, blocking=False)

# Start the pipeline
pipeline.start()

while pipeline.isRunning():
    in_rgb = q_rgb.get()
    frame = in_rgb.getCvFrame()

    # Run YOLO inference
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Display window
    cv2.imshow("OAK-D Pro - YOLOv8", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
