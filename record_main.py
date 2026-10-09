import cv2
import torch
import time

# Load YOLOv5 model (you can also use YOLOv8 with Ultralytics if you prefer)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Animals of interest
TARGET_ANIMALS = {"dog", "horse", "cow"}  # YOLOv5 doesn't label coyote/deer specifically; use 'dog'/'cow' or custom model

# Video source
cap = cv2.VideoCapture(0)  # Change to IP cam URL if needed
out = None

recording = False
last_seen_time = 0
RECORD_TIMEOUT = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)
    labels = results.xyxyn[0][:, -1].numpy()
    names = [model.names[int(i)] for i in labels]

    # Check if target animals are detected
    detected_animals = [name for name in names if name in TARGET_ANIMALS]

    current_time = time.time()

    if detected_animals:
        last_seen_time = current_time
        if not recording:
            print("🔴 Animal detected - starting recording.")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(f"recording_{int(current_time)}.avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True

    elif recording and current_time - last_seen_time > RECORD_TIMEOUT:
        print("🟢 Animal gone - stopping recording.")
        recording = False
        if out:
            out.release()
            out = None

    # Write frame to video if recording
    if recording and out:
        out.write(frame)

    # Optional: show frame with detections
    cv2.imshow("Live Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()