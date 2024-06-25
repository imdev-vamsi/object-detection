import cv2
import torch
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

import cv2
from flask import Response

def generate_frames(input_video_path):
    cap = cv2.VideoCapture(input_video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img)

        # Draw bounding boxes
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result
            label = f"{model.names[int(cls)]} {conf:.2f}"
            if label[:-5] == "motorcycle" or label[:-5] == "car":
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
