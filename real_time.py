import cv2
import time
import numpy as np
from collections import deque
from ultralytics import YOLO

model = YOLO(r'D:\WorkSpace\Python\ChuyenDe2\modelOld\01_n\best.pt')
class_names = model.names
np.random.seed(42)
colors = {i: tuple(np.random.randint(50, 255, 3).tolist()) for i in range(len(class_names))}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở camera.")
    exit()

fps_deque = deque(maxlen=30)
prev_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    input_frame = cv2.resize(frame, (640, 640))

    results = model.predict(input_frame, conf=0.7, iou=0.45, stream=True, verbose=False)

    scale_x = frame_w / 640
    scale_y = frame_h / 640

    object_count = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # Giới hạn tọa độ trong frame
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w - 1))
            y2 = max(0, min(y2, frame_h - 1))

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{class_names[cls_id]} {conf:.2f}"
            color = colors.get(cls_id, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            label_y = y1 - 10 if y1 - 20 > 10 else y1 + 20
            cv2.rectangle(frame, (x1, label_y - th - 5), (x1 + tw + 5, label_y), color, -1)
            cv2.putText(frame, label, (x1 + 2, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            object_count += 1

    # Hiển thị số lượng object và FPS
    cv2.putText(frame, f"Detected: {object_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    curr_time = time.time()
    if prev_time is not None:
        dt = curr_time - prev_time
        if dt > 0:
            fps_deque.append(1 / dt)
    prev_time = curr_time
    avg_fps = sum(fps_deque) / len(fps_deque) if fps_deque else 0
    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("YOLOv8 Real-time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
