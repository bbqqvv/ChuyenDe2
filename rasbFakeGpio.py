import cv2
import time
import random
from ultralytics import YOLO
from time import sleep

# Fake Motor class nếu không có gpiozero (test trên máy ảo)
try:
    from gpiozero import Motor

    motor_left = Motor(forward=17, backward=18)
    motor_right = Motor(forward=22, backward=23)
except ImportError:
    print("⚠️ Đang test trên máy ảo - fake Motor.")

    class FakeMotor:
        def forward(self):
            print("Motor forward")

        def backward(self):
            print("Motor backward")

        def stop(self):
            print("Motor stop")

    motor_left = FakeMotor()
    motor_right = FakeMotor()

# Load YOLOv8 model
model = YOLO(r'D:\WorkSpace\Python\ChuyenDe2\best.pt')  # Cập nhật đường dẫn mô hình của bạn

# Mở camera (0: webcam tích hợp, 1+: camera ngoài)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở camera.")
    exit()

# Tên lớp và màu ngẫu nhiên cho từng lớp
class_names = model.names
colors = {i: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for i in class_names}

# Kích thước khung hình
TARGET_SIZE = (640, 480)

# FPS tính toán
prev_time = 0

# Tuỳ chọn: Ghi lại video (bật = True nếu cần)
SAVE_VIDEO = False
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('yolo_output.avi', fourcc, 20.0, TARGET_SIZE)

# Vòng lặp chính
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Không thể đọc khung hình từ camera.")
        break

    # Resize khung hình nếu cần
    if TARGET_SIZE:
        frame = cv2.resize(frame, TARGET_SIZE)

    # Dự đoán bằng YOLOv8
    results = model.predict(source=frame, conf=0.5, iou=0.45, stream=True, verbose=False)

    object_count = 0  # Đếm số đối tượng

    # Vẽ bounding boxes và điều khiển xe dựa trên biển báo
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{class_names[cls_id]} {conf:.2f}"
            color = colors.get(cls_id, (0, 255, 0))

            # Kiểm tra nếu là biển báo "turn right" hoặc "turn left"
            if 'turn right' in class_names[cls_id].lower():
                label = "Turn Right Detected"
                motor_left.forward()  # Quay động cơ trái tiến
                motor_right.backward()  # Quay động cơ phải lùi
                sleep(1)  # Thời gian để xe quay
                motor_left.stop()  # Dừng động cơ trái
                motor_right.stop()  # Dừng động cơ phải

            elif 'turn left' in class_names[cls_id].lower():
                label = "Turn Left Detected"
                motor_left.backward()  # Quay động cơ trái lùi
                motor_right.forward()  # Quay động cơ phải tiến
                sleep(1)  # Thời gian để xe quay
                motor_left.stop()  # Dừng động cơ trái
                motor_right.stop()  # Dừng động cơ phải

            elif 'stop' in class_names[cls_id].lower():
                label = "Stop Detected"
                motor_left.stop()  # Dừng động cơ trái
                motor_right.stop()  # Dừng động cơ phải

            elif 'straight' in class_names[cls_id].lower():
                label = "Straight Detected"
                motor_left.forward()  # Động cơ trái tiến
                motor_right.forward()  # Động cơ phải tiến

            elif 'parking' in class_names[cls_id].lower():
                label = "Parking Detected"
                motor_left.stop()  # Dừng động cơ trái
                motor_right.stop()  # Dừng động cơ phải

            elif 'roundabout' in class_names[cls_id].lower():
                label = "Roundabout Detected"
                motor_left.forward()  # Quay vòng bằng cách quay 2 bánh
                motor_right.backward()  # Quay động cơ phải lùi
                sleep(2)  # Thời gian để xe quay vòng
                motor_left.stop()  # Dừng động cơ trái
                motor_right.stop()  # Dừng động cơ phải

            # Vẽ khung hình và nhãn
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            object_count += 1

    # Hiển thị số lượng đối tượng
    cv2.putText(frame, f"Detected: {object_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Hiển thị khung hình
    cv2.imshow("YOLOv8 Real-time Detection", frame)

    # Ghi video nếu bật
    if SAVE_VIDEO:
        out.write(frame)

    # Thoát bằng phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp
cap.release()
if SAVE_VIDEO:
    out.release()
cv2.destroyAllWindows()
