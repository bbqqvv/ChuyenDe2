import time
import random
import RPi.GPIO as GPIO
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import numpy as np

# --- CẤU HÌNH GPIO ---
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

IN1 = 11  # Trái
IN2 = 13
IN3 = 15  # Phải
IN4 = 12
ENA = 16  # PWM trái
ENB = 18  # PWM phải

motor_pins = [IN1, IN2, IN3, IN4, ENA, ENB]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(100)
pwmB.start(100)

# --- ĐIỀU KHIỂN ---
def stop_all():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

def forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def turn_left():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    time.sleep(1.0)

def turn_right():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    time.sleep(1.0)

# --- TRẠNG THÁI ---
current_mode = "idle"
last_action_time = 0
action_cooldown = 2  # giây

stop_detected = False
last_stop_time = 0
resume_after = 2  # giây dừng sau STOP

# --- MÔ HÌNH YOLO ---
model = YOLO('/home/vanbui262004/workspace/pios/best.pt')
class_names = model.names
colors = {i: (random.randint(50,255), random.randint(50,255), random.randint(50,255)) for i in class_names}

# --- CAMERA ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
picam2.configure(config)
picam2.start()

# --- HÀNH ĐỘNG BIỂN BÁO ---
def execute_action(label):
    global current_mode, last_action_time, stop_detected, last_stop_time
    now = time.time()

    if now - last_action_time < action_cooldown:
        return

    print(f"📸 Phát hiện biển báo: {label.upper()}")
    last_action_time = now

    if label == "stop":
        stop_all()
        current_mode = "stopped"
        stop_detected = True
        last_stop_time = now

    elif label == "turn left":
        turn_left()
        forward()
        current_mode = "forward"

    elif label == "turn right":
        turn_right()
        forward()
        current_mode = "forward"

# --- VÒNG LẶP CHÍNH ---
try:
    while True:
        if current_mode == "idle":
            forward()
            current_mode = "forward"

        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        results = model.predict(source=frame, conf=0.5, iou=0.45, stream=True, verbose=False)

        labels_seen = set()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label_name = class_names[cls_id].lower()
                labels_seen.add(label_name)

                if label_name in ["stop", "turn left", "turn right"]:
                    execute_action(label_name)

                color = colors.get(cls_id, (0,255,0))
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Nếu đã từng thấy STOP, mà sau thời gian resume thì tiếp tục đi thẳng
        if stop_detected and ("stop" not in labels_seen) and (time.time() - last_stop_time > resume_after):
            print("✅ Không còn biển STOP - Tiếp tục chạy thẳng")
            forward()
            current_mode = "forward"
            stop_detected = False

        cv2.imshow("Robot Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⛔ Dừng chương trình")

# --- DỌN DẸP ---
picam2.stop()
cv2.destroyAllWindows()
stop_all()
pwmA.stop()
pwmB.stop()
GPIO.cleanup()