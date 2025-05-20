from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Load mô hình YOLO đã train
model = YOLO(r"D:\WorkSpace\Python\ChuyenDe2\best.pt")

# Dự đoán trên ảnh
results = model(r'D:\WorkSpace\Python\ChuyenDe2\images.jpg')

# Duyệt qua kết quả
for r in results:
    boxes = r.boxes
    img = r.orig_img.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{model.names[cls_id]} {conf:.2f}"

        # Vẽ bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Vẽ nhãn
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # In ra thông tin nhãn
        print(f"Label: {model.names[cls_id]}, Confidence: {conf:.2f}")

    # Hiển thị ảnh (dùng PIL để tương thích)
    im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    im_pil.show()
    im_pil.save("result_with_labels.jpg")
