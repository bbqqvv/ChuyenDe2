# Học lại từ mô hình cũ


from ultralytics import YOLO

# Load mô hình đã huấn luyện trước
model = YOLO("D:/WorkSpace/Python/YoloV8_Test/runs/detect/train/weights/best.pt")

# Huấn luyện thêm với dữ liệu đã mở rộng
model.train(data="data.yaml", epochs=50, lr0=0.0001)
