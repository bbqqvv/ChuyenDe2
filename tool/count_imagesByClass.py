import os
from collections import Counter

# Cấu hình
dataset_root = r'D:\WorkSpace\Python\ChuyenDe2\dataset'
image_dir = os.path.join(dataset_root, 'images', 'train')
label_dir = os.path.join(dataset_root, 'labels', 'train')
allowed_classes = {0, 1, 2}
max_objects_threshold = 20

# Biến thống kê
empty_files = []
invalid_classes = set()
object_counts = []
class_counter = Counter()
too_many_objects_files = []

# Đọc tất cả file .txt trong labels
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Đổi đuôi ảnh về .txt để so sánh
image_basenames = set(os.path.splitext(f)[0] for f in image_files)
label_basenames = set(os.path.splitext(f)[0] for f in label_files)

# Check ảnh không có label
images_without_labels = image_basenames - label_basenames
labels_without_images = label_basenames - image_basenames

# Xử lý từng label file
for label_file in label_files:
    path = os.path.join(label_dir, label_file)
    with open(path, 'r') as f:
        lines = f.readlines()

    if not lines:
        empty_files.append(label_file)
    else:
        count = 0
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                if class_id not in allowed_classes:
                    invalid_classes.add(class_id)
                class_counter[class_id] += 1
                count += 1
        object_counts.append((label_file, count))
        if count > max_objects_threshold:
            too_many_objects_files.append((label_file, count))

# ==== KẾT QUẢ ====
print(f"\n📁 Tổng số file ảnh: {len(image_files)}")
print(f"📄 Tổng số file label: {len(label_files)}")

# So khớp ảnh - nhãn
print(f"\n🔍 Ảnh không có label: {len(images_without_labels)}")
if images_without_labels:
    print(" →", sorted(images_without_labels))

print(f"🔍 Label không có ảnh: {len(labels_without_images)}")
if labels_without_images:
    print(" →", sorted(labels_without_images))

# Label analysis
print(f"\n📄 File .txt rỗng (không có object): {len(empty_files)}")
if empty_files:
    print(" →", empty_files)

if invalid_classes:
    print(f"❗ Class không hợp lệ (ngoài {allowed_classes}): {sorted(invalid_classes)}")
else:
    print("✅ Không có class bất thường ngoài phạm vi cho phép.")

print("\n📊 Tổng số object theo class:")
for class_id in sorted(class_counter.keys()):
    print(f" - Class {class_id}: {class_counter[class_id]} object")

print("\n📈 Số object trong mỗi file:")
for fname, count in object_counts:
    print(f" - {fname}: {count} object")

if too_many_objects_files:
    print(f"\n⚠️ Cảnh báo! Có {len(too_many_objects_files)} file có quá nhiều object (> {max_objects_threshold}):")
    for fname, count in too_many_objects_files:
        print(f" → {fname}: {count} object")

# Tổng hợp
total_obj = sum([count for _, count in object_counts])
avg_obj = total_obj / len(object_counts) if object_counts else 0
print(f"\n📌 Tổng object: {total_obj}")
print(f"📌 Trung bình object mỗi ảnh có label: {avg_obj:.2f}")
