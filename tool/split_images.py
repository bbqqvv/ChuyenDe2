import os
import shutil
import random

# Đường dẫn dữ liệu gốc
img_dir = 'dataset/all_images'
label_dir = 'dataset/all_labels'

# Nơi lưu sau khi chia
output_base = 'dataset/'
train_img = os.path.join(output_base, 'images/train')
val_img = os.path.join(output_base, 'images/val')
train_lbl = os.path.join(output_base, 'labels/train')
val_lbl = os.path.join(output_base, 'labels/val')

# Tạo thư mục nếu chưa có
for d in [train_img, val_img, train_lbl, val_lbl]:
    os.makedirs(d, exist_ok=True)

# Tỷ lệ chia
split_ratio = 0.2  # 20% val

# Duyệt toàn bộ ảnh và chia
image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(image_files)

for img_file in image_files:
    name = os.path.splitext(img_file)[0]
    label_file = name + ".txt"
    src_img_path = os.path.join(img_dir, img_file)
    src_lbl_path = os.path.join(label_dir, label_file)

    if not os.path.exists(src_lbl_path):
        continue  # bỏ qua nếu không có label

    if random.random() < split_ratio:
        shutil.copy(src_img_path, os.path.join(val_img, img_file))
        shutil.copy(src_lbl_path, os.path.join(val_lbl, label_file))
    else:
        shutil.copy(src_img_path, os.path.join(train_img, img_file))
        shutil.copy(src_lbl_path, os.path.join(train_lbl, label_file))

print("✅ Đã chia xong dữ liệu theo tỉ lệ 80/20.")
