import os
from collections import Counter

label_dir = 'dataset/labels/train'
counter = Counter()

for file in os.listdir(label_dir):
    with open(os.path.join(label_dir, file), 'r') as f:
        for line in f:
            cls_id = int(line.strip().split()[0])
            counter[cls_id] += 1

print("📦 Phân bố nhãn trong train:")
for cls_id, count in sorted(counter.items()):
    print(f" - Lớp {cls_id}: {count} ảnh")
