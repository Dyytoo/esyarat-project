import os
import numpy as np
import cv2
from utils.preprocessing import HandDetector
from model.sign_language_model import SignLanguageModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Inisialisasi detektor tangan dari MediaPipe
detector = HandDetector()

# Membaca gambar dari folder dataset dan menyiapkan data serta label
data = []
labels = []
label_map = {}  # mapping label ke angka
dataset_path = 'dataset'
label_names = sorted(os.listdir(dataset_path))

for idx, label_name in enumerate(label_names):
    label_map[label_name] = idx
    folder = os.path.join(dataset_path, label_name)
    for img_name in os.listdir(folder):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        # Deteksi landmark tangan pada gambar
        _, landmarks_list, _ = detector.detect_hands(img)
        if len(landmarks_list) == 1:
            # Jika hanya satu tangan, landmark tangan kedua diisi noise kecil
            landmark = np.concatenate([
                np.array(landmarks_list[0]),
                np.random.normal(0, 0.01, (21, 3))
            ], axis=0)
        elif len(landmarks_list) == 2:
            # Jika dua tangan, landmark digabung
            landmark = np.concatenate([np.array(landmarks_list[0]), np.array(landmarks_list[1])], axis=0)
        else:
            continue  # Lewati jika tidak ada tangan
        if landmark.shape == (42, 3):
            data.append(landmark)
            labels.append(idx)

# Augmentasi data: menyeimbangkan jumlah data tiap kelas dengan jitter/noise
max_count = max(Counter(labels).values())
new_data = []
new_labels = []
for class_idx in set(labels):
    class_data = [d for d, l in zip(data, labels) if l == class_idx]
    n_to_add = max_count - len(class_data)
    for _ in range(n_to_add):
        base = class_data[np.random.randint(len(class_data))]
        noise = np.random.normal(0, 0.01, base.shape)
        augmented = base + noise
        new_data.append(augmented)
        new_labels.append(class_idx)
if new_data:
    data.extend(new_data)
    labels.extend(new_labels)

# Konversi data dan label ke numpy array dan one-hot encoding
data = np.array(data)
labels = np.array(labels)
labels_cat = to_categorical(labels, num_classes=len(label_names))

# Hitung class_weight untuk mengatasi data tidak seimbang
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Split data menjadi data latih dan validasi
X_train, X_val, y_train, y_val = train_test_split(
    data, labels_cat, test_size=0.2, random_state=42, stratify=labels
)

# Inisialisasi dan latih model
model = SignLanguageModel(num_classes=len(label_names))
history = model.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=8, class_weight=class_weight_dict)

# Simpan model dan label hasil training
model.save_model('model/saved_model.h5')
print("Training selesai! Model disimpan di model/saved_model.h5")
print(f"Total data yang berhasil diekstrak: {len(data)}")
for i, name in enumerate(label_names):
    print(f"Label {name}: {np.sum(labels == i)} data")

with open('model/label_names.pkl', 'wb') as f:
    pickle.dump(label_names, f)