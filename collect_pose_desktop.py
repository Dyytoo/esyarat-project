import cv2
import numpy as np
import mediapipe as mp
import os

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Folder output
output_dir = 'pose_collection'
os.makedirs(output_dir, exist_ok=True)

def get_label():
    label = input('Masukkan label/kalimat untuk pose ini: ')
    return label.strip().replace(' ', '_')

label = get_label()
print(f"Label aktif: {label}")

# Cek file output
file_path = os.path.join(output_dir, f'{label}.npy')
if os.path.exists(file_path):
    data = list(np.load(file_path, allow_pickle=True))
else:
    data = []

cap = cv2.VideoCapture(0)
print("Tekan [spasi] untuk capture pose, [l] untuk ganti label, [q] untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka kamera.")
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Gambar landmark jika ada
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Pose Collector', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        # Capture pose
        if results.multi_hand_landmarks:
            landmarks_list = []
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []
                for lm in hand_landmarks.landmark:
                    hand.append([lm.x, lm.y, lm.z])
                landmarks_list.append(hand)
            # Standarisasi: 2 tangan, jika 1 tangan, tangan kedua noise kecil
            if len(landmarks_list) == 1:
                landmark = np.concatenate([
                    np.array(landmarks_list[0]),
                    np.random.normal(0, 0.01, (21, 3))
                ], axis=0)
            elif len(landmarks_list) == 2:
                landmark = np.concatenate([
                    np.array(landmarks_list[0]),
                    np.array(landmarks_list[1])
                ], axis=0)
            else:
                print("Pose tidak valid (tidak ada/tangan lebih dari 2). Ulangi.")
                continue
            data.append(landmark)
            print(f"Pose disimpan untuk label '{label}'. Total: {len(data)}")
        else:
            print("Tidak ada tangan terdeteksi.")
    elif key == ord('l'):
        # Simpan data lama
        if data:
            np.save(file_path, np.array(data))
            print(f"Data untuk label '{label}' disimpan ke {file_path}")
        # Ganti label
        label = get_label()
        print(f"Label aktif: {label}")
        file_path = os.path.join(output_dir, f'{label}.npy')
        if os.path.exists(file_path):
            data = list(np.load(file_path, allow_pickle=True))
        else:
            data = []

# Simpan data terakhir
if data:
    np.save(file_path, np.array(data))
    print(f"Data untuk label '{label}' disimpan ke {file_path}")

cap.release()
cv2.destroyAllWindows()
print("Selesai.") 