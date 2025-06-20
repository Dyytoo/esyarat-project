import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from utils.preprocessing import HandDetector
from model.sign_language_model import SignLanguageModel
from tensorflow.keras.models import load_model
import pickle

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Esyarat - Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

# Inisialisasi state aplikasi dan model jika belum ada di session
if 'detector' not in st.session_state:
    st.session_state.detector = HandDetector()
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'sign_model' not in st.session_state:
    st.session_state.sign_model = load_model('model/saved_model.h5')
    with open('model/label_names.pkl', 'rb') as f:
        st.session_state.label_names = pickle.load(f)

# Judul dan deskripsi aplikasi
st.title("🤟 Esyarat - Sign Language Detection")
st.markdown("""
This application uses AI to detect sign language gestures through your webcam and convert them into text.
""")

# Membuat dua kolom layout utama
col1, col2 = st.columns(2)

with col1:
    st.header("Camera Feed")
    # Placeholder untuk tampilan kamera
    camera_placeholder = st.empty()
    
    # Tombol untuk mengaktifkan/mematikan kamera
    if not st.session_state.camera_active:
        if st.button("Start Camera", key="start_camera"):
            st.session_state.camera_active = True
            st.rerun()
    else:
        if st.button("Stop Camera", key="stop_camera"):
            st.session_state.camera_active = False
            st.rerun()

with col2:
    st.header("Detected Sign")
    # Placeholder untuk hasil deteksi gesture
    sign_placeholder = st.empty()
    
    # Placeholder untuk skor confidence
    confidence_placeholder = st.empty()
    
    # Placeholder untuk info deteksi tangan
    hand_info_placeholder = st.empty()

# Jika kamera aktif, mulai proses pengambilan dan prediksi frame
if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    
    try:
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera")
                break
                
            # Membalik frame agar seperti selfie
            frame = cv2.flip(frame, 1)
            
            # Deteksi tangan dan landmark
            frame, landmarks_list, hand_types = st.session_state.detector.detect_hands(frame)
            
            # Konversi frame ke RGB untuk Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Tampilkan frame kamera di aplikasi
            camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Update info deteksi tangan
            if landmarks_list:
                num_hands = len(landmarks_list)
                if num_hands == 1:
                    hand_info = f"Detected 1 hand ({hand_types[0]})"
                else:
                    hand_info = f"Detected 2 hands ({hand_types[0]} and {hand_types[1]})"
                hand_info_placeholder.write(hand_info)
                
                # Preprocessing landmark untuk input model
                processed_landmarks = st.session_state.detector.preprocess_landmarks(landmarks_list)
                if processed_landmarks is not None:
                    # Siapkan landmark dua tangan (jika satu tangan, tangan kedua diisi nol)
                    if len(processed_landmarks) == 1:
                        landmark = np.concatenate([processed_landmarks[0][0], np.zeros((21, 3))], axis=0)
                    elif len(processed_landmarks) == 2:
                        landmark = np.concatenate([processed_landmarks[0][0], processed_landmarks[1][0]], axis=0)
                    else:
                        landmark = None
                    if landmark is not None and landmark.shape == (42, 3):
                        landmark = landmark.reshape(1, 42, 3)
                        pred = st.session_state.sign_model.predict(landmark)
                        print("Prediksi model:", pred)  # Debug
                        pred_label = st.session_state.label_names[np.argmax(pred)]
                        confidence = float(np.max(pred))
                        sign_placeholder.write(f"Prediksi: {pred_label}")
                        confidence_placeholder.write(f"Confidence: {confidence:.2f}")
                    else:
                        sign_placeholder.write("Landmark tidak valid untuk prediksi.")
                        confidence_placeholder.write("")
            else:
                hand_info_placeholder.write("No hands detected")
                sign_placeholder.write("")
                confidence_placeholder.write("")
            
            # Delay kecil agar tidak membebani sistem
            time.sleep(0.1)
            
    finally:
        # Tutup kamera saat selesai
        cap.release()

# Footer aplikasi
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ for people with disabilities</p>
    <p>Esyarat - AI-Assisted Sign Language Detection</p>
</div>
""", unsafe_allow_html=True) 