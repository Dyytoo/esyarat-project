import os
import glob

# Definisikan path file yang akan direset
POSE_DATA_DIR = 'pose_collection'
MODEL_DIR = 'model'
MODEL_FILE = os.path.join(MODEL_DIR, 'saved_model.h5')
LABEL_FILE = os.path.join(MODEL_DIR, 'label_names.pkl')

def reset_all_data():
    """
    Menghapus semua data pose yang dikumpulkan, model yang dilatih,
    dan file label yang tersimpan.
    """
    print("--- Memulai Proses Reset Total ---")

    # 1. Hapus semua file .npy di pose_collection/
    pose_files = glob.glob(os.path.join(POSE_DATA_DIR, '*.npy'))
    if pose_files:
        print(f"Menghapus {len(pose_files)} file data pose dari '{POSE_DATA_DIR}/'...")
        for f in pose_files:
            os.remove(f)
        print("-> Data pose berhasil dihapus.")
    else:
        print(f"-> Tidak ada data pose di '{POSE_DATA_DIR}/'.")

    # 2. Hapus file model
    if os.path.exists(MODEL_FILE):
        print(f"Menghapus file model '{MODEL_FILE}'...")
        os.remove(MODEL_FILE)
        print("-> File model berhasil dihapus.")
    else:
        print(f"-> File model '{MODEL_FILE}' tidak ditemukan.")

    # 3. Hapus file label
    if os.path.exists(LABEL_FILE):
        print(f"Menghapus file label '{LABEL_FILE}'...")
        os.remove(LABEL_FILE)
        print("-> File label berhasil dihapus.")
    else:
        print(f"-> File label '{LABEL_FILE}' tidak ditemukan.")

    print("\n--- Proses Reset Selesai ---")
    print("Anda bisa mulai mengumpulkan data dan training dari awal.")

if __name__ == "__main__":
    # Minta konfirmasi dari user sebelum menghapus
    confirm = input("Anda yakin ingin menghapus SEMUA data pose dan model yang sudah dilatih? (y/n): ").lower()
    if confirm == 'y':
        reset_all_data()
    else:
        print("Proses reset dibatalkan.") 