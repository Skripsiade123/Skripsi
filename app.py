import streamlit as st
import pandas as pd
import zipfile
import os
import joblib

# === Membaca dataset dari ZIP ===
def load_data():
    if not os.path.exists("data"):
        with zipfile.ZipFile("Dataset.zip", 'r') as zip_ref:
            zip_ref.extractall("data")

    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.lower().endswith(".csv") and "dataset" in file.lower():
                df = pd.read_csv(os.path.join(root, file))
                df.columns = df.columns.str.strip().str.lower()
                return df

    st.error("File Dataset.csv tidak ditemukan dalam ZIP.")
    return pd.DataFrame()

df = load_data()

# === Load semua model SVM ===
# Ensure these files exist or handle their absence gracefully
try:
    model_genre = joblib.load("svm_model.pkl")
    model_tag = joblib.load("svm_model_tags.pkl")
    model_category = joblib.load("svm_model_categories.pkl")
except FileNotFoundError:
    st.error("Model files (svm_model.pkl, svm_model_tags.pkl, svm_model_categories.pkl) not found. Please ensure they are in the same directory.")
    st.stop() # Stop the app if models are not found

# === Sidebar Navigasi ===
st.sidebar.title("Navigasi")
halaman = st.sidebar.radio("Pilih Halaman:", ["Beranda", "Penjelasan Metode", "Rekomendasi Genre", "Rekomendasi Tag", "Rekomendasi Kategori"])

# === Simpan histori pilihan genre/tag/kategori ===
if "history" not in st.session_state:
    st.session_state.history = {"genre": [], "tag": [], "category": []}

# === Fungsi Rekomendasi Berdasarkan Preferensi Gabungan dengan Bobot ===
def rekomendasi_berdasarkan_histori():
    preferensi_genre = st.session_state.history["genre"]
    preferensi_tag = st.session_state.history["tag"]
    preferensi_kat = st.session_state.history["category"]

    if preferensi_genre or preferensi_tag or preferensi_kat:
        df_temp = df.copy()
        df_temp["score"] = 0

        if preferensi_genre:
            # Check if 'genre' column exists in df_temp before using isin
            if 'genre' in df_temp.columns:
                df_temp.loc[df_temp["genre"].isin(preferensi_genre), "score"] += 3
        if preferensi_tag:
            # Check if 'tag' column exists in df_temp before using isin
            if 'tag' in df_temp.columns:
                df_temp.loc[df_temp["tag"].isin(preferensi_tag), "score"] += 2
        if preferensi_kat:
            # Check if 'category' column exists in df_temp before using isin
            if 'category' in df_temp.columns:
                df_temp.loc[df_temp["category"].isin(preferensi_kat), "score"] += 1

        hasil = df_temp[df_temp["score"] > 0].sort_values(by="score", ascending=False)
        return hasil.head(10)
    else:
        return df.sample(10) # Fallback to random if no history

# === Fungsi untuk Menampilkan Game dengan Format Kartu ===
def tampilkan_game(hasil):
    if hasil.empty:
        st.write("Tidak ada game yang ditemukan berdasarkan kriteria yang dipilih.")
        return

    for i, row in hasil.iterrows():
        nama = row.get('name', 'Tidak ada nama')
        deskripsi = row.get('deskripsi', row.get('description', 'Deskripsi tidak tersedia.'))
        genre = row.get('genre', '-')
        tag = row.get('tag', '-')
        kategori = row.get('category', '-')
        gambar = row.get('img', '')

        if pd.isna(gambar) or not isinstance(gambar, str) or not gambar.startswith("http"):
            gambar = "https://via.placeholder.com/180x100.png?text=No+Image"

        st.markdown(f"""
        <div style="display: flex; gap: 20px; padding: 15px; border: 1px solid #444; border-radius: 10px; margin-bottom: 20px; background-color: #222;">
            <div style="flex-shrink: 0;">
                <img src="{gambar}" style="width: 180px; height: auto; border-radius: 10px; object-fit: cover;">
            </div>
            <div style="flex-grow: 1; color: white;">
                <h4 style="margin-bottom: 5px;">{nama}</h4>
                <p style="font-size: 14px; margin-bottom: 10px;">{deskripsi}</p>
                <p style="font-size: 13px;">
                    <strong>Genre:</strong> {genre} <br>
                    <strong>Tags:</strong> {tag} <br>
                    <strong>Kategori:</strong> {kategori}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# === Halaman Beranda ===
if halaman == "Beranda":
    st.title("🎮 Rekomendasi Game Terbaik untuk Anda")
    st.write("Berikut adalah 10 rekomendasi game terbaik berdasarkan histori interaksi Anda.")
    rekomendasi = rekomendasi_berdasarkan_histori()
    tampilkan_game(rekomendasi)

# === Halaman Penjelasan Metode ===
elif halaman == "Penjelasan Metode":
    st.title("📚 Penjelasan Metode")
    st.write("Aplikasi ini menggunakan metode **Content-Based Filtering** dengan algoritma **Support Vector Machine (SVM)**. Terdapat tiga model terpisah untuk memproses fitur Genre, Tag, dan Category berdasarkan deskripsi dan metadata game.")

# === Halaman Rekomendasi Genre ===
elif halaman == "Rekomendasi Genre":
    st.title("🎯 Rekomendasi Berdasarkan Genre")

    daftar_genre = df['genre'].dropna().unique().tolist()
    genre_pilihan = st.selectbox("Pilih genre sebagai filter awal:", sorted(daftar_genre))

    if genre_pilihan:
        # Add to history only if not already present to avoid redundant entries
        if genre_pilihan not in st.session_state.history['genre']:
            st.session_state.history['genre'].append(genre_pilihan)
        hasil = df[df['genre'] == genre_pilihan]
        st.markdown("### Rekomendasi Game berdasarkan genre")
        tampilkan_game(hasil)

# === Halaman Rekomendasi Tag ===
elif halaman == "Rekomendasi Tag":
    st.title("🏷️ Rekomendasi Berdasarkan Tag")

    daftar_tag = df['tag'].dropna().unique().tolist()
    tag_pilihan = st.selectbox("Pilih tag sebagai filter awal:", sorted(daftar_tag))

    if tag_pilihan:
        # Add to history only if not already present to avoid redundant entries
        if tag_pilihan not in st.session_state.history['tag']:
            st.session_state.history['tag'].append(tag_pilihan)
        hasil = df[df['tag'] == tag_pilihan]
        st.markdown("### Rekomendasi Game berdasarkan tag")
        tampilkan_game(hasil)

# === Halaman Rekomendasi Kategori ===
elif halaman == "Rekomendasi Kategori":
    st.title("📂 Rekomendasi Berdasarkan Kategori")

    daftar_kategori = df['category'].dropna().unique().tolist()
    kategori_pilihan = st.selectbox("Pilih kategori sebagai filter awal:", sorted(daftar_kategori))

    if kategori_pilihan:
        # Add to history only if not already present to avoid redundant entries
        if kategori_pilihan not in st.session_state.history['category']:
            st.session_state.history['category'].append(kategori_pilihan)
        hasil = df[df['category'] == kategori_pilihan]
        st.markdown("### Rekomendasi Game berdasarkan kategori")
        tampilkan_game(hasil)
