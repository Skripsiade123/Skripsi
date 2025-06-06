import streamlit as st
import pandas as pd
import zipfile
import os
import joblib

# === Membaca dataset dari ZIP ===
def load_data():
    with zipfile.ZipFile("Dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("data")
    df = pd.read_csv("data/games.csv")
    return df

df = load_data()

# === Load semua model SVM ===
model_genre = joblib.load("data/model_svm_genre.pkl")
model_tag = joblib.load("data/model_svm_tag.pkl")
model_category = joblib.load("data/model_svm_category.pkl")

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
            df_temp.loc[df_temp["genre"].isin(preferensi_genre), "score"] += 3
        if preferensi_tag:
            df_temp.loc[df_temp["tag"].isin(preferensi_tag), "score"] += 2
        if preferensi_kat:
            df_temp.loc[df_temp["category"].isin(preferensi_kat), "score"] += 1

        hasil = df_temp[df_temp["score"] > 0].sort_values(by="score", ascending=False)
        return hasil.head(10)
    else:
        return df.sample(10)

# === Halaman Beranda ===
if halaman == "Beranda":
    st.title("🎮 Rekomendasi Game Terbaik untuk Anda")
    st.write("Berikut adalah 10 rekomendasi game terbaik berdasarkan histori interaksi Anda.")
    rekomendasi = rekomendasi_berdasarkan_histori()
    st.dataframe(rekomendasi)

# === Halaman Penjelasan Metode ===
elif halaman == "Penjelasan Metode":
    st.title("📚 Penjelasan Metode")
    st.write("Aplikasi ini menggunakan metode **Content-Based Filtering** dengan algoritma **Support Vector Machine (SVM)**. Terdapat tiga model terpisah untuk memproses fitur Genre, Tag, dan Category berdasarkan deskripsi dan metadata game.")

# === Halaman Rekomendasi Genre ===
elif halaman == "Rekomendasi Genre":
    st.title("🎯 Rekomendasi Berdasarkan Genre")
    st.write("10 Game Rekomendasi Umum Berdasarkan Genre")
    st.dataframe(df.sample(10))

    st.subheader("Pilih Genre:")
    daftar_genre = df['genre'].dropna().unique().tolist()
    for genre in daftar_genre:
        if st.button(genre):
            st.session_state.history['genre'].append(genre)
            hasil = df[df['genre'] == genre]
            st.write(f"Menampilkan {len(hasil)} game dengan genre '{genre}':")
            st.dataframe(hasil)

# === Halaman Rekomendasi Tag ===
elif halaman == "Rekomendasi Tag":
    st.title("🏷️ Rekomendasi Berdasarkan Tag")
    st.write("10 Game Rekomendasi Umum Berdasarkan Tag")
    st.dataframe(df.sample(10))

    st.subheader("Pilih Tag:")
    daftar_tag = df['tag'].dropna().unique().tolist()
    for tag in daftar_tag:
        if st.button(tag):
            st.session_state.history['tag'].append(tag)
            hasil = df[df['tag'] == tag]
            st.write(f"Menampilkan {len(hasil)} game dengan tag '{tag}':")
            st.dataframe(hasil)

# === Halaman Rekomendasi Kategori ===
elif halaman == "Rekomendasi Kategori":
    st.title("📂 Rekomendasi Berdasarkan Kategori")
    st.write("10 Game Rekomendasi Umum Berdasarkan Kategori")
    st.dataframe(df.sample(10))

    st.subheader("Pilih Kategori:")
    daftar_kategori = df['category'].dropna().unique().tolist()
    for kat in daftar_kategori:
        if st.button(kat):
            st.session_state.history['category'].append(kat)
            hasil = df[df['category'] == kat]
            st.write(f"Menampilkan {len(hasil)} game dengan kategori '{kat}':")
            st.dataframe(hasil)
