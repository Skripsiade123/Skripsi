import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import zipfile
import requests
from sklearn.metrics.pairwise import linear_kernel

# --- Unduh dan ekstrak dataset dari GitHub ---
@st.cache_data
def download_and_extract_zip(url, extract_to="dataset"):
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, "Dataset.zip")

    # Download ZIP file jika belum ada
    if not os.path.exists(zip_path):
        response = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(response.content)

    # Ekstrak hanya jika belum ada Dataset.csv
    csv_path = os.path.join(extract_to, "Dataset.csv")
    if not os.path.exists(csv_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

    return csv_path

# Gunakan RAW link dari GitHub
ZIP_URL = "https://github.com/Skripsiade123/Skripsi/raw/main/Dataset.zip"

# Unduh & ekstrak otomatis
csv_file_path = download_and_extract_zip(ZIP_URL)

# --- Load Dataset ---
@st.cache_data
def load_data():
    return pd.read_csv(csv_file_path)

df = load_data()

# --- Load Models & Vectorizers ---
@st.cache_resource
def load_models():
    models = {}
    for model_name in ['svm', 'svm_categories', 'svm_tags']:
        with open(f"{model_name}.pkl", "rb") as f:
            models[model_name] = pickle.load(f)
    return models

@st.cache_resource
def load_vectorizers():
    vectorizers = {}
    for vectorizer_name in ['tfidf_vectorizer', 'tfidf_vectorizer_categories', 'tfidf_vectorizer_tags']:
        with open(f"{vectorizer_name}.pkl", "rb") as f:
            vectorizers[vectorizer_name] = pickle.load(f)
    return vectorizers

models = load_models()
vectorizers = load_vectorizers()

# --- Setup Session State ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Recommendation Function ---
def recommend_by_input(text, vectorizer, model, top_n=10):
    tfidf = vectorizer.transform([text])
    sim = linear_kernel(tfidf, model)
    top_indices = sim[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# --- Sidebar ---
page = st.sidebar.selectbox("Navigasi", [
    "Penjelasan Metode",
    "Beranda",
    "Rekomendasi Genre",
    "Rekomendasi Tag",
    "Rekomendasi Kategori",
    "Histori"
])

# --- Pages ---
if page == "Penjelasan Metode":
    st.title("Penjelasan Metode")
    st.write("""
    Sistem ini menggunakan pendekatan **Content-Based Filtering** berbasis algoritma **Support Vector Machine (SVM)**.

    Langkah-langkah utama:
    1. Menggabungkan fitur teks seperti `Genre`, `Tags`, dan `Categories`.
    2. Menggunakan **TF-IDF Vectorizer** untuk mentransformasikan teks menjadi fitur numerik.
    3. Membangun model **SVM** untuk menghitung kemiripan antar game.
    4. Menyediakan rekomendasi berdasarkan input pengguna atau histori penggunaan sebelumnya.
    """)

elif page == "Beranda":
    st.title("Rekomendasi Game untuk Anda")
    if st.session_state.history:
        st.subheader("Berdasarkan histori Anda")
        combined_text = " ".join(st.session_state.history)
        recs = recommend_by_input(combined_text, vectorizers['tfidf_vectorizer'], models['svm'])
    else:
        st.subheader("10 Game Terpopuler")
        recs = df.sample(10)
    
    for _, row in recs.iterrows():
        st.markdown(f"### {row['Name']}")
        st.image(row['Header Image'], width=300)
        st.caption(row['Short Description'])
        if st.button(f"Tambahkan ke histori - {row['Name']}"):
            st.session_state.history.append(row['Genre'] + " " + str(row['Tags']) + " " + str(row['Categories']))

elif page == "Rekomendasi Genre":
    st.title("Rekomendasi Berdasarkan Genre")
    all_genres = df['Genre'].dropna().unique()
    pilihan = st.multiselect("Pilih Genre:", all_genres)
    if pilihan:
        input_text = " ".join(pilihan)
        recs = recommend_by_input(input_text, vectorizers['tfidf_vectorizer'], models['svm'])
        st.subheader("Rekomendasi:")
        st.write(recs[['Name', 'Genre']])

elif page == "Rekomendasi Tag":
    st.title("Rekomendasi Berdasarkan Tag")
    tag_sample = df['Tags'].dropna().sample(50).tolist()
    flat_tags = list(set(", ".join(tag_sample).split(", ")))
    pilihan = st.multiselect("Pilih Tag:", flat_tags)
    if pilihan:
        input_text = " ".join(pilihan)
        recs = recommend_by_input(input_text, vectorizers['tfidf_vectorizer_tags'], models['svm_tags'])
        st.subheader("Rekomendasi:")
        st.write(recs[['Name', 'Tags']])

elif page == "Rekomendasi Kategori":
    st.title("Rekomendasi Berdasarkan Kategori")
    cat_sample = df['Categories'].dropna().sample(50).tolist()
    flat_cat = list(set(", ".join(cat_sample).split(", ")))
    pilihan = st.multiselect("Pilih Kategori:", flat_cat)
    if pilihan:
        input_text = " ".join(pilihan)
        recs = recommend_by_input(input_text, vectorizers['tfidf_vectorizer_categories'], models['svm_categories'])
        st.subheader("Rekomendasi:")
        st.write(recs[['Name', 'Categories']])

elif page == "Histori":
    st.title("Histori Pengguna")
    if st.session_state.history:
        st.write("Histori input Anda:")
        st.write(st.session_state.history)
    else:
        st.write("Belum ada histori.")
