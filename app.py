import streamlit as st
import pandas as pd
import pickle
import requests
import zipfile
import io

# URL GitHub untuk dataset dan model
DATASET_URL = "https://github.com/Skripsiade123/Skripsi/blob/main/Dataset.zip?raw=true"
SVM_GENRE_URL = "https://github.com/Skripsiade123/Skripsi/blob/main/svm_model.pkl?raw=true"
VECTORIZER_GENRE_URL = "https://github.com/Skripsiade123/Skripsi/blob/main/tfidf_vectorizer.pkl?raw=true"

SVM_TAG_URL = "https://github.com/Skripsiade123/Skripsi/blob/main/svm_model_tags.pkl?raw=true"
VECTORIZER_TAG_URL = "https://github.com/Skripsiade123/Skripsi/blob/main/tfidf_vectorizer_tags.pkl?raw=true"

SVM_CATEGORY_URL = "https://github.com/Skripsiade123/Skripsi/blob/main/svm_model_categories.pkl?raw=true"
VECTORIZER_CATEGORY_URL = "https://github.com/Skripsiade123/Skripsi/blob/main/tfidf_vectorizer_categories.pkl?raw=true"

# Fungsi untuk mengunduh dan membaca dataset dari ZIP
@st.cache
def load_data():
    response = requests.get(DATASET_URL)
    with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
        zip_ref.extractall("data")
    return pd.read_csv("data/Dataset.csv")

# Fungsi untuk memuat model SVM dan vectorizer berdasarkan kategori
@st.cache
def load_models():
    def download_model(url, filename):
        response = requests.get(url)
        with open(filename, "wb") as file:
            file.write(response.content)
        with open(filename, "rb") as file:
            return pickle.load(file)

    svm_genre = download_model(SVM_GENRE_URL, "svm_model.pkl")
    vectorizer_genre = download_model(VECTORIZER_GENRE_URL, "tfidf_vectorizer.pkl")

    svm_tag = download_model(SVM_TAG_URL, "svm_model_tags.pkl")
    vectorizer_tag = download_model(VECTORIZER_TAG_URL, "tfidf_vectorizer_tags.pkl")

    svm_category = download_model(SVM_CATEGORY_URL, "svm_model_categories.pkl")
    vectorizer_category = download_model(VECTORIZER_CATEGORY_URL, "tfidf_vectorizer_categories.pkl")

    return (svm_genre, vectorizer_genre), (svm_tag, vectorizer_tag), (svm_category, vectorizer_category)

# Muat data dan model
df = load_data()
(genre_model, genre_vectorizer), (tag_model, tag_vectorizer), (category_model, category_vectorizer) = load_models()

# Navigasi halaman
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Penjelasan Metode", "Beranda", "Rekomendasi Genre", "Rekomendasi Tag & Kategori", "Histori"])

# Halaman Penjelasan Metode
if page == "Penjelasan Metode":
    st.title("Penjelasan Metode")
    st.write("Website ini menggunakan Content-Based Filtering dengan SVM untuk memberikan rekomendasi game berdasarkan preferensi pengguna.")

# Halaman Beranda
elif page == "Beranda":
    st.title("Beranda - Rekomendasi Game")
    st.write("Berikut adalah 10 game yang direkomendasikan:")
    st.dataframe(df.sample(10))

# Halaman Rekomendasi Genre
elif page == "Rekomendasi Genre":
    st.title("Rekomendasi Berdasarkan Genre")
    genre = st.selectbox("Pilih Genre", df['genre'].unique())
    genre_features = genre_vectorizer.transform([genre])
    recommended_indices = genre_model.predict(genre_features)[:10]
    recommended_games = df.iloc[recommended_indices]
    st.dataframe(recommended_games)

# Halaman Rekomendasi Tag & Kategori
elif page == "Rekomendasi Tag & Kategori":
    st.title("Rekomendasi Berdasarkan Tag & Kategori")
    tag = st.selectbox("Pilih Tag", df['tags'].unique())
    category = st.selectbox("Pilih Kategori", df['category'].unique())

    tag_features = tag_vectorizer.transform([tag])
    category_features = category_vectorizer.transform([category])

    recommended_tag_indices = tag_model.predict(tag_features)[:10]
    recommended_category_indices = category_model.predict(category_features)[:10]

    recommended_games = df.iloc[recommended_tag_indices].append(df.iloc[recommended_category_indices]).drop_duplicates().head(10)
    st.dataframe(recommended_games)

# Halaman Histori
elif page == "Histori":
    st.title("Histori Pengguna")
    st.write("Riwayat game yang pernah dipilih pengguna akan ditampilkan di sini.")
    user_history = df.sample(10)
    st.dataframe(user_history)

    st.write("Berdasarkan histori pengguna, berikut rekomendasi tambahan:")
    st.dataframe(df.sample(10))  # Bisa dikembangkan lebih lanjut dengan model historis

