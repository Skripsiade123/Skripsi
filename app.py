import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Fungsi untuk membaca dataset langsung dari GitHub
def load_data():
    url = "https://github.com/Skripsiade123/Skripsi/raw/main/dataset.csv"
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.text))
    # Menghapus data duplikat
    df.drop_duplicates(inplace=True)
    return df

# Fungsi untuk mengunduh file dari GitHub
def download_file_from_github(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

# Fungsi untuk memuat model dan vectorizer
def load_model_and_vectorizer(model_url, vectorizer_url, model_filename, vectorizer_filename):
    download_file_from_github(model_url, model_filename)
    download_file_from_github(vectorizer_url, vectorizer_filename)
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_filename, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Fungsi untuk mendapatkan rekomendasi berdasarkan genre
def get_recommendations_by_genre(df, genre, model, vectorizer):
    genre_games = df[df['genre'] == genre]
    X = vectorizer.transform(genre_games['description'])
    y = genre_games['game_name']
    game_name = st.selectbox("Pilih game", y)
    game_index = y[y == game_name].index[0]
    prediction = model.predict(X[game_index])
    recommended_games = [df.iloc[i]['game_name'] for i in prediction if i != game_index][:10]
    return recommended_games

# Fungsi untuk mendapatkan rekomendasi berdasarkan tag
def get_recommendations_by_tag(df, tag, model, vectorizer):
    tag_games = df[df['tags'].str.contains(tag, case=False, na=False)]
    X = vectorizer.transform(tag_games['description'])
    y = tag_games['game_name']
    game_name = st.selectbox("Pilih game", y)
    game_index = y[y == game_name].index[0]
    prediction = model.predict(X[game_index])
    recommended_games = [df.iloc[i]['game_name'] for i in prediction if i != game_index][:10]
    return recommended_games

# Fungsi untuk mendapatkan rekomendasi berdasarkan kategori
def get_recommendations_by_category(df, category, model, vectorizer):
    category_games = df[df['category'] == category]
    X = vectorizer.transform(category_games['description'])
    y = category_games['game_name']
    game_name = st.selectbox("Pilih game", y)
    game_index = y[y == game_name].index[0]
    prediction = model.predict(X[game_index])
    recommended_games = [df.iloc[i]['game_name'] for i in prediction if i != game_index][:10]
    return recommended_games

# Fungsi untuk menampilkan halaman penjelasan metode
def show_method_explanation():
    st.title("Penjelasan Metode")
    st.write("""
    Aplikasi ini menggunakan metode content-based filtering dengan algoritma SVM (Support Vector Machine) untuk memberikan rekomendasi game. 
    Berikut adalah langkah-langkah yang digunakan:
    1. **Ekstraksi Fitur**: Menggunakan TF-IDF untuk mengubah deskripsi game menjadi vektor fitur.
    2. **Pembelajaran Model**: Melatih model SVM untuk mengklasifikasikan game berdasarkan fitur yang diekstraksi.
    3. **Rekomendasi**: Menggunakan model yang dilatih untuk memberikan rekomendasi game berdasarkan genre, tag, atau kategori yang dipilih pengguna.
    """)

# Fungsi untuk menampilkan halaman beranda
def show_home(df):
    st.title("Beranda")
    st.write("Berikut adalah 10 game yang direkomendasikan:")
    recommended_games = df['game_name'].sample(10).tolist()
    for game in recommended_games:
        st.write(game)

# Fungsi untuk menampilkan halaman rekomendasi genre
def show_genre_recommendations(df, model, vectorizer):
    st.title("Rekomendasi Berdasarkan Genre")
    genres = df['genre'].unique()
    selected_genre = st.selectbox("Pilih genre", genres)
    if st.button("Dapatkan Rekomendasi"):
        recommended_games = get_recommendations_by_genre(df, selected_genre, model, vectorizer)
        st.write("Rekomendasi game untuk genre", selected_genre, "adalah:")
        for game in recommended_games:
            st.write(game)

# Fungsi untuk menampilkan halaman rekomendasi tag
def show_tag_recommendations(df, model, vectorizer):
    st.title("Rekomendasi Berdasarkan Tag")
    tags = df['tags'].unique()
    selected_tag = st.selectbox("Pilih tag", tags)
    if st.button("Dapatkan Rekomendasi"):
        recommended_games = get_recommendations_by_tag(df, selected_tag, model, vectorizer)
        st.write("Rekomendasi game untuk tag", selected_tag, "adalah:")
        for game in recommended_games:
            st.write(game)

# Fungsi untuk menampilkan halaman rekomendasi kategori
def show_category_recommendations(df, model, vectorizer):
    st.title("Rekomendasi Berdasarkan Kategori")
    categories = df['category'].unique()
    selected_category = st.selectbox("Pilih kategori", categories)
    if st.button("Dapatkan Rekomendasi"):
        recommended_games = get_recommendations_by_category(df, selected_category, model, vectorizer)
        st.write("Rekomendasi game untuk kategori", selected_category, "adalah:")
        for game in recommended_games:
            st.write(game)

# Fungsi untuk menampilkan halaman histori
def show_history():
    st.title("Histori Pengguna")
    if 'history' not in st.session_state:
        st.session_state.history = []
    history = st.session_state.history
    for item in history:
        st.write(item)

# Fungsi utama
def main():
    df = load_data()

    # Muat model dan vectorizer untuk genre
    genre_model_url = "https://github.com/Skripsiade123/Skripsi/raw/main/svm_model.pkl"
    genre_vectorizer_url = "https://github.com/Skripsiade123/Skripsi/raw/main/tfidf_vectorizer.pkl"
    genre_model, genre_vectorizer = load_model_and_vectorizer(genre_model_url, genre_vectorizer_url, "svm_model.pkl", "tfidf_vectorizer.pkl")

    # Muat model dan vectorizer untuk tag
    tag_model_url = "https://github.com/Skripsiade123/Skripsi/raw/main/svm_model_tags.pkl"
    tag_vectorizer_url = "https://github.com/Skripsiade123/Skripsi/raw/main/tfidf_vectorizer_tags.pkl"
    tag_model, tag_vectorizer = load_model_and_vectorizer(tag_model_url, tag_vectorizer_url, "svm_model_tags.pkl", "tfidf_vectorizer_tags.pkl")

    # Muat model dan vectorizer untuk kategori
    category_model_url = "https://github.com/Skripsiade123/Skripsi/raw/main/svm_model_categories.pkl"
    category_vectorizer_url = "https://github.com/Skripsiade123/Skripsi/raw/main/tfidf_vectorizer_categories.pkl"
    category_model, category_vectorizer = load_model_and_vectorizer(category_model_url, category_vectorizer_url, "svm_model_categories.pkl", "tfidf_vectorizer_categories.pkl")

    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox("Pilih Halaman", ["Penjelasan Metode", "Beranda", "Rekomendasi Genre", "Rekomendasi Tag", "Rekomendasi Kategori", "Histori"])

    if page == "Penjelasan Metode":
        show_method_explanation()
    elif page == "Beranda":
        show_home(df)
    elif page == "Rekomendasi Genre":
        show_genre_recommendations(df, genre_model, genre_vectorizer)
    elif page == "Rekomendasi Tag":
        show_tag_recommendations(df, tag_model, tag_vectorizer)
    elif page == "Rekomendasi Kategori":
        show_category_recommendations(df, category_model, category_vectorizer)
    elif page == "Histori":
        show_history()

if __name__ == "__main__":
    main()
