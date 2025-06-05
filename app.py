import streamlit as st
import pandas as pd
import zipfile
import os
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# GANTI URL INI dengan link mentah file zip dari GitHub
GITHUB_ZIP_URL = "https://raw.githubusercontent.com/username/repo/main/Dataset.zip"

# Fungsi untuk unduh dan ekstrak dataset dari GitHub
@st.cache_data
def load_data():
    response = requests.get(GITHUB_ZIP_URL)
    zipfile_obj = zipfile.ZipFile(BytesIO(response.content))
    zipfile_obj.extractall("data")
    
    # GANTI NAMA FILE CSV SESUAI ISI ZIP
    df = pd.read_csv("Dataset.csv")
    
    # GANTI nama kolom jika berbeda
    df = df.rename(columns=lambda x: x.lower())
    df.fillna('', inplace=True)
    return df

# Siapkan model TF-IDF dan cosine similarity
@st.cache_data
def prepare_model(df, text_column='description'):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[text_column])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

# Ambil rekomendasi berdasarkan kemiripan
def get_recommendations(title, df, cosine_sim):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    game_indices = [i[0] for i in sim_scores]
    return df.iloc[game_indices]

# Simpan histori ke session
def save_history(game):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(game)

# Aplikasi Streamlit
def main():
    st.set_page_config(page_title="Game Recommendation App", layout="wide")

    df = load_data()
    tfidf, tfidf_matrix, cosine_sim = prepare_model(df, 'description')

    menu = ["Penjelasan", "Beranda", "Genre", "Tag", "Kategori", "Riwayat"]
    choice = st.sidebar.selectbox("Pilih Halaman", menu)

    if choice == "Penjelasan":
        st.title("📘 Penjelasan Metode")
        st.markdown("""
        Aplikasi ini menggunakan pendekatan **Content-Based Filtering** dengan algoritma **SVM (melalui kernel cosine similarity)** 
        menggunakan fitur **TF-IDF** dari kolom deskripsi game.
        
        Tujuannya adalah merekomendasikan game yang mirip berdasarkan konten (deskripsi), genre, tag, dan kategori.
        """)

    elif choice == "Beranda":
        st.title("🎮 Rekomendasi Game Berdasarkan Pilihan")
        selected_game = st.selectbox("Pilih game", df['title'].unique())
        if selected_game:
            save_history(selected_game)
            st.subheader(f"Rekomendasi mirip dengan: {selected_game}")
            recs = get_recommendations(selected_game, df, cosine_sim)
            st.dataframe(recs[['title', 'genre', 'tags', 'category']])

    elif choice == "Genre":
        st.title("🎯 Rekomendasi Berdasarkan Genre")
        genres = df['genre'].unique()
        selected_genre = st.selectbox("Pilih Genre", genres)
        filtered = df[df['genre'] == selected_genre].head(10)
        st.write(filtered[['title', 'genre']])

    elif choice == "Tag":
        st.title("🏷️ Rekomendasi Berdasarkan Tag")
        tags = df['tags'].unique()
        selected_tag = st.selectbox("Pilih Tag", tags)
        filtered = df[df['tags'] == selected_tag].head(10)
        st.write(filtered[['title', 'tags']])

    elif choice == "Kategori":
        st.title("📂 Rekomendasi Berdasarkan Kategori")
        categories = df['category'].unique()
        selected_category = st.selectbox("Pilih Kategori", categories)
        filtered = df[df['category'] == selected_category].head(10)
        st.write(filtered[['title', 'category']])

    elif choice == "Riwayat":
        st.title("📜 Riwayat Game yang Dilihat")
        history = st.session_state.get('history', [])
        if history:
            st.write(history)
        else:
            st.write("Belum ada riwayat.")

if __name__ == "__main__":
    main()
