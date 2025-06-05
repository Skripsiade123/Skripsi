import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import zipfile
import os
import requests
import io
import random

# --- Configuration ---
DATASET_URL = "https://raw.githubusercontent.com/Skripsiade123/Skripsi/main/Dataset.zip"
DATA_DIR = "temp_data" # Use a temporary directory for extraction
os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_data
def download_and_extract_data(url, extract_to):
    st.write(f"Downloading data from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        # Use BytesIO to handle the zip content in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Print contents of the zip file for debugging
            st.info(f"Contents of the zip file: {z.namelist()}")
            # Assuming there's only one CSV or the relevant CSV is easily identifiable
            csv_files_in_zip = [name for name in z.namelist() if name.endswith('.csv')]

            if not csv_files_in_zip:
                st.error("No CSV file found inside the zip file.")
                return False, None
            
            # For simplicity, let's assume the first found CSV is the one we need.
            # If your zip contains multiple CSVs, you might need a more specific logic.
            csv_file_name_in_zip = csv_files_in_zip[0]
            
            z.extract(csv_file_name_in_zip, path=extract_to)
            st.success(f"Data downloaded and extracted successfully to {os.path.join(extract_to, csv_file_name_in_zip)}!")
            return True, os.path.join(extract_to, csv_file_name_in_zip) # Return full path to the extracted CSV
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading data: {e}")
        return False, None
    except zipfile.BadZipFile:
        st.error("Downloaded file is not a valid zip file.")
        return False, None
    except Exception as e:
        st.error(f"An unexpected error occurred during extraction: {e}")
        return False, None

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    success, csv_path = download_and_extract_data(DATASET_URL, DATA_DIR)
    if not success or csv_path is None:
        st.stop() # Stop if data extraction fails

    try:
        df = pd.read_csv(csv_path)
        st.write(f"Loaded {len(df)} games from {csv_path}")
    except Exception as e:
        st.error(f"Error loading CSV file from extracted path ({csv_path}): {e}")
        st.stop()

    # Fill NaN values in relevant columns with empty strings
    df['genres'] = df['genres'].fillna('')
    df['tags'] = df['tags'].fillna('')
    df['categories'] = df['categories'].fillna('')
    df['description'] = df['description'].fillna('')
    df['name'] = df['name'].fillna('Unknown Game')

    # Combine relevant text features into a single string for TF-IDF
    df['combined_features'] = df['genres'] + ' ' + df['tags'] + ' ' + df['categories'] + ' ' + df['description']

    # TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

    return df, tfidf_matrix, tfidf_vectorizer

df_games, tfidf_matrix_games, tfidf_vectorizer_games = load_and_preprocess_data()

# --- Content-Based Filtering Logic (SVM concept for similarity) ---
def get_recommendations_from_history(game_indices_history, top_n=10):
    if not game_indices_history:
        return pd.DataFrame() # Return empty DataFrame if no history

    # Get the average TF-IDF vector for the games in history
    history_vectors = tfidf_matrix_games[game_indices_history].mean(axis=0)
    history_vectors = history_vectors.reshape(1, -1) # Reshape to (1, n_features)

    # Calculate cosine similarity between the history vector and all game vectors
    cosine_similarities = linear_kernel(history_vectors, tfidf_matrix_games).flatten()

    # Sort games by similarity score
    sim_scores = list(enumerate(cosine_similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude games already in history and get the top N recommendations
    recommended_indices = []
    # Using a set for faster lookup of already seen game names
    seen_names = set(df_games.iloc[st.session_state.history_indices]['name'].tolist()) if st.session_state.history_indices else set()

    for i, score in sim_scores:
        if i not in game_indices_history and df_games.iloc[i]['name'] not in seen_names and len(recommended_indices) < top_n:
            recommended_indices.append(i)
        if len(recommended_indices) == top_n:
            break
            
    return df_games.iloc[recommended_indices] if recommended_indices else pd.DataFrame()


def get_recommendations_by_feature(feature_type, feature_value, top_n=10):
    filtered_games = df_games[df_games[feature_type].str.contains(feature_value, case=False, na=False, regex=False)]
    return filtered_games.sample(min(top_n, len(filtered_games)), random_state=42) if not filtered_games.empty else pd.DataFrame()


# --- Streamlit Session State for History ---
if 'history' not in st.session_state:
    st.session_state.history = [] # Stores game names
if 'history_indices' not in st.session_state:
    st.session_state.history_indices = [] # Stores game DataFrame indices

def add_to_history(game_name, game_index):
    # Ensure no duplicates in history
    if game_name not in st.session_state.history:
        st.session_state.history.append(game_name)
        st.session_state.history_indices.append(game_index)

# --- Streamlit Pages ---

def page_explanation():
    st.title("Halaman Penjelasan Metode")
    st.header("Metode Rekomendasi Game")

    st.markdown("""
    Aplikasi ini adalah sistem rekomendasi game yang dibuat menggunakan **Content-Based Filtering**.
    """)

    st.subheader("Apa itu Content-Based Filtering?")
    st.markdown("""
    Content-Based Filtering bekerja dengan merekomendasikan item yang serupa dengan yang disukai pengguna di masa lalu.
    Sistem ini menganalisis atribut (konten) dari item-item yang telah berinteraksi dengan pengguna
    dan kemudian merekomendasikan item baru yang memiliki atribut serupa.
    """)

    st.subheader("Algoritma yang Digunakan")
    st.markdown("""
    Dalam aplikasi ini, kami menggunakan kombinasi teknik untuk mencapai rekomendasi berbasis konten:
    * **TF-IDF (Term Frequency-Inverse Document Frequency):** Ini adalah teknik statistik yang digunakan untuk
        mengevaluasi seberapa penting sebuah kata dalam dokumen relatif terhadap koleksi dokumen yang besar.
        Kami menggunakan TF-IDF untuk mengubah deskripsi game, genre, tag, dan kategori menjadi representasi numerik
        (vektor fitur). Vektor ini menangkap esensi konten setiap game.
    * **Cosine Similarity (Konsep SVM untuk Kesamaan Fitur):** Setelah game diwakili sebagai vektor TF-IDF,
        kami menggunakan Cosine Similarity untuk mengukur tingkat kemiripan antara dua game atau antara
        profil preferensi pengguna (gabungan vektor game yang disukai) dengan semua game yang tersedia.
        Semakin tinggi nilai Cosine Similarity, semakin mirip dua game atau game dengan preferensi pengguna.
        Meskipun SVM (Support Vector Machine) biasanya adalah algoritma klasifikasi, dalam konteks ini,
        "konsep SVM" merujuk pada penggunaan ruang fitur yang dibangun (oleh TF-IDF) untuk menemukan item serupa
        berdasarkan kedekatan dalam ruang tersebut, mirip dengan bagaimana SVM memisahkan kelas-kelas
        dengan hyperplane optimal. Kami mengukur kemiripan antara vektor fitur, yang merupakan prinsip dasar
        untuk menemukan item serupa dalam ruang dimensi tinggi.

    """)

    st.subheader("Bagaimana Rekomendasi Bekerja di Aplikasi Ini?")
    st.markdown("""
    1.  **Ekstraksi Fitur:** Informasi tentang setiap game (genre, tag, kategori, deskripsi) dikumpulkan dan diubah menjadi
        vektor numerik menggunakan TF-IDF.
    2.  **Profil Pengguna:** Ketika Anda melihat game atau menambahkannya ke riwayat Anda, aplikasi membangun "profil"
        Anda berdasarkan konten game-game tersebut. Profil ini adalah rata-rata atau gabungan dari vektor fitur
        game yang telah Anda lihat.
    3.  **Penentuan Kesamaan:** Sistem kemudian menghitung seberapa mirip setiap game lain dalam database dengan
        profil pengguna Anda menggunakan Cosine Similarity.
    4.  **Rekomendasi:** Game dengan skor kesamaan tertinggi direkomendasikan kepada Anda.
    """)

def page_home():
    st.title("Beranda")

    st.markdown("Selamat datang di sistem rekomendasi game! Temukan game baru yang mungkin Anda sukai.")

    if st.session_state.history_indices:
        st.subheader("Rekomendasi Berdasarkan Riwayat Anda")
        recommended_games = get_recommendations_from_history(st.session_state.history_indices)
        if not recommended_games.empty:
            for i, row in recommended_games.iterrows():
                st.write(f"**{row['name']}**")
                st.markdown(f"Genre: {row['genres']}")
                st.markdown(f"Tags: {row['tags']}")
                st.markdown(f"Categories: {row['categories']}")
                if st.button(f"Lihat Detail {row['name']}", key=f"home_view_{row['name']}"):
                    # Find the exact index in the original DataFrame to add to history_indices
                    original_index = df_games[df_games['name'] == row['name']].index[0]
                    add_to_history(row['name'], original_index)
                    st.toast(f"Ditambahkan ke riwayat: {row['name']}")
                st.markdown("---")
        else:
            st.info("Tidak ada rekomendasi baru berdasarkan riwayat Anda. Coba lihat beberapa game terlebih dahulu.")
    else:
        st.subheader("10 Game Populer/Acak (Untuk memulai)")
        # Display 10 random games if no history
        random_games = df_games.sample(min(10, len(df_games)), random_state=42) # Use random_state for consistent random games
        for i, row in random_games.iterrows():
            st.write(f"**{row['name']}**")
            st.markdown(f"Genre: {row['genres']}")
            st.markdown(f"Tags: {row['tags']}")
            st.markdown(f"Categories: {row['categories']}")
            if st.button(f"Lihat Detail {row['name']}", key=f"initial_view_{row['name']}"):
                original_index = df_games[df_games['name'] == row['name']].index[0]
                add_to_history(row['name'], original_index)
                st.toast(f"Ditambahkan ke riwayat: {row['name']}")
            st.markdown("---")

def page_genre_recommendation():
    st.title("Rekomendasi Berdasarkan Genre")

    all_genres = set()
    # Ensure to handle potential empty strings from fillna()
    for genres_str in df_games['genres'].dropna():
        # Split by ';' and handle multiple spaces or empty strings after split
        for genre in genres_str.split(';'):
            stripped_genre = genre.strip()
            if stripped_genre: # Add only non-empty genres
                all_genres.add(stripped_genre)

    sorted_genres = sorted(list(all_genres))
    selected_genre = st.selectbox("Pilih Genre", [""] + sorted_genres)

    if selected_genre:
        st.subheader(f"10 Game dalam Genre: {selected_genre}")
        # Make sure to use regex=False if split by literal ';'
        recommended_games = get_recommendations_by_feature('genres', selected_genre, top_n=10)
        if not recommended_games.empty:
            for i, row in recommended_games.iterrows():
                st.write(f"**{row['name']}**")
                st.markdown(f"Genre: {row['genres']}")
                st.markdown(f"Tags: {row['tags']}")
                st.markdown(f"Categories: {row['categories']}")
                if st.button(f"Lihat Detail {row['name']}", key=f"genre_view_{row['name']}"):
                    original_index = df_games[df_games['name'] == row['name']].index[0]
                    add_to_history(row['name'], original_index)
                    st.toast(f"Ditambahkan ke riwayat: {row['name']}")
                st.markdown("---")
        else:
            st.info(f"Tidak ada game yang ditemukan untuk genre '{selected_genre}'.")

def page_tag_recommendation():
    st.title("Rekomendasi Berdasarkan Tag")

    all_tags = set()
    for tags_str in df_games['tags'].dropna():
        for tag in tags_str.split(';'):
            stripped_tag = tag.strip()
            if stripped_tag:
                all_tags.add(stripped_tag)

    sorted_tags = sorted(list(all_tags))
    selected_tag = st.selectbox("Pilih Tag", [""] + sorted_tags)

    if selected_tag:
        st.subheader(f"10 Game dengan Tag: {selected_tag}")
        recommended_games = get_recommendations_by_feature('tags', selected_tag, top_n=10)
        if not recommended_games.empty:
            for i, row in recommended_games.iterrows():
                st.write(f"**{row['name']}**")
                st.markdown(f"Genre: {row['genres']}")
                st.markdown(f"Tags: {row['tags']}")
                st.markdown(f"Categories: {row['categories']}")
                if st.button(f"Lihat Detail {row['name']}", key=f"tag_view_{row['name']}"):
                    original_index = df_games[df_games['name'] == row['name']].index[0]
                    add_to_history(row['name'], original_index)
                    st.toast(f"Ditambahkan ke riwayat: {row['name']}")
                st.markdown("---")
        else:
            st.info(f"Tidak ada game yang ditemukan untuk tag '{selected_tag}'.")

def page_category_recommendation():
    st.title("Rekomendasi Berdasarkan Kategori")

    all_categories = set()
    for categories_str in df_games['categories'].dropna():
        for category in categories_str.split(';'):
            stripped_category = category.strip()
            if stripped_category:
                all_categories.add(stripped_category)

    sorted_categories = sorted(list(all_categories))
    selected_category = st.selectbox("Pilih Kategori", [""] + sorted_categories)

    if selected_category:
        st.subheader(f"10 Game dalam Kategori: {selected_category}")
        recommended_games = get_recommendations_by_feature('categories', selected_category, top_n=10)
        if not recommended_games.empty:
            for i, row in recommended_games.iterrows():
                st.write(f"**{row['name']}**")
                st.markdown(f"Genre: {row['genres']}")
                st.markdown(f"Tags: {row['tags']}")
                st.markdown(f"Categories: {row['categories']}")
                if st.button(f"Lihat Detail {row['name']}", key=f"category_view_{row['name']}"):
                    original_index = df_games[df_games['name'] == row['name']].index[0]
                    add_to_history(row['name'], original_index)
                    st.toast(f"Ditambahkan ke riwayat: {row['name']}")
                st.markdown("---")
        else:
            st.info(f"Tidak ada game yang ditemukan untuk kategori '{selected_category}'.")


def page_history():
    st.title("Riwayat Penjelajahan")
    if st.session_state.history:
        st.write("Berikut adalah game yang telah Anda lihat:")
        for game_name in st.session_state.history:
            st.write(f"- {game_name}")
    else:
        st.info("Riwayat Anda kosong. Mulai jelajahi beberapa game!")

# --- Main App Navigation ---
st.sidebar.title("Navigasi")
page_selection = st.sidebar.radio(
    "Go to",
    ("Penjelasan Metode", "Beranda", "Rekomendasi Genre", "Rekomendasi Tag", "Rekomendasi Kategori", "Riwayat")
)

if page_selection == "Penjelasan Metode":
    page_explanation()
elif page_selection == "Beranda":
    page_home()
elif page_selection == "Rekomendasi Genre":
    page_genre_recommendation()
elif page_selection == "Rekomendasi Tag":
    page_tag_recommendation()
elif page_selection == "Rekomendasi Kategori":
    page_category_recommendation()
elif page_selection == "Riwayat":
    page_history()
