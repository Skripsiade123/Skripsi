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

# --- Konfigurasi Aplikasi ---
# URL GitHub untuk unduhan otomatis (akan menjadi fallback jika tidak ada unggahan manual)
# DATASET_URL = "https://raw.githubusercontent.com/Skripsiade123/Skripsi/main/Dataset.zip" # Tidak lagi digunakan langsung
# Direktori sementara untuk mengekstrak file dari ZIP
DATA_DIR = "temp_data_extraction"
os.makedirs(DATA_DIR, exist_ok=True) # Pastikan direktori ada

# --- Fungsi untuk Mengekstrak Data dari ZIP (baik dari URL atau unggahan) ---
# Fungsi ini sekarang akan menerima objek file (dari unggahan) atau BytesIO (dari unduhan URL)
@st.cache_data
def extract_data_from_zip(zip_file_obj, extract_to_path):
    st.info(f"Mengekstrak data dari file ZIP...")
    try:
        with zipfile.ZipFile(zip_file_obj) as z:
            st.info(f"Daftar file di dalam ZIP: {z.namelist()}")
            
            csv_files_in_zip = [name for name in z.namelist() if name.lower().endswith('.csv')]

            if not csv_files_in_zip:
                st.error("Tidak ada file CSV (.csv) yang ditemukan di dalam file ZIP. "
                         "Mohon pastikan ada file CSV di dalam zip Anda.")
                return False, None
            
            csv_file_name_in_zip = csv_files_in_zip[0]
            
            st.info(f"Mengekstrak file: {csv_file_name_in_zip}...")
            z.extract(csv_file_name_in_zip, path=extract_to_path)
            
            extracted_csv_full_path = os.path.join(extract_to_path, csv_file_name_in_zip)
            st.success(f"File CSV berhasil diekstrak ke: {extracted_csv_full_path}")
            return True, extracted_csv_full_path
    except zipfile.BadZipFile:
        st.error("File yang diunggah/diunduh bukan file ZIP yang valid. Mohon periksa file ZIP Anda.")
        return False, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat ekstraksi: {e}")
        return False, None

# --- Fungsi untuk Memuat dan Pra-proses Data Game ---
@st.cache_data
def load_and_preprocess_data(uploaded_file_obj=None):
    csv_file_path = None

    if uploaded_file_obj:
        # Jika ada file yang diunggah manual
        success, csv_file_path = extract_data_from_zip(uploaded_file_obj, DATA_DIR)
    else:
        # Jika tidak ada unggahan manual, coba unduh dari GitHub (fallback)
        st.info("Tidak ada file ZIP yang diunggah. Mencoba mengunduh Dataset.zip dari GitHub sebagai gantinya.")
        github_zip_url = "https://raw.githubusercontent.com/Skripsiade123/Skripsi/main/Dataset.zip"
        try:
            response = requests.get(github_zip_url, stream=True)
            response.raise_for_status()
            zip_in_memory = io.BytesIO(response.content)
            success, csv_file_path = extract_data_from_zip(zip_in_memory, DATA_DIR)
        except requests.exceptions.RequestException as e:
            st.error(f"Error saat mengunduh Dataset.zip dari GitHub: {e}. "
                     "Mohon periksa koneksi internet Anda atau URL GitHub.")
            st.stop()
        except Exception as e:
            st.error(f"Terjadi kesalahan tak terduga saat mengunduh dari GitHub: {e}")
            st.stop()


    if not success or csv_file_path is None:
        st.stop() # Hentikan aplikasi jika gagal mengunduh/ekstrak (baik manual maupun dari GitHub)

    try:
        df = pd.read_csv(csv_file_path)
        
        if df.empty:
            st.error("File CSV yang dimuat kosong. Mohon periksa konten dataset Anda.")
            st.stop()

        st.info(f"Kolom yang ditemukan di dataset: {', '.join(df.columns)}")

        required_columns = ['genres', 'tags', 'categories', 'description', 'name']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Kolom-kolom berikut tidak ditemukan di dataset Anda: {', '.join(missing_columns)}. "
                     "Mohon pastikan nama kolom di CSV Anda sesuai (case-sensitive) dengan yang dibutuhkan.")
            st.stop()

        st.success(f"Berhasil memuat {len(df)} game dari dataset.")
        
    except pd.errors.EmptyDataError:
        st.error("File CSV yang diekstrak kosong. Mohon periksa isi file CSV Anda.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat file CSV dari jalur diekstrak ({csv_file_path}): {e}. "
                 "Check if the CSV is well-formed or if column names are correct.")
        st.stop()

    df['genres'] = df['genres'].fillna('')
    df['tags'] = df['tags'].fillna('')
    df['categories'] = df['categories'].fillna('')
    df['description'] = df['description'].fillna('')
    df['name'] = df['name'].fillna('Unknown Game')

    df['combined_features'] = df['genres'] + ' ' + df['tags'] + ' ' + df['categories'] + ' ' + df['description']

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

    return df, tfidf_matrix, tfidf_vectorizer

# --- Content-Based Filtering Logic (SVM concept for similarity) ---
def get_recommendations_from_history(game_indices_history, top_n=10):
    if not game_indices_history:
        return pd.DataFrame()

    history_vectors = tfidf_matrix_games[game_indices_history].mean(axis=0)
    history_vectors = history_vectors.reshape(1, -1)

    cosine_similarities = linear_kernel(history_vectors, tfidf_matrix_games).flatten()

    sim_scores = list(enumerate(cosine_similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommended_indices = []
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
    st.session_state.history = []
if 'history_indices' not in st.session_state:
    st.session_state.history_indices = []

def add_to_history(game_name, game_index):
    if game_name not in st.session_state.history:
        st.session_state.history.append(game_name)
        st.session_state.history_indices.append(game_index)

# --- Fungsi untuk Menampilkan Input Unggahan Dataset ---
def dataset_input_section():
    st.sidebar.header("Pilih Sumber Dataset")
    
    uploaded_file = st.sidebar.file_uploader(
        "Unggah file Dataset.zip Anda",
        type=["zip"],
        help="Silakan unggah file ZIP yang berisi dataset CSV game Anda."
    )
    
    if uploaded_file is not None:
        st.session_state['uploaded_dataset'] = uploaded_file
        st.sidebar.success("File ZIP berhasil diunggah!")
    else:
        st.session_state['uploaded_dataset'] = None
        st.sidebar.info("Tidak ada file ZIP diunggah. Aplikasi akan mencoba mengunduh dataset dari GitHub.")

# Panggil fungsi input dataset di awal aplikasi
dataset_input_section()

# Memuat dan pra-proses data saat aplikasi dimulai, dengan mempertimbangkan unggahan
# st.session_state['uploaded_dataset'] akan dilewatkan ke load_and_preprocess_data
df_games, tfidf_matrix_games, tfidf_vectorizer_games = load_and_preprocess_data(st.session_state['uploaded_dataset'])


# --- Streamlit Pages ---

def page_explanation():
    st.title("Halaman Penjelasan Metode")
    st.header("Metode Rekomendasi Game")

    st.markdown("""
    Aplikasi ini adalah sistem rekomendasi game yang dibuat menggunakan **Content-Based Filtering**.
    """)

    st.subheader("Apa itu Content-Based Filtering?")
    st.markdown("""
    Content-Based Filtering bekerja dengan merekomendasikan item yang serupa dengan yang telah disukai pengguna di masa lalu.
    Sistem ini menganalisis atribut atau "konten" dari item-item yang telah berinteraksi dengan pengguna (misalnya, game yang dilihat)
    dan kemudian merekomendasikan item baru yang memiliki atribut atau konten serupa.
    """)

    st.subheader("Algoritma yang Digunakan")
    st.markdown("""
    Dalam aplikasi ini, kami menggunakan kombinasi teknik untuk mencapai rekomendasi berbasis konten:
    * **TF-IDF (Term Frequency-Inverse Document Frequency):** Ini adalah teknik statistik yang sangat umum digunakan dalam pemrosesan bahasa alami (NLP)
        untuk mengevaluasi seberapa penting sebuah kata dalam sebuah dokumen relatif terhadap koleksi dokumen yang besar.
        Di sini, kami menggunakan TF-IDF untuk mengubah deskripsi game, genre, tag, dan kategori menjadi representasi numerik
        (disebut "vektor fitur"). Vektor ini secara efektif menangkap esensi konten setiap game.
    * **Cosine Similarity (Konsep SVM untuk Kesamaan Fitur):** Setelah setiap game diwakili sebagai vektor TF-IDF,
        kami menggunakan Cosine Similarity untuk mengukur tingkat kemiripan antara dua game atau antara
        "profil preferensi" pengguna (yang dibuat dari gabungan vektor game yang telah dilihat pengguna) dengan semua game yang tersedia.
        Semakin tinggi nilai Cosine Similarity, semakin mirip dua game atau game dengan preferensi pengguna.
        Meskipun **SVM (Support Vector Machine)** biasanya adalah algoritma klasifikasi, dalam konteks ini,
        frasa "konsep SVM" merujuk pada prinsip dasar penggunaan ruang fitur yang dibangun (oleh TF-IDF) untuk menemukan item serupa
        berdasarkan kedekatan mereka dalam ruang tersebut, mirip dengan bagaimana SVM memisahkan kelas-kelas
        dengan hyperplane optimal. Kami mengukur kemiripan antara vektor fitur, yang merupakan prinsip inti
        untuk menemukan item yang relevan dalam ruang dimensi tinggi.

    """)

    st.subheader("Bagaimana Rekomendasi Bekerja di Aplikasi Ini?")
    st.markdown("""
    1.  **Ekstraksi Fitur:** Informasi relevan tentang setiap game (seperti genre, tag, kategori, dan deskripsi) dikumpulkan dan diubah menjadi
        representasi vektor numerik menggunakan TF-IDF.
    2.  **Profil Pengguna:** Ketika Anda melihat sebuah game (dan menambahkannya ke riwayat Anda), aplikasi membangun "profil" preferensi Anda.
        Profil ini adalah rata-rata atau gabungan dari vektor fitur game-game yang telah Anda lihat dan sukai.
    3.  **Penentuan Kesamaan:** Sistem kemudian menghitung seberapa mirip setiap game lain dalam database dengan
        profil pengguna Anda menggunakan metrik Cosine Similarity.
    4.  **Rekomendasi:** Game dengan skor kesamaan tertinggi (yang paling mirip dengan profil Anda) akan direkomendasikan kepada Anda.
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
                    original_index = df_games[df_games['name'] == row['name']].index[0]
                    add_to_history(row['name'], original_index)
                    st.toast(f"Ditambahkan ke riwayat: {row['name']}")
                st.markdown("---")
        else:
            st.info("Tidak ada rekomendasi baru berdasarkan riwayat Anda. Coba lihat beberapa game terlebih dahulu.")
    else:
        st.subheader("10 Game Populer/Acak (Untuk memulai)")
        random_games = df_games.sample(min(10, len(df_games)), random_state=42)
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
    for genres_str in df_games['genres'].dropna():
        for genre in genres_str.split(';'):
            stripped_genre = genre.strip()
            if stripped_genre:
                all_genres.add(stripped_genre)

    sorted_genres = sorted(list(all_genres))
    selected_genre = st.selectbox("Pilih Genre", [""] + sorted_genres)

    if selected_genre:
        st.subheader(f"10 Game dalam Genre: {selected_genre}")
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

# --- Navigasi Utama Aplikasi ---
st.sidebar.title("Navigasi")
page_selection = st.sidebar.radio(
    "Pergi ke Halaman",
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
