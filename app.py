import streamlit as st
import pandas as pd
import zipfile
import joblib
import requests
import io
from datetime import datetime

st.set_page_config(page_title="Sistem Rekomendasi Game", layout="wide")

# Inisialisasi session state untuk histori
if 'history' not in st.session_state:
    st.session_state.history = []

# Load model dan vectorizer yang telah dilatih berdasarkan halaman
@st.cache_resource
def load_model(page):
    if page == "Rekomendasi Genre":
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        svm_model = joblib.load('svm_model.pkl')
    elif page == "Rekomendasi Tag":
        tfidf_vectorizer = joblib.load('tfidf_vectorizer_tags.pkl')
        svm_model = joblib.load('svm_model_tags.pkl')
    elif page == "Rekomendasi Kategori":
        tfidf_vectorizer = joblib.load('tfidf_vectorizer_categories.pkl')
        svm_model = joblib.load('svm_model_categories.pkl')
    else:
        tfidf_vectorizer = svm_model = None
    return tfidf_vectorizer, svm_model

# Sidebar untuk memilih sumber dataset
st.sidebar.subheader("Sumber Dataset")
use_github = st.sidebar.checkbox("Gunakan dataset dari GitHub")

if use_github:
    github_url = "https://github.com/Skripsiade123/Skripsi/raw/main/Dataset.zip"  # Ganti sesuai URL kamu
    try:
        response = requests.get(github_url)
        response.raise_for_status()
        uploaded_zip = io.BytesIO(response.content)
        st.success("Berhasil mengambil Dataset.zip dari GitHub.")
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal mengambil file dari GitHub: {e}")
        uploaded_zip = None
else:
    uploaded_zip = st.file_uploader("Upload file ZIP yang berisi Dataset.csv", type="zip")

if uploaded_zip is not None:
    try:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            if "Dataset.csv" in zip_ref.namelist():
                with zip_ref.open('Dataset.csv') as csv_file:
                    df = pd.read_csv(csv_file)

                # Preprocessing
                df['Genre'] = df['Genre'].fillna("").apply(lambda x: [i.strip() for i in x.split(",")])
                df['Tags'] = df['Tags'].fillna("").apply(lambda x: [i.strip().strip('"') for i in x.split(",")])
                df['Categories'] = df['Categories'].fillna("").apply(lambda x: [i.strip() for i in x.split(",")])

                df['combined_features'] = df['Genre'].apply(lambda x: ' '.join(x)) + ' ' + \
                                          df['Tags'].apply(lambda x: ' '.join(x)) + ' ' + \
                                          df['Categories'].apply(lambda x: ' '.join(x))

                df_cleaned = df.copy()

                # Sidebar
                st.sidebar.title("Navigasi")
                page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Rekomendasi Genre", "Rekomendasi Tag", "Rekomendasi Kategori", "Histori"])

                tfidf_vectorizer, svm_model = load_model(page)

                # Halaman Beranda
                if page == "Beranda":
                    st.title("🎮 Sistem Rekomendasi Game")
                    st.write("Berikut adalah beberapa game dari Steam:")
                    for _, row in df.head(10).iterrows():  # Batasi hanya 10 game
                        st.subheader(row['Name'])
                        st.image(row['Header Image'], width=300)
                        st.write(row['Short Description'])
                        st.write(f"**Genre:** {', '.join(row['Genre'])}")
                        st.write(f"**Tags:** {', '.join(row['Tags'])}")
                        st.write(f"**Categories:** {', '.join(row['Categories'])}")
                        st.markdown("---")

                # Rekomendasi Berdasarkan Genre
                elif page == "Rekomendasi Genre":
                    TARGET_GENRE = 'Action'
                    selected_genre = st.selectbox("Pilih genre sebagai filter awal:", sorted(set(g for genres in df['Genre'] for g in genres)))
                    filtered_games = df_cleaned[df_cleaned['Genre'].apply(lambda x: selected_genre in x)]
                    if not filtered_games.empty:
                        tfidf_filtered = tfidf_vectorizer.transform(filtered_games['combined_features'])
                        filtered_games['Predicted'] = svm_model.predict(tfidf_filtered)
                        recommended = filtered_games[filtered_games['Predicted'] == 1]
                        recommended = recommended.sample(n=min(10, len(recommended)), random_state=42)
                        st.subheader(f"Rekomendasi Game berdasarkan genre '{selected_genre}' dan validasi model untuk target '{TARGET_GENRE}':")
                        for _, row in recommended.iterrows():
                            st.subheader(row['Name'])
                            st.image(row['Header Image'], width=300)
                            st.write(row['Short Description'])
                            st.write(f"**Genre:** {', '.join(row['Genre'])}")
                            st.write(f"**Tags:** {', '.join(row['Tags'])}")
                            st.write(f"**Categories:** {', '.join(row['Categories'])}")
                            st.markdown("---")
                        st.session_state.history.append((datetime.now(), "Genre", selected_genre, recommended['Name'].tolist()))
                    else:
                        st.warning("Tidak ada game yang cocok.")

                # Rekomendasi Berdasarkan Tag
                elif page == "Rekomendasi Tag":
                    TARGET_TAG = 'Indie'
                    selected_tag = st.selectbox("Pilih tag sebagai filter awal:", sorted(set(t for tags in df['Tags'] for t in tags)))
                    filtered_games = df_cleaned[df_cleaned['Tags'].apply(lambda x: selected_tag in x)]
                    if not filtered_games.empty:
                        tfidf_filtered = tfidf_vectorizer.transform(filtered_games['combined_features'])
                        filtered_games['Predicted'] = svm_model.predict(tfidf_filtered)
                        recommended = filtered_games[filtered_games['Predicted'] == 1]
                        recommended = recommended.sample(n=min(10, len(recommended)), random_state=42)
                        st.subheader(f"Rekomendasi Game berdasarkan tag '{selected_tag}' dan validasi model untuk target '{TARGET_TAG}':")
                        for _, row in recommended.iterrows():
                            st.subheader(row['Name'])
                            st.image(row['Header Image'], width=300)
                            st.write(row['Short Description'])
                            st.write(f"**Genre:** {', '.join(row['Genre'])}")
                            st.write(f"**Tags:** {', '.join(row['Tags'])}")
                            st.write(f"**Categories:** {', '.join(row['Categories'])}")
                            st.markdown("---")
                        st.session_state.history.append((datetime.now(), "Tag", selected_tag, recommended['Name'].tolist()))
                    else:
                        st.warning("Tidak ada game yang cocok.")

                # Rekomendasi Berdasarkan Kategori
                elif page == "Rekomendasi Kategori":
                    TARGET_CAT = 'Single-player'
                    selected_cat = st.selectbox("Pilih kategori sebagai filter awal:", sorted(set(c for cats in df['Categories'] for c in cats)))
                    filtered_games = df_cleaned[df_cleaned['Categories'].apply(lambda x: selected_cat in x)]
                    if not filtered_games.empty:
                        tfidf_filtered = tfidf_vectorizer.transform(filtered_games['combined_features'])
                        filtered_games['Predicted'] = svm_model.predict(tfidf_filtered)
                        recommended = filtered_games[filtered_games['Predicted'] == 1]
                        recommended = recommended.sample(n=min(10, len(recommended)), random_state=42)
                        st.subheader(f"Rekomendasi Game berdasarkan kategori '{selected_cat}' dan validasi model untuk target '{TARGET_CAT}':")
                        for _, row in recommended.iterrows():
                            st.subheader(row['Name'])
                            st.image(row['Header Image'], width=300)
                            st.write(row['Short Description'])
                            st.write(f"**Genre:** {', '.join(row['Genre'])}")
                            st.write(f"**Tags:** {', '.join(row['Tags'])}")
                            st.write(f"**Categories:** {', '.join(row['Categories'])}")
                            st.markdown("---")
                        st.session_state.history.append((datetime.now(), "Kategori", selected_cat, recommended['Name'].tolist()))
                    else:
                        st.warning("Tidak ada game yang cocok.")

                # Halaman Histori
                elif page == "Histori":
                    st.title("📜 Histori Rekomendasi")
                    if st.session_state.history:
                        for waktu, tipe, nilai, hasil in reversed(st.session_state.history):
                            st.markdown(f"**{waktu.strftime('%Y-%m-%d %H:%M:%S')}** - Rekomendasi berdasarkan **{tipe}** '{nilai}':")
                            for nama in hasil:
                                st.write(f"- {nama}")
                            st.markdown("---")
                    else:
                        st.info("Belum ada histori rekomendasi.")

            else:
                st.error("File 'Dataset.csv' tidak ditemukan di dalam ZIP.")
    except zipfile.BadZipFile:
        st.error("File ZIP tidak valid.")
else:
    st.info("Silakan upload file ZIP yang berisi Dataset.csv atau aktifkan opsi GitHub.")
