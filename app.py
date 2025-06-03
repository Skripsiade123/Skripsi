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

# Ambil dataset hanya dari GitHub
github_url = "https://github.com/Skripsiade123/Skripsi/raw/main/Dataset.zip"
try:
    response = requests.get(github_url)
    response.raise_for_status()
    uploaded_zip = io.BytesIO(response.content)
except requests.exceptions.RequestException as e:
    st.error(f"Gagal mengambil file dari GitHub: {e}")
    uploaded_zip = None

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
                st.sidebar.title("Dashboard")
                page = st.sidebar.radio("Pilih Halaman", ["\U0001F4D8 Penjelasan Metode", "Beranda", "Rekomendasi Genre", "Rekomendasi Tag", "Rekomendasi Kategori", "Histori"])

                tfidf_vectorizer, svm_model = load_model(page)

                # Halaman Penjelasan Metode
                if page == "\U0001F4D8 Penjelasan Metode":
                    st.title("\U0001F4D8 Penjelasan Metode dan Algoritma")
                    st.markdown("""
                    Aplikasi ini menggunakan **Content-Based Filtering** untuk merekomendasikan game berdasarkan kemiripan kontennya. 
                    Berikut adalah metode dan algoritma yang digunakan:

                    ### 🔍 1. Preprocessing
                    - Data teks dari kolom `Genre`, `Tags`, dan `Categories` digabung menjadi satu kolom fitur.
                    - Teks dibersihkan dan dikonversi menjadi format numerik menggunakan **TF-IDF (Term Frequency - Inverse Document Frequency)**.

                    ### 🤖 2. Algoritma Klasifikasi
                    - Model menggunakan **Support Vector Machine (SVM)** dengan kernel linear.
                    - Model dilatih untuk mengenali game mana yang cocok untuk label tertentu (misalnya: `'Action'`, `'Indie'`, atau `'Single-player'`).
                    - Untuk setiap filter (genre/tag/kategori), model SVM memprediksi apakah game cocok (label = 1) atau tidak cocok (label = 0).

                    ### 🧠 3. Proses Rekomendasi
                    - Pengguna memilih filter seperti Genre atau Tag.
                    - Model SVM mengklasifikasikan game yang sesuai dengan filter tersebut.
                    - Game yang diklasifikasikan sebagai cocok akan ditampilkan sebagai rekomendasi.

                    ### 📂 4. Model dan Dataset
                    - Model dan vectorizer telah dilatih sebelumnya dan disimpan dalam file `.pkl`.
                    - Dataset diambil dari file ZIP langsung melalui GitHub agar aplikasi tetap ringan dan mudah diperbarui.
                    """)
                    st.success("Silakan gunakan sidebar di kiri untuk masuk ke halaman rekomendasi atau beranda.")

                # Halaman Beranda
                elif page == "Beranda":
                    st.title("🎮 Sistem Rekomendasi Game")
                    st.write("Berikut adalah beberapa game:")
                    for _, row in df.head(10).iterrows():
                        st.subheader(row['Name'])
                        st.image(row['Header Image'], width=300)
                        st.write(row['Short Description'])
                        st.write(f"**Genre:** {', '.join(row['Genre'])}")
                        st.write(f"**Tags:** {', '.join(row['Tags'])}")
                        st.write(f"**Categories:** {', '.join(row['Categories'])}")
                        st.markdown("---")

                # Rekomendasi Genre
                elif page == "Rekomendasi Genre":
                    TARGET_GENRE = 'Action'
                    selected_genre = st.selectbox("Pilih genre sebagai filter awal:", sorted(set(g for genres in df['Genre'] for g in genres)))
                    filtered_games = df_cleaned[df_cleaned['Genre'].apply(lambda x: selected_genre in x)]
                    if not filtered_games.empty:
                        tfidf_filtered = tfidf_vectorizer.transform(filtered_games['combined_features'])
                        filtered_games['Predicted'] = svm_model.predict(tfidf_filtered)
                        recommended = filtered_games[filtered_games['Predicted'] == 1].sample(n=min(10, len(filtered_games)), random_state=42)
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

                # Rekomendasi Tag
                elif page == "Rekomendasi Tag":
                    TARGET_TAG = 'Indie'
                    selected_tag = st.selectbox("Pilih tag sebagai filter awal:", sorted(set(t for tags in df['Tags'] for t in tags)))
                    filtered_games = df_cleaned[df_cleaned['Tags'].apply(lambda x: selected_tag in x)]
                    if not filtered_games.empty:
                        tfidf_filtered = tfidf_vectorizer.transform(filtered_games['combined_features'])
                        filtered_games['Predicted'] = svm_model.predict(tfidf_filtered)
                        recommended = filtered_games[filtered_games['Predicted'] == 1].sample(n=min(10, len(filtered_games)), random_state=42)
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

                # Rekomendasi Kategori
                elif page == "Rekomendasi Kategori":
                    TARGET_CAT = 'Single-player'
                    selected_cat = st.selectbox("Pilih kategori sebagai filter awal:", sorted(set(c for cats in df['Categories'] for c in cats)))
                    filtered_games = df_cleaned[df_cleaned['Categories'].apply(lambda x: selected_cat in x)]
                    if not filtered_games.empty:
                        tfidf_filtered = tfidf_vectorizer.transform(filtered_games['combined_features'])
                        filtered_games['Predicted'] = svm_model.predict(tfidf_filtered)
                        recommended = filtered_games[filtered_games['Predicted'] == 1].sample(n=min(10, len(filtered_games)), random_state=42)
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
                    st.title("\U0001F4DC Histori Rekomendasi")
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
    st.info("Silakan pastikan file Dataset.zip dari GitHub dapat diakses.")
