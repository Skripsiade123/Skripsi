import streamlit as st
import pandas as pd
import os
import json
import zipfile
import requests
import io
import joblib
from datetime import datetime
from sklearn.metrics.pairwise import linear_kernel

# Fungsi untuk membersihkan dan memformat data
def clean_and_format_data(df):
    df.rename(columns=lambda x: x.strip(), inplace=True)
    if 'Game Name' in df.columns:
        df.rename(columns={'Game Name': 'Name'}, inplace=True)
    for col in ['Tags', 'Categories', 'Genre']:
        df[col] = df[col].fillna('')
        df[col] = df[col].apply(lambda x: x.split(','))
        df[col] = df[col].apply(lambda x: [i.strip() for i in x])
    return df

# Fungsi untuk merekomendasikan game berdasarkan indeks
def recommend_games(df, cosine_sim, index):
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # ambil 10 teratas kecuali dirinya sendiri
    game_indices = [i[0] for i in sim_scores]
    return df.iloc[game_indices]

# Fungsi untuk menyimpan histori ke file
HISTORY_FILE = "history.json"

def load_saved_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history_entry(entry):
    history = load_saved_history()
    history.insert(0, entry)
    if len(history) > 20:
        history = history[:20]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

# Fungsi untuk rekomendasi dari histori
def recommend_from_history(df, history_entries):
    game_names = []
    for entry in history_entries:
        game_names.extend(entry.get("recommended", []))
    seen = set()
    game_names = [x for x in game_names if not (x in seen or seen.add(x))]
    return df[df['Name'].isin(game_names)].head(10)

# Fungsi untuk mengunduh dan membaca CSV dari file ZIP di GitHub
zip_url = "https://github.com/Skripsiade123/Skripsi/raw/main/Dataset.zip"

def load_csv_from_github_zip(zip_url):
    response = requests.get(zip_url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = z.namelist()[0]  # Asumsikan hanya ada satu file CSV
        with z.open(csv_filename) as f:
            df = pd.read_csv(f)
    return df

# Load data dan bersihkan
df = load_csv_from_github_zip(zip_url)
df = clean_and_format_data(df)

# Load model dan vectorizer dari GitHub
@st.cache_resource
def load_model_and_vectorizer(feature_type):
    base_url = "https://github.com/Skripsiade123/Skripsi/raw/main/"
    if feature_type == "Genre":
        model_file = "svm_model.pkl"
        vec_file = "tfidf_vectorizer.pkl"
    elif feature_type == "Tag":
        model_file = "svm_model_tags.pkl"
        vec_file = "tfidf_vectorizer_tags.pkl"
    elif feature_type == "Kategori":
        model_file = "svm_model_categories.pkl"
        vec_file = "tfidf_vectorizer_categories.pkl"
    else:
        return None, None

    model = joblib.load(io.BytesIO(requests.get(base_url + model_file).content))
    vectorizer = joblib.load(io.BytesIO(requests.get(base_url + vec_file).content))
    return model, vectorizer

# Streamlit App
st.sidebar.title("Navigasi")
pages = ["Penjelasan Metode", "Dashboard", "Rekomendasi Genre", "Rekomendasi Tag", "Rekomendasi Kategori", "Histori"]
page = st.sidebar.radio("Pilih Halaman", pages)

if 'history' not in st.session_state:
    st.session_state.history = []

if page == "Dashboard":
    st.title("\U0001F3AE Sistem Rekomendasi Game")

    saved_history = load_saved_history()
    if saved_history:
        st.subheader("Rekomendasi Berdasarkan Histori Anda")
        recommended_games = recommend_from_history(df, saved_history)
        if not recommended_games.empty:
            for _, row in recommended_games.iterrows():
                st.subheader(row['Name'])
                st.image(row['Header Image'], width=300)
                st.write(row['Short Description'])
                st.write(f"**Genre:** {', '.join(row['Genre'])}")
                st.write(f"**Tags:** {', '.join(row['Tags'])}")
                st.write(f"**Categories:** {', '.join(row['Categories'])}")
                st.markdown("---")
        else:
            st.info("Histori belum menghasilkan rekomendasi yang cocok.")
    else:
        st.info("Belum ada histori. Silakan eksplorasi game terlebih dahulu.")

elif page == "Rekomendasi Genre":
    st.title("Rekomendasi Berdasarkan Genre")
    genre_list = sorted(set([genre for sublist in df['Genre'] for genre in sublist]))
    selected_genre = st.selectbox("Pilih Genre", genre_list)

    df_genre = df[df['Genre'].apply(lambda x: selected_genre in x)].reset_index(drop=True)
    if not df_genre.empty:
        model, vectorizer = load_model_and_vectorizer("Genre")
        df_genre['features'] = df_genre['Tags'].apply(lambda x: ' '.join(x))
        tfidf_matrix = vectorizer.transform(df_genre['features'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        recommended = recommend_games(df_genre, cosine_sim, 0)

        for _, row in recommended.iterrows():
            st.subheader(row['Name'])
            st.image(row['Header Image'], width=300)
            st.write(row['Short Description'])
            st.write(f"**Tags:** {', '.join(row['Tags'])}")
            st.markdown("---")

        st.session_state.history.append(("Genre", selected_genre))
        save_history_entry({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Genre",
            "value": selected_genre,
            "recommended": recommended['Name'].tolist()
        })
    else:
        st.warning("Tidak ada game ditemukan untuk genre ini.")

elif page == "Rekomendasi Tag":
    st.title("Rekomendasi Berdasarkan Tag")
    tag_list = sorted(set([tag for sublist in df['Tags'] for tag in sublist]))
    selected_tag = st.selectbox("Pilih Tag", tag_list)

    df_tag = df[df['Tags'].apply(lambda x: selected_tag in x)].reset_index(drop=True)
    if not df_tag.empty:
        model, vectorizer = load_model_and_vectorizer("Tag")
        df_tag['features'] = df_tag['Categories'].apply(lambda x: ' '.join(x))
        tfidf_matrix = vectorizer.transform(df_tag['features'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        recommended = recommend_games(df_tag, cosine_sim, 0)

        for _, row in recommended.iterrows():
            st.subheader(row['Name'])
            st.image(row['Header Image'], width=300)
            st.write(row['Short Description'])
            st.write(f"**Categories:** {', '.join(row['Categories'])}")
            st.markdown("---")

        st.session_state.history.append(("Tag", selected_tag))
        save_history_entry({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Tag",
            "value": selected_tag,
            "recommended": recommended['Name'].tolist()
        })
    else:
        st.warning("Tidak ada game ditemukan untuk tag ini.")

elif page == "Rekomendasi Kategori":
    st.title("Rekomendasi Berdasarkan Kategori")
    cat_list = sorted(set([cat for sublist in df['Categories'] for cat in sublist]))
    selected_cat = st.selectbox("Pilih Kategori", cat_list)

    df_cat = df[df['Categories'].apply(lambda x: selected_cat in x)].reset_index(drop=True)
    if not df_cat.empty:
        model, vectorizer = load_model_and_vectorizer("Kategori")
        df_cat['features'] = df_cat['Tags'].apply(lambda x: ' '.join(x))
        tfidf_matrix = vectorizer.transform(df_cat['features'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        recommended = recommend_games(df_cat, cosine_sim, 0)

        for _, row in recommended.iterrows():
            st.subheader(row['Name'])
            st.image(row['Header Image'], width=300)
            st.write(row['Short Description'])
            st.write(f"**Tags:** {', '.join(row['Tags'])}")
            st.markdown("---")

        st.session_state.history.append(("Kategori", selected_cat))
        save_history_entry({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Kategori",
            "value": selected_cat,
            "recommended": recommended['Name'].tolist()
        })
    else:
        st.warning("Tidak ada game ditemukan untuk kategori ini.")

elif page == "Histori":
    st.title("Histori Rekomendasi")
    if st.session_state.history:
        for item in st.session_state.history:
            st.write(f"**{item[0]}:** {item[1]}")
    else:
        st.info("Belum ada histori rekomendasi.")

elif page == "Penjelasan Metode":
    st.title("Penjelasan Metode")
    st.write("""
        Aplikasi ini menggunakan pendekatan content-based filtering dengan algoritma Support Vector Machine (SVM).
        Fitur-fitur seperti Genre, Tags, dan Categories digunakan untuk merepresentasikan setiap game dalam bentuk
        vektor menggunakan TF-IDF. Model SVM digunakan untuk belajar dari representasi ini dan kemudian menghitung
        kemiripan antar game menggunakan cosine similarity.

        Dengan memilih salah satu Genre, Tag, atau Kategori, sistem akan memberikan 10 rekomendasi game
        yang paling mirip berdasarkan konten yang dipilih.
    """)
