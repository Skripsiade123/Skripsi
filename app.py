import streamlit as st
import pandas as pd
import zipfile
import requests
from io import BytesIO
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# URL mentah file zip dan model dari GitHub
DATA_URL = "https://raw.githubusercontent.com/Skripsiade123/Skripsi/main/Dataset.zip"
MODEL_URLS = {
    "main": "https://raw.githubusercontent.com/Skripsiade123/Skripsi/main/svm_model.pkl",
    "tags": "https://raw.githubusercontent.com/Skripsiade123/Skripsi/main/svm_model_tags.pkl",
    "categories": "https://raw.githubusercontent.com/Skripsiade123/Skripsi/main/svm_model_categories.pkl",
}
VECTORIZER_URLS = {
    "main": "https://raw.githubusercontent.com/Skripsiade123/Skripsi/main/tfidf_vectorizer.pkl",
    "tags": "https://raw.githubusercontent.com/Skripsiade123/Skripsi/main/tfidf_vectorizer_tags.pkl",
    "categories": "https://raw.githubusercontent.com/Skripsiade123/Skripsi/main/tfidf_vectorizer_categories.pkl",
}

@st.cache_data
def download_dataset():
    response = requests.get(DATA_URL)
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall("data")
    df = pd.read_csv("data/Dataset.csv")
    df = df.rename(columns=lambda x: x.lower())
    df.fillna('', inplace=True)
    return df

@st.cache_resource
def load_model_and_vectorizer(model_url, vectorizer_url):
    model = pickle.load(requests.get(model_url, stream=True).raw)
    vectorizer = pickle.load(requests.get(vectorizer_url, stream=True).raw)
    return model, vectorizer

def get_recommendations_svm(input_text, model, vectorizer, df):
    vec_input = vectorizer.transform([input_text])
    distances = model.decision_function(vec_input @ vectorizer.transform(df['description']))
    top_indices = distances.argsort()[0][-10:][::-1]
    return df.iloc[top_indices]

def save_history(item):
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    st.session_state['history'].append(item)

# Streamlit UI
def main():
    st.set_page_config(page_title="Sistem Rekomendasi Game", layout="wide")
    df = download_dataset()
    
    page = st.sidebar.selectbox("Navigasi", ["Penjelasan", "Beranda", "Genre", "Tag", "Kategori", "Riwayat"])

    if page == "Penjelasan":
        st.title("📘 Penjelasan Metode")
        st.markdown("""
        Sistem ini menggunakan pendekatan **Content-Based Filtering** dengan algoritma **Support Vector Machine (SVM)**.
        
        - Fitur deskripsi diubah menjadi representasi numerik menggunakan **TF-IDF Vectorizer**.
        - Kemudian, model **SVM** digunakan untuk mencari kemiripan antara game.
        - Model berbeda digunakan untuk rekomendasi berdasarkan deskripsi, tag, dan kategori.
        """)

    elif page == "Beranda":
        st.title("🎮 Rekomendasi Game")
        selected = st.selectbox("Pilih game", df['title'].unique())
        if selected:
            save_history(selected)
            desc = df[df['title'] == selected]['description'].values[0]
            model, vectorizer = load_model_and_vectorizer(MODEL_URLS["main"], VECTORIZER_URLS["main"])
            results = get_recommendations_svm(desc, model, vectorizer, df)
            st.dataframe(results[['title', 'genre', 'tags', 'category']])

    elif page == "Genre":
        st.title("🎯 Berdasarkan Genre")
        selected = st.selectbox("Pilih Genre", df['genre'].unique())
        filtered = df[df['genre'] == selected]
        st.write(filtered[['title', 'genre']].head(10))

    elif page == "Tag":
        st.title("🏷️ Berdasarkan Tag")
        selected = st.selectbox("Pilih Tag", df['tags'].unique())
        filtered = df[df['tags'] == selected]
        if not filtered.empty:
            model, vectorizer = load_model_and_vectorizer(MODEL_URLS["tags"], VECTORIZER_URLS["tags"])
            input_text = filtered.iloc[0]['description']
            results = get_recommendations_svm(input_text, model, vectorizer, df)
            st.dataframe(results[['title', 'tags']])
        else:
            st.warning("Tag tidak ditemukan.")

    elif page == "Kategori":
        st.title("📂 Berdasarkan Kategori")
        selected = st.selectbox("Pilih Kategori", df['category'].unique())
        filtered = df[df['category'] == selected]
        if not filtered.empty:
            model, vectorizer = load_model_and_vectorizer(MODEL_URLS["categories"], VECTORIZER_URLS["categories"])
            input_text = filtered.iloc[0]['description']
            results = get_recommendations_svm(input_text, model, vectorizer, df)
            st.dataframe(results[['title', 'category']])
        else:
            st.warning("Kategori tidak ditemukan.")

    elif page == "Riwayat":
        st.title("📜 Riwayat Pengguna")
        history = st.session_state.get('history', [])
        if history:
            st.write(history)
        else:
            st.write("Belum ada riwayat yang disimpan.")

if __name__ == "__main__":
    main()
