# app.py
import streamlit as st
import zipfile
import os
import pandas as pd
import joblib

# ===== Fungsi untuk load dataset dari zip =====
@st.cache_data

def load_data():
    with zipfile.ZipFile("Dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("data")
    df = pd.read_csv("data/Dataset.csv")
    return df

# ===== Fungsi load model =====
@st.cache_resource

def load_model():
    model_genre = joblib.load("svm_model.pkl")
    model_tags = joblib.load("svm_model_tags.pkl")
    model_categories = joblib.load("svm_model_categories.pkl")
    vec_genre = joblib.load("tfidf_vectorizer.pkl")
    vec_tags = joblib.load("tfidf_vectorizer_tags.pkl")
    vec_categories = joblib.load("tfidf_vectorizer_categories.pkl")
    return model_genre, model_tags, model_categories, vec_genre, vec_tags, vec_categories

# ===== Fungsi prediksi =====
def predict(df, model, vectorizer):
    X = vectorizer.transform(df['deskripsi'])
    pred = model.predict(X)
    df_result = df.copy()
    df_result['prediksi'] = pred
    return df_result

# ===== Halaman Beranda =====
def show_beranda(df):
    st.subheader("Rekomendasi Game Beranda")
    for _, row in df.head(10).iterrows():
        st.markdown(f"### {row['nama']}")
        st.image(row['gambar'])
        st.write(row['deskripsi'])
        st.caption(f"Genre: {row['genre']}")
        st.caption(f"Tags: {row['tags']}")
        st.caption(f"Kategori: {row['kategori']}")

# ===== Halaman Penjelasan Metode =====
def show_metode():
    st.title("Penjelasan Metode")
    st.write("""
    Website ini menggunakan pendekatan **Content-Based Filtering**.
    Deskripsi game dianalisis menggunakan **TF-IDF vectorizer** dan diklasifikasikan
    ke dalam genre, tag, dan kategori menggunakan **algoritma Support Vector Machine (SVM)**.
    Model telah dilatih sebelumnya untuk memberikan rekomendasi berdasarkan deskripsi konten game.
    """)

# ===== Halaman Rekomendasi =====
def show_rekomendasi(df, kolom, judul):
    st.title(f"Rekomendasi Game berdasarkan {judul.lower()}")
    semua_label = sorted(df['prediksi'].unique())
    selected = st.selectbox(f"Pilih {judul}", [None] + semua_label)
    if selected:
        filtered = df[df['prediksi'] == selected]
    else:
        filtered = df

    for _, row in filtered.head(10).iterrows():
        st.markdown(f"### {row['nama']}")
        st.image(row['gambar'])
        st.write(row['deskripsi'])
        st.caption(f"Genre: {row['genre']}")
        st.caption(f"Tags: {row['tags']}")
        st.caption(f"Kategori: {row['kategori']}")

# ===== Main App =====
def main():
    st.sidebar.title("Navigasi")
    menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Penjelasan Metode", "Rekomendasi Genre", "Rekomendasi Tag", "Rekomendasi Kategori"])

    df = load_data()
    model_genre, model_tags, model_categories, vec_genre, vec_tags, vec_categories = load_model()

    if menu == "Beranda":
        df_pred = predict(df, model_genre, vec_genre)
        show_beranda(df_pred)

    elif menu == "Penjelasan Metode":
        show_metode()

    elif menu == "Rekomendasi Genre":
        df_pred = predict(df, model_genre, vec_genre)
        show_rekomendasi(df_pred, 'genre', 'Genre')

    elif menu == "Rekomendasi Tag":
        df_pred = predict(df, model_tags, vec_tags)
        show_rekomendasi(df_pred, 'tags', 'Tag')

    elif menu == "Rekomendasi Kategori":
        df_pred = predict(df, model_categories, vec_categories)
        show_rekomendasi(df_pred, 'kategori', 'Kategori')

if __name__ == "__main__":
    main()
