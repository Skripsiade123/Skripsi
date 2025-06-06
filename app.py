import streamlit as st
import pandas as pd
import zipfile
import joblib
import random
from io import BytesIO

# ------------------------------
# LOAD DATASET FROM ZIP
# ------------------------------
@st.cache_data
def load_data():
    with zipfile.ZipFile("Dataset.zip", 'r') as zip_ref:
        with zip_ref.open("Dataset.csv") as file:
            df = pd.read_csv(file)
    return df

# ------------------------------
# LOAD MODELS & VECTORIZERS
# ------------------------------
@st.cache_resource
def load_models():
    model_genre = joblib.load("svm_model.pkl")
    model_tag = joblib.load("svm_model_tags.pkl")
    model_cat = joblib.load("svm_model_categories.pkl")

    vec_genre = joblib.load("tfidf_vectorizer.pkl")
    vec_tag = joblib.load("tfidf_vectorizer_tags.pkl")
    vec_cat = joblib.load("tfidf_vectorizer_categories.pkl")

    return model_genre, model_tag, model_cat, vec_genre, vec_tag, vec_cat

# ------------------------------
# PREDIKSI SEMUA DATA SEKALI SAJA
# ------------------------------
@st.cache_data
def predict_all(df, model, vectorizer):
    X = vectorizer.transform(df['short_description'].fillna(""))
    y_pred = model.predict(X)
    return y_pred

# ------------------------------
# TAMPILKAN GAME
# ------------------------------
def tampilkan_game(df):
    for _, row in df.iterrows():
        st.subheader(row['name'])
        st.caption(row.get('short_description', ''))
        st.markdown("---")

# ------------------------------
# MAIN APP
# ------------------------------
def main():
    st.set_page_config(page_title="Sistem Rekomendasi Game", layout="wide")
    menu = st.sidebar.radio("Navigasi", [
        "Beranda",
        "Penjelasan Metode",
        "Rekomendasi Genre",
        "Rekomendasi Tag",
        "Rekomendasi Kategori"
    ])

    df = load_data()
    model_genre, model_tag, model_cat, vec_genre, vec_tag, vec_cat = load_models()

    pred_genre = predict_all(df, model_genre, vec_genre)
    pred_tag = predict_all(df, model_tag, vec_tag)
    pred_cat = predict_all(df, model_cat, vec_cat)

    df['predicted_genre'] = pred_genre
    df['predicted_tag'] = pred_tag
    df['predicted_category'] = pred_cat

    if menu == "Beranda":
        st.title("🎮 Rekomendasi Game Steam (SVM Model)")
        st.markdown("Berikut 10 game yang direkomendasikan:")
        sampel = df.sample(10, random_state=42)
        tampilkan_game(sampel)

    elif menu == "Penjelasan Metode":
        st.title("📚 Penjelasan Metode")
        st.markdown("""
        Aplikasi ini menggunakan:

        - **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengubah deskripsi teks menjadi angka.
        - **SVM (Support Vector Machine)** untuk memprediksi genre, tag, dan kategori dari deskripsi game.
        - **Content-Based Filtering**: Sistem menyarankan game berdasarkan kesamaan konten (deskripsi).
        """)

    elif menu == "Rekomendasi Genre":
        st.title("🎯 Rekomendasi Berdasarkan Genre (Prediksi)")
        st.markdown("10 Rekomendasi Berdasarkan Prediksi Model:")
        sampel = df[df['predicted_genre'].notna()].sample(10, random_state=1)
        tampilkan_game(sampel)

        genre_list = sorted(df['genres'].dropna().str.split(';').explode().unique())
        selected = st.selectbox("Atau pilih genre:", genre_list)
        if selected:
            hasil = df[df['genres'].fillna('').str.contains(selected)]
            st.markdown(f"Ditemukan {len(hasil)} game dengan genre **{selected}**")
            tampilkan_game(hasil)

    elif menu == "Rekomendasi Tag":
        st.title("🏷️ Rekomendasi Berdasarkan Tag (Prediksi)")
        st.markdown("10 Rekomendasi Berdasarkan Prediksi Model:")
        sampel = df[df['predicted_tag'].notna()].sample(10, random_state=2)
        tampilkan_game(sampel)

        tag_list = sorted(df['tags'].dropna().str.split(';').explode().unique())
        selected = st.selectbox("Atau pilih tag:", tag_list)
        if selected:
            hasil = df[df['tags'].fillna('').str.contains(selected)]
            st.markdown(f"Ditemukan {len(hasil)} game dengan tag **{selected}**")
            tampilkan_game(hasil)

    elif menu == "Rekomendasi Kategori":
        st.title("📦 Rekomendasi Berdasarkan Kategori (Prediksi)")
        st.markdown("10 Rekomendasi Berdasarkan Prediksi Model:")
        sampel = df[df['predicted_category'].notna()].sample(10, random_state=3)
        tampilkan_game(sampel)

        cat_list = sorted(df['categories'].dropna().str.split(';').explode().unique())
        selected = st.selectbox("Atau pilih kategori:", cat_list)
        if selected:
            hasil = df[df['categories'].fillna('').str.contains(selected)]
            st.markdown(f"Ditemukan {len(hasil)} game dengan kategori **{selected}**")
            tampilkan_game(hasil)

if __name__ == "__main__":
    main()
