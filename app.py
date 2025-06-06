import streamlit as st
import pandas as pd
import zipfile
import io
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# ====== Fungsi untuk membaca Dataset dari zip ======
@st.cache_data
def load_dataset_from_zip(zip_path='Dataset.zip', csv_name='Dataset.csv'):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    return df

# ====== Load dataset ======
df = load_dataset_from_zip()

# ====== Load semua vectorizer dan model ======
def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

vectorizer_genre = load_pickle('tfidf_vectorizer.pkl')
vectorizer_tags = load_pickle('tfidf_vectorizer_tags.pkl')
vectorizer_categories = load_pickle('tfidf_vectorizer_categories.pkl')

model_genre = load_pickle('svm_model.pkl')
model_tags = load_pickle('svm_model_tags.pkl')
model_categories = load_pickle('svm_model_categories.pkl')

# ====== Sidebar & Judul ======
st.sidebar.title("Dashboard")
st.sidebar.markdown("""
- Penjelasan Metode  
- **Beranda**  
- Rekomendasi Genre  
- Rekomendasi Tag  
- Rekomendasi Kategori  
- Histori  
""")

st.title("Sistem Rekomendasi Game dengan SVM & Content-Based Filtering")

# ====== Input teks dari pengguna ======
user_input = st.text_area("Masukkan deskripsi/game yang Anda sukai (bebas):", "")

if user_input:
    # ====== Prediksi Genre ======
    input_vec_genre = vectorizer_genre.transform([user_input])
    pred_genre = model_genre.predict(input_vec_genre)[0]

    # ====== Prediksi Tags ======
    input_vec_tags = vectorizer_tags.transform([user_input])
    pred_tags = model_tags.predict(input_vec_tags)[0]

    # ====== Prediksi Categories ======
    input_vec_categories = vectorizer_categories.transform([user_input])
    pred_categories = model_categories.predict(input_vec_categories)[0]

    # ====== Tampilkan hasil prediksi ======
    st.success("**Hasil Prediksi Berdasarkan Input Anda:**")
    st.write(f"🎮 **Genre:** {pred_genre}")
    st.write(f"🏷️ **Tags:** {pred_tags}")
    st.write(f"📦 **Category:** {pred_categories}")

    # ====== Filter data dan tampilkan rekomendasi ======
    filtered = df[
        df['genres'].str.contains(pred_genre, na=False) &
        df['tags'].str.contains(pred_tags, na=False) &
        df['categories'].str.contains(pred_categories, na=False)
    ]

    st.subheader("🎯 Rekomendasi Game untuk Anda")
    if filtered.empty:
        st.warning("Tidak ditemukan game yang cocok.")
    else:
        for _, row in filtered.iterrows():
            st.markdown(f"### {row['name']}")
            st.write(row.get('short_description', 'Tidak ada deskripsi.'))
            st.write(f"**Genre:** {row.get('genres', '-')}")
            st.write(f"**Tags:** {row.get('tags', '-')}")
            st.write(f"**Categories:** {row.get('categories', '-')}")
            st.markdown("---")

else:
    st.info("Silakan masukkan teks terlebih dahulu untuk melihat rekomendasi.")
