import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import linear_kernel

# Load Dataset from GitHub
CSV_URL = "https://raw.githubusercontent.com/Skripsiade123/Skripsi/main/Dataset.csv"

@st.cache
def load_data():
    try:
        return pd.read_csv(CSV_URL)
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Load Models & Vectorizers
@st.cache
def load_models_and_vectorizers():
    models, vectorizers = {}, {}
    try:
        for model_name in ['svm', 'svm_categories', 'svm_tags']:
            with open(f"{model_name}.pkl", "rb") as f:
                models[model_name] = pickle.load(f)
        for vectorizer_name in ['tfidf_vectorizer', 'tfidf_vectorizer_categories', 'tfidf_vectorizer_tags']:
            with open(f"{vectorizer_name}.pkl", "rb") as f:
                vectorizers[vectorizer_name] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading models or vectorizers: {e}")
    return models, vectorizers

df = load_data()
models, vectorizers = load_models_and_vectorizers()

# Setup Session State
if "history" not in st.session_state:
    st.session_state.history = []

# Recommendation function
def recommend(text, vectorizer, model, top_n=10):
    tfidf = vectorizer.transform([text])
    sim = linear_kernel(tfidf, model)
    top_indices = sim[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# Sidebar & Pages
page = st.sidebar.selectbox("Navigasi", [
    "Penjelasan Metode",
    "Beranda",
    "Rekomendasi Genre",
    "Rekomendasi Tag",
    "Rekomendasi Kategori",
    "Histori"
])

def display_recommendations(recs):
    for _, row in recs.iterrows():
        st.markdown(f"### {row['Name']}")
        if 'Header Image' in row and pd.notna(row['Header Image']):
            st.image(row['Header Image'], width=300)
        if 'Short Description' in row and pd.notna(row['Short Description']):
            st.caption(row['Short Description'])
        if st.button(f"Tambahkan ke histori - {row['Name']}"):
            st.session_state.history.append(f"{row['Genre']} {row['Tags']} {row['Categories']}")

if page == "Beranda":
    st.title("Rekomendasi Game untuk Anda")
    if st.session_state.history:
        st.subheader("Berdasarkan histori Anda")
        combined_text = " ".join(st.session_state.history)
        recs = recommend(combined_text, vectorizers['tfidf_vectorizer'], models['svm'])
    else:
        st.subheader("10 Game Terpopuler")
        recs = df.sample(10)
    
    display_recommendations(recs)
